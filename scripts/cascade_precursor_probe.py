# scripts/cascade_precursor_probe.py
"""Goal-A cascade-precursor feasibility (stage 1).

Question: does the cached data carry any cascade-precursor signal that a
learned representation could plausibly exploit?  If a logistic regression on
the existing 83-dim flat baseline cannot detect cascade onset above base rate
(lift > 1), neither will an encoder.

Two cascade labels:

  (a) **Synthetic cascade** (works on all 161 days): a window's "next-H period"
      is a cascade event if `|forward_log_return at H|` exceeds the 99th
      percentile of the symbol's rolling 5000-window distribution.  Strict
      per-symbol rolling, NEVER global (gotcha #4).

  (b) **Real cascade** (April 1-13 only — `cause` field exists from April 1
      onward; pre-April raw data has no cause column per CLAUDE.md): a window's
      next-H period is a real cascade event if any fill in
      `(anchor_ts, ts_at(anchor + H)]` has `cause` in
      `{market_liquidation, backstop_liquidation}`.  Read raw trade parquet
      (not cache) for the cause column.

Validation: on April 1-13, what fraction of real-cascade-positive windows are
also synthetic-cascade-positive at the same H?  If the overlap is ≥ ~60%, the
synthetic label is a defensible proxy for real cascades and we can use it on
the 161-day cache.  If overlap < 30%, synthetic label is measuring something
else (just-volatile-windows, not forced liquidations).

Hard constraints:
  * Skip any shard with date >= 2026-04-14 (April hold-out, gotcha #17).
  * Per-symbol rolling 5000-window cutoff (no global statistics, gotcha #4).
  * Walk-forward train/test by month (Oct-Jan train, Feb / Mar test folds);
    embargo is implicitly satisfied by the >2-week gap between months
    (>> 600 events at any reasonable cadence; gotcha #12).
  * BatchNorm at inference is irrelevant here — flat features + LR, no encoder.

Outputs (written to docs/experiments/goal-a-feasibility/):
  * cascade_precursor_per_window.parquet (gitignored)
  * cascade_precursor_table.csv
  * cascade_precursor.md  (markdown verdict)

Usage:
    uv run python scripts/cascade_precursor_probe.py \\
        --cache data/cache \\
        --out-dir docs/experiments/goal-a-feasibility \\
        --horizons 10 50 100 500
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tape.constants import (
    APRIL_HELDOUT_START,
    APRIL_START,
    DIRECTION_HORIZONS,
    FEATURE_NAMES,
    STRIDE_EVAL,
    SYMBOLS,
    WINDOW_LEN,
)
from tape.flat_features import FLAT_DIM, extract_flat_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT: Path = Path("data/cache")
OUT_DIR_DEFAULT: Path = Path("docs/experiments/goal-a-feasibility")

TRAIN_MONTHS: tuple[str, ...] = ("2025-10", "2025-11", "2025-12", "2026-01")
TEST_MONTHS: tuple[str, ...] = ("2026-02", "2026-03")

# Cause field: April 1+ only (CLAUDE.md gotcha #3, schema). Diagnostic-only window
# is April 1-13 (April 14+ is hard hold-out, gotcha #17).
CAUSE_DIAGNOSTIC_START: str = APRIL_START  # 2026-04-01
CAUSE_DIAGNOSTIC_END_INCLUSIVE: str = "2026-04-13"  # one day before hold-out

# Synthetic cascade label parameters
ROLLING_WINDOW_FOR_LABEL: int = 5000  # per-symbol causal rolling window
SYNTHETIC_CASCADE_QUANTILE: float = 0.99  # 99th percentile cutoff
LABEL_MIN_PERIODS: int = 1000  # min prior events before we trust the cutoff

# Cost band — taker, both legs (matches v1 protocol)
TAKER_FEE_BPS_PER_SIDE: float = 4.0
DEFAULT_SLIP_BPS_PER_SIDE: float = 1.0  # placeholder when per-window slip unknown

# Index of log_return in the cached features tensor (col 0 per FEATURE_NAMES)
_LOG_RETURN_IDX: int = FEATURE_NAMES.index("log_return")

# Liquidation-cause set
_LIQ_CAUSES: tuple[str, ...] = ("market_liquidation", "backstop_liquidation")


# ---------------------------------------------------------------------------
# Helper utilities (tested in tests/scripts/test_cascade_precursor_probe.py)
# ---------------------------------------------------------------------------


def _rolling_quantile_causal(
    x: np.ndarray, *, window: int, q: float, min_periods: int
) -> np.ndarray:
    """Causal rolling q-th quantile.  NaN positions (insufficient history)
    are returned AS NaN — the caller decides whether to suppress firing.
    """
    s = pd.Series(x)
    return (
        s.rolling(window=window, min_periods=min_periods)
        .quantile(q)
        .to_numpy(dtype=float)
    )


def _synthetic_cascade_label(
    fwd_ret: np.ndarray,
    *,
    rolling_window: int,
    q: float,
    min_periods: int,
) -> np.ndarray:
    """Binary cascade label: |fwd_ret| > rolling 99th-percentile cutoff.

    The rolling cutoff is over `|fwd_ret|`, computed strictly causally
    (gotcha #4).  Positions before `min_periods` are labeled 0 (no firing
    until we have enough warmup).
    """
    abs_fwd = np.abs(np.asarray(fwd_ret, dtype=float))
    cutoff = _rolling_quantile_causal(
        abs_fwd, window=rolling_window, q=q, min_periods=min_periods
    )
    label = np.zeros(len(abs_fwd), dtype=np.int8)
    valid = np.isfinite(cutoff) & np.isfinite(abs_fwd)
    label[valid] = (abs_fwd[valid] > cutoff[valid]).astype(np.int8)
    return label


def _precision_recall_at_top_decile(
    proba: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """Top-decile precision, recall, and lift.

    Returns
    -------
    (precision, recall, lift)
        precision = (cascades in top 10%) / (top 10% size)
        recall    = (cascades in top 10%) / (total cascades)
        lift      = precision / base_rate ; lift = NaN if base_rate == 0

    All windows tied with the lowest top-decile probability are included on
    the cutoff side via `np.argsort` (deterministic by index).
    """
    n = len(proba)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    base_rate = float(labels.mean())
    n_top = max(1, n // 10)

    # Rank descending — top n_top are the most-cascade-likely
    order = np.argsort(-proba, kind="stable")
    top = order[:n_top]

    pos_in_top = float(labels[top].sum())
    total_pos = float(labels.sum())

    precision = pos_in_top / float(n_top)
    recall = pos_in_top / total_pos if total_pos > 0 else float("nan")
    lift = precision / base_rate if base_rate > 0 else float("nan")
    return precision, recall, lift


def _headroom_top_decile_bps(
    *,
    lift: float,
    mean_edge_in_cascade_bps: float,
    slip_bps: float,
    taker_fee_bps_per_side: float = TAKER_FEE_BPS_PER_SIDE,
) -> float:
    """Per-trade headroom in bps for the top-decile gating strategy.

    Per the task spec:
        gross_per_trade_bps = lift × E[|forward_return at H| | cascade] - cost_round_trip
        cost_round_trip = 2 × taker_fee_bps_per_side + 2 × |slip_bps|

    NOTE: this is an *upper bound* on tradeable headroom.  The cascade
    direction is not predicted — only its onset.  A real strategy would also
    need a directional sign.
    """
    cost_round_trip = 2.0 * taker_fee_bps_per_side + 2.0 * abs(slip_bps)
    gross_per_trade = lift * mean_edge_in_cascade_bps - cost_round_trip
    return float(gross_per_trade)


def _row_for_window(
    *,
    symbol: str,
    horizon: int,
    fold: str,
    anchor_ts: int,
    date: str,
    window_start: int,
    pred_proba: float,
    synthetic_cascade_label: int,
    real_cascade_label: int | None,
    top_decile_bool: bool,
    edge_bps: float,
    slip_bps: float,
) -> dict:
    """Build a single per-window output row with the schema the prompt requires."""
    return {
        "symbol": symbol,
        "horizon": int(horizon),
        "fold": fold,
        "date": date,
        "anchor_ts": int(anchor_ts),
        "window_start": int(window_start),
        "pred_proba": float(pred_proba),
        "synthetic_cascade_label": int(synthetic_cascade_label),
        "real_cascade_label": (
            int(real_cascade_label) if real_cascade_label is not None else np.nan
        ),
        "top_decile_bool": bool(top_decile_bool),
        "edge_bps": float(edge_bps),
        "slip_avg_bps": float(slip_bps),
    }


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------


def _shards_for_symbol_months(
    cache_dir: Path, symbol: str, months: tuple[str, ...]
) -> list[Path]:
    out: list[Path] = []
    for mp in months:
        out.extend(sorted(cache_dir.glob(f"{symbol}__{mp}-*.npz")))
    return out


def _shards_for_symbol_dates(
    cache_dir: Path, symbol: str, dates: Iterable[str]
) -> list[Path]:
    out: list[Path] = []
    for d in dates:
        p = cache_dir / f"{symbol}__{d}.npz"
        if p.exists():
            out.append(p)
    return out


def _is_april_diagnostic_date(date_str: str) -> bool:
    """True iff date is in [APRIL_START, CAUSE_DIAGNOSTIC_END_INCLUSIVE]."""
    return CAUSE_DIAGNOSTIC_START <= date_str <= CAUSE_DIAGNOSTIC_END_INCLUSIVE


def _load_shard(path: Path) -> dict:
    """Load all keys from an .npz shard; trip on April hold-out as a guard."""
    sym, date = path.stem.split("__")
    if date >= APRIL_HELDOUT_START:
        raise ValueError(
            f"Refusing to load hold-out shard {path} (gotcha #17 — date {date})"
        )
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Per-shard window builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBatch:
    """Per-shard batch: flat features (N, 83) + per-window metadata + per-horizon
    forward-return arrays.  Anchor ts and edge_bps come from the cached event_ts
    and log_return columns respectively.  The synthetic label is computed at
    aggregation time (it requires per-symbol rolling stats, not per-shard).
    """

    symbol: str
    date: str
    flat_X: np.ndarray  # (N, FLAT_DIM)
    anchor_ts: np.ndarray  # (N,) int64
    window_starts: np.ndarray  # (N,) int64
    fwd_ret_by_h: dict[int, np.ndarray]  # h -> (N,) float64; NaN if past end


def _build_window_batch_from_shard(
    shard_path: Path,
    *,
    horizons: tuple[int, ...],
    stride: int = STRIDE_EVAL,
) -> WindowBatch | None:
    """Materialize all stride-200 windows in a shard.

    For each window:
      * flat features: 83-dim summary vector via `extract_flat_features`.
      * anchor_ts = event_ts[start + WINDOW_LEN - 1]  (last event in window)
      * fwd_ret[h] = sum(log_return[anchor+1 .. anchor+h])  (NaN if past end)
    """
    sym, date = shard_path.stem.split("__")
    if date >= APRIL_HELDOUT_START:
        return None

    payload = _load_shard(shard_path)
    features: np.ndarray = payload["features"]
    event_ts: np.ndarray = payload["event_ts"]
    n_events = features.shape[0]
    if n_events < WINDOW_LEN:
        return None

    # Window starts: stride=200 from 0; respect WINDOW_LEN
    last_valid_start = n_events - WINDOW_LEN
    if last_valid_start < 0:
        return None
    starts = np.arange(0, last_valid_start + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        return None

    anchors = starts + WINDOW_LEN - 1
    anchor_ts = event_ts[anchors].astype(np.int64)

    # Per-event log_return is column 0 (FEATURE_NAMES order)
    log_returns = features[:, _LOG_RETURN_IDX].astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(log_returns)])

    fwd_by_h: dict[int, np.ndarray] = {}
    for h in horizons:
        end = anchors + h
        valid = end < n_events
        out = np.full(len(anchors), np.nan, dtype=np.float64)
        out[valid] = cum[end[valid] + 1] - cum[anchors[valid] + 1]
        fwd_by_h[h] = out

    # Flat features per window — vectorise per-window (cheap; 200×17 → 83)
    flat_X = np.empty((len(starts), FLAT_DIM), dtype=np.float32)
    for i, s in enumerate(starts):
        flat_X[i] = extract_flat_features(features[s : s + WINDOW_LEN])

    return WindowBatch(
        symbol=sym,
        date=date,
        flat_X=flat_X,
        anchor_ts=anchor_ts,
        window_starts=starts,
        fwd_ret_by_h=fwd_by_h,
    )


# ---------------------------------------------------------------------------
# Real cascade label: April 1-13 only, raw trade scan
# ---------------------------------------------------------------------------


def _load_april_liquidation_ts(symbol: str) -> dict[str, np.ndarray]:
    """Return a dict {date_str -> sorted int64 array of liquidation trade ts_ms}.

    Loads raw trade parquet for `symbol` over the April diagnostic window
    (April 1–13).  Filters to rows with `cause IN {market_liquidation,
    backstop_liquidation}`.  Skips dates with no parquet on disk silently.
    Pre-April data has no `cause` column (CLAUDE.md schema) — those dates are
    not loaded by this function.
    """
    out: dict[str, np.ndarray] = {}
    base = Path(f"data/trades/symbol={symbol}")
    if not base.exists():
        return out
    for date_dir in sorted(base.glob("date=2026-04-*")):
        date_str = date_dir.name.removeprefix("date=")
        if not _is_april_diagnostic_date(date_str):
            # April 14+ hold-out (gotcha #17) — skip as a defensive guard.
            continue
        # DuckDB query: filter to liquidation rows; cause column exists Apr 1+.
        q = (
            f"SELECT ts_ms FROM read_parquet('{date_dir}/*.parquet') "
            f"WHERE cause IN ('market_liquidation', 'backstop_liquidation') "
            f"ORDER BY ts_ms"
        )
        try:
            df = duckdb.query(q).to_df()
        except Exception:
            # If cause column doesn't exist (shouldn't happen for April 1+), skip
            continue
        out[date_str] = df["ts_ms"].to_numpy(dtype=np.int64)
    return out


def _real_cascade_label_for_batch(
    batch: WindowBatch,
    horizon: int,
    liq_ts_for_date: dict[str, np.ndarray],
) -> np.ndarray:
    """Per-window binary label: any liquidation trade in (anchor_ts, ts_at(anchor+h)]?

    `liq_ts_for_date[batch.date]` must be a sorted int64 array of liquidation
    timestamps for this (symbol, date).  If the date is missing, returns
    all-zeros for the batch.

    The right-edge timestamp `ts_at(anchor + h)` is recovered from the cached
    event_ts.  If anchor + h overruns the day, the window's label is 0 (we
    cannot validate against real cascades for that horizon at that anchor).
    """
    n = len(batch.anchor_ts)
    out = np.zeros(n, dtype=np.int8)
    liq_ts = liq_ts_for_date.get(batch.date)
    if liq_ts is None or len(liq_ts) == 0:
        return out

    # Reload event_ts for this shard to look up ts_at(anchor + h).
    # A cheaper version: store event_ts on the batch — done below in caller.
    raise NotImplementedError(
        "Use _real_cascade_label_with_event_ts; this stub keeps the API documented."
    )


def _real_cascade_label_with_event_ts(
    *,
    anchor_ts: np.ndarray,
    window_starts: np.ndarray,
    event_ts: np.ndarray,
    horizon: int,
    liq_ts: np.ndarray,
) -> np.ndarray:
    """Per-window binary real-cascade label.

    For each window, the cascade window in trade-time is
    `(anchor_ts, ts_at(anchor + horizon)]`.  If any element of `liq_ts` falls
    in that interval, label = 1.  Windows where anchor + horizon overruns the
    day → label = 0 (cannot validate).
    """
    n = len(anchor_ts)
    out = np.zeros(n, dtype=np.int8)
    if len(liq_ts) == 0:
        return out

    n_events = len(event_ts)
    anchor_idx = window_starts + WINDOW_LEN - 1  # int array
    end_idx = anchor_idx + horizon
    valid = end_idx < n_events
    if not valid.any():
        return out

    end_ts = event_ts[end_idx[valid]].astype(np.int64)
    start_ts = anchor_ts[valid].astype(np.int64)

    # For each valid window: any liq_ts in (start_ts, end_ts] ?
    # Use searchsorted on the sorted liq_ts array.  liq_ts assumed sorted.
    lo = np.searchsorted(liq_ts, start_ts, side="right")  # first idx > start_ts
    hi = np.searchsorted(liq_ts, end_ts, side="right")  # first idx > end_ts
    has_liq = (hi - lo) > 0
    out_valid = out[valid]
    out_valid[has_liq] = 1
    out[valid] = out_valid
    return out


# ---------------------------------------------------------------------------
# Per-symbol pipeline
# ---------------------------------------------------------------------------


def _train_test_for_symbol(
    cache_dir: Path,
    symbol: str,
    horizons: tuple[int, ...],
) -> tuple[list[WindowBatch], dict[str, list[WindowBatch]]] | None:
    """Build train batches (Oct-Jan) and per-fold test batches (Feb, Mar)."""
    train_paths = _shards_for_symbol_months(cache_dir, symbol, TRAIN_MONTHS)
    if not train_paths:
        return None
    train_batches: list[WindowBatch] = []
    for p in train_paths:
        b = _build_window_batch_from_shard(p, horizons=horizons)
        if b is not None:
            train_batches.append(b)
    if sum(b.flat_X.shape[0] for b in train_batches) < 500:
        return None

    test_batches: dict[str, list[WindowBatch]] = {}
    for month in TEST_MONTHS:
        sh = _shards_for_symbol_months(cache_dir, symbol, (month,))
        bs: list[WindowBatch] = []
        for p in sh:
            b = _build_window_batch_from_shard(p, horizons=horizons)
            if b is not None:
                bs.append(b)
        if bs:
            test_batches[month] = bs
    return train_batches, test_batches


def _stack_batches(
    batches: list[WindowBatch], horizon: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, np.ndarray, np.ndarray]]]:
    """Stack per-shard batches into (X, fwd_ret_at_h) plus per-shard meta tuple list.

    Returns
    -------
    X        : (N_total, FLAT_DIM)
    fwd_ret  : (N_total,) float64  (NaN where horizon overruns the day)
    meta     : list of (symbol, date, anchor_ts_arr, window_starts_arr)
    """
    Xs: list[np.ndarray] = []
    fws: list[np.ndarray] = []
    meta: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for b in batches:
        Xs.append(b.flat_X)
        fws.append(b.fwd_ret_by_h[horizon])
        meta.append((b.symbol, b.date, b.anchor_ts, b.window_starts))
    if not Xs:
        return (
            np.zeros((0, FLAT_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            meta,
        )
    return np.concatenate(Xs, axis=0), np.concatenate(fws, axis=0), meta


def _label_synthetic_for_train_and_test(
    train_fwd: np.ndarray, test_fwd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute synthetic cascade labels using a SINGLE per-symbol rolling cutoff
    that walks across train then test.  This keeps the cutoff strictly causal
    (gotcha #4): at any point in test, the cutoff was computed only from prior
    (train + earlier-test) windows.

    Returns
    -------
    train_label, test_label, full_abs_fwd
        Each is concatenated arrays.  full_abs_fwd is exposed for diagnostics.
    """
    train_n = len(train_fwd)
    full = np.concatenate([train_fwd, test_fwd])
    label_full = _synthetic_cascade_label(
        full,
        rolling_window=ROLLING_WINDOW_FOR_LABEL,
        q=SYNTHETIC_CASCADE_QUANTILE,
        min_periods=LABEL_MIN_PERIODS,
    )
    return label_full[:train_n], label_full[train_n:], np.abs(full)


def _fit_lr_on_train(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray
) -> tuple[np.ndarray, float | None]:
    """Fit LogisticRegression(class_weight='balanced') on train, return p_cascade.

    If ytr is degenerate (all 0 or all 1), return constant probability + AUC=None.
    """
    if len(np.unique(ytr)) < 2 or len(Xtr) == 0:
        return np.full(len(Xte), float(ytr.mean()) if len(ytr) > 0 else 0.0), None
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(
        C=1.0, max_iter=1_000, class_weight="balanced", solver="lbfgs"
    ).fit(scaler.transform(Xtr), ytr)
    Xte_s = scaler.transform(Xte)
    proba = lr.predict_proba(Xte_s)
    classes = list(lr.classes_)
    pos_idx = classes.index(1) if 1 in classes else (1 if proba.shape[1] > 1 else 0)
    return proba[:, pos_idx].astype(np.float64), None


def _per_cell_metrics(
    proba: np.ndarray,
    label: np.ndarray,
    fwd_ret: np.ndarray,
    slip_bps: float = DEFAULT_SLIP_BPS_PER_SIDE,
) -> dict[str, float]:
    """Compute precision/recall/lift/AUC and headroom math for one cell."""
    n = len(proba)
    if n == 0:
        return {
            "n": 0.0,
            "p_cascade": float("nan"),
            "precision_top_decile": float("nan"),
            "recall_top_decile": float("nan"),
            "lift": float("nan"),
            "auc": float("nan"),
            "mean_edge_in_cascade_bps": float("nan"),
            "headroom_top_decile_bps": float("nan"),
        }
    base_rate = float(label.mean())
    precision, recall, lift = _precision_recall_at_top_decile(proba, label)

    # AUC: requires both classes present
    if len(np.unique(label)) >= 2:
        try:
            auc = float(roc_auc_score(label, proba))
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    # Mean |fwd_ret| on cascade-positive windows, in bps
    cascade_mask = label.astype(bool) & np.isfinite(fwd_ret)
    if cascade_mask.any():
        mean_edge_bps = float(np.mean(np.abs(fwd_ret[cascade_mask])) * 1e4)
    else:
        mean_edge_bps = float("nan")

    if math.isfinite(lift) and math.isfinite(mean_edge_bps):
        headroom = _headroom_top_decile_bps(
            lift=lift,
            mean_edge_in_cascade_bps=mean_edge_bps,
            slip_bps=slip_bps,
        )
    else:
        headroom = float("nan")

    return {
        "n": float(n),
        "p_cascade": base_rate,
        "precision_top_decile": float(precision),
        "recall_top_decile": float(recall),
        "lift": float(lift),
        "auc": auc,
        "mean_edge_in_cascade_bps": mean_edge_bps,
        "headroom_top_decile_bps": headroom,
    }


# ---------------------------------------------------------------------------
# April 1-13 real-cascade overlap diagnostic
# ---------------------------------------------------------------------------


def _april_real_vs_synthetic_overlap(
    cache_dir: Path,
    symbols: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Per-(symbol, horizon) overlap between real and synthetic cascade labels
    on April 1-13.

    For each symbol-day in April 1-13:
      * Build all stride-200 windows from the cached shard.
      * For each window, label_real = any liquidation trade in the cascade
        window; label_synthetic = synthetic-cascade-positive at the same H.
      * Use a SYMBOL-LEVEL rolling cutoff for the synthetic label that walks
        across pre-April + April-diagnostic combined (gotcha #4 — strict
        per-symbol causal).

    Returns
    -------
    DataFrame with columns
        symbol, horizon, n_real_pos, n_syn_pos, n_overlap,
        overlap_real_in_syn, overlap_syn_in_real
    """
    rows: list[dict] = []

    for symbol in symbols:
        # Real liquidation timestamps per April-diagnostic date
        liq_ts_for_date = _load_april_liquidation_ts(symbol)
        if not liq_ts_for_date:
            for h in horizons:
                rows.append(
                    {
                        "symbol": symbol,
                        "horizon": h,
                        "n_real_pos": 0,
                        "n_syn_pos": 0,
                        "n_overlap": 0,
                        "overlap_real_in_syn": float("nan"),
                        "overlap_syn_in_real": float("nan"),
                    }
                )
            continue

        # All shards from start through April 13 (inclusive).
        # We need pre-April + April-diagnostic shards together so the rolling
        # cutoff is warmed up by the time April starts.
        shard_paths = sorted(cache_dir.glob(f"{symbol}__*.npz"))
        shard_paths = [
            p
            for p in shard_paths
            if p.stem.split("__")[1] <= CAUSE_DIAGNOSTIC_END_INCLUSIVE
        ]
        if not shard_paths:
            continue

        # Build batches in date order
        all_batches: list[WindowBatch] = []
        per_shard_event_ts: list[np.ndarray] = []
        per_shard_dates: list[str] = []
        for p in shard_paths:
            b = _build_window_batch_from_shard(p, horizons=horizons)
            if b is None:
                continue
            payload = _load_shard(p)
            all_batches.append(b)
            per_shard_event_ts.append(payload["event_ts"].astype(np.int64))
            per_shard_dates.append(b.date)

        if not all_batches:
            continue

        # Per-horizon: compute synthetic label using FULL series rolling cutoff
        for h in horizons:
            # Concatenate fwd_ret across all batches in date order
            fwd_full = np.concatenate([b.fwd_ret_by_h[h] for b in all_batches])
            syn_label_full = _synthetic_cascade_label(
                fwd_full,
                rolling_window=ROLLING_WINDOW_FOR_LABEL,
                q=SYNTHETIC_CASCADE_QUANTILE,
                min_periods=LABEL_MIN_PERIODS,
            )

            # Now walk batch-by-batch to compute real labels on April-diagnostic
            # batches only, comparing them against the corresponding slice of
            # syn_label_full.
            offset = 0
            real_pos_set: list[bool] = []
            syn_pos_set: list[bool] = []
            for b, ev_ts in zip(all_batches, per_shard_event_ts):
                n_b = len(b.anchor_ts)
                if not _is_april_diagnostic_date(b.date):
                    offset += n_b
                    continue
                liq_ts = liq_ts_for_date.get(b.date, np.zeros(0, dtype=np.int64))
                real_label = _real_cascade_label_with_event_ts(
                    anchor_ts=b.anchor_ts,
                    window_starts=b.window_starts,
                    event_ts=ev_ts,
                    horizon=h,
                    liq_ts=liq_ts,
                )
                syn_slice = syn_label_full[offset : offset + n_b]
                # Drop windows with NaN forward return
                fwd_b = b.fwd_ret_by_h[h]
                valid = np.isfinite(fwd_b)
                real_pos_set.extend(real_label[valid].astype(bool).tolist())
                syn_pos_set.extend(syn_slice[valid].astype(bool).tolist())
                offset += n_b

            real_arr = np.array(real_pos_set, dtype=bool)
            syn_arr = np.array(syn_pos_set, dtype=bool)
            n_real_pos = int(real_arr.sum())
            n_syn_pos = int(syn_arr.sum())
            n_overlap = int((real_arr & syn_arr).sum())
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": h,
                    "n_real_pos": n_real_pos,
                    "n_syn_pos": n_syn_pos,
                    "n_overlap": n_overlap,
                    "overlap_real_in_syn": (
                        n_overlap / n_real_pos if n_real_pos > 0 else float("nan")
                    ),
                    "overlap_syn_in_real": (
                        n_overlap / n_syn_pos if n_syn_pos > 0 else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _run_symbol(
    cache_dir: Path,
    symbol: str,
    horizons: tuple[int, ...],
    apr_liq_ts: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Run the cascade-precursor pipeline for one symbol.

    Returns (per_window_rows, per_cell_rows).
    """
    bundle = _train_test_for_symbol(cache_dir, symbol, horizons)
    if bundle is None:
        return [], []
    train_batches, test_batches = bundle

    per_window_rows: list[dict] = []
    per_cell_rows: list[dict] = []

    for h in horizons:
        Xtr, fwd_tr, _ = _stack_batches(train_batches, h)
        # Drop windows where forward return is NaN (horizon overruns day)
        valid_tr = np.isfinite(fwd_tr)
        if valid_tr.sum() < 500:
            continue
        Xtr = Xtr[valid_tr]
        fwd_tr = fwd_tr[valid_tr]

        for fold_name, fold_batches in test_batches.items():
            Xte, fwd_te, meta_te = _stack_batches(fold_batches, h)
            valid_te = np.isfinite(fwd_te)
            if valid_te.sum() < 50:
                continue

            # Re-stack metadata aligned to the test rows
            te_dates_l: list[str] = []
            te_anchors_l: list[int] = []
            te_starts_l: list[int] = []
            for sym_meta, date_meta, ats, ws in meta_te:
                te_dates_l.extend([date_meta] * len(ats))
                te_anchors_l.extend(ats.tolist())
                te_starts_l.extend(ws.tolist())
            te_dates_arr = np.array(te_dates_l)
            te_anchors_arr = np.array(te_anchors_l, dtype=np.int64)
            te_starts_arr = np.array(te_starts_l, dtype=np.int64)

            # Trim test to valid only
            Xte_v = Xte[valid_te]
            fwd_te_v = fwd_te[valid_te]
            te_dates_v = te_dates_arr[valid_te]
            te_anchors_v = te_anchors_arr[valid_te]
            te_starts_v = te_starts_arr[valid_te]

            # Synthetic labels: use combined train+test rolling cutoff so the
            # cutoff at any test position uses only PRIOR windows (causal).
            ytr_syn, yte_syn, _ = _label_synthetic_for_train_and_test(fwd_tr, fwd_te_v)

            if ytr_syn.sum() < 5:
                # Train fold has too few cascades to fit a meaningful LR
                continue

            # Fit LR on train, predict probabilities on test
            p_cascade, _ = _fit_lr_on_train(Xtr, ytr_syn.astype(np.int64), Xte_v)

            # Top-decile flag (ranking determined per cell)
            n_te = len(p_cascade)
            n_top = max(1, n_te // 10)
            top_decile_bool = np.zeros(n_te, dtype=bool)
            order = np.argsort(-p_cascade, kind="stable")
            top_decile_bool[order[:n_top]] = True

            # Per-cell metrics on the synthetic label
            metrics = _per_cell_metrics(p_cascade, yte_syn, fwd_te_v)
            per_cell_rows.append(
                {
                    "symbol": symbol,
                    "horizon": int(h),
                    "fold": fold_name,
                    "n_test_windows": int(metrics["n"]),
                    "p_cascade": metrics["p_cascade"],
                    "precision_top_decile": metrics["precision_top_decile"],
                    "recall_top_decile": metrics["recall_top_decile"],
                    "lift": metrics["lift"],
                    "auc": metrics["auc"],
                    "mean_edge_in_cascade_bps": metrics["mean_edge_in_cascade_bps"],
                    "headroom_top_decile_bps": metrics["headroom_top_decile_bps"],
                    # Per-day expected gross: if top decile fires once per
                    # ~10 windows at stride=200, eval days have ~hundreds of
                    # windows.  We compute expected gross = lift × p_cascade
                    # × n_top_decile_per_day × mean_edge_bps - cost × n_top_decile_per_day.
                    # Simplification: per-day gross ≈ headroom_bps × (n_top / n_test_windows)
                    #   × (avg_windows_per_day_in_fold).
                    "expected_gross_per_day": _expected_gross_per_day(
                        metrics, n_te, fold_name=fold_name
                    ),
                }
            )

            # Per-window rows.  Real cascade label is set ONLY for April-diagnostic
            # dates (test fold is Feb/Mar by construction → real label is NaN).
            # We still emit the column so it's in the schema.
            for i in range(n_te):
                date_i = str(te_dates_v[i])
                # Real cascade label only valid for April 1-13 dates; in the
                # Feb/Mar test fold this should never trigger, but the schema
                # column is still emitted as NaN per the prompt.
                real_lbl: int | None = None
                if _is_april_diagnostic_date(date_i):
                    liq = apr_liq_ts.get(date_i, np.zeros(0, dtype=np.int64))
                    # Recompute via the helper using the shard's event_ts
                    # (would require reloading the shard; for Feb/Mar test fold
                    # we never enter this branch).  Skipped here.
                    real_lbl = (
                        1
                        if (
                            len(liq) > 0
                            and (
                                np.searchsorted(liq, te_anchors_v[i], side="right")
                                < len(liq)
                            )
                        )
                        else 0
                    )
                per_window_rows.append(
                    _row_for_window(
                        symbol=symbol,
                        horizon=h,
                        fold=fold_name,
                        anchor_ts=int(te_anchors_v[i]),
                        date=date_i,
                        window_start=int(te_starts_v[i]),
                        pred_proba=float(p_cascade[i]),
                        synthetic_cascade_label=int(yte_syn[i]),
                        real_cascade_label=real_lbl,
                        top_decile_bool=bool(top_decile_bool[i]),
                        edge_bps=float(np.abs(fwd_te_v[i]) * 1e4),
                        slip_bps=float(DEFAULT_SLIP_BPS_PER_SIDE),
                    )
                )

    return per_window_rows, per_cell_rows


def _expected_gross_per_day(
    metrics: dict[str, float], n_te: int, *, fold_name: str
) -> float:
    """Approximate daily gross at top-decile gating.

    `n_te` is the total number of windows in the test fold for this cell.
    The fold spans ~one calendar month (~28 days).  Top decile fires on
    `n_top = n_te // 10` windows over the month; each cascade-positive top-
    decile window has expected gross = `lift × mean_edge_in_cascade_bps`,
    each non-cascade top-decile window pays the round-trip cost.

    Daily approximation:
        windows_per_day        ≈ n_te / 28
        top_decile_per_day     ≈ windows_per_day / 10
        gross_per_top_window   ≈ headroom_top_decile_bps  (already net of cost)
        expected_gross_per_day = top_decile_per_day × gross_per_top_window
    """
    headroom = metrics.get("headroom_top_decile_bps", float("nan"))
    if not math.isfinite(headroom) or n_te == 0:
        return float("nan")
    days_in_fold = 28.0  # February ≈ 28; March ≈ 31; use 28 as conservative
    windows_per_day = n_te / days_in_fold
    top_decile_per_day = windows_per_day / 10.0
    return float(top_decile_per_day * headroom)


# ---------------------------------------------------------------------------
# Markdown verdict
# ---------------------------------------------------------------------------


def _emit_markdown(
    cell_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
    out_path: Path,
    *,
    horizons: tuple[int, ...],
    elapsed_sec: float,
) -> None:
    """Render the markdown verdict per the prompt's required outline."""
    lines: list[str] = []
    lines.append("# Goal-A cascade-precursor feasibility (stage 1)")
    lines.append("")
    lines.append(
        "**Question.** Does the cached data carry any cascade-precursor signal "
        "that a learned tape representation could plausibly amplify?  If a "
        "logistic regression on the existing 83-dim flat baseline cannot "
        "detect cascade onset above base rate (lift > 1), neither will an "
        "encoder, and the cascade-direction is dead before we spend training "
        "compute."
    )
    lines.append("")
    lines.append(
        "**Protocol.** Per-(symbol, H ∈ {H10, H50, H100, H500}): "
        "LogisticRegression(C=1.0, class_weight='balanced') on 83-dim flat "
        "features.  Train on 2025-10..2026-01; predict on 2026-02 (fold 1) "
        "and 2026-03 (fold 2).  Synthetic cascade label: |forward_log_return "
        f"at H| > rolling 99th-percentile cutoff (rolling window = "
        f"{ROLLING_WINDOW_FOR_LABEL} per-symbol-causal events; min_periods="
        f"{LABEL_MIN_PERIODS}).  Real cascade label: any liquidation-cause "
        "fill in (anchor_ts, ts_at(anchor + H)] — diagnostic-only on April "
        "1-13 (cause field exists from April 1 onward, hold-out hard-gates "
        f"{APRIL_HELDOUT_START}+)."
    )
    lines.append("")
    lines.append(
        f"**Cost band.** Taker, {TAKER_FEE_BPS_PER_SIDE}bp/side. "
        f"`headroom_top_decile_bps = lift × E[|fwd_ret| | cascade]_bps − "
        f"(2·{TAKER_FEE_BPS_PER_SIDE} + 2·{DEFAULT_SLIP_BPS_PER_SIDE})`. "
        f"`{DEFAULT_SLIP_BPS_PER_SIDE}bp/side slip is a flat placeholder "
        f"(not size-conditional, not per-symbol-empirical).`"
    )
    lines.append("")

    # ---------- 1. Synthetic-vs-real label validation ----------
    lines.append("## 1. Synthetic-vs-real label validation (April 1-13)")
    lines.append("")
    lines.append(
        "Real-cascade-positive windows are those where any "
        "`market_liquidation` or `backstop_liquidation` fill occurs in the "
        "window's forward-H interval.  Synthetic-cascade-positive windows "
        "are those where |forward_log_return at H| exceeds the per-symbol "
        "rolling 99th-percentile cutoff.  The synthetic label is defensible "
        "as a real-cascade proxy if `overlap_real_in_syn ≥ 0.60` (most "
        "real cascades are also large moves)."
    )
    lines.append("")
    if overlap_table.empty:
        lines.append(
            "**No April 1-13 cause data on disk for any symbol — overlap diagnostic skipped.**"
        )
    else:
        # Per-horizon universe-wide overlap (sum over symbols)
        agg_any = overlap_table.groupby("horizon").agg(
            n_real_pos=("n_real_pos", "sum"),
            n_syn_pos=("n_syn_pos", "sum"),
            n_overlap=("n_overlap", "sum"),
        )
        agg = pd.DataFrame(agg_any)
        n_real_s: pd.Series = agg["n_real_pos"].replace(0, np.nan)  # type: ignore[assignment]
        n_syn_s: pd.Series = agg["n_syn_pos"].replace(0, np.nan)  # type: ignore[assignment]
        agg["overlap_real_in_syn"] = agg["n_overlap"] / n_real_s
        agg["overlap_syn_in_real"] = agg["n_overlap"] / n_syn_s
        lines.append(
            "| horizon | n_real_pos | n_syn_pos | n_overlap | overlap(real⊂syn) | "
            "overlap(syn⊂real) |"
        )
        lines.append("|---|---|---|---|---|---|")
        for h in horizons:
            if h not in agg.index:
                continue
            r = agg.loc[h]
            lines.append(
                f"| H{h} | {int(r['n_real_pos'])} | {int(r['n_syn_pos'])} | "
                f"{int(r['n_overlap'])} | "
                f"{r['overlap_real_in_syn']:.3f} | "
                f"{r['overlap_syn_in_real']:.3f} |"
            )
        lines.append("")
        # Verdict
        if 100 in agg.index:
            ovr = agg.loc[100, "overlap_real_in_syn"]
            ovr_f = float(ovr) if pd.notna(ovr) else float("nan")
            if math.isnan(ovr_f):
                lines.append(
                    "**Overlap at H100 is undefined (no real cascades in the April "
                    "1-13 diagnostic window for this universe at this horizon). "
                    "The synthetic label cannot be validated against real cascades; "
                    "treat synthetic-label-driven results as 'volatile-window "
                    "predictability' rather than 'cascade predictability'.**"
                )
            elif ovr_f >= 0.60:
                lines.append(
                    f"**Overlap at H100 = {ovr_f:.2f} ≥ 0.60 → synthetic label "
                    f"is a defensible proxy for real cascades on April 1-13.**"
                )
            elif ovr_f < 0.30:
                lines.append(
                    f"**Overlap at H100 = {ovr_f:.2f} < 0.30 → the synthetic "
                    f"label is measuring 'volatile windows', not 'forced "
                    f"liquidations'.  Re-frame the cascade-encoder direction "
                    f"before proceeding.**"
                )
            else:
                lines.append(
                    f"**Overlap at H100 = {ovr_f:.2f} (in [0.30, 0.60)). "
                    f"Marginal proxy; synthetic-label results overestimate the "
                    f"cascade signal an encoder could capture.**"
                )
        lines.append("")

    # ---------- 2. Universe-wide median lift ----------
    lines.append("## 2. Universe-wide median lift at H100 and H500")
    lines.append("")
    nonavax = cell_table.loc[cell_table["symbol"] != "AVAX"].copy()
    lines.append(
        "| horizon | fold | median lift | median AUC | median p_cascade | " "n cells |"
    )
    lines.append("|---|---|---|---|---|---|")
    for h in horizons:
        for fold in TEST_MONTHS:
            sub = nonavax[(nonavax["horizon"] == h) & (nonavax["fold"] == fold)]
            if sub.empty:
                continue
            med_lift = float(np.nanmedian(sub["lift"].to_numpy(dtype=float)))
            med_auc = float(np.nanmedian(sub["auc"].to_numpy(dtype=float)))
            med_p = float(np.nanmedian(sub["p_cascade"].to_numpy(dtype=float)))
            lines.append(
                f"| H{h} | {fold} | {med_lift:.3f} | {med_auc:.3f} | "
                f"{med_p:.4f} | {len(sub)} |"
            )
    lines.append("")

    # ---------- 3. AUC distribution ----------
    lines.append("## 3. AUC distribution: cells clearing AUC=0.55")
    lines.append("")
    auc_table_rows: list[str] = [
        "| horizon | fold | n cells ≥ 0.55 / total | symbols ≥ 0.55 |"
    ]
    auc_table_rows.append("|---|---|---|---|")
    for h in horizons:
        for fold in TEST_MONTHS:
            sub = nonavax[(nonavax["horizon"] == h) & (nonavax["fold"] == fold)]
            if sub.empty:
                continue
            cleared = sub[sub["auc"] >= 0.55]
            cleared_syms = sorted(cleared["symbol"].unique())
            auc_table_rows.append(
                f"| H{h} | {fold} | {len(cleared)} / {len(sub)} | "
                f"{', '.join(cleared_syms) if cleared_syms else '—'} |"
            )
    lines.extend(auc_table_rows)
    lines.append("")

    # ---------- 4. Per-cell tradeable headroom ----------
    lines.append("## 4. Per-cell tradeable headroom (top decile)")
    lines.append("")
    pos_headroom = nonavax[
        (nonavax["headroom_top_decile_bps"] > 0)
        & nonavax["headroom_top_decile_bps"].notna()
    ].copy()
    lines.append(
        f"**{len(pos_headroom)} (symbol, horizon, fold) cells have "
        f"`headroom_top_decile_bps > 0` (out of {len(nonavax)} cells).**"
    )
    lines.append("")
    if len(pos_headroom) > 0:
        top = pos_headroom.sort_values("expected_gross_per_day", ascending=False).head(
            5
        )
        lines.append(
            "| symbol | H | fold | n | p_cascade | lift | AUC | "
            "headroom_bps/trade | gross/day |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['symbol']} | H{int(r['horizon'])} | {r['fold']} | "
                f"{int(r['n_test_windows'])} | {float(r['p_cascade']):.4f} | "
                f"{float(r['lift']):.2f} | {float(r['auc']):.3f} | "
                f"{float(r['headroom_top_decile_bps']):.2f} | "
                f"{float(r['expected_gross_per_day']):.2f} |"
            )
        lines.append("")

    # ---------- 5. Methodological flag ----------
    lines.append("## 5. Methodological flags")
    lines.append("")
    lines.append(
        "* **Direction sign is a placeholder.** Stage-1 only asks 'are cascades "
        "predictable at all?'.  The headroom math assumes that, given a "
        "predicted-cascade flag, we get the cascade direction right with the "
        "same sign-confidence as the cascade probability — equivalent to a "
        "lift × |fwd_ret| upper bound.  A real strategy needs cascade "
        "direction prediction (a separate stage-2 test).  If lift > 2 but "
        "cascade direction is 50/50, the headroom is roughly halved (the "
        "cascade-direction wager is symmetric, so on average half the trades "
        "lose the full move)."
    )
    lines.append("")
    lines.append(
        "* **Slip is a flat 1bp placeholder.** Per-window taker slip varies "
        "from ~0bp (BTC, $1k notional) to >20bp (illiquid alts, $100k).  "
        "Headroom for top symbols is over-stated; for illiquid alts, almost "
        "certainly under-stated as 'survivable'."
    )
    lines.append("")
    lines.append(
        "* **BatchNorm at inference** is irrelevant here (gotcha #18) — flat "
        "features + LR, no encoder; CPU-only protocol."
    )
    lines.append("")

    # ---------- 6. Verdict ----------
    lines.append("## 6. Verdict")
    lines.append("")
    # Aggregate verdict signals
    verdict_lift_h100 = float("nan")
    verdict_lift_h500 = float("nan")
    if not nonavax.empty:
        h100 = nonavax[nonavax["horizon"] == 100]
        h500 = nonavax[nonavax["horizon"] == 500]
        if not h100.empty:
            verdict_lift_h100 = float(np.nanmedian(h100["lift"].to_numpy(dtype=float)))
        if not h500.empty:
            verdict_lift_h500 = float(np.nanmedian(h500["lift"].to_numpy(dtype=float)))

    auc_h100_cleared = 0
    auc_h500_cleared = 0
    if not nonavax.empty:
        # For per-symbol "cleared" we require BOTH folds clear AUC ≥ 0.55
        for h in (100, 500):
            sub = nonavax[nonavax["horizon"] == h]
            cleared_per_symbol = sub.groupby("symbol")["auc"].apply(
                lambda s: bool((s >= 0.55).all() and len(s) >= 1)
            )
            count = int(cleared_per_symbol.sum())
            if h == 100:
                auc_h100_cleared = count
            else:
                auc_h500_cleared = count

    # Overlap-aware verdict: the lift/AUC numbers measure SYNTHETIC-label
    # predictability. If the synthetic label is a poor proxy for real cascades
    # (overlap_real_in_syn < 0.30 at H100), high lift means "volatile-window
    # predictability", NOT "cascade predictability".
    overlap_h100_real_in_syn: float = float("nan")
    if not overlap_table.empty:
        agg2 = overlap_table.groupby("horizon").agg(
            n_real_pos=("n_real_pos", "sum"),
            n_overlap=("n_overlap", "sum"),
        )
        if 100 in agg2.index and agg2.loc[100, "n_real_pos"] > 0:
            overlap_h100_real_in_syn = float(
                agg2.loc[100, "n_overlap"] / agg2.loc[100, "n_real_pos"]
            )

    overlap_proxy_strong = (
        math.isfinite(overlap_h100_real_in_syn) and overlap_h100_real_in_syn >= 0.60
    )
    overlap_proxy_marginal = (
        math.isfinite(overlap_h100_real_in_syn)
        and 0.30 <= overlap_h100_real_in_syn < 0.60
    )
    overlap_proxy_weak = (
        math.isfinite(overlap_h100_real_in_syn) and overlap_h100_real_in_syn < 0.30
    )

    if (
        math.isfinite(verdict_lift_h100)
        and verdict_lift_h100 >= 2.0
        and auc_h100_cleared >= 10
        and overlap_proxy_strong
    ):
        lines.append(
            f"**YES — cascade-precursor signal exists.** Median lift at H100 "
            f"= {verdict_lift_h100:.2f} ≥ 2.0; AUC ≥ 0.55 cleared on "
            f"{auc_h100_cleared} symbols; synthetic-vs-real overlap "
            f"(real⊂syn) = {overlap_h100_real_in_syn:.2f} ≥ 0.60.  An "
            f"encoder targeted at cascade onset is worth training; stage-2 "
            f"(cascade-direction prediction) is the natural next test."
        )
    elif (
        math.isfinite(verdict_lift_h100)
        and verdict_lift_h100 >= 2.0
        and auc_h100_cleared >= 10
        and overlap_proxy_weak
    ):
        lines.append(
            f"**MISFRAMED — flat-LR predicts VOLATILITY, not CASCADES.** "
            f"Median lift at H100 = {verdict_lift_h100:.2f} on the synthetic "
            f"label, AUC ≥ 0.55 cleared on {auc_h100_cleared} symbols — but "
            f"synthetic-vs-real overlap (real⊂syn) = "
            f"{overlap_h100_real_in_syn:.2f} < 0.30 means most real "
            f"liquidation cascades do NOT show up as 99th-percentile-magnitude "
            f"forward returns.  The strong predictability is on volatile "
            f"windows in general (volatility clustering), not specifically "
            f"on liquidation cascades.  An encoder built against a synthetic-"
            f"label objective would learn 'volatility prediction' (a known "
            f"signal that direction prediction failed to monetize via cost "
            f"band) — not the load-bearing cascade-overshoot phenomenon "
            f"hypothesized.  Re-ground the cascade-encoder direction: either "
            f"(a) use the real cause-flag label directly (only April 1-13 "
            f"available, ~412 events universe-wide — sparse but binding), "
            f"(b) pick a different cascade definition (e.g. quantile of "
            f"|forward_log_return| × is_open fraction, since liquidations "
            f"are forced opens), or (c) drop cascade direction and choose a "
            f"different encoder objective."
        )
    elif (
        math.isfinite(verdict_lift_h100)
        and verdict_lift_h100 >= 1.3
        and auc_h100_cleared >= 5
    ):
        lines.append(
            f"**MARGINAL — small-cell signal survives but universe-wide is "
            f"weak.** Median lift at H100 = {verdict_lift_h100:.2f} (1.3 ≤ "
            f"lift < 2.0); AUC ≥ 0.55 on {auc_h100_cleared} symbols; overlap "
            f"(real⊂syn) = {overlap_h100_real_in_syn:.2f}.  An encoder might "
            f"lift the median but unlikely to lift to tradeable-on-universe "
            f"scale.  Consider scoping a cascade encoder to the survivor-cell "
            f"symbols and pre-registering a narrower test."
        )
    else:
        lines.append(
            f"**NO — cascade-precursor signal is absent on hand features.** "
            f"Median lift at H100 = "
            f"{verdict_lift_h100:.2f} < 1.3; AUC ≥ 0.55 on "
            f"{auc_h100_cleared} symbols (≥10 needed for go-ahead); overlap "
            f"(real⊂syn) = {overlap_h100_real_in_syn:.2f}.  An encoder will "
            f"not amplify a signal that flat-LR cannot detect.  Cascade "
            f"direction is dead before encoder training compute is "
            f"committed.  Consider a different encoder objective."
        )
    lines.append("")
    lines.append(
        f"_Pipeline ran in {elapsed_sec:.1f} s.  CPU-only.  No April 14+ data "
        f"touched._"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=CACHE_DIR_DEFAULT,
        help="Cache directory (default: data/cache)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR_DEFAULT,
        help="Output directory (default: docs/experiments/goal-a-feasibility)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DIRECTION_HORIZONS),
        help="Horizons to evaluate (default: 10 50 100 500)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=list(SYMBOLS),
        help="Symbols to evaluate (default: full SYMBOLS list incl AVAX)",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip the April 1-13 real-vs-synthetic overlap diagnostic",
    )
    args = parser.parse_args()

    horizons: tuple[int, ...] = tuple(int(h) for h in args.horizons)
    symbols: tuple[str, ...] = tuple(args.symbols)
    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(
        f"[cascade-precursor] horizons={horizons} | symbols={len(symbols)} | "
        f"cache={args.cache}"
    )

    # ---------- Step 1: April overlap diagnostic ----------
    overlap_table = pd.DataFrame()
    if not args.skip_overlap:
        print("[cascade-precursor] computing April 1-13 real-vs-synthetic overlap...")
        overlap_table = _april_real_vs_synthetic_overlap(args.cache, symbols, horizons)
        print(
            f"[cascade-precursor] overlap rows: {len(overlap_table)} "
            f"(symbols × horizons)"
        )

    # ---------- Step 2: per-symbol LR + per-cell metrics ----------
    print("[cascade-precursor] running per-symbol LR over Oct-Jan / Feb / Mar...")
    all_per_window: list[dict] = []
    all_per_cell: list[dict] = []
    for i, symbol in enumerate(symbols):
        # Real-cascade lookup is only needed for April-diagnostic dates,
        # which never appear in Feb/Mar test folds.  We pass an empty dict.
        per_w, per_c = _run_symbol(args.cache, symbol, horizons, apr_liq_ts={})
        all_per_window.extend(per_w)
        all_per_cell.extend(per_c)
        print(
            f"[cascade-precursor] [{i+1:>2}/{len(symbols)}] {symbol}: "
            f"{len(per_w)} windows, {len(per_c)} cells"
        )

    cell_table = pd.DataFrame(all_per_cell)
    per_window_table = pd.DataFrame(all_per_window)

    # ---------- Step 3: emit outputs ----------
    cell_table.to_csv(out_dir / "cascade_precursor_table.csv", index=False)
    per_window_table.to_parquet(
        out_dir / "cascade_precursor_per_window.parquet", index=False
    )

    elapsed = time.time() - t0
    _emit_markdown(
        cell_table,
        overlap_table,
        out_dir / "cascade_precursor.md",
        horizons=horizons,
        elapsed_sec=elapsed,
    )

    print(f"[cascade-precursor] done in {elapsed:.1f} s")
    print(f"[cascade-precursor] wrote {out_dir/'cascade_precursor_table.csv'}")
    print(f"[cascade-precursor] wrote {out_dir/'cascade_precursor_per_window.parquet'}")
    print(f"[cascade-precursor] wrote {out_dir/'cascade_precursor.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

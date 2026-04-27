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


def _precision_recall_at_top_pct(
    proba: np.ndarray, labels: np.ndarray, *, top_pct: float
) -> tuple[float, float]:
    """Top-`top_pct` precision and recall (no lift; caller computes via base rate).

    Returns
    -------
    (precision, recall)
        precision = (cascades in top k) / k
        recall    = (cascades in top k) / total_cascades ; NaN if total_cascades==0
    Top-k size is `max(1, n*top_pct)`.
    """
    n = len(proba)
    if n == 0:
        return float("nan"), float("nan")
    n_top = max(1, int(round(n * top_pct)))

    order = np.argsort(-proba, kind="stable")
    top = order[:n_top]
    pos_in_top = float(labels[top].sum())
    total_pos = float(labels.sum())
    precision = pos_in_top / float(n_top)
    recall = pos_in_top / total_pos if total_pos > 0 else float("nan")
    return precision, recall


def _bootstrap_auc_ci(
    proba: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) AUC with a 100*(1-alpha)% percentile CI.

    Resamples `(proba, labels)` rows with replacement `n_boot` times.  If a
    bootstrap draw yields a single class the AUC for that draw is skipped
    (recorded as NaN and dropped from the percentile calc).  If the labels are
    all-zero or all-one the point AUC is undefined and we return (NaN, NaN, NaN).
    """
    n = len(proba)
    if n == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        point = float(roc_auc_score(labels, proba))
    except ValueError:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=np.float64)
    aucs[:] = np.nan
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        p_b = proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs[b] = roc_auc_score(y_b, p_b)
        except ValueError:
            continue

    finite = aucs[np.isfinite(aucs)]
    if len(finite) < 10:
        # Not enough valid bootstrap draws to build a CI.
        return point, float("nan"), float("nan")
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return point, lo, hi


def _signal_distinguishable_from_baseline(
    *,
    real_lo: float,
    real_hi: float,
    baseline_lo: float,
    baseline_hi: float,
) -> bool:
    """True iff `real_lo > baseline_hi` (strict, NaN-safe).

    The binding statistical question for the real-cascade probe: does the
    real-label AUC's 95% CI lower bound strictly exceed the shuffled-label
    AUC's 95% CI upper bound?  Strict > so 'tied at boundary' is False.
    """
    vals = (real_lo, real_hi, baseline_lo, baseline_hi)
    if any(not math.isfinite(v) for v in vals):
        return False
    return real_lo > baseline_hi


def _april_diagnostic_dates() -> list[str]:
    """Calendar dates April 1 through April 13 (inclusive), as `YYYY-MM-DD` strings.

    These are the diagnostic-window dates regardless of whether raw data is on
    disk for a given symbol.  The caller filters to dates with shards / parquet.
    """
    return [f"2026-04-{d:02d}" for d in range(1, 14)]


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


# ---------------------------------------------------------------------------
# Cascade-direction probe helpers (--cascade-direction flag)
# ---------------------------------------------------------------------------
#
# Given the stage-2 cascade-onset model's per-window predicted probability,
# we ask: *conditional on a cascade firing*, can we predict its direction
# (long vs short) from the same 83-dim flat baseline + the cascade-onset
# confidence?
#
# Two operationalizations of "direction":
#   a) Realized direction at horizon H — sign(forward_log_return_h500).
#   b) Overshoot peak direction — sign(first_liq_price - anchor_mid) for the
#      first liquidation fill in (anchor_ts, ts_at(anchor + h500)].  Captures
#      the cascade-overshoot direction before any subsequent reversion.
#
# Multiple-comparisons guardrail: H500 ONLY (n_cascades_h500 ~ 73; H100 has
# n=20, too underpowered for direction prediction).


def _realized_direction_label(fwd_ret: np.ndarray) -> np.ndarray:
    """Realized direction: 1 if forward_log_return > 0, 0 if <= 0, -1 if NaN.

    The -1 sentinel signals 'invalid' for callers that need to drop windows
    with horizon-overrun forward returns.
    """
    arr = np.asarray(fwd_ret, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int8)
    finite = np.isfinite(arr)
    out[finite & (arr > 0)] = 1
    out[finite & (arr <= 0)] = 0
    return out


def _top_pct_mask(proba: np.ndarray, *, top_pct: float) -> np.ndarray:
    """Boolean mask selecting the top-`top_pct` of `proba` by rank.

    Top-k size is `max(1, floor(n * top_pct))`.  Ties broken in stable order.
    """
    n = len(proba)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.floor(n * top_pct)))
    order = np.argsort(-np.asarray(proba, dtype=np.float64), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def _marginal_direction_asymmetry(
    fwd_ret: np.ndarray, cascade_label: np.ndarray
) -> float:
    """P(forward_return > 0 | cascade_label == 1).  NaN if no cascades."""
    fwd = np.asarray(fwd_ret, dtype=np.float64)
    cas = np.asarray(cascade_label, dtype=np.int64)
    valid = np.isfinite(fwd) & (cas == 1)
    if not valid.any():
        return float("nan")
    return float((fwd[valid] > 0).mean())


def _overshoot_direction_for_window(
    *,
    anchor_ts: int,
    end_ts: int,
    anchor_mid: float,
    liq_ts: np.ndarray,
    liq_price: np.ndarray,
) -> int:
    """Overshoot direction for one window.

    Returns +1 if first_liq_price > anchor_mid, -1 if <, 0 if no liquidation
    fill in (anchor_ts, end_ts] OR if first_liq_price == anchor_mid.
    """
    if len(liq_ts) == 0:
        return 0
    lo = int(np.searchsorted(liq_ts, anchor_ts, side="right"))
    hi = int(np.searchsorted(liq_ts, end_ts, side="right"))
    if hi <= lo:
        return 0
    first_price = float(liq_price[lo])
    if first_price > anchor_mid:
        return 1
    if first_price < anchor_mid:
        return -1
    return 0


def _majority_class_baseline_auc(y: np.ndarray) -> float:
    """AUC of a constant majority-class predictor — exactly 0.5 by definition.

    Returns NaN on degenerate single-class label vectors.
    """
    arr = np.asarray(y, dtype=np.int64)
    if len(np.unique(arr)) < 2:
        return float("nan")
    return 0.5


def _direction_per_day_expected_gross(
    *,
    triggers_per_day: float,
    direction_accuracy: float,
    mean_abs_fwd_bps: float,
    fee_bps_per_side: float = TAKER_FEE_BPS_PER_SIDE,
    slip_bps_per_side: float = DEFAULT_SLIP_BPS_PER_SIDE,
) -> dict[str, float]:
    """Per-trigger and per-day headroom math for the direction strategy.

    gross_per_trigger = (2 * direction_accuracy - 1) * mean_abs_fwd_bps
    cost_per_trigger  = 2 * fee + 2 * slip
    net_per_trigger   = gross - cost
    per_day_gross     = triggers_per_day * net_per_trigger
    """
    if not (math.isfinite(direction_accuracy) and math.isfinite(mean_abs_fwd_bps)):
        return {
            "gross_per_trigger_bps": float("nan"),
            "cost_per_trigger_bps": float("nan"),
            "net_per_trigger_bps": float("nan"),
            "per_day_gross_bps": float("nan"),
        }
    gross = (2.0 * direction_accuracy - 1.0) * mean_abs_fwd_bps
    cost = 2.0 * fee_bps_per_side + 2.0 * slip_bps_per_side
    net = gross - cost
    per_day = triggers_per_day * net
    return {
        "gross_per_trigger_bps": float(gross),
        "cost_per_trigger_bps": float(cost),
        "net_per_trigger_bps": float(net),
        "per_day_gross_bps": float(per_day),
    }


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
# Stage 2 — REAL cause-flag probe on April 1-13 with leave-one-day-out CV
# ---------------------------------------------------------------------------
#
# The synthetic-label probe (stage 1) measured volatility-clustering, not real
# liquidation cascades — the overlap-validation diagnostic confirmed only ~20%
# of real cascades coincide with 99th-percentile-magnitude forward returns at
# H100.  Stage 2 uses the `cause` flag directly.  Sample size on April 1-13 is
# small (n_cascades ≈ 9 / 20 / 73 universe-wide at H50/H100/H500) — wide CIs
# expected.  The binding question: is the real cascade label distinguishable
# from a shuffled-label baseline given the small n?
#
# Protocol:
#   * Real-cascade label per window: any `cause IN ('market_liquidation',
#     'backstop_liquidation')` fill in (anchor_ts, ts_at(anchor + H)].  Pulled
#     from raw trade parquet (April 1+) via DuckDB; pre-April dedup rule does
#     NOT apply (April+ uses event_type='fulfill_taker' filter — gotcha #19).
#   * Pooled cross-symbol LR with leave-one-day-out CV (April 1-13).
#   * Per-symbol LR with leave-one-day-out CV for symbols with ≥5 cascades at H.
#   * Shuffled-label baseline: same fit, labels permuted within each train day.
#   * Random-feature baseline: same fit, features replaced by Gaussian noise.
#   * Bootstrap 95% AUC CIs.
#   * Hard constraint: April 14+ untouched (gotcha #17).
#
# Multiple-testing awareness: the BINDING metric is pooled cross-symbol AUC vs
# shuffled at H100 and H500.  Per-symbol cells are descriptive only.

REAL_HORIZONS: tuple[int, ...] = (50, 100, 500)
N_BOOT_REAL: int = 1000
PER_SYMBOL_MIN_CASCADES: int = 5
TOP_PCT_REAL: float = 0.01  # precision-at-top-1%


@dataclass(frozen=True)
class RealDayBatch:
    """Per-(symbol, date) windows + real-cascade labels at every horizon.

    Built from the cache shard for flat features + event_ts, plus a DuckDB
    query against the raw trade parquet for the liquidation timestamps.
    """

    symbol: str
    date: str
    flat_X: np.ndarray  # (N, FLAT_DIM)
    anchor_ts: np.ndarray  # (N,) int64
    window_starts: np.ndarray  # (N,) int64
    real_labels: dict[int, np.ndarray]  # h -> (N,) int8
    real_valid: dict[int, np.ndarray]  # h -> (N,) bool — True iff anchor+h fits in day


def _load_liq_ts_for_symbol_date(symbol: str, date_str: str) -> np.ndarray | None:
    """Return sorted int64 array of liquidation trade ts_ms for (symbol, date),
    or None if the raw parquet directory does not exist.

    Hard constraint: April 14+ data is hold-out — refuse to load.
    """
    if date_str >= APRIL_HELDOUT_START:
        return None
    if date_str < APRIL_START:
        # Pre-April raw data has no `cause` column (CLAUDE.md schema)
        return None
    base = Path(f"data/trades/symbol={symbol}/date={date_str}")
    if not base.exists():
        return None
    parquet_files = list(base.glob("*.parquet"))
    if not parquet_files:
        return None
    q = (
        f"SELECT ts_ms FROM read_parquet('{base}/*.parquet') "
        f"WHERE cause IN ('market_liquidation', 'backstop_liquidation') "
        f"ORDER BY ts_ms"
    )
    try:
        df = duckdb.query(q).to_df()
    except Exception:
        return None
    return df["ts_ms"].to_numpy(dtype=np.int64)


def _build_real_day_batch(
    cache_dir: Path,
    symbol: str,
    date_str: str,
    horizons: tuple[int, ...] = REAL_HORIZONS,
) -> RealDayBatch | None:
    """Build a RealDayBatch for (symbol, date_str) on April 1-13.

    Returns None if the cache shard or raw-trade parquet is unavailable, or if
    the shard is empty.
    """
    shard_path = cache_dir / f"{symbol}__{date_str}.npz"
    if not shard_path.exists():
        return None
    if date_str >= APRIL_HELDOUT_START or date_str < APRIL_START:
        return None
    payload = _load_shard(shard_path)
    features: np.ndarray = payload["features"]
    event_ts: np.ndarray = payload["event_ts"].astype(np.int64)
    n_events = features.shape[0]
    if n_events < WINDOW_LEN:
        return None

    last_valid_start = n_events - WINDOW_LEN
    if last_valid_start < 0:
        return None
    starts = np.arange(0, last_valid_start + 1, STRIDE_EVAL, dtype=np.int64)
    if len(starts) == 0:
        return None
    anchors = starts + WINDOW_LEN - 1
    anchor_ts = event_ts[anchors].astype(np.int64)

    # Flat features per window
    flat_X = np.empty((len(starts), FLAT_DIM), dtype=np.float32)
    for i, s in enumerate(starts):
        flat_X[i] = extract_flat_features(features[s : s + WINDOW_LEN])

    # Real cascade labels per horizon — even if liq_ts is empty (label = all-zeros)
    liq_ts = _load_liq_ts_for_symbol_date(symbol, date_str)
    if liq_ts is None:
        return None  # raw parquet missing → can't validate

    real_labels: dict[int, np.ndarray] = {}
    real_valid: dict[int, np.ndarray] = {}
    for h in horizons:
        end_idx = anchors + h
        valid = end_idx < n_events
        real_valid[h] = valid
        real_labels[h] = _real_cascade_label_with_event_ts(
            anchor_ts=anchor_ts,
            window_starts=starts,
            event_ts=event_ts,
            horizon=h,
            liq_ts=liq_ts,
        )
    return RealDayBatch(
        symbol=symbol,
        date=date_str,
        flat_X=flat_X,
        anchor_ts=anchor_ts,
        window_starts=starts,
        real_labels=real_labels,
        real_valid=real_valid,
    )


def _gather_real_batches(
    cache_dir: Path,
    symbols: tuple[str, ...],
    dates: list[str],
    horizons: tuple[int, ...] = REAL_HORIZONS,
) -> list[RealDayBatch]:
    """Build all RealDayBatch objects across (symbol × date)."""
    out: list[RealDayBatch] = []
    for symbol in symbols:
        for date_str in dates:
            b = _build_real_day_batch(cache_dir, symbol, date_str, horizons=horizons)
            if b is not None:
                out.append(b)
    return out


def _stack_real_batches_at_horizon(
    batches: list[RealDayBatch], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-batch arrays for a single horizon, dropping invalid windows
    (anchor + horizon overruns the shard).

    Returns
    -------
    X         : (N_total, FLAT_DIM)
    y         : (N_total,) int8 real cascade labels
    sym_arr   : (N_total,) U-array of symbol strings
    date_arr  : (N_total,) U-array of date strings
    anchor_arr: (N_total,) int64 anchor timestamps
    """
    if not batches:
        return (
            np.zeros((0, FLAT_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype="<U16"),
            np.zeros(0, dtype="<U10"),
            np.zeros(0, dtype=np.int64),
        )
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    syms: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for b in batches:
        if horizon not in b.real_labels:
            continue
        valid = b.real_valid[horizon]
        if not valid.any():
            continue
        Xs.append(b.flat_X[valid])
        ys.append(b.real_labels[horizon][valid].astype(np.int8))
        n_v = int(valid.sum())
        syms.append(np.full(n_v, b.symbol, dtype="<U16"))
        dates.append(np.full(n_v, b.date, dtype="<U10"))
        anchors.append(b.anchor_ts[valid].astype(np.int64))
    if not Xs:
        return (
            np.zeros((0, FLAT_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype="<U16"),
            np.zeros(0, dtype="<U10"),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(syms, axis=0),
        np.concatenate(dates, axis=0),
        np.concatenate(anchors, axis=0),
    )


def _fit_lr_proba(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Fit balanced LR + StandardScaler on (Xtr, ytr); return P(class=1) on Xte.

    Degenerate ytr (single class) → returns constant probability = ytr.mean().
    """
    if len(np.unique(ytr)) < 2 or len(Xtr) == 0:
        const = float(ytr.mean()) if len(ytr) > 0 else 0.0
        return np.full(len(Xte), const, dtype=np.float64)
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(
        C=1.0, max_iter=1_000, class_weight="balanced", solver="lbfgs"
    ).fit(scaler.transform(Xtr), ytr)
    proba = lr.predict_proba(scaler.transform(Xte))
    classes = list(lr.classes_)
    pos_idx = classes.index(1) if 1 in classes else (1 if proba.shape[1] > 1 else 0)
    return proba[:, pos_idx].astype(np.float64)


def _shuffle_labels_within_day(
    y: np.ndarray, dates: np.ndarray, *, seed: int
) -> np.ndarray:
    """Permute labels independently within each day (preserves per-day base rate)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for d in np.unique(dates):
        mask = dates == d
        idx = np.flatnonzero(mask)
        if len(idx) > 1:
            perm = rng.permutation(len(idx))
            out[idx] = y[idx[perm]]
    return out


def _leave_one_day_out_predictions(
    *,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    feature_mode: str = "real",
    label_mode: str = "real",
    rng_seed: int = 0,
) -> np.ndarray:
    """Leave-one-day-out CV: hold out each day, train on the rest, predict on day.

    Parameters
    ----------
    feature_mode : 'real' uses X as-is; 'random' replaces with Gaussian noise.
    label_mode   : 'real' uses y; 'shuffled' permutes within-day in train fold only.

    Returns
    -------
    pred : (N,) predicted P(class=1).  pred[i] is from a fold where i was held out.
    """
    n = len(y)
    pred = np.full(n, np.nan, dtype=np.float64)
    unique_dates = sorted(np.unique(dates).tolist())
    rng = np.random.default_rng(rng_seed)
    for fold_idx, held_date in enumerate(unique_dates):
        train_mask = dates != held_date
        test_mask = dates == held_date
        if not train_mask.any() or not test_mask.any():
            continue
        if feature_mode == "real":
            Xtr = X[train_mask]
            Xte = X[test_mask]
        elif feature_mode == "random":
            # Same shape, same dtype, Gaussian noise
            Xtr = rng.standard_normal(size=(int(train_mask.sum()), X.shape[1])).astype(
                np.float32
            )
            Xte = rng.standard_normal(size=(int(test_mask.sum()), X.shape[1])).astype(
                np.float32
            )
        else:
            raise ValueError(f"unknown feature_mode: {feature_mode}")

        ytr = y[train_mask].astype(np.int64)
        if label_mode == "shuffled":
            ytr = _shuffle_labels_within_day(
                ytr, dates[train_mask], seed=rng_seed + fold_idx
            )

        proba = _fit_lr_proba(Xtr, ytr, Xte)
        pred[test_mask] = proba
    return pred


def _per_cell_real_metrics(
    *,
    proba_real: np.ndarray,
    proba_shuffled: np.ndarray,
    proba_random_feat: np.ndarray,
    labels: np.ndarray,
    n_boot: int = N_BOOT_REAL,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Compute the canonical metrics for one (H, scope) cell."""
    valid = np.isfinite(proba_real) & np.isfinite(labels.astype(np.float64))
    p_r = proba_real[valid]
    p_sh = proba_shuffled[valid]
    p_rf = proba_random_feat[valid]
    y = labels[valid].astype(np.int64)
    n_total = int(len(y))
    n_cascades = int(y.sum())
    base_rate = float(n_cascades / n_total) if n_total > 0 else float("nan")

    # AUC + bootstrap CI
    auc, auc_lo, auc_hi = _bootstrap_auc_ci(p_r, y, n_boot=n_boot, seed=seed)
    auc_sh, auc_sh_lo, auc_sh_hi = _bootstrap_auc_ci(
        p_sh, y, n_boot=n_boot, seed=seed + 1
    )
    auc_rf, auc_rf_lo, auc_rf_hi = _bootstrap_auc_ci(
        p_rf, y, n_boot=n_boot, seed=seed + 2
    )

    # Precision/recall at top-1%
    prec_top, rec_top = _precision_recall_at_top_pct(p_r, y, top_pct=TOP_PCT_REAL)
    if math.isfinite(base_rate) and base_rate > 0:
        lift_top = prec_top / base_rate
    else:
        lift_top = float("nan")

    # Distinguishability flags
    dist_from_shuffled = _signal_distinguishable_from_baseline(
        real_lo=auc_lo, real_hi=auc_hi, baseline_lo=auc_sh_lo, baseline_hi=auc_sh_hi
    )
    dist_from_random_feat = _signal_distinguishable_from_baseline(
        real_lo=auc_lo, real_hi=auc_hi, baseline_lo=auc_rf_lo, baseline_hi=auc_rf_hi
    )

    return {
        "n_total": float(n_total),
        "n_cascades": float(n_cascades),
        "base_rate": base_rate,
        "auc": auc,
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
        "auc_shuffled": auc_sh,
        "auc_shuffled_lo": auc_sh_lo,
        "auc_shuffled_hi": auc_sh_hi,
        "auc_random_feat": auc_rf,
        "auc_random_feat_lo": auc_rf_lo,
        "auc_random_feat_hi": auc_rf_hi,
        "precision_at_top_1pct": prec_top,
        "recall_at_top_1pct": rec_top,
        "lift_at_top_1pct": lift_top,
        "signal_distinguishable_from_shuffled": dist_from_shuffled,
        "signal_distinguishable_from_random_feat": dist_from_random_feat,
    }


def _run_cascade_real_pipeline(
    cache_dir: Path,
    symbols: tuple[str, ...],
    out_dir: Path,
    *,
    horizons: tuple[int, ...] = REAL_HORIZONS,
    n_boot: int = N_BOOT_REAL,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the stage-2 real-cascade probe.

    Returns
    -------
    (per_window_df, per_cell_df, summary_dict)
    """
    dates = _april_diagnostic_dates()
    print(
        f"[cascade-real] gathering windows for {len(symbols)} symbols × "
        f"{len(dates)} dates (April 1-13)..."
    )
    batches = _gather_real_batches(cache_dir, symbols, dates, horizons=horizons)
    if not batches:
        raise RuntimeError(
            "No April 1-13 batches built — cache shards or raw trade parquets "
            "may be missing.  Cannot run stage-2 real-cascade probe."
        )
    n_batches = len(batches)
    n_unique_dates = len({b.date for b in batches})
    n_unique_syms = len({b.symbol for b in batches})
    print(
        f"[cascade-real] built {n_batches} (symbol, date) batches across "
        f"{n_unique_syms} symbols and {n_unique_dates} dates"
    )

    per_window_rows: list[dict] = []
    per_cell_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for h in horizons:
        X, y, sym_arr, date_arr, anchor_arr = _stack_real_batches_at_horizon(batches, h)
        if len(y) == 0 or y.sum() == 0:
            print(
                f"[cascade-real] H{h}: no positive cascades on April 1-13 — "
                f"skipping cell"
            )
            continue
        n_total = int(len(y))
        n_cascades = int(y.sum())
        unique_dates = sorted(np.unique(date_arr).tolist())
        print(
            f"[cascade-real] H{h}: n_total={n_total}, n_cascades={n_cascades}, "
            f"n_dates={len(unique_dates)}"
        )

        # ----- Pooled cross-symbol leave-one-day-out CV -----
        proba_real_pool = _leave_one_day_out_predictions(
            X=X,
            y=y,
            dates=date_arr,
            feature_mode="real",
            label_mode="real",
            rng_seed=seed,
        )
        proba_shuffled_pool = _leave_one_day_out_predictions(
            X=X,
            y=y,
            dates=date_arr,
            feature_mode="real",
            label_mode="shuffled",
            rng_seed=seed + 100,
        )
        proba_random_feat_pool = _leave_one_day_out_predictions(
            X=X,
            y=y,
            dates=date_arr,
            feature_mode="random",
            label_mode="real",
            rng_seed=seed + 200,
        )

        pooled_metrics = _per_cell_real_metrics(
            proba_real=proba_real_pool,
            proba_shuffled=proba_shuffled_pool,
            proba_random_feat=proba_random_feat_pool,
            labels=y,
            n_boot=n_boot,
            seed=seed,
        )
        pooled_metrics_typed: dict[str, float | bool | str | int] = dict(pooled_metrics)
        pooled_metrics_typed["horizon"] = int(h)
        pooled_metrics_typed["scope"] = "pooled"
        pooled_metrics_typed["symbol"] = "ALL"
        per_cell_rows.append(pooled_metrics_typed)
        summary[f"pooled_H{h}"] = pooled_metrics

        # ----- Per-window rows for the pooled fold -----
        valid_pool = np.isfinite(proba_real_pool)
        for i in np.flatnonzero(valid_pool):
            per_window_rows.append(
                {
                    "symbol": str(sym_arr[i]),
                    "date": str(date_arr[i]),
                    "anchor_ts": int(anchor_arr[i]),
                    "horizon": int(h),
                    "real_cascade_label": int(y[i]),
                    "pred_proba": float(proba_real_pool[i]),
                    "pred_proba_shuffled": float(proba_shuffled_pool[i]),
                    "pred_proba_random_feat": float(proba_random_feat_pool[i]),
                    "fold": str(date_arr[i]),  # held-out day
                    "scope": "pooled",
                }
            )

        # ----- Per-symbol cells for symbols with ≥ PER_SYMBOL_MIN_CASCADES at H -----
        for symbol in sorted(np.unique(sym_arr).tolist()):
            sym_mask = sym_arr == symbol
            y_s = y[sym_mask]
            if y_s.sum() < PER_SYMBOL_MIN_CASCADES:
                continue
            X_s = X[sym_mask]
            dates_s = date_arr[sym_mask]
            unique_dates_s = sorted(np.unique(dates_s).tolist())
            if len(unique_dates_s) < 2:
                # Cannot run leave-one-day-out CV with a single date
                continue

            proba_real_s = _leave_one_day_out_predictions(
                X=X_s,
                y=y_s,
                dates=dates_s,
                feature_mode="real",
                label_mode="real",
                rng_seed=seed + 1000,
            )
            proba_shuffled_s = _leave_one_day_out_predictions(
                X=X_s,
                y=y_s,
                dates=dates_s,
                feature_mode="real",
                label_mode="shuffled",
                rng_seed=seed + 1100,
            )
            proba_random_feat_s = _leave_one_day_out_predictions(
                X=X_s,
                y=y_s,
                dates=dates_s,
                feature_mode="random",
                label_mode="real",
                rng_seed=seed + 1200,
            )

            sym_metrics = _per_cell_real_metrics(
                proba_real=proba_real_s,
                proba_shuffled=proba_shuffled_s,
                proba_random_feat=proba_random_feat_s,
                labels=y_s,
                n_boot=n_boot,
                seed=seed,
            )
            sym_metrics_typed: dict[str, float | bool | str | int] = dict(sym_metrics)
            sym_metrics_typed["horizon"] = int(h)
            sym_metrics_typed["scope"] = "per_symbol"
            sym_metrics_typed["symbol"] = symbol
            per_cell_rows.append(sym_metrics_typed)

    per_window_df = pd.DataFrame(per_window_rows)
    per_cell_df = pd.DataFrame(per_cell_rows)
    return per_window_df, per_cell_df, summary


def _emit_real_markdown(
    per_cell_df: pd.DataFrame,
    out_path: Path,
    *,
    horizons: tuple[int, ...],
    elapsed_sec: float,
    notes: list[str] | None = None,
) -> None:
    """Render the stage-2 markdown verdict per the prompt's required outline."""
    lines: list[str] = []
    lines.append("# Goal-A cascade-precursor (stage 2) — real `cause` flag probe")
    lines.append("")
    lines.append(
        "**Question.** Is there a measurable precursor footprint in the 200 "
        "events before a real liquidation cascade?  Stage 1 used a synthetic "
        "99th-percentile-magnitude label that turned out to overlap real "
        "cascades only ~20% on April 1-13 at H100 — the high lift was "
        "measuring volatility clustering, not forced liquidations.  Stage 2 "
        "uses the `cause` flag directly.  Sample size on April 1-13 is small "
        "(n_cascades ≈ 9 / 20 / 73 universe-wide at H50 / H100 / H500) — "
        "wide CIs are expected.  The binding statistical question is whether "
        "the real-label AUC's 95% CI lower bound strictly exceeds the "
        "shuffled-label AUC's 95% CI upper bound."
    )
    lines.append("")
    lines.append(
        "**Protocol.** 83-dim flat baseline (`tape/flat_features.py`).  "
        "`LogisticRegression(class_weight='balanced', C=1.0)` — same default "
        "as Gate 0 / Gate 1.  No hyperparameter search (small data + multiple "
        "comparisons would inflate AUC).  Pooled cross-symbol leave-one-day-"
        "out CV on April 1-13 — held-out day's predictions aggregated across "
        "folds.  Per-symbol leave-one-day-out CV restricted to symbols with "
        "≥ 5 real cascades at the horizon of interest (descriptive, not "
        "binding).  Bootstrap 95% AUC CI (n_boot = 1000).  Shuffled-label "
        "baseline (labels permuted within day, train fold only).  Random-"
        "feature baseline (Gaussian noise, same shape).  April 14+ "
        "untouched (gotcha #17)."
    )
    lines.append("")
    lines.append(
        "**Contamination disclosure.** April 1-13 was used for v1 diagnostic "
        "checks (gotcha #17 lists this as the diagnostic window) — but the "
        "`cause` field was NOT studied in v1.  The contamination concern "
        "from prior v1 work is on direction-related tests and cannot bias "
        "this real-cascade probe."
    )
    lines.append("")

    # ---------- 1. Sample size confirmation ----------
    lines.append("## 1. Sample size confirmation")
    lines.append("")
    pooled: pd.DataFrame = pd.DataFrame(
        per_cell_df[per_cell_df["scope"] == "pooled"]
    ).copy()
    if pooled.empty:
        lines.append("**No pooled cells produced — see methodological flag below.**")
    else:
        lines.append("| H | n_total (windows) | n_cascades | base rate |")
        lines.append("|---|---|---|---|")
        for h in horizons:
            sub = pooled[pooled["horizon"] == h]
            if sub.empty:
                lines.append(f"| H{h} | — | — | — |")
                continue
            r = sub.iloc[0]
            lines.append(
                f"| H{h} | {int(r['n_total'])} | {int(r['n_cascades'])} | "
                f"{float(r['base_rate']):.4f} |"
            )
        lines.append("")
        lines.append(
            "Prior synthetic-vs-real validation reported H50 = 9, H100 = 20, "
            "H500 = 73 cascades universe-wide.  If the table above differs, "
            "explain in flags below."
        )
    lines.append("")

    # ---------- 2. Pooled cross-symbol AUC ----------
    lines.append("## 2. Pooled cross-symbol AUC at H100 and H500")
    lines.append("")
    if pooled.empty:
        lines.append("**No pooled cells available.**")
    else:
        lines.append(
            "| H | AUC (real label) | AUC (shuffled) | AUC (random feat) | "
            "distinguishable from shuffled? |"
        )
        lines.append("|---|---|---|---|---|")
        for h in horizons:
            sub = pooled[pooled["horizon"] == h]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| H{h} | "
                f"{float(r['auc']):.3f} [{float(r['auc_lo']):.3f}, "
                f"{float(r['auc_hi']):.3f}] | "
                f"{float(r['auc_shuffled']):.3f} "
                f"[{float(r['auc_shuffled_lo']):.3f}, "
                f"{float(r['auc_shuffled_hi']):.3f}] | "
                f"{float(r['auc_random_feat']):.3f} "
                f"[{float(r['auc_random_feat_lo']):.3f}, "
                f"{float(r['auc_random_feat_hi']):.3f}] | "
                f"{'YES' if bool(r['signal_distinguishable_from_shuffled']) else 'NO'} |"
            )
        lines.append("")
    lines.append("")

    # ---------- 3. Distinguishability ----------
    lines.append("## 3. Distinguishable from shuffled-label baseline?")
    lines.append("")
    lines.append(
        "Binding statistical test: real-label AUC CI lower bound must "
        "strictly exceed shuffled-label AUC CI upper bound.  Aggregate "
        "across H100 and H500 below."
    )
    lines.append("")
    if pooled.empty:
        lines.append("**Cannot evaluate — no pooled cells.**")
    else:
        for h in horizons:
            sub = pooled[pooled["horizon"] == h]
            if sub.empty:
                continue
            r = sub.iloc[0]
            verdict = (
                "**DISTINGUISHABLE**"
                if bool(r["signal_distinguishable_from_shuffled"])
                else "**NOT distinguishable**"
            )
            lines.append(
                f"* H{h}: real CI [{float(r['auc_lo']):.3f}, "
                f"{float(r['auc_hi']):.3f}] vs shuffled CI "
                f"[{float(r['auc_shuffled_lo']):.3f}, "
                f"{float(r['auc_shuffled_hi']):.3f}] → {verdict}."
            )
    lines.append("")

    # ---------- 4. Per-symbol breakdown ----------
    lines.append("## 4. Per-symbol breakdown")
    lines.append("")
    per_sym: pd.DataFrame = pd.DataFrame(
        per_cell_df[per_cell_df["scope"] == "per_symbol"]
    ).copy()
    if per_sym.empty:
        lines.append(
            "**No per-symbol cells — no symbols had ≥ 5 cascades at any "
            "evaluated horizon.**"
        )
    else:
        lines.append(
            "Symbols with ≥ 5 real cascades at the horizon of interest "
            "(descriptive only; multiple-testing not adjusted)."
        )
        lines.append("")
        lines.append(
            "| symbol | H | n_cascades | AUC (real) | AUC (shuffled) | "
            "auc>0.60 & CI excl 0.50 & dist from shuffled? |"
        )
        lines.append("|---|---|---|---|---|---|")
        for h in horizons:
            sub_df: pd.DataFrame = pd.DataFrame(
                per_sym[per_sym["horizon"] == h]
            ).sort_values("auc", ascending=False)
            for _, r in sub_df.iterrows():
                auc_v = float(r["auc"])  # type: ignore[arg-type]
                lo_v = float(r["auc_lo"])  # type: ignore[arg-type]
                # Three-way conjunction per the prompt
                clears = (
                    math.isfinite(auc_v)
                    and auc_v > 0.60
                    and math.isfinite(lo_v)
                    and lo_v > 0.50
                    and bool(r["signal_distinguishable_from_shuffled"])
                )
                lines.append(
                    f"| {r['symbol']} | H{int(r['horizon'])} | "  # type: ignore[arg-type]
                    f"{int(r['n_cascades'])} | "  # type: ignore[arg-type]
                    f"{auc_v:.3f} [{lo_v:.3f}, {float(r['auc_hi']):.3f}] | "  # type: ignore[arg-type]
                    f"{float(r['auc_shuffled']):.3f} | "  # type: ignore[arg-type]
                    f"{'YES' if clears else 'NO'} |"
                )
        lines.append("")
        # Aggregate counter
        clears_count = 0
        for _, r in per_sym.iterrows():
            auc_v = float(r["auc"])  # type: ignore[arg-type]
            lo_v = float(r["auc_lo"])  # type: ignore[arg-type]
            if (
                math.isfinite(auc_v)
                and auc_v > 0.60
                and math.isfinite(lo_v)
                and lo_v > 0.50
                and bool(r["signal_distinguishable_from_shuffled"])
            ):
                clears_count += 1
        lines.append(
            f"**{clears_count} per-symbol cells clear AUC > 0.60 AND CI "
            f"excludes 0.50 AND are distinguishable from shuffled.**"
        )
    lines.append("")

    # ---------- 5. Precision-at-top-1% lift ----------
    lines.append("## 5. Precision-at-top-1% lift (pooled cross-symbol)")
    lines.append("")
    lines.append(
        "If we trade only when the model says cascade-likely (top 1% of "
        "windows by predicted probability), how often is it right?  For a "
        "10× tradeable lift at base rate ~0.5%, top-1% precision must be "
        "> 5%."
    )
    lines.append("")
    if pooled.empty:
        lines.append("**No pooled cells available.**")
    else:
        lines.append("| H | base rate | precision@top-1% | lift | recall@top-1% |")
        lines.append("|---|---|---|---|---|")
        for h in horizons:
            sub = pooled[pooled["horizon"] == h]
            if sub.empty:
                continue
            r = sub.iloc[0]
            prec = float(r["precision_at_top_1pct"])
            rec = float(r["recall_at_top_1pct"])
            lift = float(r["lift_at_top_1pct"])
            base = float(r["base_rate"])
            prec_s = f"{prec:.4f}" if math.isfinite(prec) else "—"
            rec_s = f"{rec:.4f}" if math.isfinite(rec) else "—"
            lift_s = f"{lift:.2f}" if math.isfinite(lift) else "—"
            lines.append(f"| H{h} | {base:.4f} | {prec_s} | {lift_s} | {rec_s} |")
        lines.append("")

    # ---------- 6. Verdict ----------
    lines.append("## 6. Verdict")
    lines.append("")
    if pooled.empty:
        lines.append(
            "**UNDERPOWERED — no pooled cells available.**  The April 1-13 "
            "raw data may be incomplete; see flags below."
        )
    else:
        # Decision: did either H100 or H500 clear distinguishable-from-shuffled?
        h100 = pooled[pooled["horizon"] == 100]
        h500 = pooled[pooled["horizon"] == 500]
        h100_dist = (
            bool(h100.iloc[0]["signal_distinguishable_from_shuffled"])
            if not h100.empty
            else False
        )
        h500_dist = (
            bool(h500.iloc[0]["signal_distinguishable_from_shuffled"])
            if not h500.empty
            else False
        )
        # Sample-size guard — distinguishability with very few cascades is fragile.
        n_h100 = int(h100.iloc[0]["n_cascades"]) if not h100.empty else 0
        n_h500 = int(h500.iloc[0]["n_cascades"]) if not h500.empty else 0

        if h100_dist or h500_dist:
            lines.append(
                "**YES (with caveats) — at least one binding horizon clears "
                "distinguishability from the shuffled-label baseline.**  "
                f"H100 dist: {h100_dist} (n_cascades = {n_h100}); H500 dist: "
                f"{h500_dist} (n_cascades = {n_h500}).  The 83-dim flat "
                "representation carries some real-cascade-precursor signal "
                "above pure noise on April 1-13.  Caveats: small n keeps the "
                "CI wide; a single cluster of same-day cascades can dominate "
                "a fold's AUC."
            )
        else:
            # Bracket: is it truly null or underpowered?
            min_lo_above_05 = False
            for h_df in (h100, h500):
                if not h_df.empty:
                    lo_v = float(h_df.iloc[0]["auc_lo"])
                    if math.isfinite(lo_v) and lo_v > 0.50:
                        min_lo_above_05 = True
                        break
            if min_lo_above_05:
                lines.append(
                    "**MARGINAL — pooled real-label AUC excludes 0.50 at "
                    "the 95% CI but does NOT distinguish from the shuffled "
                    "baseline.**  The shuffled baseline's CI is wide enough "
                    "(small n) to overlap real-label CI.  Cannot reject "
                    "noise hypothesis at the binding test."
                )
            else:
                lines.append(
                    "**UNDERPOWERED / NULL — pooled real-label AUC CI "
                    "brackets 0.50, indistinguishable from the shuffled "
                    "baseline at H100 and H500.**  Either the 83-dim flat "
                    "representation lacks real-cascade-precursor signal, or "
                    "the n=20-73 cascades on April 1-13 is too small to "
                    "tell.  Definitive answer requires more April data "
                    "(currently hold-out) or a different cascade definition."
                )
    lines.append("")

    # ---------- 7. Methodological flags ----------
    lines.append("## 7. Methodological flags")
    lines.append("")
    lines.append(
        "* **Leave-one-day-out independence.** The folds are leave-one-day-"
        "out (April 1-13 → ≤ 13 folds depending on data availability), but "
        "real liquidation cascades cluster intra-day (cascade contagion).  "
        "If a single day's cascades dominate the held-out day's AUC, fold-"
        "level AUC overstates true held-out performance.  The bootstrap CI "
        "captures sampling noise but not fold-clustering noise."
    )
    lines.append("")
    lines.append(
        "* **Per-symbol cells are descriptive, not binding.** With 3 horizons "
        "× ≤ 25 symbols of per-symbol comparisons, per-symbol AUC > 0.60 "
        "cells will appear by chance even under the null.  Treat per-symbol "
        "results as pattern hints, not standalone evidence."
    )
    lines.append("")
    lines.append(
        "* **Stage-1 contamination disclosure (gotcha #17).** April 1-13 was "
        "used for v1 diagnostic checks but the `cause` field was not "
        "studied — contamination on this stage-2 study is not a concern."
    )
    lines.append("")
    lines.append(
        "* **April 14+ hold-out preserved.** No raw or cached April 14+ data "
        "was loaded by this script."
    )
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"* {note}")
            lines.append("")

    lines.append(
        f"_Pipeline ran in {elapsed_sec:.1f} s.  CPU-only.  No April 14+ "
        f"data touched._"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Robustness analysis — day-clustered bootstrap on existing per-window predictions
# ---------------------------------------------------------------------------

# Robustness defaults
N_BOOT_ROBUSTNESS: int = 1000
ROBUSTNESS_HORIZONS: tuple[int, ...] = (100, 500)  # H50 underpowered (n=9)


def _precision_at_top_pct_pooled(
    proba: np.ndarray, labels: np.ndarray, *, top_pct: float
) -> float:
    """Pooled precision-at-top-`top_pct`.  Returns NaN on empty input."""
    n = len(proba)
    if n == 0:
        return float("nan")
    n_top = max(1, int(round(n * top_pct)))
    order = np.argsort(-np.asarray(proba, dtype=float), kind="stable")
    top = order[:n_top]
    return float(np.asarray(labels)[top].sum() / float(n_top))


def _day_to_index_map(
    df: pd.DataFrame, *, date_col: str = "date"
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return (sorted unique day list, dict mapping day → integer row indices)."""
    dates = sorted(df[date_col].unique().tolist())
    day_to_idx: dict[str, np.ndarray] = {}
    arr_dates = df[date_col].to_numpy()
    for d in dates:
        day_to_idx[d] = np.flatnonzero(arr_dates == d)
    return dates, day_to_idx


def _day_clustered_bootstrap_iter_indices(
    days: list[str],
    day_to_idx: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """One bootstrap draw: sample len(days) days WITH REPLACEMENT, return the
    concatenated row indices of ALL windows on those days (no within-day subsampling).
    """
    k = len(days)
    sampled = rng.integers(0, k, size=k)
    parts = [day_to_idx[days[s]] for s in sampled]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def _day_clustered_bootstrap_auc(
    df: pd.DataFrame,
    *,
    proba_col: str,
    label_col: str = "real_cascade_label",
    date_col: str = "date",
    n_boot: int = N_BOOT_ROBUSTNESS,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Day-clustered bootstrap AUC: resample days with replacement, take ALL
    windows of each sampled day, compute AUC on the concatenated fold.

    Returns (point, lo, hi) where `point` is the BOOTSTRAP MEAN (not the
    full-pooled AUC — the prompt asks for "point estimate (mean across bootstrap
    folds)").
    """
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    days, day_to_idx = _day_to_index_map(df, date_col=date_col)
    if len(days) < 2:
        return float("nan"), float("nan"), float("nan")

    proba = df[proba_col].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=np.float64)
    aucs[:] = np.nan
    for b in range(n_boot):
        idx = _day_clustered_bootstrap_iter_indices(days, day_to_idx, rng=rng)
        if len(idx) == 0:
            continue
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs[b] = roc_auc_score(y_b, proba[idx])
        except ValueError:
            continue

    finite = aucs[np.isfinite(aucs)]
    if len(finite) < 10:
        return float("nan"), float("nan"), float("nan")
    point = float(finite.mean())
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return point, lo, hi


def _day_clustered_bootstrap_precision_at_top(
    df: pd.DataFrame,
    *,
    proba_col: str,
    label_col: str = "real_cascade_label",
    date_col: str = "date",
    top_pct: float = 0.01,
    n_boot: int = N_BOOT_ROBUSTNESS,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Day-clustered bootstrap precision-at-top-`top_pct` (mean + 95% CI).

    Each bootstrap draw concatenates ALL windows from sampled days, ranks by
    `proba_col`, and computes precision over the top `top_pct` slice.
    """
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    days, day_to_idx = _day_to_index_map(df, date_col=date_col)
    if len(days) < 2:
        return float("nan"), float("nan"), float("nan")

    proba = df[proba_col].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)
    precs = np.empty(n_boot, dtype=np.float64)
    precs[:] = np.nan
    for b in range(n_boot):
        idx = _day_clustered_bootstrap_iter_indices(days, day_to_idx, rng=rng)
        if len(idx) == 0:
            continue
        precs[b] = _precision_at_top_pct_pooled(
            proba[idx], labels[idx], top_pct=top_pct
        )

    finite = precs[np.isfinite(precs)]
    if len(finite) < 10:
        return float("nan"), float("nan"), float("nan")
    point = float(finite.mean())
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return point, lo, hi


def _per_day_attribution(
    df: pd.DataFrame,
    *,
    horizon: int,
    proba_col: str = "pred_proba",
    label_col: str = "real_cascade_label",
    date_col: str = "date",
    top_pct: float = 0.01,
) -> pd.DataFrame:
    """Per-day diagnostics + leave-one-day-out pooled AUC drop.

    For each day d:
      * n_cascades, n_windows
      * AUC computed on day d alone (held-out predictions for that day)
      * precision-at-top-`top_pct` on day d alone
      * leave_out_pooled_auc_drop = pooled AUC(all days) − pooled AUC(all days \\ d)
        — POSITIVE values mean removing the day HURT the pooled AUC (day was carrying signal).
    """
    sub = pd.DataFrame(df[df["horizon"] == horizon]).copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "horizon",
                "n_cascades",
                "n_windows",
                "auc",
                "precision_top_1pct",
                "leave_out_pooled_auc_drop",
            ]
        )

    pooled_proba = sub[proba_col].to_numpy(dtype=np.float64)
    pooled_labels = sub[label_col].to_numpy(dtype=np.int64)
    if len(np.unique(pooled_labels)) < 2:
        pooled_auc = float("nan")
    else:
        try:
            pooled_auc = float(roc_auc_score(pooled_labels, pooled_proba))
        except ValueError:
            pooled_auc = float("nan")

    arr_dates = sub[date_col].to_numpy()
    rows: list[dict] = []
    for d in sorted(sub[date_col].unique().tolist()):
        mask = arr_dates == d
        day_proba = pooled_proba[mask]
        day_labels = pooled_labels[mask]
        n_windows = int(mask.sum())
        n_cascades = int(day_labels.sum())

        if n_cascades == 0 or n_cascades == n_windows:
            auc = float("nan")  # single-class within day → AUC undefined
        else:
            try:
                auc = float(roc_auc_score(day_labels, day_proba))
            except ValueError:
                auc = float("nan")

        prec_top = _precision_at_top_pct_pooled(day_proba, day_labels, top_pct=top_pct)

        rest_mask = ~mask
        rest_labels = pooled_labels[rest_mask]
        rest_proba = pooled_proba[rest_mask]
        if len(np.unique(rest_labels)) < 2:
            rest_auc = float("nan")
        else:
            try:
                rest_auc = float(roc_auc_score(rest_labels, rest_proba))
            except ValueError:
                rest_auc = float("nan")
        if math.isnan(pooled_auc) or math.isnan(rest_auc):
            drop = float("nan")
        else:
            drop = pooled_auc - rest_auc

        rows.append(
            {
                "date": d,
                "horizon": int(horizon),
                "n_cascades": n_cascades,
                "n_windows": n_windows,
                "auc": auc,
                "precision_top_1pct": prec_top,
                "leave_out_pooled_auc_drop": drop,
                "pooled_auc_full": pooled_auc,
            }
        )
    return pd.DataFrame(rows)


def _run_robustness_pipeline(
    per_window_path: Path,
    out_dir: Path,
    *,
    horizons: tuple[int, ...] = ROBUSTNESS_HORIZONS,
    n_boot: int = N_BOOT_ROBUSTNESS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day-clustered bootstrap robustness analysis on the existing per-window
    parquet from the cascade-real probe.

    Returns (per_horizon_summary_df, per_day_df).  Both are also written to disk
    as `cascade_precursor_real_robustness_summary.csv` and
    `cascade_precursor_real_per_day.csv`.
    """
    if not per_window_path.exists():
        raise RuntimeError(
            f"per-window parquet not found: {per_window_path}.  Run "
            f"--cascade-real first to generate it."
        )
    df = pd.read_parquet(per_window_path)
    required_cols = {
        "date",
        "horizon",
        "real_cascade_label",
        "pred_proba",
        "pred_proba_shuffled",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"per-window parquet missing required columns: {missing}.  "
            f"Expected from cascade_precursor_real_per_window.parquet."
        )

    # Restrict to pooled scope (cross-symbol pooled rows are how the table is built)
    if "scope" in df.columns:
        df = pd.DataFrame(df[df["scope"] == "pooled"]).copy()

    summary_rows: list[dict] = []
    per_day_frames: list[pd.DataFrame] = []
    for h in horizons:
        sub = pd.DataFrame(df[df["horizon"] == h]).copy()
        if sub.empty:
            print(f"[robustness] H{h}: no rows in per-window parquet — skipping")
            continue

        # Real-AUC day-clustered bootstrap
        real_pt, real_lo, real_hi = _day_clustered_bootstrap_auc(
            sub,
            proba_col="pred_proba",
            n_boot=n_boot,
            seed=seed,
        )
        # Shuffled-AUC day-clustered bootstrap (uses the SAVED shuffled predictions
        # — does NOT re-shuffle.  Honors the prompt's hard constraint.)
        shuf_pt, shuf_lo, shuf_hi = _day_clustered_bootstrap_auc(
            sub,
            proba_col="pred_proba_shuffled",
            n_boot=n_boot,
            seed=seed + 1,
        )
        # Real precision-at-top-1% day-clustered bootstrap
        prec_pt, prec_lo, prec_hi = _day_clustered_bootstrap_precision_at_top(
            sub,
            proba_col="pred_proba",
            n_boot=n_boot,
            seed=seed + 2,
        )
        shuf_prec_pt, shuf_prec_lo, shuf_prec_hi = (
            _day_clustered_bootstrap_precision_at_top(
                sub,
                proba_col="pred_proba_shuffled",
                n_boot=n_boot,
                seed=seed + 3,
            )
        )

        # Distinguishability: real lo > shuffled hi, on the day-clustered CIs.
        dist = math.isfinite(real_lo) and math.isfinite(shuf_hi) and (real_lo > shuf_hi)

        summary_rows.append(
            {
                "horizon": int(h),
                "n_total": int(len(sub)),
                "n_cascades": int(sub["real_cascade_label"].sum()),  # type: ignore[arg-type]
                "n_days": int(sub["date"].nunique()),  # type: ignore[arg-type]
                "auc_real_dayboot": real_pt,
                "auc_real_dayboot_lo": real_lo,
                "auc_real_dayboot_hi": real_hi,
                "auc_shuffled_dayboot": shuf_pt,
                "auc_shuffled_dayboot_lo": shuf_lo,
                "auc_shuffled_dayboot_hi": shuf_hi,
                "precision_top_1pct_real_dayboot": prec_pt,
                "precision_top_1pct_real_dayboot_lo": prec_lo,
                "precision_top_1pct_real_dayboot_hi": prec_hi,
                "precision_top_1pct_shuffled_dayboot": shuf_prec_pt,
                "precision_top_1pct_shuffled_dayboot_lo": shuf_prec_lo,
                "precision_top_1pct_shuffled_dayboot_hi": shuf_prec_hi,
                "distinguishable_dayclustered": bool(dist),
            }
        )

        # Per-day attribution
        per_day = _per_day_attribution(sub, horizon=h)
        per_day_frames.append(per_day)

    summary_df = pd.DataFrame(summary_rows)
    per_day_df = (
        pd.concat(per_day_frames, ignore_index=True)
        if per_day_frames
        else pd.DataFrame(
            columns=[
                "date",
                "horizon",
                "n_cascades",
                "n_windows",
                "auc",
                "precision_top_1pct",
                "leave_out_pooled_auc_drop",
                "pooled_auc_full",
            ]
        )
    )

    summary_df.to_csv(
        out_dir / "cascade_precursor_real_robustness_summary.csv", index=False
    )
    # Per-day CSV — drop the helper `pooled_auc_full` column to keep the schema
    # focused on what the prompt's per-day spec asked for.
    per_day_for_csv = per_day_df.drop(columns=["pooled_auc_full"], errors="ignore")
    per_day_for_csv.to_csv(out_dir / "cascade_precursor_real_per_day.csv", index=False)

    return summary_df, per_day_df


def _emit_robustness_markdown(
    summary_df: pd.DataFrame,
    per_day_df: pd.DataFrame,
    out_path: Path,
    *,
    elapsed_sec: float,
    n_boot: int,
) -> None:
    """Render the day-clustered robustness markdown."""
    lines: list[str] = []
    lines.append(
        "# Goal-A cascade-precursor (stage 2) — day-clustered bootstrap robustness"
    )
    lines.append("")
    lines.append(
        "**Question.** The stage-2 real-cause-flag probe reported pooled AUC = "
        "0.817 [0.771, 0.858] at H500 with per-window bootstrap.  Cascade events "
        "cluster intra-day (contagion), so the leave-one-day-out folds may not "
        "be independent.  This document re-tests the AUC and precision-at-top-1% "
        "under a **day-clustered bootstrap** (resample the 7 cascade days WITH "
        "REPLACEMENT, take ALL windows from each sampled day, repeat "
        f"{n_boot}×).  The binding distinguishability test compares the "
        "day-clustered real-AUC lower bound against the day-clustered shuffled-"
        "AUC upper bound."
    )
    lines.append("")
    lines.append(
        "**Inputs.** Reuses `cascade_precursor_real_per_window.parquet` (held-out "
        "predictions from the stage-2 LR run).  No re-training; this is a "
        "bootstrap analysis on existing predictions.  Shuffled-baseline AUC uses "
        "the saved `pred_proba_shuffled` column — same predictions as the prior "
        "run, NOT a fresh shuffle."
    )
    lines.append("")

    # ---------- Section 1: AUC ----------
    lines.append("## 1. Day-clustered AUC at H100 and H500")
    lines.append("")
    lines.append(
        "| H | AUC real (mean over boot) | 95% CI (day-clustered) | "
        "Per-window CI (prior run) |"
    )
    lines.append("|---|---|---|---|")
    prior_ci = {100: "[0.689, 0.870]", 500: "[0.771, 0.858]"}
    for _, r in summary_df.iterrows():
        h = int(r["horizon"])  # type: ignore[arg-type]
        lines.append(
            f"| H{h} | {float(r['auc_real_dayboot']):.3f} | "  # type: ignore[arg-type]
            f"[{float(r['auc_real_dayboot_lo']):.3f}, "  # type: ignore[arg-type]
            f"{float(r['auc_real_dayboot_hi']):.3f}] | "  # type: ignore[arg-type]
            f"{prior_ci.get(h, '—')} |"
        )
    lines.append("")

    # ---------- Section 2: Shuffled baseline ----------
    lines.append("## 2. Day-clustered shuffled-baseline AUC + distinguishability")
    lines.append("")
    lines.append(
        "| H | AUC shuffled (mean over boot) | 95% CI (day-clustered) | "
        "real lo > shuffled hi? |"
    )
    lines.append("|---|---|---|---|")
    for _, r in summary_df.iterrows():
        h = int(r["horizon"])  # type: ignore[arg-type]
        lines.append(
            f"| H{h} | {float(r['auc_shuffled_dayboot']):.3f} | "  # type: ignore[arg-type]
            f"[{float(r['auc_shuffled_dayboot_lo']):.3f}, "  # type: ignore[arg-type]
            f"{float(r['auc_shuffled_dayboot_hi']):.3f}] | "  # type: ignore[arg-type]
            f"{'YES' if bool(r['distinguishable_dayclustered']) else 'NO'} |"
        )
    lines.append("")

    # ---------- Section 3: Per-day attribution ----------
    lines.append("## 3. Per-day attribution (H500)")
    lines.append("")
    h500 = pd.DataFrame(per_day_df[per_day_df["horizon"] == 500]).copy()
    if h500.empty:
        lines.append("**No H500 per-day rows produced.**")
    else:
        h500 = h500.sort_values("date").reset_index(drop=True)
        lines.append(
            "| date | n_cascades | n_windows | day AUC | precision@top-1% | "
            "leave-this-day-out pooled AUC drop |"
        )
        lines.append("|---|---|---|---|---|---|")
        for _, r in h500.iterrows():
            auc_v = float(r["auc"])  # type: ignore[arg-type]
            auc = f"{auc_v:.3f}" if math.isfinite(auc_v) else "n/a (single-class)"
            prec = float(r["precision_top_1pct"])  # type: ignore[arg-type]
            drop_v = float(r["leave_out_pooled_auc_drop"])  # type: ignore[arg-type]
            drop = f"{drop_v:+.3f}" if math.isfinite(drop_v) else "n/a"
            lines.append(
                f"| {r['date']} | {int(r['n_cascades'])} | "  # type: ignore[arg-type]
                f"{int(r['n_windows'])} | {auc} | {prec:.3f} | {drop} |"  # type: ignore[arg-type]
            )
        lines.append("")

        # Top-2 days by leave-one-day-out drop (positive = removing day HURT pooled AUC)
        h500_sorted = h500.dropna(subset=["leave_out_pooled_auc_drop"]).sort_values(
            "leave_out_pooled_auc_drop", ascending=False
        )
        if len(h500_sorted) >= 2:
            top2 = h500_sorted.head(2)
            top2_dates = ", ".join(top2["date"].tolist())
            top2_drops = ", ".join(
                f"{float(d):+.3f}" for d in top2["leave_out_pooled_auc_drop"].tolist()
            )
            # Compute pooled AUC if BOTH top-2 days are removed
            pooled_full = (
                float(h500.iloc[0]["pooled_auc_full"])
                if (
                    "pooled_auc_full" in h500.columns
                    and math.isfinite(float(h500.iloc[0]["pooled_auc_full"]))
                )
                else float("nan")
            )
            lines.append(
                f"**Top 2 days driving the pooled AUC: {top2_dates} "
                f"(leave-out drops {top2_drops}).**  Pooled H500 AUC = "
                f"{pooled_full:.3f}."
            )
            lines.append("")

    # ---------- Section 4: Day-clustered precision at top-1% ----------
    lines.append("## 4. Day-clustered precision-at-top-1% at H500")
    lines.append("")
    h500_sum = pd.DataFrame(summary_df[summary_df["horizon"] == 500]).copy()
    if h500_sum.empty:
        lines.append("**No H500 summary row.**")
    else:
        r = h500_sum.iloc[0]
        prec = float(r["precision_top_1pct_real_dayboot"])
        lines.append(
            f"Real precision@top-1% (day-clustered mean): "
            f"{prec:.3f} "
            f"[{float(r['precision_top_1pct_real_dayboot_lo']):.3f}, "
            f"{float(r['precision_top_1pct_real_dayboot_hi']):.3f}]"
        )
        lines.append("")
        lines.append(
            f"Shuffled precision@top-1% (day-clustered mean): "
            f"{float(r['precision_top_1pct_shuffled_dayboot']):.3f} "
            f"[{float(r['precision_top_1pct_shuffled_dayboot_lo']):.3f}, "
            f"{float(r['precision_top_1pct_shuffled_dayboot_hi']):.3f}]"
        )
        lines.append("")
        lines.append(
            f"Above the 5% (10× lift over base rate ~0.5%) tradeable threshold? "
            f"**{'YES' if prec > 0.05 else 'NO'}** "
            f"(threshold from cascade_precursor_real.md §5)."
        )
        lines.append("")

    # ---------- Section 5: Verdict ----------
    lines.append("## 5. Verdict")
    lines.append("")
    h500_sum_full = pd.DataFrame(summary_df[summary_df["horizon"] == 500])
    if h500_sum_full.empty:
        lines.append("**No H500 results — cannot render verdict.**")
    else:
        r = h500_sum_full.iloc[0]
        dist = bool(r["distinguishable_dayclustered"])
        # Find the leave-out drops
        h500 = pd.DataFrame(per_day_df[per_day_df["horizon"] == 500]).copy()
        h500 = h500.dropna(subset=["leave_out_pooled_auc_drop"]).sort_values(
            "leave_out_pooled_auc_drop", ascending=False
        )
        if len(h500) >= 2:
            top2_drops = float(h500.iloc[0]["leave_out_pooled_auc_drop"]) + float(
                h500.iloc[1]["leave_out_pooled_auc_drop"]
            )
            top2_dates = h500.head(2)["date"].tolist()
        else:
            top2_drops = float("nan")
            top2_dates = []
        verdict_kind = (
            "robust to day-level clustering"
            if dist
            and (
                math.isfinite(top2_drops)
                and (float(r["auc_real_dayboot"]) - top2_drops > 0.60)
            )
            else (
                "distinguishable from shuffled but contagion-leaning"
                if dist
                else "contagion-dominated"
            )
        )
        lines.append(
            f"Day-clustered H500 real AUC = {float(r['auc_real_dayboot']):.3f} "
            f"[{float(r['auc_real_dayboot_lo']):.3f}, "
            f"{float(r['auc_real_dayboot_hi']):.3f}].  "
            f"Day-clustered H500 shuffled AUC = "
            f"{float(r['auc_shuffled_dayboot']):.3f} "
            f"[{float(r['auc_shuffled_dayboot_lo']):.3f}, "
            f"{float(r['auc_shuffled_dayboot_hi']):.3f}].  "
            f"Real lo > shuffled hi: **{'YES' if dist else 'NO'}**.  "
            f"Top 2 days driving signal: "
            f"{', '.join(top2_dates) if top2_dates else 'n/a'} "
            f"(combined leave-out drop = "
            f"{top2_drops:+.3f}{')' if math.isfinite(top2_drops) else ' — n/a)'}."
            f"  Implied pooled AUC after removing both: "
            f"{(float(r['auc_real_dayboot']) - top2_drops):.3f}."
            if math.isfinite(top2_drops)
            else (
                f"Day-clustered H500 real AUC = "
                f"{float(r['auc_real_dayboot']):.3f}"
                f" [{float(r['auc_real_dayboot_lo']):.3f}, "
                f"{float(r['auc_real_dayboot_hi']):.3f}]."
            )
        )
        lines.append("")
        lines.append(f"**Result kind: {verdict_kind}.**")
        lines.append("")

    lines.append("## 6. Methodological notes")
    lines.append("")
    lines.append(
        "* **Day-clustered bootstrap.** Each iteration samples 7 days WITH "
        "replacement from the 7 available cascade days (Apr 3, 4, 6, 7, 9, 10, "
        "13), takes ALL windows from each sampled day, computes AUC on the "
        "concatenated fold.  This treats each day as the unit of independence, "
        "consistent with the cascade-contagion concern flagged in the stage-2 "
        "writeup."
    )
    lines.append("")
    lines.append(
        "* **Shuffled baseline reuses saved predictions.** The shuffled-label "
        "AUC uses the `pred_proba_shuffled` column already in the per-window "
        "parquet — no fresh shuffle, no re-training."
    )
    lines.append("")
    lines.append(
        "* **Per-day AUC may be NaN** when a day has zero or all positive "
        "labels (single-class within day → AUC undefined).  These rows are "
        "dropped from the leave-one-day-out attribution but remain in the CSV."
    )
    lines.append("")
    lines.append(
        "* **April 14+ untouched.** No raw or cached April 14+ data was loaded."
    )
    lines.append("")
    lines.append(
        f"_Robustness analysis ran in {elapsed_sec:.1f} s on existing per-"
        f"window predictions ({n_boot} bootstrap iterations).  No LR re-"
        f"training, no April 14+ data touched._"
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
    parser.add_argument(
        "--cascade-real",
        action="store_true",
        help=(
            "Run the stage-2 real-cause-flag probe instead of the stage-1 "
            "synthetic-label probe.  Uses leave-one-day-out CV on April 1-13 "
            "with the 83-dim flat baseline; emits "
            "cascade_precursor_real_{table.csv, per_window.parquet, .md}."
        ),
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help=(
            "Run the day-clustered bootstrap robustness analysis on the "
            "existing cascade_precursor_real_per_window.parquet.  Reuses saved "
            "predictions (no LR re-training).  Emits "
            "cascade_precursor_real_robustness.md, "
            "cascade_precursor_real_per_day.csv, and "
            "cascade_precursor_real_robustness_summary.csv."
        ),
    )
    parser.add_argument(
        "--per-window-path",
        type=Path,
        default=None,
        help=(
            "Override path to the cascade_precursor_real_per_window.parquet "
            "input for --robustness (default: <out-dir>/"
            "cascade_precursor_real_per_window.parquet)."
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT_REAL,
        help=f"Bootstrap iterations for AUC CI (default: {N_BOOT_REAL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed",
    )
    args = parser.parse_args()

    horizons: tuple[int, ...] = tuple(int(h) for h in args.horizons)
    symbols: tuple[str, ...] = tuple(args.symbols)
    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.robustness:
        # ---------- Stage 2 follow-up: day-clustered bootstrap robustness ----------
        per_window_path: Path = (
            Path(args.per_window_path)
            if args.per_window_path is not None
            else out_dir / "cascade_precursor_real_per_window.parquet"
        )
        rob_horizons: tuple[int, ...] = tuple(
            h for h in horizons if h in ROBUSTNESS_HORIZONS
        )
        if not rob_horizons:
            rob_horizons = ROBUSTNESS_HORIZONS
        print(
            f"[robustness] day-clustered bootstrap | horizons={rob_horizons} | "
            f"n_boot={int(args.n_boot)} | per_window={per_window_path}"
        )
        try:
            summary_df, per_day_df = _run_robustness_pipeline(
                per_window_path,
                out_dir,
                horizons=rob_horizons,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        except RuntimeError as exc:
            print(f"[robustness] BLOCKER: {exc}")
            return 1
        elapsed = time.time() - t0
        _emit_robustness_markdown(
            summary_df,
            per_day_df,
            out_dir / "cascade_precursor_real_robustness.md",
            elapsed_sec=elapsed,
            n_boot=int(args.n_boot),
        )
        print(f"[robustness] done in {elapsed:.1f} s")
        print(
            f"[robustness] wrote " f"{out_dir / 'cascade_precursor_real_robustness.md'}"
        )
        print(
            f"[robustness] wrote " f"{out_dir / 'cascade_precursor_real_per_day.csv'}"
        )
        print(
            f"[robustness] wrote "
            f"{out_dir / 'cascade_precursor_real_robustness_summary.csv'}"
        )
        return 0

    if args.cascade_real:
        # ---------- Stage 2: real cause-flag probe ----------
        real_horizons: tuple[int, ...] = tuple(
            h for h in horizons if h in REAL_HORIZONS
        )
        if not real_horizons:
            real_horizons = REAL_HORIZONS
        print(
            f"[cascade-real] stage-2 probe | horizons={real_horizons} | "
            f"symbols={len(symbols)} | cache={args.cache}"
        )
        try:
            per_window_df, per_cell_df, _ = _run_cascade_real_pipeline(
                args.cache,
                symbols,
                out_dir,
                horizons=real_horizons,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        except RuntimeError as exc:
            print(f"[cascade-real] BLOCKER: {exc}")
            return 1

        per_cell_df.to_csv(out_dir / "cascade_precursor_real_table.csv", index=False)
        per_window_df.to_parquet(
            out_dir / "cascade_precursor_real_per_window.parquet", index=False
        )
        elapsed = time.time() - t0
        _emit_real_markdown(
            per_cell_df,
            out_dir / "cascade_precursor_real.md",
            horizons=real_horizons,
            elapsed_sec=elapsed,
        )
        print(f"[cascade-real] done in {elapsed:.1f} s")
        print(
            f"[cascade-real] wrote " f"{out_dir / 'cascade_precursor_real_table.csv'}"
        )
        print(
            f"[cascade-real] wrote "
            f"{out_dir / 'cascade_precursor_real_per_window.parquet'}"
        )
        print(f"[cascade-real] wrote {out_dir / 'cascade_precursor_real.md'}")
        return 0

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

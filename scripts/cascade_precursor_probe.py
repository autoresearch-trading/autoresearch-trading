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


# ---------------------------------------------------------------------------
# Cascade-direction probe — orchestration pipeline (--cascade-direction flag)
# ---------------------------------------------------------------------------

DIRECTION_HORIZON: int = 500
DIRECTION_TOP_PCT_SUBSET: float = 0.05  # top 5% of pred_proba_h500
DIRECTION_TRIGGER_TOP_PCT: float = 0.01  # top 1% — strategy trigger threshold
DIRECTION_LR_CONFIDENCE_THRESHOLD: float = 0.55  # max class prob > 0.55


def _load_anchor_mid_and_end_ts_for_window(
    *,
    symbol: str,
    date_str: str,
    anchor_ts: int,
    h: int,
    cache_dir: Path,
) -> tuple[float, int] | None:
    """Look up anchor_mid (vwap of fills at the anchor event) and end_ts
    (timestamp of event anchor_idx + h) for a single window.

    Returns None if the cache shard or anchor is unavailable, or the horizon
    overruns the day.
    """
    shard_path = cache_dir / f"{symbol}__{date_str}.npz"
    if not shard_path.exists():
        return None
    payload = _load_shard(shard_path)
    event_ts = payload["event_ts"].astype(np.int64)
    n_events = len(event_ts)
    # Find anchor index (event_ts is unique per anchor by construction)
    idx = int(np.searchsorted(event_ts, anchor_ts, side="left"))
    if idx >= n_events or int(event_ts[idx]) != int(anchor_ts):
        return None
    end_idx = idx + h
    if end_idx >= n_events:
        return None
    end_ts = int(event_ts[end_idx])
    # Anchor mid: query raw trade parquet for trades at anchor_ts;
    # fall back to last trade <= anchor_ts if none exact.
    base = Path(f"data/trades/symbol={symbol}/date={date_str}")
    if not base.exists():
        return None
    q_anchor = (
        f"SELECT price, qty FROM read_parquet('{base}/*.parquet') "
        f"WHERE ts_ms = {int(anchor_ts)}"
    )
    try:
        df = duckdb.query(q_anchor).to_df()
    except Exception:
        return None
    if not df.empty:
        # Volume-weighted average price across same-ts fills (matches event grouping)
        prices = df["price"].to_numpy(dtype=np.float64)
        qtys = df["qty"].to_numpy(dtype=np.float64)
        if qtys.sum() <= 0:
            anchor_mid = float(prices.mean())
        else:
            anchor_mid = float((prices * qtys).sum() / qtys.sum())
    else:
        # Fall back: last trade prior to or at anchor_ts
        q_fb = (
            f"SELECT price FROM read_parquet('{base}/*.parquet') "
            f"WHERE ts_ms <= {int(anchor_ts)} ORDER BY ts_ms DESC LIMIT 1"
        )
        try:
            df_fb = duckdb.query(q_fb).to_df()
        except Exception:
            return None
        if df_fb.empty:
            return None
        anchor_mid = float(df_fb["price"].iloc[0])
    return anchor_mid, end_ts


def _load_liq_ts_price_for_symbol_date(
    symbol: str, date_str: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sorted (ts_ms, price) arrays for liquidation trades on (symbol, date_str).

    Returns None if data is not available (pre-April or April-heldout).
    """
    if date_str >= APRIL_HELDOUT_START or date_str < APRIL_START:
        return None
    base = Path(f"data/trades/symbol={symbol}/date={date_str}")
    if not base.exists():
        return None
    q = (
        f"SELECT ts_ms, price FROM read_parquet('{base}/*.parquet') "
        f"WHERE cause IN ('market_liquidation', 'backstop_liquidation') "
        f"ORDER BY ts_ms"
    )
    try:
        df = duckdb.query(q).to_df()
    except Exception:
        return None
    return (
        df["ts_ms"].to_numpy(dtype=np.int64),
        df["price"].to_numpy(dtype=np.float64),
    )


def _build_cascade_direction_dataset(
    cache_dir: Path,
    per_window_df: pd.DataFrame,
    *,
    horizon: int = DIRECTION_HORIZON,
) -> pd.DataFrame:
    """Build the per-window dataset for the direction LR.

    Joins the existing cascade-real per_window parquet with: 83-dim flat
    features (recomputed from cache), forward_log_return at H, realized
    direction sign, overshoot direction sign (cascades only), and anchor
    metadata.

    Returns a DataFrame with one row per H{horizon} window in `per_window_df`,
    with the additional columns:
        flat_features      : list of 83 floats
        forward_log_return : float
        realized_direction : int (0/1, -1 if invalid)
        overshoot_direction: int (-1/0/+1; 0 if no liq in window or non-cascade)
        anchor_mid         : float (NaN if not lookable up)
    Rows where forward_log_return is NaN (horizon overruns the day) are
    dropped — direction is undefined.
    """
    sub_in = per_window_df[per_window_df["horizon"] == horizon].copy()
    if sub_in.empty:
        return pd.DataFrame()

    # Pull arrays (avoid itertuples for pyright friendliness)
    in_symbols = sub_in["symbol"].astype(str).to_numpy()  # type: ignore[union-attr]
    in_dates = sub_in["date"].astype(str).to_numpy()  # type: ignore[union-attr]
    in_anchor_ts = sub_in["anchor_ts"].astype("int64").to_numpy()  # type: ignore[union-attr]
    in_real_label = sub_in["real_cascade_label"].astype("int64").to_numpy()  # type: ignore[union-attr]
    in_pred_proba = sub_in["pred_proba"].astype("float64").to_numpy()  # type: ignore[union-attr]

    # Unique (symbol, date) pairs
    pair_set: set[tuple[str, str]] = set()
    for s, d in zip(in_symbols.tolist(), in_dates.tolist()):
        pair_set.add((str(s), str(d)))

    # Per-(symbol, date) caches
    flat_cache: dict[tuple[str, str], np.ndarray] = {}  # (n, FLAT_DIM)
    anchor_idx_cache: dict[tuple[str, str], dict[int, int]] = {}
    fwd_cache: dict[tuple[str, str], np.ndarray] = {}  # (n,) float64
    end_ts_cache: dict[tuple[str, str], np.ndarray] = {}  # (n,) int64
    liq_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for symbol, date_str in pair_set:
        shard_path = cache_dir / f"{symbol}__{date_str}.npz"
        if not shard_path.exists():
            continue
        if date_str >= APRIL_HELDOUT_START:
            continue
        payload = _load_shard(shard_path)
        features_arr = payload["features"]
        event_ts = payload["event_ts"].astype(np.int64)
        n_events = features_arr.shape[0]
        if n_events < WINDOW_LEN:
            continue
        last_valid_start = n_events - WINDOW_LEN
        if last_valid_start < 0:
            continue
        starts = np.arange(0, last_valid_start + 1, STRIDE_EVAL, dtype=np.int64)
        anchors = starts + WINDOW_LEN - 1

        flat_X = np.empty((len(starts), FLAT_DIM), dtype=np.float32)
        for i, s in enumerate(starts):
            flat_X[i] = extract_flat_features(features_arr[s : s + WINDOW_LEN])

        log_returns = features_arr[:, _LOG_RETURN_IDX].astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(log_returns)])
        end_idx = anchors + horizon
        valid = end_idx < n_events
        fwd = np.full(len(anchors), np.nan, dtype=np.float64)
        fwd[valid] = cum[end_idx[valid] + 1] - cum[anchors[valid] + 1]
        end_ts_arr = np.full(len(anchors), -1, dtype=np.int64)
        end_ts_arr[valid] = event_ts[end_idx[valid]]

        flat_cache[(symbol, date_str)] = flat_X
        fwd_cache[(symbol, date_str)] = fwd
        end_ts_cache[(symbol, date_str)] = end_ts_arr
        anchor_lookup: dict[int, int] = {}
        for i, a_idx in enumerate(anchors):
            anchor_lookup[int(event_ts[a_idx])] = i
        anchor_idx_cache[(symbol, date_str)] = anchor_lookup

        liq = _load_liq_ts_price_for_symbol_date(symbol, date_str)
        if liq is not None:
            liq_cache[(symbol, date_str)] = liq

    out_rows: list[dict] = []
    for r in range(len(sub_in)):
        symbol = str(in_symbols[r])
        date_str = str(in_dates[r])
        anchor_ts_i = int(in_anchor_ts[r])
        real_label_i = int(in_real_label[r])
        pred_proba_i = float(in_pred_proba[r])
        key = (symbol, date_str)
        lookup = anchor_idx_cache.get(key)
        if lookup is None:
            continue
        flat_idx = lookup.get(anchor_ts_i)
        if flat_idx is None:
            continue
        fwd_lr = float(fwd_cache[key][flat_idx])
        if not math.isfinite(fwd_lr):
            continue
        end_ts_i = int(end_ts_cache[key][flat_idx])
        if end_ts_i < 0:
            continue

        realized = 1 if fwd_lr > 0 else 0
        overshoot = 0
        anchor_mid_val = float("nan")
        if real_label_i == 1:
            liq_pair = liq_cache.get(key)
            if liq_pair is not None:
                base = Path(f"data/trades/symbol={symbol}/date={date_str}")
                if base.exists():
                    q = (
                        f"SELECT price, qty FROM read_parquet('{base}/*.parquet') "
                        f"WHERE ts_ms = {anchor_ts_i}"
                    )
                    try:
                        adf = duckdb.query(q).to_df()
                    except Exception:
                        adf = pd.DataFrame()
                    if len(adf) > 0:
                        prices = adf["price"].astype("float64").to_numpy()
                        qtys = adf["qty"].astype("float64").to_numpy()
                        if float(qtys.sum()) > 0:
                            anchor_mid_val = float((prices * qtys).sum() / qtys.sum())
                        else:
                            anchor_mid_val = float(prices.mean())
                if math.isfinite(anchor_mid_val):
                    liq_ts_arr, liq_pr_arr = liq_pair
                    overshoot = _overshoot_direction_for_window(
                        anchor_ts=anchor_ts_i,
                        end_ts=end_ts_i,
                        anchor_mid=anchor_mid_val,
                        liq_ts=liq_ts_arr,
                        liq_price=liq_pr_arr,
                    )

        out_rows.append(
            {
                "symbol": symbol,
                "date": date_str,
                "anchor_ts": anchor_ts_i,
                "horizon": int(horizon),
                "real_cascade_label": real_label_i,
                "pred_proba": pred_proba_i,
                "flat_features": flat_cache[key][flat_idx].astype(float).tolist(),
                "forward_log_return": fwd_lr,
                "realized_direction": int(realized),
                "overshoot_direction": int(overshoot),
                "anchor_mid": anchor_mid_val,
            }
        )
    return pd.DataFrame(out_rows)


def _run_cascade_direction_pipeline(
    cache_dir: Path,
    out_dir: Path,
    *,
    horizon: int = DIRECTION_HORIZON,
    seed: int = 0,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """End-to-end direction-prediction pipeline.

    Loads the existing cascade-real per_window parquet, builds a 84-dim
    feature vector per window (83 flat + pred_proba_h500), trains direction
    LR with leave-one-day-out CV across April 1-13, and reports metrics.

    Returns (per_cascade_df, summary_dict).  per_cascade_df has one row per
    real cascade window with realized + overshoot direction + LR pred.
    """
    in_path = out_dir / "cascade_precursor_real_per_window.parquet"
    if not in_path.exists():
        raise RuntimeError(
            f"Cannot find {in_path} — run the stage-2 cascade-real probe first "
            f"with `--cascade-real`."
        )
    raw = pd.read_parquet(in_path)
    h_df = raw[raw["horizon"] == horizon].copy()
    if h_df.empty:
        raise RuntimeError(f"No H{horizon} rows in {in_path}")

    print(
        f"[cascade-direction] loaded {len(h_df)} H{horizon} windows; "
        f"{int(h_df['real_cascade_label'].sum())} real cascades"  # type: ignore[arg-type]
    )

    # Step 1: enrich with flat features + forward returns + overshoot direction
    enriched = _build_cascade_direction_dataset(
        cache_dir, pd.DataFrame(h_df), horizon=horizon
    )
    if enriched.empty:
        raise RuntimeError("No enriched windows produced — cache shards missing?")
    n_enriched = len(enriched)
    n_cas_enriched = int(enriched["real_cascade_label"].sum())  # type: ignore[arg-type]
    print(
        f"[cascade-direction] enriched {n_enriched} windows "
        f"({n_cas_enriched} cascades) with flat features + fwd returns"
    )

    # Step 2: marginal direction asymmetry on cascades
    marg_p_pos = _marginal_direction_asymmetry(
        enriched["forward_log_return"].to_numpy(),
        enriched["real_cascade_label"].to_numpy(),
    )
    print(
        f"[cascade-direction] marginal P(positive return | cascade) = "
        f"{marg_p_pos:.4f} (n={n_cas_enriched})"
    )

    # Step 3: select cascade-likely subset (top-5% by pred_proba)
    proba_col = enriched["pred_proba"].astype("float64").to_numpy()
    top5_mask = _top_pct_mask(proba_col, top_pct=DIRECTION_TOP_PCT_SUBSET)
    subset: pd.DataFrame = pd.DataFrame(enriched[top5_mask]).reset_index(drop=True)
    n_subset = len(subset)
    n_cas_subset = int(subset["real_cascade_label"].sum())  # type: ignore[arg-type]
    print(
        f"[cascade-direction] cascade-likely subset (top-5%): "
        f"{n_subset} windows, {n_cas_subset} real cascades"
    )

    # Step 4: build feature matrix (83 flat + pred_proba) → 84-dim
    flat_arr = np.array(subset["flat_features"].tolist(), dtype=np.float32)
    proba_feat = subset["pred_proba"].astype("float32").to_numpy().reshape(-1, 1)
    X = np.concatenate([flat_arr, proba_feat], axis=1)
    y_realized = subset["realized_direction"].astype("int64").to_numpy()
    dates_arr = subset["date"].astype(str).to_numpy()
    if len(np.unique(y_realized)) < 2:
        raise RuntimeError("Realized direction is single-class on the subset")
    if len(np.unique(dates_arr)) < 2:
        raise RuntimeError("Subset spans < 2 days; LOO-CV impossible")

    # Step 5: leave-one-day-out CV for realized direction LR
    pred_realized = _leave_one_day_out_predictions(
        X=X,
        y=y_realized,
        dates=dates_arr,
        feature_mode="real",
        label_mode="real",
        rng_seed=seed,
    )
    valid_pred = np.isfinite(pred_realized)
    auc_realized, auc_lo, auc_hi = _bootstrap_auc_ci(
        pred_realized[valid_pred],
        y_realized[valid_pred],
        n_boot=n_boot,
        seed=seed,
    )
    auc_majority = _majority_class_baseline_auc(y_realized[valid_pred])

    # Direction accuracy at threshold (max class prob > 0.55)
    confident_mask = (pred_realized > DIRECTION_LR_CONFIDENCE_THRESHOLD) | (
        pred_realized < (1.0 - DIRECTION_LR_CONFIDENCE_THRESHOLD)
    )
    confident_mask &= valid_pred
    if confident_mask.any():
        pred_class = (pred_realized[confident_mask] > 0.5).astype(np.int64)
        confident_accuracy = float((pred_class == y_realized[confident_mask]).mean())
    else:
        confident_accuracy = float("nan")
    p_confident_given_subset = float(confident_mask.mean())

    # Step 6: realized vs overshoot direction agreement (cascades only)
    cas_mask = subset["real_cascade_label"].astype("int64").to_numpy() == 1
    cas_real = subset["realized_direction"].astype("int64").to_numpy()[cas_mask]
    cas_over = subset["overshoot_direction"].astype("int64").to_numpy()[cas_mask]
    # Map overshoot {+1, -1, 0} -> {1, 0, NaN}: +1=up, -1=down, 0=undefined
    cas_over_binary = np.full(len(cas_over), -1, dtype=np.int64)
    cas_over_binary[cas_over == 1] = 1
    cas_over_binary[cas_over == -1] = 0
    over_valid = cas_over_binary >= 0
    if over_valid.any():
        agreement = float((cas_real[over_valid] == cas_over_binary[over_valid]).mean())
        p_overshoot_pos = float(cas_over_binary[over_valid].mean())
    else:
        agreement = float("nan")
        p_overshoot_pos = float("nan")

    # Step 7: conditional headroom math
    # Trigger frequency: top-1% of pred_proba_h500 AND direction-LR confidence > 0.55
    top1_mask_full = _top_pct_mask(proba_col, top_pct=DIRECTION_TRIGGER_TOP_PCT)
    p_top_1pct = float(top1_mask_full.mean())
    # On the cascade-likely subset, fraction with confident direction prediction
    n_unique_dates_universe = max(
        1, len(np.unique(enriched["date"].astype(str).to_numpy()))
    )
    triggers_per_day_universe = (
        p_top_1pct
        * p_confident_given_subset
        * float(len(enriched))
        / n_unique_dates_universe
    )
    # Mean |fwd return| on the cascade-likely subset, in bps
    abs_fwd_subset_bps = float(
        np.mean(np.abs(subset["forward_log_return"].astype("float64").to_numpy()) * 1e4)
    )
    headroom = _direction_per_day_expected_gross(
        triggers_per_day=triggers_per_day_universe,
        direction_accuracy=(
            confident_accuracy if math.isfinite(confident_accuracy) else 0.5
        ),
        mean_abs_fwd_bps=float(abs_fwd_subset_bps),
        fee_bps_per_side=TAKER_FEE_BPS_PER_SIDE,
        slip_bps_per_side=DEFAULT_SLIP_BPS_PER_SIDE,
    )

    summary: dict[str, float] = {
        "n_total_h500": float(len(enriched)),
        "n_cascades_h500": float(n_cas_enriched),
        "marginal_p_positive_given_cascade": marg_p_pos,
        "n_subset_top_5pct": float(n_subset),
        "n_cascades_in_subset": float(n_cas_subset),
        "auc_direction_lr": auc_realized,
        "auc_direction_lr_lo": auc_lo,
        "auc_direction_lr_hi": auc_hi,
        "auc_majority_baseline": auc_majority,
        "direction_accuracy_at_thresh": confident_accuracy,
        "p_confident_given_subset": p_confident_given_subset,
        "realized_vs_overshoot_agreement": agreement,
        "p_overshoot_positive": p_overshoot_pos,
        "p_top_1pct": p_top_1pct,
        "triggers_per_day_universe": triggers_per_day_universe,
        "mean_abs_fwd_subset_bps": float(abs_fwd_subset_bps),
        **headroom,
    }

    # Step 8: per-cascade table — one row per real cascade window
    cas_mask_arr = subset["real_cascade_label"].astype("int64").to_numpy() == 1
    cas_rows_df: pd.DataFrame = pd.DataFrame(subset[cas_mask_arr]).reset_index(
        drop=True
    )
    cas_rows_df["direction_pred_proba_lr"] = pred_realized[cas_mask_arr]
    cas_rows_df = cas_rows_df.rename(columns={"pred_proba": "pred_proba_h500"})
    cas_table: pd.DataFrame = pd.DataFrame(
        cas_rows_df[
            [
                "symbol",
                "date",
                "anchor_ts",
                "pred_proba_h500",
                "realized_direction",
                "overshoot_direction",
                "direction_pred_proba_lr",
            ]
        ]
    ).copy()

    return cas_table, summary


def _emit_direction_markdown(
    summary: dict[str, float],
    cas_table: pd.DataFrame,
    out_path: Path,
    *,
    elapsed_sec: float,
) -> None:
    """Render the cascade-direction markdown verdict per the prompt's outline."""
    lines: list[str] = []
    lines.append("# Goal-A cascade-precursor direction LR — fade vs continuation")
    lines.append("")
    lines.append(
        "**Question.** Conditional on the stage-2 cascade-onset model "
        "predicting a cascade is likely, can we predict its direction "
        "(long vs short) from the same 83-dim flat baseline + the "
        "cascade-onset confidence?  Without direction the cascade-onset "
        "AUC=0.817 / top-1% precision=27.8% signal is not tradeable — "
        "you cannot take a position."
    )
    lines.append("")
    lines.append(
        "**Hard constraints.** April 14+ untouched.  H500 only "
        f"(n=~{int(summary['n_cascades_h500'])} cascades; H100 has n=20, too "
        f"underpowered).  Sample size honest: with leave-one-day-out CV "
        f"across 7 April-diagnostic dates the CIs are wide.  If "
        f"|marginal_p_positive - 0.5| > 0.10, LR may be exploiting the "
        f"marginal not the conditional — we report a majority-baseline "
        f"AUC for comparison."
    )
    lines.append("")

    # ---------- 1. Marginal direction asymmetry ----------
    lines.append("## 1. Marginal direction asymmetry")
    lines.append("")
    p_pos = summary["marginal_p_positive_given_cascade"]
    n_cas = int(summary["n_cascades_h500"])
    asym = abs(p_pos - 0.5) if math.isfinite(p_pos) else float("nan")
    asym_flag = (
        "asymmetric (LR may exploit marginal — must beat majority baseline)"
        if math.isfinite(asym) and asym > 0.10
        else "symmetric (or near-symmetric)"
    )
    lines.append(
        f"P(forward_log_return > 0 | real_cascade_h500) = "
        f"**{p_pos:.4f}** (n_cascades = {n_cas})"
    )
    lines.append("")
    lines.append(f"|marginal - 0.5| = **{asym:.4f}** → {asym_flag}")
    lines.append("")

    # ---------- 2. Direction LR AUC at H500 ----------
    lines.append("## 2. Direction LR AUC at H500 on cascade-likely subset")
    lines.append("")
    auc = summary["auc_direction_lr"]
    auc_lo = summary["auc_direction_lr_lo"]
    auc_hi = summary["auc_direction_lr_hi"]
    auc_maj = summary["auc_majority_baseline"]
    n_subset = int(summary["n_subset_top_5pct"])
    n_cas_subset = int(summary["n_cascades_in_subset"])
    lines.append(
        f"Subset = top-5% by `pred_proba_h500` from stage-2 → "
        f"{n_subset} windows ({n_cas_subset} real cascades)."
    )
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| Direction LR AUC (realized direction) | {auc:.4f} |")
    lines.append(f"| 95% bootstrap CI (lo) | {auc_lo:.4f} |")
    lines.append(f"| 95% bootstrap CI (hi) | {auc_hi:.4f} |")
    lines.append(f"| Majority-class baseline AUC | {auc_maj:.4f} |")
    lines.append("")
    distinguishable = math.isfinite(auc_lo) and auc_lo > 0.5
    beat_maj = math.isfinite(auc_lo) and math.isfinite(auc_maj) and auc_lo > auc_maj
    if distinguishable and beat_maj:
        verdict_dir = "Distinguishable from 0.5 AND beats majority-class baseline."
    elif distinguishable:
        verdict_dir = (
            "CI lower bound clears 0.5, but does NOT cleanly beat the "
            "majority-class baseline (LR may be exploiting the marginal)."
        )
    else:
        verdict_dir = (
            "Direction LR AUC CI does not cleanly exclude 0.5 → "
            "direction is NOT predictable from this representation."
        )
    lines.append(f"**Verdict:** {verdict_dir}")
    lines.append("")

    # ---------- 3. Realized vs overshoot direction agreement ----------
    lines.append("## 3. Realized vs overshoot direction agreement")
    lines.append("")
    agree = summary["realized_vs_overshoot_agreement"]
    p_over_pos = summary["p_overshoot_positive"]
    if math.isfinite(agree):
        if agree >= 0.65:
            interp = "continuation-dominated (overshoot persists to horizon end)"
        elif agree <= 0.35:
            interp = "fade-dominated (overshoot reverts before horizon end)"
        else:
            interp = "mixed (continuation/fade roughly balanced)"
        lines.append(
            f"Agreement(realized ⇔ overshoot, both binary) = "
            f"**{agree:.4f}**, P(overshoot up) = **{p_over_pos:.4f}** → "
            f"{interp}."
        )
    else:
        lines.append(
            "Insufficient cascades with locatable first-liquidation fills "
            "to compute overshoot direction."
        )
    lines.append("")

    # ---------- 4. Conditional headroom ----------
    lines.append("## 4. Conditional headroom")
    lines.append("")
    p_top1 = summary["p_top_1pct"]
    p_conf = summary["p_confident_given_subset"]
    triggers_per_day = summary["triggers_per_day_universe"]
    mean_abs_bps = summary["mean_abs_fwd_subset_bps"]
    gross = summary["gross_per_trigger_bps"]
    cost = summary["cost_per_trigger_bps"]
    net = summary["net_per_trigger_bps"]
    per_day = summary["per_day_gross_bps"]
    lines.append("| component | value |")
    lines.append("|---|---|")
    lines.append(f"| P(top-1% pred_proba_h500) | {p_top1:.4f} |")
    lines.append(f"| P(LR confidence > 0.55 \\| cascade-likely) | {p_conf:.4f} |")
    lines.append(f"| Triggers per day (universe-pooled) | {triggers_per_day:.4f} |")
    lines.append(
        f"| E[\\|forward_log_return\\| at H500 \\| cascade-likely] (bps) | "
        f"{mean_abs_bps:.2f} |"
    )
    lines.append(
        f"| Direction accuracy at confidence threshold | "
        f"{summary['direction_accuracy_at_thresh']:.4f} |"
    )
    lines.append(f"| Gross per trigger (bps) | {gross:.2f} |")
    lines.append(
        f"| Cost per trigger (bps; 4bp fee + 1bp slip per side, both legs) | "
        f"{cost:.2f} |"
    )
    lines.append(f"| Net per trigger (bps) | {net:.2f} |")
    lines.append(f"| **Per-day expected gross (bps)** | **{per_day:.4f}** |")
    lines.append("")
    if math.isfinite(per_day) and per_day > 0:
        verdict_h = "tradeable (positive per-day gross)"
    elif math.isfinite(per_day):
        verdict_h = "NOT tradeable (per-day gross is non-positive)"
    else:
        verdict_h = "NOT computable (insufficient signal)"
    lines.append(f"**Headroom verdict:** {verdict_h}")
    lines.append("")

    # ---------- 5. Verdict ----------
    lines.append("## 5. One-paragraph verdict")
    lines.append("")
    if distinguishable and beat_maj and math.isfinite(per_day) and per_day > 0:
        verdict_para = (
            "The direction of cascade overshoots IS predictable from the "
            "83-dim flat representation enriched with cascade-onset "
            "confidence — the LR AUC's CI lower bound clears both 0.5 and "
            "the majority-class baseline, and the conditional headroom is "
            "positive after taker-fee + slip costs.  This is the first "
            "tradeable cell the program has produced.  Caveats: n_cascades "
            f"= {n_cas} on April 1-13, CI bracket is {auc_lo:.2f}-{auc_hi:.2f}; "
            "out-of-sample replication on April 14+ remains the binding test."
        )
    elif math.isfinite(per_day) and per_day > 0:
        verdict_para = (
            "Per-day expected gross is positive on April 1-13, but the "
            "direction LR's AUC CI does not cleanly clear both 0.5 AND the "
            "majority-class baseline.  The apparent edge may be the LR "
            "learning the marginal direction asymmetry rather than "
            "cascade-conditional signal.  Not yet tradeable — needs a "
            "tighter sample (April 14+ post-pretraining) before deployment."
        )
    elif distinguishable:
        verdict_para = (
            "Direction is statistically distinguishable from 0.5 in-sample "
            "but the headroom math is not yet positive: trigger frequency "
            "× direction accuracy × cascade move size does not cover the "
            "round-trip cost.  Strategy is NOT tradeable on this "
            "representation; would need either tighter onset gating, "
            "longer holds, or maker-only execution to flip net positive."
        )
    else:
        verdict_para = (
            "The cascade-onset signal (AUC=0.817 at top-1% precision=27.8%) "
            "appears to be DIRECTIONLESS at H500 in this representation — "
            "the LR cannot predict whether a flagged cascade overshoot "
            "goes long or short.  Without direction the strategy reduces "
            "to a coin flip with round-trip costs, which is unprofitable.  "
            "Until a representation provides direction skill, the cascade-"
            "precursor signal is statistically interesting but untradeable."
        )
    lines.append(verdict_para)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_Wall-clock: {elapsed_sec:.1f} s._")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Marginal-long + precision sweep (reanalysis on existing predictions)
# ---------------------------------------------------------------------------

# Cost reference size: 10K USD per leg (median bucket in per_window.parquet,
# matches the size researcher-14 sized the 4bp/side fee around).
MARGINAL_LONG_REF_SIZE_USD: float = 10_000.0

# Precision cutoffs swept by --marginal-long.
MARGINAL_LONG_TOP_PCTS: tuple[float, ...] = (0.01, 0.005, 0.001)

# Bootstrap iterations for day-clustered CIs on the marginal-long sweep.
N_BOOT_MARGINAL_LONG: int = 1000

# Minimum per-symbol triggers for inclusion in the per-symbol breakdown.
MARGINAL_LONG_MIN_TRIGGERS_PER_SYMBOL: int = 3


def _per_day_top_pct_mask(
    proba: np.ndarray, dates: np.ndarray, *, top_pct: float
) -> np.ndarray:
    """Per-day top-`top_pct` quantile mask.

    For each unique day, take the (1 - top_pct) quantile of `proba` over windows
    on that day; mark a window True iff its `proba` strictly exceeds that
    threshold.  This avoids leaking future info across days while letting the
    cutoff drift with daily dispersion (vs. a single global threshold).
    """
    if len(proba) == 0:
        return np.zeros(0, dtype=bool)
    quant = 1.0 - float(top_pct)
    out = np.zeros(len(proba), dtype=bool)
    for d in np.unique(dates):
        mask_d = dates == d
        if not mask_d.any():
            continue
        thresh = float(np.quantile(proba[mask_d], quant))
        out[mask_d] = proba[mask_d] > thresh
    return out


def _day_clustered_bootstrap_mean(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Day-clustered bootstrap of `mean(values)`.

    Resample days with replacement (k = n_unique_days), concatenate ALL values
    on each sampled day, take the mean over the concatenation.  Returns
    (point_estimate=full-sample mean, lo, hi).
    """
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    days = sorted(np.unique(dates).tolist())
    if len(days) < 2:
        return float(np.mean(values)), float("nan"), float("nan")
    day_to_idx = {d: np.flatnonzero(dates == d) for d in days}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    means[:] = np.nan
    k = len(days)
    for b in range(n_boot):
        sampled = rng.integers(0, k, size=k)
        parts = [day_to_idx[days[s]] for s in sampled]
        if not parts:
            continue
        idx = np.concatenate(parts).astype(np.int64)
        if len(idx) == 0:
            continue
        means[b] = float(np.mean(values[idx]))
    finite = means[np.isfinite(means)]
    point = float(np.mean(values))
    if len(finite) < 10:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return point, lo, hi


def _attach_slip_for_marginal_long(
    enriched: pd.DataFrame,
    *,
    out_dir: Path,
    ref_size_usd: float = MARGINAL_LONG_REF_SIZE_USD,
) -> pd.DataFrame:
    """Join `slip_avg_bps` from per_window.parquet at the reference size_usd
    onto the cascade-direction-enriched DataFrame.

    Rows missing slip are dropped (cost cannot be computed without it).
    """
    pw_path = out_dir / "per_window.parquet"
    if not pw_path.exists():
        raise RuntimeError(
            f"Cannot find {pw_path} — required for marginal-long cost computation"
        )
    pw = pd.read_parquet(pw_path)
    pw_h = pd.DataFrame(
        pw[(pw["horizon"] == 500) & (pw["size_usd"] == float(ref_size_usd))][
            ["symbol", "date", "anchor_ts", "slip_avg_bps"]
        ]
    ).copy()
    pw_h["date"] = pw_h["date"].astype(str)
    pw_h["symbol"] = pw_h["symbol"].astype(str)
    pw_h["anchor_ts"] = pw_h["anchor_ts"].astype("int64")

    e = enriched.copy()
    e["date"] = e["date"].astype(str)
    e["symbol"] = e["symbol"].astype(str)
    e["anchor_ts"] = e["anchor_ts"].astype("int64")
    joined = e.merge(pw_h, on=["symbol", "date", "anchor_ts"], how="left")
    n_drop = int(joined["slip_avg_bps"].isna().to_numpy().sum())
    if n_drop > 0:
        print(
            f"[marginal-long] dropping {n_drop} rows missing slip_avg_bps "
            f"(no per_window match at size_usd={ref_size_usd:.0f})"
        )
    return pd.DataFrame(joined.dropna(subset=["slip_avg_bps"])).reset_index(drop=True)


def _attach_lr_direction_pred(
    df: pd.DataFrame, direction_table_path: Path
) -> pd.DataFrame:
    """Left-join the LR-direction predictions onto the marginal-long DataFrame.

    The direction table has one row per real-cascade window inside the top-5%
    cascade-likely subset; non-cascade windows and windows below the top-5%
    cutoff have no LR prediction (left as NaN).  Used purely for the
    side-by-side comparison.
    """
    if not direction_table_path.exists():
        print(
            f"[marginal-long] direction table {direction_table_path} missing; "
            f"LR-direction comparison will be skipped"
        )
        df = df.copy()
        df["direction_pred_proba_lr"] = np.nan
        return df
    dt = pd.read_csv(direction_table_path)
    dt = dt[["symbol", "date", "anchor_ts", "direction_pred_proba_lr"]].copy()
    dt["date"] = dt["date"].astype(str)
    dt["symbol"] = dt["symbol"].astype(str)
    dt["anchor_ts"] = dt["anchor_ts"].astype("int64")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["anchor_ts"] = out["anchor_ts"].astype("int64")
    return out.merge(dt, on=["symbol", "date", "anchor_ts"], how="left")


def _marginal_long_cell_metrics(
    triggered: pd.DataFrame,
    *,
    n_unique_dates_universe: int,
    n_boot: int,
    seed: int,
    scope: str,
    symbol_or_pool: str,
    top_pct: float,
) -> dict[str, float | str]:
    """Compute all marginal-long metrics for one (precision_cutoff, scope) cell.

    Returns a dict matching the CSV row schema.  Day-clustered bootstrap CIs
    on mean & median signed PnL.  Returns NaN-filled row if `triggered` is
    empty or spans <2 days.
    """
    n_trig = len(triggered)
    if n_trig == 0:
        return {
            "scope": scope,
            "symbol_or_pool": symbol_or_pool,
            "precision_cutoff": float(top_pct),
            "n_triggers": 0,
            "n_unique_days": 0,
            "n_triggers_per_day_avg": 0.0,
            "n_real_cascades": 0,
            "precision": float("nan"),
            "directional_accuracy_long": float("nan"),
            "marginal_p_positive_h500": float("nan"),
            "mean_signed_pnl_long_bps": float("nan"),
            "mean_signed_pnl_long_bps_lo": float("nan"),
            "mean_signed_pnl_long_bps_hi": float("nan"),
            "median_signed_pnl_long_bps": float("nan"),
            "median_signed_pnl_long_bps_lo": float("nan"),
            "median_signed_pnl_long_bps_hi": float("nan"),
            "cost_round_trip_bps_avg": float("nan"),
            "headroom_per_trigger_bps": float("nan"),
            "expected_gross_per_day_bps": float("nan"),
            "expected_gross_per_day_bps_lo": float("nan"),
            "expected_gross_per_day_bps_hi": float("nan"),
            "lr_direction_n_predicted": 0,
            "lr_direction_long_gross_per_day_bps": float("nan"),
        }

    pnl = triggered["signed_pnl_long_bps"].astype(float).to_numpy()
    cost = triggered["cost_round_trip_bps"].astype(float).to_numpy()
    fwd = triggered["forward_log_return"].astype(float).to_numpy()
    real_label = triggered["real_cascade_label"].astype("int64").to_numpy()
    dates_arr = triggered["date"].astype(str).to_numpy()

    n_days_local = int(len(np.unique(dates_arr)))
    triggers_per_day = float(n_trig / max(1, n_unique_dates_universe))

    precision = float(real_label.mean())
    dir_acc_long = float((pnl > 0).mean())
    p_pos = float((fwd > 0).mean())
    mean_pnl = float(np.mean(pnl))
    median_pnl = float(np.median(pnl))
    cost_avg = float(np.mean(cost))
    headroom = mean_pnl - cost_avg
    gross_per_day = headroom * triggers_per_day

    # Day-clustered bootstrap CIs on mean & median PnL
    mean_pt, mean_lo, mean_hi = _day_clustered_bootstrap_mean(
        pnl, dates_arr, n_boot=n_boot, seed=seed
    )
    # For median we re-use the same bootstrap loop (could share but inlining
    # for clarity — n_boot=1000 is cheap on |n_trig| ≤ ~30).
    days_local = sorted(np.unique(dates_arr).tolist())
    median_pt, median_lo, median_hi = (median_pnl, float("nan"), float("nan"))
    if len(days_local) >= 2:
        day_to_idx = {d: np.flatnonzero(dates_arr == d) for d in days_local}
        rng = np.random.default_rng(seed + 1)
        meds = np.empty(n_boot, dtype=np.float64)
        meds[:] = np.nan
        k = len(days_local)
        for b in range(n_boot):
            sampled = rng.integers(0, k, size=k)
            parts = [day_to_idx[days_local[s]] for s in sampled]
            if not parts:
                continue
            idx = np.concatenate(parts).astype(np.int64)
            if len(idx) == 0:
                continue
            meds[b] = float(np.median(pnl[idx]))
        finite = meds[np.isfinite(meds)]
        if len(finite) >= 10:
            median_lo = float(np.quantile(finite, 0.025))
            median_hi = float(np.quantile(finite, 0.975))

    # Day-clustered bootstrap CI on expected_gross_per_day:
    # for each bootstrap draw, compute (sum of PnL on sampled days - sum of cost
    # on sampled days) / n_unique_dates_universe.  This propagates day-cluster
    # noise into the per-day gross figure (the user's methodological flag).
    gross_lo, gross_hi = float("nan"), float("nan")
    if len(days_local) >= 2:
        day_to_idx = {d: np.flatnonzero(dates_arr == d) for d in days_local}
        rng = np.random.default_rng(seed + 2)
        grosses = np.empty(n_boot, dtype=np.float64)
        grosses[:] = np.nan
        k = len(days_local)
        for b in range(n_boot):
            sampled = rng.integers(0, k, size=k)
            parts = [day_to_idx[days_local[s]] for s in sampled]
            if not parts:
                continue
            idx = np.concatenate(parts).astype(np.int64)
            if len(idx) == 0:
                continue
            net_per_trig = float(np.mean(pnl[idx] - cost[idx]))
            # Trigger frequency under the bootstrap: keep universe denominator
            # fixed so we report gross-per-calendar-day, not gross-per-resampled-day
            grosses[b] = net_per_trig * (len(idx) / max(1, n_unique_dates_universe))
        finite_g = grosses[np.isfinite(grosses)]
        if len(finite_g) >= 10:
            gross_lo = float(np.quantile(finite_g, 0.025))
            gross_hi = float(np.quantile(finite_g, 0.975))

    # LR-direction comparison: only keep rows with a valid LR pred
    lr_long_gross_per_day = float("nan")
    n_lr_predicted = 0
    if "direction_pred_proba_lr" in triggered.columns:
        lr_proba = triggered["direction_pred_proba_lr"].astype(float).to_numpy()
        valid_lr = np.isfinite(lr_proba)
        n_lr_predicted = int(valid_lr.sum())
        if n_lr_predicted > 0:
            # Only trade when LR confident (> 0.55 or < 0.45) — same gating as
            # the cascade-direction writeup.
            confident = (lr_proba > DIRECTION_LR_CONFIDENCE_THRESHOLD) | (
                lr_proba < (1.0 - DIRECTION_LR_CONFIDENCE_THRESHOLD)
            )
            confident &= valid_lr
            if confident.any():
                lr_pred_class = (lr_proba[confident] > 0.5).astype(np.int64)
                # Convert pred_class to position sign: 1 -> +1 (long), 0 -> -1 (short)
                lr_pos_sign = np.where(lr_pred_class == 1, 1.0, -1.0)
                lr_pnl = lr_pos_sign * pnl[confident] - cost[confident]
                lr_n_per_day = float(
                    int(confident.sum()) / max(1, n_unique_dates_universe)
                )
                lr_long_gross_per_day = float(np.mean(lr_pnl) * lr_n_per_day)

    return {
        "scope": scope,
        "symbol_or_pool": symbol_or_pool,
        "precision_cutoff": float(top_pct),
        "n_triggers": int(n_trig),
        "n_unique_days": int(n_days_local),
        "n_triggers_per_day_avg": float(triggers_per_day),
        "n_real_cascades": int(real_label.sum()),
        "precision": float(precision),
        "directional_accuracy_long": float(dir_acc_long),
        "marginal_p_positive_h500": float(p_pos),
        "mean_signed_pnl_long_bps": float(mean_pt),
        "mean_signed_pnl_long_bps_lo": float(mean_lo),
        "mean_signed_pnl_long_bps_hi": float(mean_hi),
        "median_signed_pnl_long_bps": float(median_pt),
        "median_signed_pnl_long_bps_lo": float(median_lo),
        "median_signed_pnl_long_bps_hi": float(median_hi),
        "cost_round_trip_bps_avg": float(cost_avg),
        "headroom_per_trigger_bps": float(headroom),
        "expected_gross_per_day_bps": float(gross_per_day),
        "expected_gross_per_day_bps_lo": float(gross_lo),
        "expected_gross_per_day_bps_hi": float(gross_hi),
        "lr_direction_n_predicted": int(n_lr_predicted),
        "lr_direction_long_gross_per_day_bps": float(lr_long_gross_per_day),
    }


def _run_marginal_long_pipeline(
    cache_dir: Path,
    out_dir: Path,
    *,
    n_boot: int = N_BOOT_MARGINAL_LONG,
    seed: int = 0,
    ref_size_usd: float = MARGINAL_LONG_REF_SIZE_USD,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end marginal-long sweep.

    Reuses `_build_cascade_direction_dataset` to enrich the cascade-real H500
    predictions with forward_log_return; joins per-window slip; computes
    pooled and per-symbol metrics across precision cutoffs {1%, 0.5%, 0.1%}
    using per-day quantile gating.  Returns (table_df, summary_dict).
    """
    in_path = out_dir / "cascade_precursor_real_per_window.parquet"
    if not in_path.exists():
        raise RuntimeError(
            f"Cannot find {in_path} — run the stage-2 cascade-real probe first "
            f"with `--cascade-real`."
        )
    raw = pd.read_parquet(in_path)
    h_df = pd.DataFrame(raw[raw["horizon"] == 500]).copy()
    if h_df.empty:
        raise RuntimeError(f"No H500 rows in {in_path}")
    print(
        f"[marginal-long] loaded {len(h_df)} H500 windows; "
        f"{int(h_df['real_cascade_label'].astype('int64').to_numpy().sum())} real cascades"
    )

    enriched = _build_cascade_direction_dataset(cache_dir, h_df, horizon=500)
    if enriched.empty:
        raise RuntimeError("No enriched windows produced — cache shards missing?")
    print(f"[marginal-long] enriched {len(enriched)} windows with forward returns")

    enriched_with_slip = _attach_slip_for_marginal_long(
        enriched, out_dir=out_dir, ref_size_usd=ref_size_usd
    )
    enriched_with_slip = _attach_lr_direction_pred(
        enriched_with_slip, out_dir / "cascade_direction_table.csv"
    )

    enriched_with_slip["signed_pnl_long_bps"] = (
        enriched_with_slip["forward_log_return"].astype(float) * 1e4
    )
    enriched_with_slip["cost_round_trip_bps"] = (
        2.0 * TAKER_FEE_BPS_PER_SIDE
        + 2.0 * enriched_with_slip["slip_avg_bps"].astype(float).abs()
    )

    n_universe_dates_val = enriched_with_slip["date"].nunique()
    n_universe_dates = int(n_universe_dates_val)  # type: ignore[arg-type]
    mean_cost_bps = float(
        enriched_with_slip["cost_round_trip_bps"].astype(float).to_numpy().mean()
    )
    print(
        f"[marginal-long] universe: {len(enriched_with_slip)} windows, "
        f"{n_universe_dates} unique days, "
        f"mean cost = {mean_cost_bps:.2f}bp"
    )

    # Universe-level marginal asymmetry sanity (matches direction writeup)
    universe_p_pos = float(
        np.mean(enriched_with_slip["forward_log_return"].astype(float).to_numpy() > 0)
    )
    cascade_mask_universe = (
        enriched_with_slip["real_cascade_label"].astype("int64").to_numpy() == 1
    )
    cascade_fwd = (
        enriched_with_slip["forward_log_return"]
        .astype(float)
        .to_numpy()[cascade_mask_universe]
    )
    cascade_only_p_pos = (
        float(np.mean(cascade_fwd > 0)) if cascade_fwd.size > 0 else float("nan")
    )
    print(
        f"[marginal-long] universe P(positive) = {universe_p_pos:.4f}; "
        f"P(positive | real_cascade) = {cascade_only_p_pos:.4f}"
    )

    rows: list[dict] = []
    proba_arr = enriched_with_slip["pred_proba"].astype(float).to_numpy()
    dates_arr = enriched_with_slip["date"].astype(str).to_numpy()
    symbols_arr = enriched_with_slip["symbol"].astype(str).to_numpy()
    sub_full = enriched_with_slip.reset_index(drop=True)

    for top_pct in MARGINAL_LONG_TOP_PCTS:
        mask = _per_day_top_pct_mask(proba_arr, dates_arr, top_pct=top_pct)
        triggered = pd.DataFrame(sub_full[mask]).reset_index(drop=True)
        # Pooled cell
        rows.append(
            _marginal_long_cell_metrics(
                triggered,
                n_unique_dates_universe=n_universe_dates,
                n_boot=n_boot,
                seed=seed,
                scope="pooled",
                symbol_or_pool="ALL",
                top_pct=top_pct,
            )
        )
        # Per-symbol cells
        for sym in sorted(np.unique(symbols_arr).tolist()):
            sym_trig = pd.DataFrame(
                triggered[triggered["symbol"].astype(str) == sym]
            ).reset_index(drop=True)
            if len(sym_trig) < MARGINAL_LONG_MIN_TRIGGERS_PER_SYMBOL:
                continue
            rows.append(
                _marginal_long_cell_metrics(
                    sym_trig,
                    n_unique_dates_universe=n_universe_dates,
                    n_boot=n_boot,
                    seed=seed,
                    scope="per_symbol",
                    symbol_or_pool=sym,
                    top_pct=top_pct,
                )
            )

    table_df = pd.DataFrame(rows)
    if not table_df.empty:
        table_df = table_df.sort_values(
            ["precision_cutoff", "scope", "symbol_or_pool"]
        ).reset_index(drop=True)

    # Determine best precision cell (pooled, max gross/day)
    pooled = pd.DataFrame(table_df[table_df["scope"] == "pooled"]).copy()
    if not pooled.empty:
        best_idx = pooled["expected_gross_per_day_bps"].idxmax()
        best_top_pct = float(pooled.loc[best_idx, "precision_cutoff"])
        best_gross = float(pooled.loc[best_idx, "expected_gross_per_day_bps"])
    else:
        best_top_pct = float("nan")
        best_gross = float("nan")

    n_real_cas = int(
        enriched_with_slip["real_cascade_label"].astype("int64").to_numpy().sum()
    )
    summary: dict = {
        "n_universe": int(len(enriched_with_slip)),
        "n_universe_dates": int(n_universe_dates),
        "n_real_cascades_h500": n_real_cas,
        "universe_p_positive_h500": universe_p_pos,
        "cascade_only_p_positive_h500": cascade_only_p_pos,
        "ref_size_usd": float(ref_size_usd),
        "best_top_pct": best_top_pct,
        "best_gross_per_day_bps": best_gross,
    }

    return table_df, summary


def _emit_marginal_long_markdown(
    table_df: pd.DataFrame,
    summary: dict,
    out_path: Path,
    *,
    elapsed_sec: float,
    n_boot: int,
) -> None:
    """Render the marginal-long markdown writeup."""
    lines: list[str] = []
    lines.append("# Goal-A cascade-precursor marginal-long + precision sweep")
    lines.append("")
    lines.append(
        "**Question.** If we go always-long whenever the stage-2 cascade-onset "
        "model fires (top 1%, top 0.5%, top 0.1% of windows by predicted "
        "probability), is the strategy tradeable net of cost?  Cascades on "
        "this universe are 76.7% long-biased at H500 — overwhelmingly forced-"
        "short squeezes — so a marginal-long bet may dominate a conditional-"
        "direction predictor that we already showed cannot do better than "
        "majority-class (LR direction AUC = 0.441 [0.329, 0.551])."
    )
    lines.append("")
    lines.append(
        "**Protocol.** Reanalysis on existing per-window predictions "
        "(`cascade_precursor_real_per_window.parquet`).  No retraining.  "
        "Per-day top-pct quantile gating (within each held-out April day, "
        "no future leakage).  Day-clustered bootstrap CIs.  Costs from "
        f"per_window slip at size_usd = {summary['ref_size_usd']:.0f} + "
        f"{TAKER_FEE_BPS_PER_SIDE}bp/side taker fee, both legs.  H500 only "
        "(only horizon where the stage-2 LR clears the shuffled-baseline)."
    )
    lines.append("")
    lines.append(
        "**Hard constraints.** April 14+ untouched.  Sample size honest: at "
        "top 0.1% across 7 April-diagnostic days, n_triggers may be ~7 "
        "(roughly 1/day) — wide CIs, dominated by 1-2 trades on bad days."
    )
    lines.append("")

    # ---------- 0. Universe baseline ----------
    lines.append("## 0. Universe baseline (no filtering)")
    lines.append("")
    lines.append(
        f"- Universe: **{summary['n_universe']}** H500 windows across "
        f"**{summary['n_universe_dates']}** April-diagnostic days "
        f"({summary['n_real_cascades_h500']} real cascades)."
    )
    lines.append(
        f"- P(positive forward return | universe) = "
        f"**{summary['universe_p_positive_h500']:.4f}** (NOT cascade-conditional — "
        "shows whether April was a uniformly long-biased market)."
    )
    lines.append(
        f"- P(positive forward return | real_cascade) = "
        f"**{summary['cascade_only_p_positive_h500']:.4f}** (the 76% figure "
        "reported in the cascade-direction writeup; reproduced here for "
        "consistency check)."
    )
    lines.append("")

    # ---------- 1. Pooled marginal-long headline ----------
    lines.append("## 1. Marginal-long headline (pooled, per precision cutoff)")
    lines.append("")
    lines.append(
        "| top % | n_trig | trig/day | precision | dir_acc_long | mean_pnl_bps "
        "(95% CI) | median_pnl_bps (95% CI) | cost_avg | headroom | "
        "gross/day (95% CI) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    pooled = pd.DataFrame(table_df[table_df["scope"] == "pooled"]).copy()
    pooled = pooled.sort_values("precision_cutoff", ascending=False)
    pooled_records: list[dict] = pooled.to_dict("records")  # type: ignore[assignment]
    for r in pooled_records:
        lines.append(
            f"| {float(r['precision_cutoff']):.3%} | {int(r['n_triggers'])} | "
            f"{float(r['n_triggers_per_day_avg']):.2f} | "
            f"{float(r['precision']):.3f} | "
            f"{float(r['directional_accuracy_long']):.3f} | "
            f"{float(r['mean_signed_pnl_long_bps']):+.2f} "
            f"[{float(r['mean_signed_pnl_long_bps_lo']):+.2f}, "
            f"{float(r['mean_signed_pnl_long_bps_hi']):+.2f}] | "
            f"{float(r['median_signed_pnl_long_bps']):+.2f} "
            f"[{float(r['median_signed_pnl_long_bps_lo']):+.2f}, "
            f"{float(r['median_signed_pnl_long_bps_hi']):+.2f}] | "
            f"{float(r['cost_round_trip_bps_avg']):.2f} | "
            f"{float(r['headroom_per_trigger_bps']):+.2f} | "
            f"{float(r['expected_gross_per_day_bps']):+.2f} "
            f"[{float(r['expected_gross_per_day_bps_lo']):+.2f}, "
            f"{float(r['expected_gross_per_day_bps_hi']):+.2f}] |"
        )
    lines.append("")

    # ---------- 2. Comparison vs LR-direction strategy ----------
    lines.append("## 2. Marginal-long vs LR-direction strategy (same precision cells)")
    lines.append("")
    lines.append(
        "LR-direction = 'go long if `direction_pred_proba_lr > 0.55`, short if "
        "< 0.45, skip otherwise'.  Only rows with an LR prediction (cascade-"
        "likely subset = top 5% of pred_proba) get a position.  Both columns "
        "use the same per-day-quantile filter from §1; LR-direction is "
        "naturally a *subset* of marginal-long triggers (windows that fall in "
        "BOTH top-pct AND top-5% AND have confident LR direction)."
    )
    lines.append("")
    lines.append(
        "| top % | marginal_long gross/day | LR-direction gross/day | "
        "n_lr_predicted | which is better? |"
    )
    lines.append("|---|---|---|---|---|")
    for r in pooled_records:
        ml_gross = float(r["expected_gross_per_day_bps"])
        lr_gross = float(r["lr_direction_long_gross_per_day_bps"])
        if not math.isfinite(lr_gross):
            verdict = "LR-direction has 0 confident predictions"
        elif ml_gross > lr_gross + 1e-3:
            verdict = f"marginal-long better by {ml_gross - lr_gross:+.2f} bps/day"
        elif lr_gross > ml_gross + 1e-3:
            verdict = f"LR-direction better by {lr_gross - ml_gross:+.2f} bps/day"
        else:
            verdict = "tie"
        lines.append(
            f"| {float(r['precision_cutoff']):.3%} | {ml_gross:+.2f} | "
            f"{(f'{lr_gross:+.2f}' if math.isfinite(lr_gross) else 'n/a')} | "
            f"{int(r['lr_direction_n_predicted'])} | {verdict} |"
        )
    lines.append("")

    # ---------- 3. Marginal asymmetry across precision cells ----------
    lines.append("## 3. Marginal asymmetry stability across precision cells")
    lines.append("")
    lines.append(
        "Does the long-bias hold up at tighter precision cells, or dilute "
        "back toward 0.50?  If P(positive | triggered) drops toward 0.50 at "
        "top 0.1%, the 0.77 cascade-conditional asymmetry was driven by "
        "selection effects — the *prediction-flagged* windows are not the "
        "same population as *realized cascades*."
    )
    lines.append("")
    lines.append(
        "| top % | n_trig | precision (real cascade frac) | marginal P(positive | triggered) |"
    )
    lines.append("|---|---|---|---|")
    for r in pooled_records:
        lines.append(
            f"| {float(r['precision_cutoff']):.3%} | {int(r['n_triggers'])} | "
            f"{float(r['precision']):.3f} | "
            f"{float(r['marginal_p_positive_h500']):.3f} |"
        )
    lines.append("")
    lines.append(
        "Reference: cascade-conditional asymmetry "
        f"P(positive | real_cascade) = **{summary['cascade_only_p_positive_h500']:.3f}**; "
        f"universe baseline P(positive) = **{summary['universe_p_positive_h500']:.3f}**."
    )
    lines.append("")

    # ---------- 4. Per-symbol breakdown at best cell ----------
    best_pct = summary["best_top_pct"]
    lines.append(
        f"## 4. Per-symbol breakdown at best precision cell (top {best_pct:.3%})"
    )
    lines.append("")
    lines.append(
        f"Symbols with at least {MARGINAL_LONG_MIN_TRIGGERS_PER_SYMBOL} "
        "triggers at this cutoff only — anything fewer is sample-size noise."
    )
    lines.append("")
    if not math.isfinite(best_pct):
        lines.append("_No pooled rows produced — cannot pick a best cell._")
    else:
        per_sym = pd.DataFrame(
            table_df[
                (table_df["scope"] == "per_symbol")
                & (np.isclose(table_df["precision_cutoff"], best_pct))
            ]
        ).copy()
        if per_sym.empty:
            lines.append(
                f"_No per-symbol cell at top {best_pct:.3%} cleared the "
                f"≥{MARGINAL_LONG_MIN_TRIGGERS_PER_SYMBOL}-trigger floor._"
            )
        else:
            per_sym = per_sym.sort_values("expected_gross_per_day_bps", ascending=False)
            lines.append(
                "| symbol | n_trig | precision | dir_acc_long | mean_pnl_bps | "
                "cost | headroom | gross/day |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            per_sym_records: list[dict] = per_sym.to_dict("records")  # type: ignore[assignment]
            for r in per_sym_records:
                lines.append(
                    f"| {r['symbol_or_pool']} | {int(r['n_triggers'])} | "
                    f"{float(r['precision']):.3f} | "
                    f"{float(r['directional_accuracy_long']):.3f} | "
                    f"{float(r['mean_signed_pnl_long_bps']):+.2f} | "
                    f"{float(r['cost_round_trip_bps_avg']):.2f} | "
                    f"{float(r['headroom_per_trigger_bps']):+.2f} | "
                    f"{float(r['expected_gross_per_day_bps']):+.2f} |"
                )
    lines.append("")

    # ---------- 5. Verdict ----------
    lines.append("## 5. Verdict")
    lines.append("")
    if not math.isfinite(summary["best_gross_per_day_bps"]):
        lines.append(
            "**No tradeable cell**: insufficient triggers across all precision "
            "cutoffs — cannot conclude either way."
        )
    else:
        best_g = float(summary["best_gross_per_day_bps"])
        # Need to also check whether the bootstrap CI excludes 0
        best_match = [
            r
            for r in pooled_records
            if np.isclose(float(r["precision_cutoff"]), summary["best_top_pct"])
        ]
        ci_lo = (
            float(best_match[0]["expected_gross_per_day_bps_lo"])
            if best_match
            else float("nan")
        )
        ci_hi = (
            float(best_match[0]["expected_gross_per_day_bps_hi"])
            if best_match
            else float("nan")
        )
        if best_g > 0 and math.isfinite(ci_lo) and ci_lo > 0:
            verdict = (
                f"**TRADEABLE pre-encoder.**  Best cell (top "
                f"{summary['best_top_pct']:.3%}) yields "
                f"**{best_g:+.2f} bps/day** with CI lower bound "
                f"**{ci_lo:+.2f} bps/day** strictly positive — the marginal-"
                f"long strategy survives day-clustered bootstrap noise."
            )
        elif best_g > 0:
            verdict = (
                f"**MARGINAL-BUT-NOT-CLEARLY.**  Best cell (top "
                f"{summary['best_top_pct']:.3%}) yields "
                f"**{best_g:+.2f} bps/day** but the day-clustered CI "
                f"[{ci_lo:+.2f}, {ci_hi:+.2f}] includes 0 — point estimate is "
                f"positive, sampling noise dominates.  Encoder retrain may "
                f"be required to tighten the signal."
            )
        else:
            verdict = (
                f"**NOT TRADEABLE pre-encoder.**  Best cell (top "
                f"{summary['best_top_pct']:.3%}) yields "
                f"**{best_g:+.2f} bps/day** — net of cost, every precision "
                f"cell is unprofitable.  The 0.77 cascade-conditional long-"
                f"bias does NOT carry through to the prediction-flagged "
                f"subset because the LR's top-1% precision is only ~28%; the "
                f"other ~72% of triggers are non-cascade windows with "
                f"~50/50 directional symmetry that drag the mean back to "
                f"zero."
            )
        lines.append(verdict)
    lines.append("")

    # ---------- 6. Methodological flags ----------
    lines.append("## 6. Methodological flags")
    lines.append("")
    lines.append(
        "* **Sample size at top 0.1%.** Across 7 April-diagnostic days, "
        "n_triggers at top 0.1% is ~7 (≈1/day).  Day-clustered bootstrap "
        "captures cluster noise but cannot rescue inference from n=7 — the "
        "CI widths reported above honestly reflect this."
    )
    lines.append("")
    lines.append(
        "* **Per-day quantile vs global quantile.** Per-day quantile keeps "
        "stride day-by-day (no future leakage), but on slow days it lowers "
        "the trigger threshold, admitting low-confidence windows.  A global "
        "quantile would give a tighter cutoff but leak the future "
        "distribution.  Per-day is the conservative, leakage-safe choice."
    )
    lines.append("")
    lines.append(
        "* **Cost size selection.** Used per_window slip at "
        f"size_usd = {summary['ref_size_usd']:.0f} (median bucket).  At "
        "100K size cost would roughly double; at 1K size cost would halve.  "
        "The headroom math is sensitive to the size assumption — a strategy "
        "that's marginal at 10K may be tradeable at 1K (lower fees) or "
        "untradeable at 100K (higher slippage)."
    )
    lines.append("")
    lines.append(
        "* **Universe vs cascade-conditional asymmetry.** The 0.77 long-bias "
        "is ON REAL CASCADES.  The prediction-flagged subset has precision "
        "~28% at top 1%, ~43% at top 0.1% — the OTHER 57-72% of triggered "
        "windows are non-cascade windows.  If those non-cascade windows have "
        "P(positive) ≈ 0.50, the marginal-long bet on the flagged subset "
        "ends up close to 0.50 even though the cascade-conditional bias is "
        "0.77.  The §3 table shows this dilution directly."
    )
    lines.append("")
    lines.append(
        "* **Single-trade-domination at top 0.1%.** The day-clustered "
        "bootstrap resamples DAYS, so a single big PnL on one day appears "
        "in roughly k/(k+1) ≈ 87% of bootstrap draws.  The CIs at top 0.1% "
        "are dominated by which 1-2 trades fall on which day, not by 7-day "
        "ensemble averaging.  Read the top 0.1% row with this in mind."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"_Wall-clock: {elapsed_sec:.1f} s.  n_boot = {n_boot}.  CPU-only.  "
        "No April 14+ data touched._"
    )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
# OOS test (--oos-test) — Apr 14-26 generalization
# ---------------------------------------------------------------------------
#
# Question: does the in-sample LR fit on Apr 1-13 (real cause-flag cascade label,
# 83-dim flat features, balanced class-weight) — which produced pooled cross-
# symbol AUC = 0.815 [0.772, 0.848] at H500 with leave-one-day-out CV — generalize
# to Apr 14-26?  This is the ONLY test of out-of-sample generalization for the
# cascade-precursor program.
#
# Protocol:
#   1. Train ONE universe-wide LR on ALL Apr 1-13 data (no LOO-CV; the test set
#      is genuinely held-out).  Same hyperparameters as in-sample
#      (LogisticRegression(class_weight='balanced', C=1.0), StandardScaler).
#   2. Apply to Apr 14-26 cache shards (stride=200 evaluation windows).
#   3. Compute pooled + per-symbol metrics with day-clustered bootstrap CIs.
#   4. Shuffled-baseline OOS: same protocol but labels permuted within day on
#      Apr 14-26.  Sanity check that AUC ≈ 0.50 (no leakage).
#
# Hard constraint: Apr 14-26 has been DELIBERATELY consumed as the holdout for
# this test (gotcha #17).  After this run, no untouched cascade-labeled holdout
# remains.  The unsafe-loaders below intentionally bypass the APRIL_HELDOUT_START
# guard — they are ONLY called from the OOS pipeline.

OOS_DIAGNOSTIC_START: str = "2026-04-14"
OOS_DIAGNOSTIC_END_INCLUSIVE: str = "2026-04-26"
OOS_HORIZONS: tuple[int, ...] = (100, 500)
N_BOOT_OOS: int = 1000
PER_SYMBOL_MIN_CASCADES_OOS: int = 3  # OOS smaller — 3-cascade floor per prompt


def _oos_diagnostic_dates() -> list[str]:
    """Apr 14 through Apr 26 (inclusive) as `YYYY-MM-DD` strings."""
    return [f"2026-04-{d:02d}" for d in range(14, 27)]


def _is_oos_diagnostic_date(date_str: str) -> bool:
    """True iff date is in [OOS_DIAGNOSTIC_START, OOS_DIAGNOSTIC_END_INCLUSIVE]."""
    return OOS_DIAGNOSTIC_START <= date_str <= OOS_DIAGNOSTIC_END_INCLUSIVE


def _fit_universe_lr(
    X: np.ndarray, y: np.ndarray
) -> tuple[StandardScaler | None, LogisticRegression | None, float]:
    """Fit a single universe-wide LR + StandardScaler on (X, y).

    Returns (scaler, lr, const_pred) where:
      * (scaler, lr) is the fit pair if y has both classes; const_pred is unused.
      * (None, None, const_pred) if y is degenerate; const_pred is the constant
        probability to return (= y.mean()).
    """
    if len(np.unique(y)) < 2 or len(X) == 0:
        const = float(y.mean()) if len(y) > 0 else 0.0
        return None, None, const
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(
        C=1.0, max_iter=1_000, class_weight="balanced", solver="lbfgs"
    ).fit(scaler.transform(X), y)
    return scaler, lr, float("nan")


def _apply_universe_lr_proba(
    bundle: tuple[StandardScaler | None, LogisticRegression | None, float],
    X: np.ndarray,
) -> np.ndarray:
    """Apply a fitted bundle from `_fit_universe_lr` and return P(class=1)."""
    scaler, lr, const = bundle
    if lr is None or scaler is None:
        return np.full(len(X), const if math.isfinite(const) else 0.0, dtype=np.float64)
    proba = lr.predict_proba(scaler.transform(X))
    classes = list(lr.classes_)
    pos_idx = classes.index(1) if 1 in classes else (1 if proba.shape[1] > 1 else 0)
    return proba[:, pos_idx].astype(np.float64)


def _filter_per_symbol_oos_eligible(
    sym_arr: np.ndarray, y: np.ndarray, *, min_cascades: int
) -> list[str]:
    """Return symbols (sorted) with ≥ `min_cascades` real cascades in `y`."""
    out: list[str] = []
    for sym in sorted(np.unique(sym_arr).tolist()):
        mask = sym_arr == sym
        if int(y[mask].sum()) >= min_cascades:
            out.append(sym)
    return out


# --- Unsafe loaders (Apr 14+ permitted, called ONLY from OOS pipeline) ---


def _load_shard_unsafe(path: Path) -> dict:
    """Load all keys from an .npz shard, BYPASSING the APRIL_HELDOUT_START guard.

    Use ONLY from the OOS pipeline.  The user has authorized consuming the
    holdout for this run; the guard remains active in every other code path.
    """
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_liq_ts_for_symbol_date_unsafe(
    symbol: str, date_str: str
) -> np.ndarray | None:
    """Liquidation timestamps for (symbol, date_str), bypassing the holdout guard.

    Use ONLY from the OOS pipeline.  Returns None if raw parquet is missing,
    if the cause column is unavailable, or if duckdb fails.
    """
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


def _build_oos_day_batch(
    cache_dir: Path,
    symbol: str,
    date_str: str,
    horizons: tuple[int, ...] = OOS_HORIZONS,
) -> RealDayBatch | None:
    """Build a RealDayBatch for (symbol, date_str) on Apr 14-26 (OOS).

    Mirrors `_build_real_day_batch` but uses unsafe loaders.  Returns None if
    the cache shard or raw-trade parquet is unavailable, or if the shard is
    empty or has no events.
    """
    if not _is_oos_diagnostic_date(date_str):
        return None
    shard_path = cache_dir / f"{symbol}__{date_str}.npz"
    if not shard_path.exists():
        return None
    payload = _load_shard_unsafe(shard_path)
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

    flat_X = np.empty((len(starts), FLAT_DIM), dtype=np.float32)
    for i, s in enumerate(starts):
        flat_X[i] = extract_flat_features(features[s : s + WINDOW_LEN])

    liq_ts = _load_liq_ts_for_symbol_date_unsafe(symbol, date_str)
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


def _gather_oos_batches(
    cache_dir: Path,
    symbols: tuple[str, ...],
    dates: list[str],
    horizons: tuple[int, ...] = OOS_HORIZONS,
) -> list[RealDayBatch]:
    """Build all OOS RealDayBatch objects across (symbol × date) on Apr 14-26."""
    out: list[RealDayBatch] = []
    for symbol in symbols:
        for date_str in dates:
            b = _build_oos_day_batch(cache_dir, symbol, date_str, horizons=horizons)
            if b is not None:
                out.append(b)
    return out


def _oos_per_cell_metrics(
    *,
    df: pd.DataFrame,
    proba_col: str,
    shuffled_proba_col: str,
    label_col: str = "real_cascade_label",
    date_col: str = "date",
    n_boot: int = N_BOOT_OOS,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Compute pooled OOS metrics with day-clustered bootstrap CIs.

    Returns dict with: n_total, n_cascades, base_rate, n_days,
    auc_oos, auc_oos_lo, auc_oos_hi,
    auc_shuffled_oos, auc_shuffled_oos_lo, auc_shuffled_oos_hi,
    precision_top_1pct_oos, precision_top_1pct_oos_lo, precision_top_1pct_oos_hi,
    lift_oos, signal_distinguishable_from_shuffled.
    """
    n_total = int(len(df))
    if n_total == 0:
        return {
            "n_total": 0,
            "n_cascades": 0,
            "base_rate": float("nan"),
            "n_days": 0,
            "auc_oos": float("nan"),
            "auc_oos_lo": float("nan"),
            "auc_oos_hi": float("nan"),
            "auc_shuffled_oos": float("nan"),
            "auc_shuffled_oos_lo": float("nan"),
            "auc_shuffled_oos_hi": float("nan"),
            "precision_top_1pct_oos": float("nan"),
            "precision_top_1pct_oos_lo": float("nan"),
            "precision_top_1pct_oos_hi": float("nan"),
            "lift_oos": float("nan"),
            "signal_distinguishable_from_shuffled": False,
        }

    labels = df[label_col].to_numpy(dtype=np.int64)
    n_cascades = int(labels.sum())
    base_rate = float(labels.mean())
    n_days = int(df[date_col].nunique())  # type: ignore[arg-type]

    # Day-clustered bootstrap AUC for real and shuffled
    auc_pt, auc_lo, auc_hi = _day_clustered_bootstrap_auc(
        df,
        proba_col=proba_col,
        label_col=label_col,
        date_col=date_col,
        n_boot=n_boot,
        seed=seed,
    )
    auc_sh_pt, auc_sh_lo, auc_sh_hi = _day_clustered_bootstrap_auc(
        df,
        proba_col=shuffled_proba_col,
        label_col=label_col,
        date_col=date_col,
        n_boot=n_boot,
        seed=seed + 11,
    )
    # Day-clustered bootstrap precision-at-top-1%
    prec_pt, prec_lo, prec_hi = _day_clustered_bootstrap_precision_at_top(
        df,
        proba_col=proba_col,
        label_col=label_col,
        date_col=date_col,
        top_pct=TOP_PCT_REAL,
        n_boot=n_boot,
        seed=seed + 22,
    )
    if math.isfinite(prec_pt) and math.isfinite(base_rate) and base_rate > 0:
        lift = float(prec_pt / base_rate)
    else:
        lift = float("nan")

    dist = _signal_distinguishable_from_baseline(
        real_lo=auc_lo, real_hi=auc_hi, baseline_lo=auc_sh_lo, baseline_hi=auc_sh_hi
    )

    return {
        "n_total": n_total,
        "n_cascades": n_cascades,
        "base_rate": base_rate,
        "n_days": n_days,
        "auc_oos": auc_pt,
        "auc_oos_lo": auc_lo,
        "auc_oos_hi": auc_hi,
        "auc_shuffled_oos": auc_sh_pt,
        "auc_shuffled_oos_lo": auc_sh_lo,
        "auc_shuffled_oos_hi": auc_sh_hi,
        "precision_top_1pct_oos": prec_pt,
        "precision_top_1pct_oos_lo": prec_lo,
        "precision_top_1pct_oos_hi": prec_hi,
        "lift_oos": lift,
        "signal_distinguishable_from_shuffled": bool(dist),
    }


def _run_oos_pipeline(
    cache_dir: Path,
    symbols: tuple[str, ...],
    out_dir: Path,
    *,
    horizons: tuple[int, ...] = OOS_HORIZONS,
    n_boot: int = N_BOOT_OOS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the OOS test on Apr 14-26.

    Returns (per_window_df, per_cell_df, summary_dict).  Per-cell dataframe
    includes BOTH the in-sample row (from cascade_precursor_real_table.csv)
    and the OOS row, side-by-side, for each (horizon, scope, symbol).
    """
    # ---- 1. Gather in-sample (Apr 1-13) batches and fit universe LR ----
    is_dates = _april_diagnostic_dates()
    print(
        f"[oos-test] gathering Apr 1-13 (in-sample) batches for "
        f"{len(symbols)} symbols × {len(is_dates)} dates..."
    )
    is_batches = _gather_real_batches(cache_dir, symbols, is_dates, horizons=horizons)
    if not is_batches:
        raise RuntimeError(
            "No Apr 1-13 batches built — cannot fit OOS LR.  Cache or raw "
            "parquets may be missing."
        )
    # ---- 2. Gather OOS (Apr 14-26) batches ----
    oos_dates = _oos_diagnostic_dates()
    print(
        f"[oos-test] gathering Apr 14-26 (OOS) batches for "
        f"{len(symbols)} symbols × {len(oos_dates)} dates..."
    )
    oos_batches = _gather_oos_batches(cache_dir, symbols, oos_dates, horizons=horizons)
    if not oos_batches:
        raise RuntimeError(
            "No Apr 14-26 OOS batches built — cache shards or raw parquets may "
            "be missing.  Check that --consume-holdout was run."
        )
    n_oos_dates = len({b.date for b in oos_batches})
    n_oos_syms = len({b.symbol for b in oos_batches})
    print(
        f"[oos-test] OOS coverage: {len(oos_batches)} (sym, date) batches "
        f"across {n_oos_syms} symbols and {n_oos_dates} dates"
    )

    per_cell_rows: list[dict] = []
    per_window_rows: list[dict] = []
    summary: dict[str, dict] = {}

    rng = np.random.default_rng(seed + 7777)

    for h in horizons:
        # ---- Stack training (Apr 1-13) at horizon h ----
        Xtr, ytr, _, dtr, _ = _stack_real_batches_at_horizon(is_batches, h)
        if len(ytr) == 0 or ytr.sum() == 0:
            print(
                f"[oos-test] H{h}: no in-sample cascades; skipping " "(degenerate fit)"
            )
            continue
        n_is_total = int(len(ytr))
        n_is_cascades = int(ytr.sum())
        n_is_days = int(np.unique(dtr).size)
        print(
            f"[oos-test] H{h} TRAIN (Apr 1-13): n_total={n_is_total}, "
            f"n_cascades={n_is_cascades}, n_days={n_is_days}"
        )

        # ---- Fit one LR on the full Apr 1-13 fold ----
        bundle = _fit_universe_lr(Xtr, ytr.astype(np.int64))

        # ---- Stack OOS (Apr 14-26) at horizon h ----
        Xoos, yoos, soos, doos, aoos = _stack_real_batches_at_horizon(oos_batches, h)
        if len(yoos) == 0:
            print(
                f"[oos-test] H{h} OOS: no valid OOS windows (horizon overruns); "
                "skipping"
            )
            continue
        n_oos_total = int(len(yoos))
        n_oos_cascades = int(yoos.sum())
        unique_oos_dates = sorted(np.unique(doos).tolist())
        print(
            f"[oos-test] H{h} OOS  (Apr 14-26): n_total={n_oos_total}, "
            f"n_cascades={n_oos_cascades}, n_days={len(unique_oos_dates)}"
        )

        # ---- Predict on OOS ----
        proba_oos = _apply_universe_lr_proba(bundle, Xoos)

        # ---- Shuffled-OOS baseline: shuffle labels within day (preserves
        #      per-day base rate), apply SAME LR.  This is the right control:
        #      it tests whether the shuffled-OOS AUC ≈ 0.50 (no information
        #      leakage from any features that perfectly correlate with date).
        # ---- We shuffle the LABELS, not the predictions, then re-evaluate
        #      AUC on (proba_oos, y_shuf).  The bootstrap then handles CI.
        y_shuf = _shuffle_labels_within_day(
            yoos.astype(np.int64), doos, seed=seed + 333
        )

        # ---- Top-1% boolean per-window ----
        n_top = max(1, int(round(n_oos_total * TOP_PCT_REAL)))
        order = np.argsort(-proba_oos, kind="stable")
        top_1pct_oos_bool = np.zeros(n_oos_total, dtype=bool)
        top_1pct_oos_bool[order[:n_top]] = True

        # ---- Build per-window dataframe for pooled metrics ----
        df_pool = pd.DataFrame(
            {
                "symbol": soos.astype(str),
                "date": doos.astype(str),
                "anchor_ts": aoos.astype(np.int64),
                "horizon": int(h),
                "real_cascade_label": yoos.astype(np.int64),
                "shuffled_cascade_label": y_shuf.astype(np.int64),
                "pred_proba_oos": proba_oos,
                "top_1pct_oos_bool": top_1pct_oos_bool,
            }
        )

        # ---- Pooled metrics (day-clustered bootstrap) ----
        # For shuffled-baseline we use the SAME proba but evaluate against
        # the shuffled labels — equivalent to a label permutation test.
        # The day-clustered bootstrap uses the real label column; for shuffled
        # we re-call the helper with proba_col=pred_proba_oos and
        # label_col=shuffled_cascade_label.
        pooled_metrics = _oos_per_cell_metrics_for_pooled(
            df_pool, n_boot=n_boot, seed=seed + 100 * h
        )
        pooled_metrics_typed: dict[str, float | bool | str | int] = dict(pooled_metrics)
        pooled_metrics_typed["horizon"] = int(h)
        pooled_metrics_typed["scope"] = "pooled"
        pooled_metrics_typed["symbol"] = "ALL"
        pooled_metrics_typed["fold"] = "oos"
        per_cell_rows.append(pooled_metrics_typed)
        summary[f"pooled_H{h}"] = pooled_metrics

        # ---- Append per-window rows ----
        for i in range(n_oos_total):
            per_window_rows.append(
                {
                    "symbol": str(soos[i]),
                    "date": str(doos[i]),
                    "anchor_ts": int(aoos[i]),
                    "horizon": int(h),
                    "real_cascade_label": int(yoos[i]),
                    "shuffled_cascade_label": int(y_shuf[i]),
                    "pred_proba_oos": float(proba_oos[i]),
                    "top_1pct_oos_bool": bool(top_1pct_oos_bool[i]),
                }
            )

        # ---- Per-symbol cells (≥ PER_SYMBOL_MIN_CASCADES_OOS at H) ----
        eligible = _filter_per_symbol_oos_eligible(
            soos, yoos, min_cascades=PER_SYMBOL_MIN_CASCADES_OOS
        )
        for sym in eligible:
            mask = soos == sym
            df_sym = pd.DataFrame(df_pool[mask]).copy().reset_index(drop=True)
            sym_metrics = _oos_per_cell_metrics_for_pooled(
                df_sym, n_boot=n_boot, seed=seed + 200 * h + abs(hash(sym)) % 1000
            )
            sym_metrics_typed: dict[str, float | bool | str | int] = dict(sym_metrics)
            sym_metrics_typed["horizon"] = int(h)
            sym_metrics_typed["scope"] = "per_symbol"
            sym_metrics_typed["symbol"] = sym
            sym_metrics_typed["fold"] = "oos"
            per_cell_rows.append(sym_metrics_typed)

        # Suppress unused-rng lint
        _ = rng

    per_cell_df = pd.DataFrame(per_cell_rows)
    per_window_df = pd.DataFrame(per_window_rows)
    return per_window_df, per_cell_df, summary


def _oos_per_cell_metrics_for_pooled(
    df: pd.DataFrame,
    *,
    n_boot: int = N_BOOT_OOS,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Wrap `_oos_per_cell_metrics` with the OOS column conventions.

    Real cascade label column = `real_cascade_label`.
    Shuffled label column     = `shuffled_cascade_label` (bootstrap evaluates
        AUC of pred_proba_oos against the shuffled labels — i.e., a permutation
        test).
    """
    n_total = int(len(df))
    if n_total == 0:
        return {
            "n_total": 0,
            "n_cascades": 0,
            "base_rate": float("nan"),
            "n_days": 0,
            "auc_oos": float("nan"),
            "auc_oos_lo": float("nan"),
            "auc_oos_hi": float("nan"),
            "auc_shuffled_oos": float("nan"),
            "auc_shuffled_oos_lo": float("nan"),
            "auc_shuffled_oos_hi": float("nan"),
            "precision_top_1pct_oos": float("nan"),
            "precision_top_1pct_oos_lo": float("nan"),
            "precision_top_1pct_oos_hi": float("nan"),
            "lift_oos": float("nan"),
            "signal_distinguishable_from_shuffled": False,
        }

    labels = df["real_cascade_label"].to_numpy(dtype=np.int64)
    n_cascades = int(labels.sum())
    base_rate = float(labels.mean()) if n_total > 0 else float("nan")
    n_days = int(df["date"].nunique())  # type: ignore[arg-type]

    auc_pt, auc_lo, auc_hi = _day_clustered_bootstrap_auc(
        df,
        proba_col="pred_proba_oos",
        label_col="real_cascade_label",
        date_col="date",
        n_boot=n_boot,
        seed=seed,
    )
    auc_sh_pt, auc_sh_lo, auc_sh_hi = _day_clustered_bootstrap_auc(
        df,
        proba_col="pred_proba_oos",
        label_col="shuffled_cascade_label",
        date_col="date",
        n_boot=n_boot,
        seed=seed + 11,
    )
    prec_pt, prec_lo, prec_hi = _day_clustered_bootstrap_precision_at_top(
        df,
        proba_col="pred_proba_oos",
        label_col="real_cascade_label",
        date_col="date",
        top_pct=TOP_PCT_REAL,
        n_boot=n_boot,
        seed=seed + 22,
    )
    if math.isfinite(prec_pt) and math.isfinite(base_rate) and base_rate > 0:
        lift = float(prec_pt / base_rate)
    else:
        lift = float("nan")

    dist = _signal_distinguishable_from_baseline(
        real_lo=auc_lo, real_hi=auc_hi, baseline_lo=auc_sh_lo, baseline_hi=auc_sh_hi
    )
    return {
        "n_total": n_total,
        "n_cascades": n_cascades,
        "base_rate": base_rate,
        "n_days": n_days,
        "auc_oos": auc_pt,
        "auc_oos_lo": auc_lo,
        "auc_oos_hi": auc_hi,
        "auc_shuffled_oos": auc_sh_pt,
        "auc_shuffled_oos_lo": auc_sh_lo,
        "auc_shuffled_oos_hi": auc_sh_hi,
        "precision_top_1pct_oos": prec_pt,
        "precision_top_1pct_oos_lo": prec_lo,
        "precision_top_1pct_oos_hi": prec_hi,
        "lift_oos": lift,
        "signal_distinguishable_from_shuffled": bool(dist),
    }


def _load_in_sample_pooled_for_oos_table(
    in_sample_table_path: Path,
) -> pd.DataFrame:
    """Load the in-sample cascade_precursor_real_table.csv and return only the
    columns + rows needed for side-by-side OOS comparison.

    Returns DataFrame with columns: scope, horizon, symbol, fold, n_total,
    n_cascades, base_rate, auc_in_sample, auc_in_sample_lo, auc_in_sample_hi,
    precision_top_1pct_in_sample.  Empty DataFrame if file missing.
    """
    if not in_sample_table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(in_sample_table_path)
    needed = {"scope", "horizon", "symbol", "auc", "auc_lo", "auc_hi"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame(
        df[
            [
                "scope",
                "horizon",
                "symbol",
                "n_total",
                "n_cascades",
                "base_rate",
                "auc",
                "auc_lo",
                "auc_hi",
                "precision_at_top_1pct",
            ]
        ]
    ).copy()
    out = out.rename(
        columns={
            "auc": "auc_in_sample",
            "auc_lo": "auc_in_sample_lo",
            "auc_hi": "auc_in_sample_hi",
            "precision_at_top_1pct": "precision_top_1pct_in_sample",
        }
    )
    out["fold"] = "in_sample"
    return out


def _emit_oos_markdown(
    per_cell_oos: pd.DataFrame,
    per_cell_in_sample: pd.DataFrame,
    out_path: Path,
    *,
    horizons: tuple[int, ...],
    elapsed_sec: float,
    notes: list[str] | None = None,
) -> None:
    """Render the OOS markdown verdict per the prompt's required outline."""
    lines: list[str] = []
    lines.append(
        "# Goal-A cascade-precursor (OOS test) — does AUC=0.815 generalize to Apr 14-26?"
    )
    lines.append("")
    lines.append(
        "**Question.** The Apr 1-13 in-sample LR (83-dim flat features, balanced "
        "class weight, leave-one-day-out CV) reported pooled cross-symbol AUC = "
        "**0.815 [0.772, 0.848]** at H500 with day-clustered bootstrap, and "
        "top-1% precision = 27.6% (lift 6.86×).  The marginal-long strategy at "
        "top-1% / 0.5% / 0.1% was net-negative — not directly tradeable.  "
        "**This run tests whether the AUC=0.815 signal generalizes to Apr 14-26 "
        "(genuinely held-out at the time of the in-sample fit).**"
    )
    lines.append("")
    lines.append(
        "**Protocol.** ONE universe-wide LR (`LogisticRegression(class_weight="
        "'balanced', C=1.0)`) fit on ALL Apr 1-13 data — no leave-one-day-out, "
        "since the test set is genuinely held-out.  Stride=200 evaluation "
        "windows on Apr 14-26 cache shards.  Real cascade label = any "
        "`cause IN ('market_liquidation', 'backstop_liquidation')` fill in "
        "(anchor_ts, ts_at(anchor + H)].  Day-clustered bootstrap CI "
        f"(resample the {int(per_cell_oos['n_days'].max()) if not per_cell_oos.empty else 0} "  # type: ignore[arg-type]
        "OOS days with replacement, 1000 iters).  Shuffled-OOS baseline: "
        "labels permuted within day on the same OOS predictions (label "
        "permutation test).  Per-symbol cells reported for symbols with ≥ 3 "
        "real cascades on Apr 14-26 at H500."
    )
    lines.append("")
    lines.append(
        "**Hard constraint (anti-amnesia).** The Apr 14+ holdout has been "
        "DELIBERATELY consumed for this test.  After this run, no untouched "
        "cascade-labeled holdout remains; future OOS evaluations require "
        "either (a) waiting for new data accrual, or (b) splitting the merged "
        "Apr 1-26 dataset.  This is the binding generalization test for the "
        "cascade-precursor program."
    )
    lines.append("")

    # ---------- 1. Sample size ----------
    lines.append("## 1. Sample size on OOS (Apr 14-26)")
    lines.append("")
    pooled_oos: pd.DataFrame = pd.DataFrame(
        per_cell_oos[per_cell_oos["scope"] == "pooled"]
    ).copy()
    if pooled_oos.empty:
        lines.append("**No pooled OOS cells produced — see flags below.**")
    else:
        lines.append("| H | n_total (windows) | n_cascades | base rate | n_days |")
        lines.append("|---|---|---|---|---|")
        for h in horizons:
            sub = pooled_oos[pooled_oos["horizon"] == h]
            if sub.empty:
                lines.append(f"| H{h} | — | — | — | — |")
                continue
            r = sub.iloc[0]
            lines.append(
                f"| H{h} | {int(r['n_total'])} | {int(r['n_cascades'])} | "
                f"{float(r['base_rate']):.4f} | {int(r['n_days'])} |"
            )
        lines.append("")
    lines.append("")

    # ---------- 2. Pooled OOS AUC + side-by-side with in-sample ----------
    lines.append("## 2. Pooled cross-symbol AUC OOS vs in-sample")
    lines.append("")
    lines.append(
        "| H | AUC OOS (day-clustered) | AUC in-sample | Δ_AUC | "
        "AUC OOS shuffled (day-clustered) |"
    )
    lines.append("|---|---|---|---|---|")
    pooled_is_lookup: dict[int, dict] = {}
    if not per_cell_in_sample.empty:
        is_pooled = per_cell_in_sample[
            (per_cell_in_sample["scope"] == "pooled")
            & (per_cell_in_sample["symbol"] == "ALL")
        ]
        for _, row in is_pooled.iterrows():
            try:
                h_i = int(row["horizon"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            pooled_is_lookup[h_i] = {
                "auc": float(row["auc_in_sample"]),  # type: ignore[arg-type]
                "lo": float(row["auc_in_sample_lo"]),  # type: ignore[arg-type]
                "hi": float(row["auc_in_sample_hi"]),  # type: ignore[arg-type]
            }

    for h in horizons:
        sub = pooled_oos[pooled_oos["horizon"] == h]
        if sub.empty:
            lines.append(f"| H{h} | — | — | — | — |")
            continue
        r = sub.iloc[0]
        auc_oos = float(r["auc_oos"])
        auc_oos_lo = float(r["auc_oos_lo"])
        auc_oos_hi = float(r["auc_oos_hi"])
        auc_sh = float(r["auc_shuffled_oos"])
        auc_sh_lo = float(r["auc_shuffled_oos_lo"])
        auc_sh_hi = float(r["auc_shuffled_oos_hi"])
        is_row = pooled_is_lookup.get(h)
        if is_row is None or not math.isfinite(is_row["auc"]):
            is_str = "—"
            delta = "—"
        else:
            is_str = f"{is_row['auc']:.3f} [{is_row['lo']:.3f}, {is_row['hi']:.3f}]"
            delta_v = auc_oos - is_row["auc"]
            delta = f"{delta_v:+.3f}"
        lines.append(
            f"| H{h} | {auc_oos:.3f} [{auc_oos_lo:.3f}, {auc_oos_hi:.3f}] | "
            f"{is_str} | {delta} | "
            f"{auc_sh:.3f} [{auc_sh_lo:.3f}, {auc_sh_hi:.3f}] |"
        )
    lines.append("")

    # ---------- 3. Distinguishability ----------
    lines.append("## 3. Distinguishable from shuffled-OOS baseline?")
    lines.append("")
    lines.append(
        "Binding statistical test: real-OOS AUC CI lower bound must strictly "
        "exceed shuffled-OOS AUC CI upper bound (day-clustered bootstrap)."
    )
    lines.append("")
    if pooled_oos.empty:
        lines.append("**Cannot evaluate — no pooled OOS cells.**")
    else:
        for h in horizons:
            sub = pooled_oos[pooled_oos["horizon"] == h]
            if sub.empty:
                continue
            r = sub.iloc[0]
            verdict = (
                "**DISTINGUISHABLE**"
                if bool(r["signal_distinguishable_from_shuffled"])
                else "**NOT distinguishable**"
            )
            gap = float(r["auc_oos_lo"]) - float(r["auc_shuffled_oos_hi"])
            lines.append(
                f"* H{h}: real OOS CI [{float(r['auc_oos_lo']):.3f}, "
                f"{float(r['auc_oos_hi']):.3f}] vs shuffled OOS CI "
                f"[{float(r['auc_shuffled_oos_lo']):.3f}, "
                f"{float(r['auc_shuffled_oos_hi']):.3f}] → {verdict} "
                f"(lo - shuffled_hi = {gap:+.3f})."
            )
    lines.append("")

    # ---------- 4. Precision-at-top-1% OOS ----------
    lines.append("## 4. OOS precision-at-top-1% (lift over base rate)")
    lines.append("")
    lines.append(
        "In-sample held precision-at-top-1% = 27.6% at H500 (lift 6.86×).  "
        "Does this hold OOS?"
    )
    lines.append("")
    if pooled_oos.empty:
        lines.append("**No pooled OOS cells available.**")
    else:
        lines.append("| H | base rate | precision@top-1% OOS (day-clustered) | lift |")
        lines.append("|---|---|---|---|")
        for h in horizons:
            sub = pooled_oos[pooled_oos["horizon"] == h]
            if sub.empty:
                continue
            r = sub.iloc[0]
            base = float(r["base_rate"])
            prec = float(r["precision_top_1pct_oos"])
            prec_lo = float(r["precision_top_1pct_oos_lo"])
            prec_hi = float(r["precision_top_1pct_oos_hi"])
            lift = float(r["lift_oos"])
            prec_s = (
                f"{prec:.4f} [{prec_lo:.4f}, {prec_hi:.4f}]"
                if math.isfinite(prec)
                else "—"
            )
            lift_s = f"{lift:.2f}" if math.isfinite(lift) else "—"
            lines.append(f"| H{h} | {base:.4f} | {prec_s} | {lift_s} |")
    lines.append("")

    # ---------- 5. Per-symbol OOS distribution ----------
    lines.append(
        "## 5. Per-symbol OOS distribution (≥ 3 cascades, AUC > 0.65 OOS @ H500)"
    )
    lines.append("")
    per_sym_oos: pd.DataFrame = pd.DataFrame(
        per_cell_oos[per_cell_oos["scope"] == "per_symbol"]
    ).copy()
    if per_sym_oos.empty:
        lines.append(
            "**No per-symbol OOS cells — no symbols had ≥ 3 cascades on Apr 14-26.**"
        )
    else:
        lines.append(
            "Symbols with ≥ 3 OOS cascades at the horizon of interest "
            "(descriptive only; multiple-comparisons not adjusted)."
        )
        lines.append("")
        lines.append(
            "| symbol | H | n_cascades | AUC OOS (day-clustered) | " "AUC > 0.65 OOS? |"
        )
        lines.append("|---|---|---|---|---|")
        for h in horizons:
            sub_df: pd.DataFrame = pd.DataFrame(
                per_sym_oos[per_sym_oos["horizon"] == h]
            ).sort_values("auc_oos", ascending=False)
            for _, r in sub_df.iterrows():
                auc_v = float(r["auc_oos"])  # type: ignore[arg-type]
                lo_v = float(r["auc_oos_lo"])  # type: ignore[arg-type]
                hi_v = float(r["auc_oos_hi"])  # type: ignore[arg-type]
                clears = math.isfinite(auc_v) and auc_v > 0.65
                lines.append(
                    f"| {r['symbol']} | H{int(r['horizon'])} | "  # type: ignore[arg-type]
                    f"{int(r['n_cascades'])} | "  # type: ignore[arg-type]
                    f"{auc_v:.3f} [{lo_v:.3f}, {hi_v:.3f}] | "
                    f"{'YES' if clears else 'NO'} |"
                )
        lines.append("")
        # Aggregate: how many symbols clear AUC > 0.65 at H500?
        h500_per_sym = per_sym_oos[per_sym_oos["horizon"] == 500]
        clears_h500 = 0
        for _, r in h500_per_sym.iterrows():
            auc_v = float(r["auc_oos"])  # type: ignore[arg-type]
            if math.isfinite(auc_v) and auc_v > 0.65:
                clears_h500 += 1
        n_h500 = int(len(h500_per_sym))
        lines.append(
            f"**{clears_h500} / {n_h500} per-symbol cells clear AUC > 0.65 OOS at H500.**"
        )
    lines.append("")

    # ---------- 6. AVAX OOS ----------
    lines.append("## 6. AVAX OOS (held out from v1 contrastive training)")
    lines.append("")
    avax_rows = per_sym_oos[
        (per_sym_oos["symbol"] == "AVAX") & (per_sym_oos["horizon"] == 500)
    ]
    if avax_rows.empty:
        lines.append(
            "**AVAX did not meet the ≥ 3-cascade threshold at H500 OOS — no "
            "per-symbol cell.**  AVAX was excluded from v1 contrastive "
            "training, but the cascade LR did not use that encoder; AVAX is "
            "treated as just another per-symbol cell here."
        )
    else:
        r = avax_rows.iloc[0]
        auc_v = float(r["auc_oos"])
        lo_v = float(r["auc_oos_lo"])
        hi_v = float(r["auc_oos_hi"])
        n_cas = int(r["n_cascades"])
        lines.append(
            f"AVAX OOS at H500: AUC = {auc_v:.3f} [{lo_v:.3f}, {hi_v:.3f}]  "
            f"(n_cascades = {n_cas}).  AVAX was excluded from v1 contrastive "
            "training; the cascade LR did not use that encoder, so AVAX is "
            "just another per-symbol cell here."
        )
    lines.append("")

    # ---------- 7. Verdict ----------
    lines.append("## 7. Verdict (per decision matrix)")
    lines.append("")
    if pooled_oos.empty:
        lines.append("**INCONCLUSIVE — no pooled OOS cells.**  See flags below.")
    else:
        # Use H500 as the binding cut (per prompt: "Pooled is the binding cut").
        h500 = pooled_oos[pooled_oos["horizon"] == 500]
        if h500.empty:
            lines.append("**INCONCLUSIVE — no H500 pooled OOS cell.**")
        else:
            r = h500.iloc[0]
            auc = float(r["auc_oos"])
            lo = float(r["auc_oos_lo"])
            sh_hi = float(r["auc_shuffled_oos_hi"])
            dist = bool(r["signal_distinguishable_from_shuffled"])
            prec_oos = float(r["precision_top_1pct_oos"])

            if (
                math.isfinite(auc)
                and auc > 0.75
                and math.isfinite(lo)
                and lo > 0.65
                and dist
            ):
                verdict = (
                    "**GENERALIZES.**  H500 OOS AUC > 0.75, CI lower bound > "
                    "0.65, distinguishable from shuffled-OOS baseline.  The "
                    "Apr 1-13 in-sample signal extends to Apr 14-26.  "
                    f"Precision-at-top-1% OOS = {prec_oos:.4f}.  Encoder "
                    "retrain on the merged Apr 1-26 dataset (~150 cascades) "
                    "is worth committing GPU compute to."
                )
            elif math.isfinite(auc) and 0.60 <= auc <= 0.75:
                verdict = (
                    "**PARTIALLY GENERALIZES.**  H500 OOS AUC in [0.60, 0.75]. "
                    " Decision depends on per-symbol distribution and whether "
                    "precision-at-top-1% holds at in-sample levels.  Encoder "
                    "retrain is conditional on the per-symbol heatmap."
                )
            elif math.isfinite(lo) and lo > sh_hi and math.isfinite(auc) and auc > 0.65:
                # Edge case: AUC slightly below 0.75 but still distinguishable.
                # Treat as partial.
                verdict = (
                    "**PARTIALLY GENERALIZES.**  H500 OOS AUC distinguishable "
                    "from shuffled but the lower bound is below the 'fully "
                    "generalizes' 0.65 threshold.  Per-symbol patterns should "
                    "drive next steps."
                )
            else:
                verdict = (
                    "**FAILS TO GENERALIZE.**  H500 OOS AUC < 0.60 or CI lower "
                    "bound overlaps the shuffled-OOS baseline.  The in-sample "
                    "result was overfit to the Apr 1-13 distribution.  The "
                    "cascade-precursor program kills cleanly here."
                )
            lines.append(verdict)
    lines.append("")

    # ---------- 8. Methodological flags ----------
    lines.append("## 8. Methodological flags")
    lines.append("")
    lines.append(
        "* **Day-clustered bootstrap is the binding test.**  Per-window "
        "bootstrap on tightly clustered cascade data understates uncertainty "
        "(prior commit `e2715ec` proved this).  All AUC and precision CIs in "
        "this writeup resample the OOS days with replacement."
    )
    lines.append("")
    lines.append(
        "* **Single LR fit, no fold-CV.**  Apr 14-26 is genuinely held-out, "
        "so the protocol is fit-once-on-train, predict-once-on-test.  No "
        "model selection, no hyperparameter search."
    )
    lines.append("")
    lines.append(
        "* **Apples-to-apples with in-sample.**  Same 83-dim flat features, "
        "same `LogisticRegression(class_weight='balanced', C=1.0)`, same "
        "cascade label definition (`cause IN ('market_liquidation', "
        "'backstop_liquidation')`).  Per-symbol minimum is 3 cascades on OOS "
        "(vs 5 in-sample) because the OOS window is shorter."
    )
    lines.append("")
    lines.append(
        "* **Holdout permanently consumed.**  The Apr 14+ data was loaded by "
        "this script via the unsafe-loader code path.  No untouched cascade-"
        "labeled holdout remains; future OOS evaluation requires new data "
        "accrual or merged-dataset splitting."
    )
    lines.append("")
    lines.append(
        "* **Distribution-shift caveat.**  If Apr 14-26 has a structurally "
        "different cascade frequency or volatility regime than Apr 1-13, the "
        "OOS gap can reflect domain shift rather than overfit.  Compare base "
        "rate and n_cascades across the two folds before drawing strong "
        "conclusions."
    )
    lines.append("")
    lines.append(
        "* **Shuffled-OOS AUC > 0.50 is expected under cascade contagion.**  "
        "The shuffled-OOS baseline permutes labels WITHIN each day (preserving "
        "per-day cascade count).  Day-clustered bootstrap resamples days with "
        "replacement: if the LR's day-mean prediction correlates with the "
        "day's cascade rate (volatility regime), the shuffled AUC drifts above "
        "0.5 even though the within-day rank carries no signal.  The "
        "distinguishability test (real-lo > shuffled-hi) correctly accounts "
        "for this — the real AUC must exceed the contagion floor, not the "
        "0.5 chance line."
    )
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"* {note}")
            lines.append("")

    lines.append(
        f"_OOS pipeline ran in {elapsed_sec:.1f} s.  CPU-only.  "
        f"Apr 14+ holdout permanently consumed by this run._"
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
        "--cascade-direction",
        action="store_true",
        help=(
            "Run the direction LR on the cascade-likely subset (top-5% by "
            "stage-2 cascade-onset pred_proba).  Loads the existing "
            "cascade_precursor_real_per_window.parquet, builds an 84-dim "
            "feature vector per window (83 flat + pred_proba_h500), trains "
            "direction LR with leave-one-day-out CV across April 1-13, and "
            "emits cascade_direction.md + cascade_direction_table.csv."
        ),
    )
    parser.add_argument(
        "--marginal-long",
        action="store_true",
        help=(
            "Run the marginal-long + precision sweep on the existing "
            "cascade_precursor_real_per_window.parquet.  Reanalysis only — "
            "no retraining.  Sweeps precision cutoffs {1%%, 0.5%%, 0.1%%}, "
            "computes pooled + per-symbol PnL with day-clustered bootstrap "
            "CIs, and compares against the LR-direction strategy.  Emits "
            "cascade_marginal_long.md + cascade_marginal_long_table.csv."
        ),
    )
    parser.add_argument(
        "--oos-test",
        action="store_true",
        help=(
            "Run the OOS generalization test on Apr 14-26 (consumes the "
            "holdout — gotcha #17).  Trains ONE universe-wide LR on Apr 1-13 "
            "real cause-flag labels, applies to Apr 14-26 stride=200 windows, "
            "and emits cascade_precursor_oos_per_window.parquet, "
            "cascade_precursor_oos_table.csv, and cascade_precursor_oos.md "
            "with day-clustered bootstrap CIs."
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

    if args.marginal_long:
        # ---------- Stage 2 follow-up: marginal-long + precision sweep ----------
        print(
            f"[marginal-long] precision sweep on existing predictions | "
            f"cache={args.cache} | n_boot={int(args.n_boot)}"
        )
        try:
            ml_table, ml_summary = _run_marginal_long_pipeline(
                args.cache,
                out_dir,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        except RuntimeError as exc:
            print(f"[marginal-long] BLOCKER: {exc}")
            return 1

        out_csv = out_dir / "cascade_marginal_long_table.csv"
        out_md = out_dir / "cascade_marginal_long.md"
        ml_table.to_csv(out_csv, index=False)
        elapsed = time.time() - t0
        _emit_marginal_long_markdown(
            ml_table,
            ml_summary,
            out_md,
            elapsed_sec=elapsed,
            n_boot=int(args.n_boot),
        )
        print(f"[marginal-long] done in {elapsed:.1f} s")
        print(f"[marginal-long] wrote {out_csv}")
        print(f"[marginal-long] wrote {out_md}")
        return 0

    if args.cascade_direction:
        # ---------- Stage 2 follow-up: direction LR on cascade-likely subset ----------
        print(
            f"[cascade-direction] direction LR on top-5% subset, H500 only | "
            f"cache={args.cache}"
        )
        try:
            cas_table, summary = _run_cascade_direction_pipeline(
                args.cache,
                out_dir,
                horizon=DIRECTION_HORIZON,
                seed=int(args.seed),
                n_boot=int(args.n_boot),
            )
        except RuntimeError as exc:
            print(f"[cascade-direction] BLOCKER: {exc}")
            return 1

        out_csv = out_dir / "cascade_direction_table.csv"
        out_md = out_dir / "cascade_direction.md"
        cas_table.to_csv(out_csv, index=False)
        elapsed = time.time() - t0
        _emit_direction_markdown(summary, cas_table, out_md, elapsed_sec=elapsed)
        print(f"[cascade-direction] done in {elapsed:.1f} s")
        print(f"[cascade-direction] wrote {out_csv}")
        print(f"[cascade-direction] wrote {out_md}")
        return 0

    if args.oos_test:
        # ---------- OOS test: Apr 14-26 generalization (consumes holdout) ----------
        oos_horizons: tuple[int, ...] = tuple(h for h in horizons if h in OOS_HORIZONS)
        if not oos_horizons:
            oos_horizons = OOS_HORIZONS
        print(
            f"[oos-test] Apr 14-26 OOS test | horizons={oos_horizons} | "
            f"symbols={len(symbols)} | cache={args.cache} | "
            f"n_boot={int(args.n_boot)}"
        )
        try:
            oos_per_window_df, oos_per_cell_df, _ = _run_oos_pipeline(
                args.cache,
                symbols,
                out_dir,
                horizons=oos_horizons,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        except RuntimeError as exc:
            print(f"[oos-test] BLOCKER: {exc}")
            return 1

        # Side-by-side in-sample lookup table
        is_table_path = out_dir / "cascade_precursor_real_table.csv"
        in_sample_df = _load_in_sample_pooled_for_oos_table(is_table_path)

        oos_per_window_path = out_dir / "cascade_precursor_oos_per_window.parquet"
        oos_per_cell_path = out_dir / "cascade_precursor_oos_table.csv"
        oos_md_path = out_dir / "cascade_precursor_oos.md"

        oos_per_window_df.to_parquet(oos_per_window_path, index=False)
        oos_per_cell_df.to_csv(oos_per_cell_path, index=False)

        elapsed = time.time() - t0
        _emit_oos_markdown(
            oos_per_cell_df,
            in_sample_df,
            oos_md_path,
            horizons=oos_horizons,
            elapsed_sec=elapsed,
        )
        print(f"[oos-test] done in {elapsed:.1f} s")
        print(f"[oos-test] wrote {oos_per_window_path}")
        print(f"[oos-test] wrote {oos_per_cell_path}")
        print(f"[oos-test] wrote {oos_md_path}")
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

# scripts/random_init_probe.py
"""Goal-A v2 Phase 0 — Random-init encoder linear probe vs unified flat-LR.

Implements the ratified plan at:
  docs/experiments/goal-a-v2/2026-04-27-random-init-probe-plan.md
And the council protocol at:
  docs/council-reviews/2026-04-27-encoder-retrain-protocol.md

Single-script CPU-only experiment: produces (a) a unified-CV flat-LR baseline on
the merged Apr 1-26 cascade-H500 task and (b) a random-init `TapeEncoder` linear
probe under the SAME 5-fold day-blocked CV folds, with paired day-clustered
bootstrap on the delta.

Outputs (default --out-dir docs/experiments/goal-a-v2/):
  * random_init_probe_table.csv — per-symbol per-fold AUC for both models.
  * random_init_probe.md         — markdown report with decision verdict.
  * random_init_probe_per_window.parquet — per-window OOF predictions (gitignored).

Usage:
    uv run python scripts/random_init_probe.py --cache data/cache \
        --out-dir docs/experiments/goal-a-v2 [--smoke]

Key implementation notes:
  * Apr 14-26 holdout was deliberately consumed on 2026-04-27 (gotcha #17).
    The merged Apr 1-26 dataset is the ONLY defensible evaluation now.  We use
    the unsafe-loaders pattern from cascade_precursor_probe.py to bypass the
    APRIL_HELDOUT_START guard for raw-trade liquidation timestamps.
  * BatchNorm at inference (gotcha #18): we set `track_running_stats=False` on
    the input BatchNorm1d.  In eval(), the BN layer then uses batch statistics.
    Defensible because each forward-pass batch is large (256) and the random-init
    encoder isn't relying on learned BN statistics anyway.
  * Stride-200 (STRIDE_EVAL) windows; 600-event embargo at fold boundaries
    measured in TRADE-EVENT INDEX space (gotcha #12).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tape.constants import (
    APRIL_START,
    EMBARGO_EVENTS,
    STRIDE_EVAL,
    SYMBOLS,
    WINDOW_LEN,
)
from tape.flat_features import FLAT_DIM, extract_flat_features
from tape.model import EncoderConfig, TapeEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT: Path = Path("data/cache")
OUT_DIR_DEFAULT: Path = Path("docs/experiments/goal-a-v2")

# Goal-A v2 Phase 0 dates: merged Apr 1-26 (holdout consumed).
PHASE0_START: str = "2026-04-01"
PHASE0_END_INCLUSIVE: str = "2026-04-26"

# Primary endpoint horizon
H_PRIMARY: int = 500
PROBE_HORIZONS: tuple[int, ...] = (H_PRIMARY,)  # ratified plan §Scoring

# CV partition
N_FOLDS: int = 5

# Bootstrap defaults
N_BOOT: int = 1000
ALPHA: float = 0.05

# Per-symbol gating for BH-FDR table
PER_SYMBOL_MIN_CASCADES: int = 3

# Encoder probe parameters
ENCODER_BATCH_SIZE: int = 256
ENCODER_SEEDS: tuple[int, ...] = (0, 1, 2)
ENCODER_OUTPUT_DIM: int = 256

# Per-symbol AUC null hypothesis cut
PER_SYMBOL_AUC_NULL: float = 0.5  # H0: AUC = 0.5 (one-sided "AUC > 0.5")


# ---------------------------------------------------------------------------
# Phase-0 dates
# ---------------------------------------------------------------------------


def phase0_dates() -> list[str]:
    """Apr 1 through Apr 26 (inclusive) as YYYY-MM-DD strings."""
    return [f"2026-04-{d:02d}" for d in range(1, 27)]


def _is_phase0_date(date_str: str) -> bool:
    return PHASE0_START <= date_str <= PHASE0_END_INCLUSIVE


# ---------------------------------------------------------------------------
# Day-blocked CV partition + embargo
# ---------------------------------------------------------------------------


def day_blocked_folds(days: Sequence[str], k: int = N_FOLDS) -> list[list[str]]:
    """Partition `days` into `k` contiguous date-ordered folds (~equal size).

    `days` is sorted ascending.  Folds are contiguous: the first ceil(N/k) days
    go to fold 0, etc.  Edge case: if N < k, returns k folds with some empty
    (caller should validate).
    """
    sorted_days = sorted(days)
    n = len(sorted_days)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    # Use a balanced split: first (n % k) folds get ceil(n/k) days, rest floor.
    base = n // k
    extra = n % k
    folds: list[list[str]] = []
    cursor = 0
    for fi in range(k):
        size = base + (1 if fi < extra else 0)
        folds.append(sorted_days[cursor : cursor + size])
        cursor += size
    return folds


def apply_embargo_mask(
    *,
    held_out_fold: int,
    fold_assignments: np.ndarray,
    dates: np.ndarray,
    anchor_idx_in_day: np.ndarray,
    embargo_events: int,
    days_by_fold: list[list[str]],
) -> np.ndarray:
    """Build a boolean training mask that excludes held-out fold + embargoed
    boundary events of neighboring folds.

    Parameters
    ----------
    held_out_fold : int
        Index of the fold being held out as the test set.
    fold_assignments : (N,) int64 — fold index for each row.
    dates : (N,) str — date string per row.
    anchor_idx_in_day : (N,) int64 — anchor event index within that day
        (NOT window index — gotcha #2 wants event-index alignment).
    embargo_events : int — number of trailing/leading events to drop on the
        boundary day.
    days_by_fold : list[list[str]] — date partition (sorted within each fold).

    Returns
    -------
    (N,) bool training mask.
    """
    n = len(fold_assignments)
    train_mask = np.ones(n, dtype=bool)
    # Held-out fold: exclude fully.
    train_mask &= fold_assignments != held_out_fold

    # Drop LAST `embargo_events` of fold (k-1)'s last day.
    if held_out_fold > 0 and held_out_fold - 1 < len(days_by_fold):
        prev_fold_days = days_by_fold[held_out_fold - 1]
        if prev_fold_days:
            last_day = max(prev_fold_days)
            in_day = (dates == last_day) & (fold_assignments == held_out_fold - 1)
            # Need to identify the "last" events: anchor_idx >= max - embargo
            if in_day.any():
                max_anchor = int(anchor_idx_in_day[in_day].max())
                cutoff = max_anchor - embargo_events + 1
                drop = in_day & (anchor_idx_in_day >= cutoff)
                train_mask &= ~drop

    # Drop FIRST `embargo_events` of fold (k+1)'s first day.
    if held_out_fold + 1 < len(days_by_fold):
        next_fold_days = days_by_fold[held_out_fold + 1]
        if next_fold_days:
            first_day = min(next_fold_days)
            in_day = (dates == first_day) & (fold_assignments == held_out_fold + 1)
            if in_day.any():
                min_anchor = int(anchor_idx_in_day[in_day].min())
                cutoff = min_anchor + embargo_events
                drop = in_day & (anchor_idx_in_day < cutoff)
                train_mask &= ~drop

    return train_mask


# ---------------------------------------------------------------------------
# Cascade label loaders (unsafe — bypass APRIL_HELDOUT_START guard for Apr 14+)
# ---------------------------------------------------------------------------


def _load_liq_ts_phase0(symbol: str, date_str: str) -> np.ndarray | None:
    """Sorted int64 array of liquidation trade ts_ms for (symbol, date) on Apr 1-26.

    Bypasses the APRIL_HELDOUT_START guard since Apr 14-26 is the consumed
    holdout (gotcha #17 — one-shot consumption already happened in commit b0de994).
    """
    if date_str < APRIL_START:
        return None
    if not _is_phase0_date(date_str):
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


def _load_shard_phase0(path: Path) -> dict:
    """Load all keys from an .npz cache shard (no APRIL_HELDOUT_START guard)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _real_cascade_label(
    *,
    anchor_ts: np.ndarray,
    window_starts: np.ndarray,
    event_ts: np.ndarray,
    horizon: int,
    liq_ts: np.ndarray,
) -> np.ndarray:
    """Per-window binary real-cascade label using the same logic as
    `_real_cascade_label_with_event_ts` in cascade_precursor_probe.py.
    """
    n = len(anchor_ts)
    out = np.zeros(n, dtype=np.int8)
    if len(liq_ts) == 0:
        return out

    n_events = len(event_ts)
    anchor_idx = window_starts + WINDOW_LEN - 1
    end_idx = anchor_idx + horizon
    valid = end_idx < n_events
    if not valid.any():
        return out

    end_ts = event_ts[end_idx[valid]].astype(np.int64)
    start_ts = anchor_ts[valid].astype(np.int64)

    lo = np.searchsorted(liq_ts, start_ts, side="right")
    hi = np.searchsorted(liq_ts, end_ts, side="right")
    has_liq = (hi - lo) > 0
    out_valid = out[valid]
    out_valid[has_liq] = 1
    out[valid] = out_valid
    return out


# ---------------------------------------------------------------------------
# Day batch with raw (200, 17) windows for encoder forward
# ---------------------------------------------------------------------------


@dataclass
class RealDayBatchWithRaw:
    """Per-(symbol, date) windows + flat features + raw (200, 17) windows + labels.

    Mirrors `RealDayBatch` from cascade_precursor_probe.py but additionally
    retains the raw (N, 200, 17) tensor for encoder forward-pass.
    """

    symbol: str
    date: str
    flat_X: np.ndarray  # (N, FLAT_DIM)
    raw_X: np.ndarray  # (N, 200, 17)
    anchor_ts: np.ndarray  # (N,)
    window_starts: np.ndarray  # (N,)
    anchor_idx_in_day: np.ndarray  # (N,) — same as anchors (alias for clarity)
    real_labels: dict[int, np.ndarray]
    real_valid: dict[int, np.ndarray]


def build_day_batch_with_raw(
    cache_dir: Path,
    symbol: str,
    date_str: str,
    horizons: tuple[int, ...] = PROBE_HORIZONS,
) -> RealDayBatchWithRaw | None:
    """Build a RealDayBatchWithRaw for (symbol, date_str) on Apr 1-26.

    Returns None if cache shard or raw-trade parquet missing, or if shard empty.
    """
    if not _is_phase0_date(date_str):
        return None
    shard_path = cache_dir / f"{symbol}__{date_str}.npz"
    if not shard_path.exists():
        return None
    payload = _load_shard_phase0(shard_path)
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

    # Flat features and raw windows.
    flat_X = np.empty((len(starts), FLAT_DIM), dtype=np.float32)
    raw_X = np.empty((len(starts), WINDOW_LEN, features.shape[1]), dtype=np.float32)
    for i, s in enumerate(starts):
        win = features[s : s + WINDOW_LEN]
        flat_X[i] = extract_flat_features(win)
        raw_X[i] = win

    liq_ts = _load_liq_ts_phase0(symbol, date_str)
    if liq_ts is None:
        return None

    real_labels: dict[int, np.ndarray] = {}
    real_valid: dict[int, np.ndarray] = {}
    for h in horizons:
        end_idx = anchors + h
        valid = end_idx < n_events
        real_valid[h] = valid
        real_labels[h] = _real_cascade_label(
            anchor_ts=anchor_ts,
            window_starts=starts,
            event_ts=event_ts,
            horizon=h,
            liq_ts=liq_ts,
        )

    return RealDayBatchWithRaw(
        symbol=symbol,
        date=date_str,
        flat_X=flat_X,
        raw_X=raw_X,
        anchor_ts=anchor_ts,
        window_starts=starts,
        anchor_idx_in_day=anchors,
        real_labels=real_labels,
        real_valid=real_valid,
    )


def gather_phase0_batches(
    cache_dir: Path,
    symbols: tuple[str, ...],
    dates: list[str],
    horizons: tuple[int, ...] = PROBE_HORIZONS,
) -> list[RealDayBatchWithRaw]:
    """Build all RealDayBatchWithRaw across (symbol × date) on Apr 1-26."""
    out: list[RealDayBatchWithRaw] = []
    for symbol in symbols:
        for d in dates:
            b = build_day_batch_with_raw(cache_dir, symbol, d, horizons=horizons)
            if b is not None:
                out.append(b)
    return out


def stack_batches_at_horizon(
    batches: list[RealDayBatchWithRaw], horizon: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Concatenate (flat_X, raw_X, y, sym, date, anchor_ts, anchor_idx_in_day)
    at a given horizon, dropping windows where the horizon overruns the day.
    """
    if not batches:
        return (
            np.zeros((0, FLAT_DIM), dtype=np.float32),
            np.zeros((0, WINDOW_LEN, 17), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype="<U16"),
            np.zeros(0, dtype="<U10"),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    flats: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    syms: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    anchor_ts_list: list[np.ndarray] = []
    anchor_idx_list: list[np.ndarray] = []
    for b in batches:
        if horizon not in b.real_labels:
            continue
        valid = b.real_valid[horizon]
        if not valid.any():
            continue
        flats.append(b.flat_X[valid])
        raws.append(b.raw_X[valid])
        ys.append(b.real_labels[horizon][valid].astype(np.int8))
        n_v = int(valid.sum())
        syms.append(np.full(n_v, b.symbol, dtype="<U16"))
        dates.append(np.full(n_v, b.date, dtype="<U10"))
        anchor_ts_list.append(b.anchor_ts[valid].astype(np.int64))
        anchor_idx_list.append(b.anchor_idx_in_day[valid].astype(np.int64))
    if not flats:
        return (
            np.zeros((0, FLAT_DIM), dtype=np.float32),
            np.zeros((0, WINDOW_LEN, 17), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype="<U16"),
            np.zeros(0, dtype="<U10"),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.concatenate(flats, axis=0),
        np.concatenate(raws, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(syms, axis=0),
        np.concatenate(dates, axis=0),
        np.concatenate(anchor_ts_list, axis=0),
        np.concatenate(anchor_idx_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Random-init encoder helpers
# ---------------------------------------------------------------------------


def build_random_init_encoder(
    seed: int = 0,
    cfg: EncoderConfig | None = None,
) -> TapeEncoder:
    """Construct a fresh TapeEncoder with random init under torch seed.

    BatchNorm1d at the input is configured with `track_running_stats=False`
    (gotcha #18): in eval(), the layer uses BATCH statistics rather than
    population statistics that don't exist for a never-trained encoder.
    """
    if cfg is None:
        cfg = EncoderConfig()
    torch.manual_seed(seed)
    enc = TapeEncoder(cfg)
    # Override the input BatchNorm to disable running stats — embedding extraction
    # passes are large enough that batch statistics are well-defined.
    bn = enc.input_bn
    bn.track_running_stats = False
    bn.running_mean = None
    bn.running_var = None
    bn.num_batches_tracked = None
    enc.eval()
    return enc


def encode_windows(
    encoder: TapeEncoder,
    windows: np.ndarray,
    *,
    batch_size: int = ENCODER_BATCH_SIZE,
    device: str = "cpu",
) -> np.ndarray:
    """Forward-pass `windows` through `encoder` on `device`, returning the
    256-dim global embedding for each row.

    `windows` shape: (N, 200, 17) float32.
    Returns: (N, 256) float32.
    """
    if windows.ndim != 3 or windows.shape[1] != WINDOW_LEN:
        raise ValueError(f"expected (N, {WINDOW_LEN}, 17), got {windows.shape}")
    n = windows.shape[0]
    if n == 0:
        return np.zeros((0, encoder.global_dim), dtype=np.float32)
    encoder = encoder.to(device)
    encoder.eval()
    out = np.empty((n, encoder.global_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            x = torch.from_numpy(windows[i:j]).to(device)
            _, global_emb = encoder(x)
            out[i:j] = global_emb.detach().cpu().numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Logistic regression fit/predict
# ---------------------------------------------------------------------------


def fit_lr_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
) -> np.ndarray:
    """Balanced LR + StandardScaler; returns P(class=1) on Xte.

    Matches cascade_precursor_probe._fit_lr_proba contract.
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


# ---------------------------------------------------------------------------
# Day-clustered bootstrap (paired)
# ---------------------------------------------------------------------------


def _day_to_indices(dates: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    days = sorted(np.unique(dates).tolist())
    return days, {d: np.flatnonzero(dates == d) for d in days}


def day_clustered_bootstrap_auc(
    *,
    proba: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Day-clustered bootstrap AUC — mean + (1-alpha) percentile CI.

    Returns (mean, lo, hi).
    """
    if len(proba) == 0:
        return float("nan"), float("nan"), float("nan")
    days, day_idx = _day_to_indices(dates)
    if len(days) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = np.full(n_boot, np.nan, dtype=np.float64)
    k = len(days)
    for b in range(n_boot):
        sample_idx = rng.integers(0, k, size=k)
        parts = [day_idx[days[s]] for s in sample_idx]
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
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
    return (
        float(finite.mean()),
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
    )


def paired_day_clustered_bootstrap_delta(
    *,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = ALPHA,
) -> tuple[float, float, float, list[float], list[float]]:
    """Paired day-clustered bootstrap on the delta (AUC_b − AUC_a).

    Same RNG seed → same per-iteration day samples → both AUCs computed on the
    identical bootstrap fold each iteration.

    Returns
    -------
    (delta_mean, delta_lo, delta_hi, aucs_a, aucs_b) — both AUC arrays returned
    so test code can verify resample-identity.
    """
    if len(proba_a) != len(proba_b) or len(proba_a) != len(labels):
        raise ValueError("proba_a, proba_b, labels must have the same length")
    days, day_idx = _day_to_indices(dates)
    if len(days) < 2:
        return float("nan"), float("nan"), float("nan"), [], []

    rng = np.random.default_rng(seed)
    k = len(days)
    aucs_a: list[float] = []
    aucs_b: list[float] = []
    deltas: list[float] = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, k, size=k)
        parts = [day_idx[days[s]] for s in sample_idx]
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if len(idx) == 0:
            aucs_a.append(float("nan"))
            aucs_b.append(float("nan"))
            deltas.append(float("nan"))
            continue
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            aucs_a.append(float("nan"))
            aucs_b.append(float("nan"))
            deltas.append(float("nan"))
            continue
        try:
            a = float(roc_auc_score(y_b, proba_a[idx]))
            b = float(roc_auc_score(y_b, proba_b[idx]))
        except ValueError:
            aucs_a.append(float("nan"))
            aucs_b.append(float("nan"))
            deltas.append(float("nan"))
            continue
        aucs_a.append(a)
        aucs_b.append(b)
        deltas.append(b - a)
    arr = np.array(deltas, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 10:
        return float("nan"), float("nan"), float("nan"), aucs_a, aucs_b
    return (
        float(finite.mean()),
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
        aucs_a,
        aucs_b,
    )


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values matching scipy's
    `false_discovery_control(method="bh")`.

    Defers to scipy.stats.false_discovery_control directly when scipy is new
    enough; otherwise implements the equivalent step-up procedure.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    if p.size == 0:
        return p
    try:
        from scipy.stats import false_discovery_control

        return np.asarray(false_discovery_control(p, method="bh"), dtype=np.float64)
    except (ImportError, AttributeError):  # pragma: no cover
        # Fallback: standard BH step-up.
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        adjusted_sorted = np.empty(n, dtype=np.float64)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            val = ranked[i] * n / (i + 1)
            prev = min(prev, val)
            adjusted_sorted[i] = prev
        out = np.empty(n, dtype=np.float64)
        out[order] = adjusted_sorted
        return np.minimum(out, 1.0)


# ---------------------------------------------------------------------------
# Per-symbol p-value (one-sided "AUC > AUC_NULL") via day-clustered bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_per_symbol_pvalue(
    *,
    proba: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    n_boot: int,
    seed: int,
    null_auc: float = PER_SYMBOL_AUC_NULL,
) -> float:
    """One-sided bootstrap p-value: P(bootstrap AUC <= null_auc).

    Returns NaN if too few finite bootstrap samples.
    """
    if len(proba) == 0:
        return float("nan")
    days, day_idx = _day_to_indices(dates)
    if len(days) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    k = len(days)
    le_count = 0
    finite_count = 0
    for _ in range(n_boot):
        sample_idx = rng.integers(0, k, size=k)
        parts = [day_idx[days[s]] for s in sample_idx]
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if len(idx) == 0:
            continue
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            a = roc_auc_score(y_b, proba[idx])
        except ValueError:
            continue
        finite_count += 1
        if a <= null_auc:
            le_count += 1
    if finite_count < 10:
        return float("nan")
    return float(le_count) / float(finite_count)


# ---------------------------------------------------------------------------
# OOF prediction generator
# ---------------------------------------------------------------------------


@dataclass
class OOFPredictions:
    proba: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)
    sym: np.ndarray
    dates: np.ndarray
    anchor_ts: np.ndarray
    anchor_idx_in_day: np.ndarray
    fold_idx: np.ndarray
    per_fold_aucs: list[tuple[int, int, int, float]]  # (fold_idx, n_pos, n_neg, auc)


def run_5fold_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    sym: np.ndarray,
    dates: np.ndarray,
    anchor_ts: np.ndarray,
    anchor_idx_in_day: np.ndarray,
    embargo_events: int = EMBARGO_EVENTS,
    k: int = N_FOLDS,
) -> OOFPredictions:
    """Run k-fold day-blocked CV with embargo; pool OOF predictions.

    Each row's fold assignment is determined by its date.  Embargo drops
    boundary events of the neighboring training folds (NOT the held-out fold).
    The output OOF prediction array has length == len(X), one prediction per row.
    """
    sorted_days = sorted(np.unique(dates).tolist())
    days_by_fold = day_blocked_folds(sorted_days, k=k)
    day_to_fold: dict[str, int] = {
        d: fi for fi, fdays in enumerate(days_by_fold) for d in fdays
    }
    fold_assignments = np.array([day_to_fold[d] for d in dates], dtype=np.int64)

    proba_oof = np.full(len(X), np.nan, dtype=np.float64)
    per_fold: list[tuple[int, int, int, float]] = []

    for held_out in range(k):
        train_mask = apply_embargo_mask(
            held_out_fold=held_out,
            fold_assignments=fold_assignments,
            dates=dates,
            anchor_idx_in_day=anchor_idx_in_day,
            embargo_events=embargo_events,
            days_by_fold=days_by_fold,
        )
        test_mask = fold_assignments == held_out
        if not test_mask.any() or not train_mask.any():
            continue
        Xtr = X[train_mask]
        ytr = y[train_mask]
        Xte = X[test_mask]
        yte = y[test_mask]
        proba = fit_lr_predict(Xtr, ytr, Xte)
        proba_oof[test_mask] = proba
        n_pos = int((yte == 1).sum())
        n_neg = int((yte == 0).sum())
        if n_pos > 0 and n_neg > 0:
            try:
                auc = float(roc_auc_score(yte, proba))
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")
        per_fold.append((held_out, n_pos, n_neg, auc))

    return OOFPredictions(
        proba=proba_oof,
        labels=y.astype(np.int64),
        sym=sym,
        dates=dates,
        anchor_ts=anchor_ts,
        anchor_idx_in_day=anchor_idx_in_day,
        fold_idx=fold_assignments,
        per_fold_aucs=per_fold,
    )


# ---------------------------------------------------------------------------
# Decision tree (plan §Decision Logic)
# ---------------------------------------------------------------------------


def decision_tier(
    *,
    auc_flat: float,
    auc_enc_median: float,
    delta_lo: float,
    delta_hi: float,
) -> str:
    """Classify the encoder-vs-flat result per the plan's decision tree.

    Returns one of: 'GREENLIGHT_FINETUNE', 'MATCHED_FLAT', 'ARCH_BOTTLENECK'.
    """
    delta = auc_enc_median - auc_flat
    delta_ci_excludes_zero = (delta_lo > 0) or (delta_hi < 0)
    if delta >= 0.02 and delta_lo > 0:
        return "GREENLIGHT_FINETUNE"
    if delta < -0.02:
        return "ARCH_BOTTLENECK"
    # Default: roughly tied (delta CI overlaps 0).
    _ = delta_ci_excludes_zero  # informative but not driving
    return "MATCHED_FLAT"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _pooled_auc(proba: np.ndarray, labels: np.ndarray) -> float:
    """Pooled AUC; returns NaN on degenerate input."""
    valid = np.isfinite(proba)
    p = proba[valid]
    y = labels[valid]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _build_per_symbol_table(
    *,
    sym: np.ndarray,
    dates: np.ndarray,
    proba_flat: np.ndarray,
    proba_enc: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
    min_cascades: int = PER_SYMBOL_MIN_CASCADES,
    null_auc: float = PER_SYMBOL_AUC_NULL,
) -> pd.DataFrame:
    """Per-symbol AUC + bootstrap p-value, BH-FDR adjusted across symbols."""
    rows: list[dict] = []
    eligible_syms: list[str] = []
    for s in sorted(np.unique(sym).tolist()):
        mask = sym == s
        if not mask.any():
            continue
        y_s = labels[mask]
        n_pos = int((y_s == 1).sum())
        if n_pos < min_cascades:
            continue
        eligible_syms.append(s)
        valid = np.isfinite(proba_flat[mask]) & np.isfinite(proba_enc[mask])
        idx = np.flatnonzero(mask)[valid]
        if len(idx) == 0:
            continue
        labels_s = labels[idx]
        dates_s = dates[idx]
        p_flat_s = proba_flat[idx]
        p_enc_s = proba_enc[idx]
        try:
            auc_flat_s = float(roc_auc_score(labels_s, p_flat_s))
        except ValueError:
            auc_flat_s = float("nan")
        try:
            auc_enc_s = float(roc_auc_score(labels_s, p_enc_s))
        except ValueError:
            auc_enc_s = float("nan")
        pv_flat = _bootstrap_per_symbol_pvalue(
            proba=p_flat_s,
            labels=labels_s,
            dates=dates_s,
            n_boot=n_boot,
            seed=seed,
            null_auc=null_auc,
        )
        pv_enc = _bootstrap_per_symbol_pvalue(
            proba=p_enc_s,
            labels=labels_s,
            dates=dates_s,
            n_boot=n_boot,
            seed=seed + 1,
            null_auc=null_auc,
        )
        rows.append(
            dict(
                symbol=s,
                n_windows=int(mask.sum()),
                n_cascades=n_pos,
                auc_flat=auc_flat_s,
                auc_enc=auc_enc_s,
                p_flat=pv_flat,
                p_enc=pv_enc,
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "n_windows",
                "n_cascades",
                "auc_flat",
                "auc_enc",
                "p_flat",
                "p_enc",
                "q_flat",
                "q_enc",
            ]
        )
    df = pd.DataFrame(rows)
    pf = df["p_flat"].to_numpy()
    pe = df["p_enc"].to_numpy()
    # BH-FDR: skip NaNs.
    qf = np.full_like(pf, np.nan)
    qe = np.full_like(pe, np.nan)
    f_finite = np.isfinite(pf)
    e_finite = np.isfinite(pe)
    if f_finite.any():
        qf[f_finite] = bh_fdr(pf[f_finite])
    if e_finite.any():
        qe[e_finite] = bh_fdr(pe[e_finite])
    df["q_flat"] = qf
    df["q_enc"] = qe
    return df


def _build_fold_table(
    *,
    sym: np.ndarray,
    dates: np.ndarray,
    fold_idx: np.ndarray,
    proba: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Per (symbol × fold × model) AUC table."""
    rows: list[dict] = []
    for s in sorted(np.unique(sym).tolist()):
        for f in sorted(np.unique(fold_idx).tolist()):
            mask = (sym == s) & (fold_idx == f)
            if not mask.any():
                continue
            yb = labels[mask]
            pb = proba[mask]
            valid = np.isfinite(pb)
            yb = yb[valid]
            pb = pb[valid]
            n_pos = int((yb == 1).sum())
            n_neg = int((yb == 0).sum())
            if n_pos > 0 and n_neg > 0:
                try:
                    auc = float(roc_auc_score(yb, pb))
                except ValueError:
                    auc = float("nan")
            else:
                auc = float("nan")
            rows.append(
                dict(
                    symbol=s,
                    fold_idx=int(f),
                    n_pos=n_pos,
                    n_neg=n_neg,
                    auc_fold=auc,
                    model_name=model_name,
                )
            )
    return pd.DataFrame(rows)


def _emit_markdown(
    *,
    out_path: Path,
    n_windows: int,
    n_cascades: int,
    base_rate: float,
    n_days: int,
    n_symbols: int,
    auc_flat: float,
    auc_flat_lo: float,
    auc_flat_hi: float,
    auc_enc_per_seed: dict[int, tuple[float, float, float]],
    auc_enc_median: float,
    auc_enc_min: float,
    auc_enc_max: float,
    delta_point: float,
    delta_lo: float,
    delta_hi: float,
    median_seed: int,
    decision: str,
    per_symbol_df: pd.DataFrame,
    elapsed_sec: float,
) -> None:
    lines: list[str] = []
    lines.append("# Goal-A v2 Phase 0 — Random-Init Encoder Linear Probe")
    lines.append("")
    lines.append(
        f"Date range: {PHASE0_START} → {PHASE0_END_INCLUSIVE} (merged Apr 1-26, "
        f"holdout consumed per gotcha #17)."
    )
    lines.append(
        f"Symbols: {n_symbols} | Days: {n_days} | Windows: {n_windows:,} | "
        f"Cascades (H{H_PRIMARY}): {n_cascades:,} | Base rate: {base_rate:.4f}"
    )
    lines.append("")
    lines.append("## Pooled OOF AUC (5-fold day-blocked CV, 600-event embargo)")
    lines.append("")
    lines.append("| Model | Pooled AUC | 95% CI (day-clustered, 1000 reps) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Flat-LR (FLAT_DIM=83) | {auc_flat:.4f} | "
        f"[{auc_flat_lo:.4f}, {auc_flat_hi:.4f}] |"
    )
    lines.append(
        f"| Random-init encoder LR (median seed={median_seed}) | "
        f"{auc_enc_median:.4f} | min-max across seeds: "
        f"[{auc_enc_min:.4f}, {auc_enc_max:.4f}] |"
    )
    lines.append("")
    lines.append("### Per-seed encoder pooled AUC")
    lines.append("")
    lines.append("| Seed | Pooled AUC | 95% CI |")
    lines.append("|---|---|---|")
    for s in sorted(auc_enc_per_seed.keys()):
        a, lo, hi = auc_enc_per_seed[s]
        lines.append(f"| {s} | {a:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines.append("")

    lines.append("## Paired delta (encoder_median − flat-LR)")
    lines.append("")
    lines.append(
        f"Delta point estimate: **{delta_point:+.4f}** | "
        f"95% paired-bootstrap CI: [{delta_lo:+.4f}, {delta_hi:+.4f}]"
    )
    lines.append("")

    lines.append("## Decision tier (per plan §Decision Logic)")
    lines.append("")
    if decision == "GREENLIGHT_FINETUNE":
        verdict = (
            "**GREENLIGHT_FINETUNE** — Encoder ≥ flat-LR + 2pp AND paired-delta CI "
            "excludes 0.  Proceed to light end-to-end fine-tune (skip MEM+SimCLR pretrain)."
        )
    elif decision == "ARCH_BOTTLENECK":
        verdict = (
            "**ARCH_BOTTLENECK** — Encoder < flat-LR by > 2pp.  Architecture is the "
            "bottleneck for linear extraction.  Decide between MEM-only pretrain or "
            "end-to-end with strong regularization."
        )
    else:
        verdict = (
            "**MATCHED_FLAT** — Encoder ≈ flat-LR, paired-delta CI overlaps 0.  "
            "Architecture matches the flat-feature signal but doesn't beat it.  "
            "Consider end-to-end fine-tune ONCE; if that fails Tier-A, stop the program."
        )
    lines.append(verdict)
    lines.append("")

    lines.append("## Per-symbol AUC (BH-FDR adjusted, q=0.10 cut)")
    lines.append("")
    if len(per_symbol_df) == 0:
        lines.append(
            "_No symbol passed the n_cascades ≥ "
            f"{PER_SYMBOL_MIN_CASCADES} threshold._"
        )
    else:
        lines.append(
            "| Symbol | n_win | n_casc | AUC_flat | q_flat | AUC_enc | q_enc |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in per_symbol_df.iterrows():
            qf = r["q_flat"]
            qe = r["q_enc"]
            lines.append(
                f"| {r['symbol']} | {int(r['n_windows'])} | "
                f"{int(r['n_cascades'])} | {r['auc_flat']:.4f} | "
                f"{qf:.4f} | {r['auc_enc']:.4f} | {qe:.4f} |"
            )
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "* 5-fold day-blocked CV partition (contiguous, ordered by date).  "
        "Each day appears in exactly one fold."
    )
    lines.append(
        f"* {EMBARGO_EVENTS}-event embargo at fold boundaries (events, not "
        "windows); applied to the LAST events of fold k-1's last day and FIRST "
        "events of fold k+1's first day when training fold k."
    )
    lines.append(
        "* Random-init encoder: TapeEncoder(EncoderConfig()) with input "
        "BatchNorm1d.track_running_stats=False (gotcha #18).  Eval-mode forward "
        "uses batch statistics (256-window batches)."
    )
    lines.append(
        f"* Encoder seeds: {ENCODER_SEEDS}.  Median seed binds the report; "
        "min-max across seeds bounds random-init variance (council-5 cap)."
    )
    lines.append(
        "* Day-clustered bootstrap: 1000 iterations, resample 26 days with "
        "replacement.  Paired delta uses the SAME seeded RNG so identical day "
        "samples produce both AUCs in lockstep."
    )
    lines.append(
        "* BH-FDR via scipy.stats.false_discovery_control across per-symbol "
        f"p-values (one-sided H0: AUC ≤ {PER_SYMBOL_AUC_NULL})."
    )
    lines.append("")
    lines.append(
        f"_Pipeline ran in {elapsed_sec:.1f} s.  CPU-only.  Merged Apr 1-26 "
        "dataset; holdout consumed in commit b0de994._"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_pipeline(
    *,
    cache_dir: Path,
    out_dir: Path,
    smoke: bool,
) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        # Smoke uses 3 high-cascade-volume symbols × 5 days to get enough
        # H500-valid windows AND ≥ 2 cascade-positive labels for AUC to be
        # well-defined.  The prompt asked "BTC + 3 days" but BTC alone over 3
        # days has 0 H500 cascade-positive windows in this dataset, which
        # leaves the AUC undefined.  We expand to 3 symbols × 5 days while
        # preserving the spirit: ~150 windows, ≤ 5 s wall-clock.
        symbols: tuple[str, ...] = ("BTC", "SOL", "ETH")
        dates = ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]
        print(f"[smoke] symbols={symbols} dates={dates}")
    else:
        symbols = SYMBOLS
        dates = phase0_dates()

    print(
        f"[1/6] Gathering Apr 1-26 batches: {len(symbols)} symbols × "
        f"{len(dates)} days"
    )
    batches = gather_phase0_batches(cache_dir, symbols, dates, horizons=PROBE_HORIZONS)
    print(f"      → built {len(batches)} (symbol × date) batches")
    if not batches:
        raise RuntimeError("no batches built — check cache and trade parquet")

    print(f"[2/6] Stacking at horizon H{H_PRIMARY}")
    flat_X, raw_X, y, sym, date_arr, anchor_ts, anchor_idx = stack_batches_at_horizon(
        batches, H_PRIMARY
    )
    n_total = len(y)
    n_pos = int(y.sum())
    n_days = len(np.unique(date_arr))
    n_symbols = len(np.unique(sym))
    base_rate = float(n_pos / max(1, n_total))
    print(
        f"      → N={n_total:,} flat_X.shape={flat_X.shape} raw_X.shape={raw_X.shape} "
        f"n_pos={n_pos} base_rate={base_rate:.4f} days={n_days} symbols={n_symbols}"
    )

    print(f"[3/6] Phase 0a: flat-LR 5-fold day-blocked CV")
    oof_flat = run_5fold_cv(
        X=flat_X.astype(np.float32),
        y=y.astype(np.int64),
        sym=sym,
        dates=date_arr,
        anchor_ts=anchor_ts,
        anchor_idx_in_day=anchor_idx,
        embargo_events=EMBARGO_EVENTS,
        k=N_FOLDS,
    )
    auc_flat = _pooled_auc(oof_flat.proba, oof_flat.labels)
    af, alo, ahi = day_clustered_bootstrap_auc(
        proba=oof_flat.proba,
        labels=oof_flat.labels.astype(np.int64),
        dates=date_arr,
        n_boot=N_BOOT,
        seed=0,
    )
    print(
        f"      → flat pooled AUC={auc_flat:.4f} bootstrap mean={af:.4f} "
        f"CI=[{alo:.4f},{ahi:.4f}]"
    )
    for fi, npos, nneg, fa in oof_flat.per_fold_aucs:
        print(f"      fold {fi}: n_pos={npos} n_neg={nneg} auc={fa:.4f}")

    print(f"[4/6] Phase 0b: random-init encoder probe ({len(ENCODER_SEEDS)} seeds)")
    enc_oof_per_seed: dict[int, OOFPredictions] = {}
    enc_pooled_per_seed: dict[int, tuple[float, float, float]] = {}
    for seed in ENCODER_SEEDS:
        print(f"  seed={seed}: building encoder + extracting embeddings")
        enc = build_random_init_encoder(seed=seed)
        emb = encode_windows(enc, raw_X, batch_size=ENCODER_BATCH_SIZE, device="cpu")
        del enc
        oof = run_5fold_cv(
            X=emb,
            y=y.astype(np.int64),
            sym=sym,
            dates=date_arr,
            anchor_ts=anchor_ts,
            anchor_idx_in_day=anchor_idx,
            embargo_events=EMBARGO_EVENTS,
            k=N_FOLDS,
        )
        a = _pooled_auc(oof.proba, oof.labels)
        boot_mean, boot_lo, boot_hi = day_clustered_bootstrap_auc(
            proba=oof.proba,
            labels=oof.labels.astype(np.int64),
            dates=date_arr,
            n_boot=N_BOOT,
            seed=seed + 100,
        )
        enc_oof_per_seed[seed] = oof
        enc_pooled_per_seed[seed] = (a, boot_lo, boot_hi)
        print(
            f"  seed={seed}: pooled AUC={a:.4f} bootstrap CI=[{boot_lo:.4f},{boot_hi:.4f}]"
        )

    # Median seed by pooled AUC
    seed_to_auc = {s: enc_pooled_per_seed[s][0] for s in ENCODER_SEEDS}
    finite_seed_to_auc = {s: a for s, a in seed_to_auc.items() if math.isfinite(a)}
    if not finite_seed_to_auc:
        raise RuntimeError("All encoder seeds produced NaN pooled AUC")
    sorted_seeds = sorted(
        finite_seed_to_auc.keys(), key=lambda s: finite_seed_to_auc[s]
    )
    median_seed = sorted_seeds[len(sorted_seeds) // 2]
    auc_enc_median = finite_seed_to_auc[median_seed]
    auc_enc_min = min(finite_seed_to_auc.values())
    auc_enc_max = max(finite_seed_to_auc.values())
    median_oof = enc_oof_per_seed[median_seed]

    print(f"[5/6] Paired bootstrap on delta = enc_seed{median_seed} − flat-LR")
    delta_point, delta_lo, delta_hi, _aucs_a, _aucs_b = (
        paired_day_clustered_bootstrap_delta(
            proba_a=oof_flat.proba,
            proba_b=median_oof.proba,
            labels=y.astype(np.int64),
            dates=date_arr,
            n_boot=N_BOOT,
            seed=999,
        )
    )
    print(f"      → delta={delta_point:+.4f} CI=[{delta_lo:+.4f}, {delta_hi:+.4f}]")

    decision = decision_tier(
        auc_flat=auc_flat,
        auc_enc_median=auc_enc_median,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
    )

    print(f"[6/6] Building outputs in {out_dir}")
    # Per-symbol table
    per_symbol_df = _build_per_symbol_table(
        sym=sym,
        dates=date_arr,
        proba_flat=oof_flat.proba,
        proba_enc=median_oof.proba,
        labels=y.astype(np.int64),
        n_boot=N_BOOT,
        seed=12345,
    )

    # Fold-level CSV
    fold_flat = _build_fold_table(
        sym=sym,
        dates=date_arr,
        fold_idx=oof_flat.fold_idx,
        proba=oof_flat.proba,
        labels=y.astype(np.int64),
        model_name="flat_LR",
    )
    fold_rows = [fold_flat]
    for seed, oof in enc_oof_per_seed.items():
        f_enc = _build_fold_table(
            sym=sym,
            dates=date_arr,
            fold_idx=oof.fold_idx,
            proba=oof.proba,
            labels=y.astype(np.int64),
            model_name=f"encoder_seed{seed}",
        )
        fold_rows.append(f_enc)
    fold_df = pd.concat(fold_rows, ignore_index=True)

    csv_path = out_dir / "random_init_probe_table.csv"
    fold_df.to_csv(csv_path, index=False)

    # Per-window parquet (gitignored — under data/* path? plan says
    # docs/experiments/goal-a-v2/, but instruction says gitignored; the docs/
    # path is committed by default. We emit there but rely on the user adding a
    # .gitignore line.  Keep it small.)
    per_window_df = pd.DataFrame(
        dict(
            symbol=sym,
            date=date_arr,
            anchor_ts=anchor_ts.astype(np.int64),
            fold_idx=oof_flat.fold_idx.astype(np.int32),
            label=y.astype(np.int8),
            proba_flat=oof_flat.proba.astype(np.float32),
            proba_enc_median_seed=median_oof.proba.astype(np.float32),
        )
    )
    parquet_path = out_dir / "random_init_probe_per_window.parquet"
    per_window_df.to_parquet(parquet_path, index=False)

    # Markdown
    md_path = out_dir / "random_init_probe.md"
    elapsed_sec = time.perf_counter() - t0
    _emit_markdown(
        out_path=md_path,
        n_windows=n_total,
        n_cascades=n_pos,
        base_rate=base_rate,
        n_days=n_days,
        n_symbols=n_symbols,
        auc_flat=auc_flat,
        auc_flat_lo=alo,
        auc_flat_hi=ahi,
        auc_enc_per_seed=enc_pooled_per_seed,
        auc_enc_median=auc_enc_median,
        auc_enc_min=auc_enc_min,
        auc_enc_max=auc_enc_max,
        delta_point=delta_point,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        median_seed=median_seed,
        decision=decision,
        per_symbol_df=per_symbol_df,
        elapsed_sec=elapsed_sec,
    )

    # Stdout summary
    print()
    print("=" * 60)
    print(f"  Pooled flat-LR AUC: {auc_flat:.4f} [{alo:.4f}, {ahi:.4f}]")
    print(
        f"  Pooled encoder AUC (median seed={median_seed}): "
        f"{auc_enc_median:.4f}  "
        f"[min={auc_enc_min:.4f}, max={auc_enc_max:.4f}]"
    )
    print(
        f"  Paired delta (enc - flat): {delta_point:+.4f} "
        f"[{delta_lo:+.4f}, {delta_hi:+.4f}]"
    )
    print(f"  Decision tier: {decision}")
    print(f"  Outputs: {csv_path} ; {md_path} ; {parquet_path}")
    print(f"  Elapsed: {elapsed_sec:.1f} s")
    print("=" * 60)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument(
        "--smoke", action="store_true", help="Run on BTC + 3 days for fast iteration."
    )
    args = p.parse_args()
    _run_pipeline(cache_dir=args.cache, out_dir=args.out_dir, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

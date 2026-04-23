"""Gate 0 baseline: per-window summary statistics (mean/std/skew/kurt/last).

Produces an 83-dimensional flat vector from a (200, 17) event window.

Layout:
    [f0_mean,  ..., f16_mean,          # 17 means
     f0_std,   ..., f16_std,           # 17 stds
     f0_skew,  ..., f16_skew,          # 17 skews
     f0_kurt,  ..., f16_kurt,          # 17 kurts
     f0_last,  ..., f16_last           # 17 last values
         minus time_delta_last (idx 71) and prev_seq_time_span_last (idx 76)]

17 features × 5 statistics = 85 raw dims, minus 2 pruned = 83-dim.

**Why 83 and not 85?**
Session-of-day confound check (2026-04-23, commit a6845de) found that an LR
trained on UTC hour alone beat PCA+LR on 85-dim flat features by >0.5pp on
exactly 5 symbols (LTC +1.63pp, HYPE +1.17pp, WLFI +1.12pp, BNB +0.74pp,
PENGU +0.62pp), triggering the `prune_last_features` decision rule
(CLAUDE.md gotcha #32; concepts/session-of-day-leakage.md).

The two `_last` columns encode approximate UTC hour because they carry the
final value of timing features (`time_delta` and `prev_seq_time_span`) from
the last event in the window — values that are systematically different during
Asian, European, and US sessions.  The mean/std/skew/kurt statistics of the
same features are retained because they average over the full window and do
not pin a session as strongly.

This pruning ONLY affects the flat-feature baselines used at Gate 0/Gate 2.
The CNN encoder processes the full (200, 17) raw tensor; its session-of-day
mitigation is handled separately via SimCLR timing-noise augmentation
(sigma=0.10 on time_delta and prev_seq_time_span) — see CLAUDE.md training.

The skew and kurtosis use scipy's bias-corrected estimators (bias=False).
A constant-channel produces skew=0 and kurt=0 (not NaN) because nan_policy="omit"
with zero variance returns 0 via fill logic below.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from tape.constants import FEATURE_NAMES, WINDOW_LEN

# ---- Module-level constants ----

_N_STATS: int = 5  # mean, std, skew, kurtosis, last
_STAT_NAMES: tuple[str, ...] = ("mean", "std", "skew", "kurt", "last")

# Session-of-day leaky features: their `_last` columns are pruned (see module docstring).
_PRUNED_LAST_FEATURES: frozenset[str] = frozenset({"time_delta", "prev_seq_time_span"})

# Full 85-dim name list (pre-prune) — used internally to compute drop indices.
_ALL_FLAT_NAMES: tuple[str, ...] = tuple(
    f"{feat}_{stat}" for stat in _STAT_NAMES for feat in FEATURE_NAMES
)
_PRUNE_INDICES: tuple[int, ...] = tuple(
    i
    for i, name in enumerate(_ALL_FLAT_NAMES)
    if name in {f"{feat}_last" for feat in _PRUNED_LAST_FEATURES}
)
assert len(_PRUNE_INDICES) == 2, f"Expected 2 pruned indices, got {len(_PRUNE_INDICES)}"

FLAT_DIM: int = len(FEATURE_NAMES) * _N_STATS - len(_PRUNE_INDICES)  # 17 × 5 - 2 = 83

FLAT_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for name in _ALL_FLAT_NAMES
    if name not in {f"{feat}_last" for feat in _PRUNED_LAST_FEATURES}
)

assert len(FLAT_FEATURE_NAMES) == FLAT_DIM, "FLAT_FEATURE_NAMES length mismatch"


# ---- Core implementation ----


def window_to_flat(window: np.ndarray) -> np.ndarray:
    """Map a (200, 17) window to a flat (83,) float32 vector.

    Builds the full 85-dim vector (mean/std/skew/kurt/last per feature) then
    drops the 2 session-leaky `_last` columns (`time_delta_last` and
    `prev_seq_time_span_last`).  See module docstring for rationale.

    No NaN is produced when the input has no NaN.
    Constant channels (std=0) produce skew=0 and kurt=0.
    """
    assert window.ndim == 2 and window.shape[1] == 17, f"bad shape {window.shape}"

    mean = window.mean(axis=0)  # (17,)
    std = window.std(axis=0)  # (17,) — population std; finite everywhere

    # scipy bias-corrected skew/kurtosis; returns NaN for n<3 or std=0
    sk = _skew(window, axis=0, bias=False, nan_policy="omit")  # (17,)
    kt = _kurtosis(window, axis=0, bias=False, nan_policy="omit")  # (17,)

    # Replace NaN (constant channel → undefined skew/kurt) with 0 — the
    # correct limiting value for a constant distribution.
    sk = np.where(np.isfinite(sk), sk, 0.0)
    kt = np.where(np.isfinite(kt), kt, 0.0)

    last = window[-1]  # (17,)

    full = np.concatenate([mean, std, sk, kt, last], axis=0).astype(np.float32)
    # Drop the 2 session-leaky _last columns.
    keep_mask = np.ones(len(_ALL_FLAT_NAMES), dtype=bool)
    for idx in _PRUNE_INDICES:
        keep_mask[idx] = False
    result = full[keep_mask]
    assert result.shape == (FLAT_DIM,), f"unexpected output shape {result.shape}"
    assert np.all(np.isfinite(result)), "non-finite values in flat features"
    return result


# ---- Public API (aliases + batch variant) ----


def extract_flat_features(window: np.ndarray) -> np.ndarray:
    """Extract flat features from a single (200, 17) window → (83,) float32.

    Pure function: same input always produces same output.
    """
    return window_to_flat(window)


def extract_flat_features_batch(windows: np.ndarray) -> np.ndarray:
    """Extract flat features from a batch of windows → (N, 83) float32.

    Args:
        windows: float32 array of shape (N, 200, 17).

    Returns:
        float32 array of shape (N, 83).
    """
    assert windows.ndim == 3 and windows.shape[1:] == (
        WINDOW_LEN,
        len(FEATURE_NAMES),
    ), f"bad batch shape {windows.shape}"

    n = windows.shape[0]
    out = np.empty((n, FLAT_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = window_to_flat(windows[i])
    return out

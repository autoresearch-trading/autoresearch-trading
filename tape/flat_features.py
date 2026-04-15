"""Gate 0 baseline: per-window summary statistics (mean/std/skew/kurt/last).

Produces an 85-dimensional flat vector from a (200, 17) event window:
    [f0_mean, ..., f16_mean,
     f0_std,  ..., f16_std,
     f0_skew, ..., f16_skew,
     f0_kurt, ..., f16_kurt,
     f0_last, ..., f16_last]

17 features × 5 statistics = 85-dim.

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

FLAT_DIM: int = len(FEATURE_NAMES) * _N_STATS  # 17 × 5 = 85

FLAT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{feat}_{stat}" for stat in _STAT_NAMES for feat in FEATURE_NAMES
)

assert len(FLAT_FEATURE_NAMES) == FLAT_DIM, "FLAT_FEATURE_NAMES length mismatch"


# ---- Core implementation ----


def window_to_flat(window: np.ndarray) -> np.ndarray:
    """Map a (200, 17) window to a flat (85,) float32 vector.

    This is the plan's reference implementation, preserved verbatim as the
    primary entry point.  No NaN is produced when the input has no NaN.
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

    result = np.concatenate([mean, std, sk, kt, last], axis=0).astype(np.float32)
    assert result.shape == (FLAT_DIM,), f"unexpected output shape {result.shape}"
    assert np.all(np.isfinite(result)), "non-finite values in flat features"
    return result


# ---- Public API (aliases + batch variant) ----


def extract_flat_features(window: np.ndarray) -> np.ndarray:
    """Extract flat features from a single (200, 17) window → (85,) float32.

    Pure function: same input always produces same output.
    """
    return window_to_flat(window)


def extract_flat_features_batch(windows: np.ndarray) -> np.ndarray:
    """Extract flat features from a batch of windows → (N, 85) float32.

    Args:
        windows: float32 array of shape (N, 200, 17).

    Returns:
        float32 array of shape (N, 85).
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

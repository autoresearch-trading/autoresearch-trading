# tape/labels.py
"""Direction labels (downstream probing) + Wyckoff self-labels (diagnostic).

Spec §Label Semantics: direction label at event i = sign(vwap[i+h] - vwap[i])
where `vwap` is EVENT VWAP (not last-fill price). Last `h` events have no
label and are masked (sentinel = 0, mask = False).

Wyckoff labels (all binary 0/1 per event, no lookahead):
- stress          : log_spread > p90(rolling) AND |depth_ratio| > p90(rolling)
- informed_flow   : kyle_lambda > p75(rolling) AND |cum_ofi_5| > p50(rolling)
                    AND sign-consistent over 3 consecutive snapshots
- climax          : z_qty > 2 AND z_ret > 2 (strict)
- spring          : min(log_return[i-50:i+1]) < -SPRING_SIGMA_MULT * rolling_σ
                    AND effort_vs_result[i] > 1.0
                    AND is_open[i] > 0.5
                    AND mean(log_return[i-9:i+1]) > 0  (recent recovery)
                    SPRING_SIGMA_MULT = 3.0 (recalibrated — prereq #4)
- absorption      : effort_vs_result > p90(rolling) AND |log_return| < p50(rolling)

All rolling windows use ROLLING_WINDOW (1000 events), causal.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from tape.constants import (
    DEPTH_PCTL,
    DIRECTION_HORIZONS,
    INFORMED_KYLE_PCTL,
    INFORMED_OFI_PCTL,
    ROLLING_WINDOW,
    SPRING_LOOKBACK,
    SPRING_PRIOR_LEN,
    SPRING_SIGMA_MULT,
    STRESS_PCTL,
)

# ---------------------------------------------------------------------------
# TypedDict for direction labels
# ---------------------------------------------------------------------------


class DirectionLabels(TypedDict):
    h10: np.ndarray
    h50: np.ndarray
    h100: np.ndarray
    h500: np.ndarray
    mask_h10: np.ndarray
    mask_h50: np.ndarray
    mask_h100: np.ndarray
    mask_h500: np.ndarray


# ---------------------------------------------------------------------------
# Direction labels
# ---------------------------------------------------------------------------


def compute_direction_labels(vwap: np.ndarray) -> DirectionLabels:
    """Compute binary direction labels at each horizon in DIRECTION_HORIZONS.

    Args:
        vwap: 1-D float array of event VWAPs (length n).

    Returns:
        DirectionLabels dict.  For each horizon h:
          - ``h{h}``: int8 array of length n.  1 = up, 0 = down-or-flat.
                      Last h positions are the zero sentinel (not valid).
          - ``mask_h{h}``: bool array of length n.  True = valid label,
                           False = no forward data (tail of series / day).
    """
    vwap = np.asarray(vwap, dtype=float)
    n = len(vwap)
    out: DirectionLabels = {}  # type: ignore[typeddict-item]

    for h in DIRECTION_HORIZONS:
        labels = np.zeros(n, dtype=np.int8)
        mask = np.zeros(n, dtype=bool)
        if n > h:
            fwd = vwap[h:]
            cur = vwap[:-h]
            labels[: n - h] = (fwd > cur).astype(np.int8)  # 1=up, 0=down-or-flat
            mask[: n - h] = True
        out[f"h{h}"] = labels  # type: ignore[literal-required]
        out[f"mask_h{h}"] = mask  # type: ignore[literal-required]

    return out


# ---------------------------------------------------------------------------
# Wyckoff labels
# ---------------------------------------------------------------------------


def compute_wyckoff_labels(
    *,
    log_return: np.ndarray,
    effort_vs_result: np.ndarray,
    is_open: np.ndarray,
    climax_score: np.ndarray,
    z_qty: np.ndarray,
    z_ret: np.ndarray,
    log_spread: np.ndarray,
    depth_ratio: np.ndarray,
    kyle_lambda: np.ndarray,
    cum_ofi_5: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the five Wyckoff self-labels for a sequence of order events.

    All inputs must be 1-D float arrays of the same length n.
    All outputs are int8 arrays in {0, 1}, length n.

    Keyword-only arguments enforce explicit naming at call sites.
    """
    log_return = np.asarray(log_return, dtype=float)
    effort_vs_result = np.asarray(effort_vs_result, dtype=float)
    is_open = np.asarray(is_open, dtype=float)
    z_qty = np.asarray(z_qty, dtype=float)
    z_ret = np.asarray(z_ret, dtype=float)
    log_spread = np.asarray(log_spread, dtype=float)
    depth_ratio = np.asarray(depth_ratio, dtype=float)
    kyle_lambda = np.asarray(kyle_lambda, dtype=float)
    cum_ofi_5 = np.asarray(cum_ofi_5, dtype=float)

    n = len(log_return)
    out: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # stress — joint p90 crossing of log_spread and |depth_ratio|
    # ------------------------------------------------------------------
    ls_pct = _rolling_percentile(log_spread, ROLLING_WINDOW, STRESS_PCTL)
    dr_pct = _rolling_percentile(np.abs(depth_ratio), ROLLING_WINDOW, DEPTH_PCTL)
    out["stress"] = ((log_spread > ls_pct) & (np.abs(depth_ratio) > dr_pct)).astype(
        np.int8
    )

    # ------------------------------------------------------------------
    # informed_flow — elevated kyle_lambda + persistent directional OFI
    #   3-snapshot sign consistency means cum_ofi_5[i], [i-1], [i-2]
    #   all share the same non-zero sign.
    # ------------------------------------------------------------------
    kl_pct = _rolling_percentile(kyle_lambda, ROLLING_WINDOW, INFORMED_KYLE_PCTL)
    of_pct = _rolling_percentile(np.abs(cum_ofi_5), ROLLING_WINDOW, INFORMED_OFI_PCTL)
    fires = (kyle_lambda > kl_pct) & (np.abs(cum_ofi_5) > of_pct)
    signs = np.sign(cum_ofi_5)
    consistent = np.zeros(n, dtype=bool)
    consistent[2:] = (
        (signs[2:] == signs[1:-1]) & (signs[2:] == signs[:-2]) & (signs[2:] != 0)
    )
    out["informed_flow"] = (fires & consistent).astype(np.int8)

    # ------------------------------------------------------------------
    # climax — simultaneous extreme qty and return z-scores
    # ------------------------------------------------------------------
    out["climax"] = ((z_qty > 2.0) & (z_ret > 2.0)).astype(np.int8)

    # ------------------------------------------------------------------
    # spring — downside probe + absorption at low + recovery
    #   cond1: min(log_return[i-49:i+1]) < -SPRING_SIGMA_MULT * rolling_σ
    #   cond2: effort_vs_result[i] > 1.0
    #   cond3: is_open[i] > 0.5
    #   cond4: mean(log_return[i-9:i+1]) > 0  (10-event recent window)
    # ------------------------------------------------------------------
    _lr_series = pd.Series(log_return)
    roll_std: np.ndarray = (
        pd.Series(_lr_series.rolling(ROLLING_WINDOW, min_periods=10).std())
        .fillna(1e-10)
        .to_numpy(dtype=float)
    )

    min_ret: np.ndarray = pd.Series(
        _lr_series.rolling(SPRING_LOOKBACK, min_periods=SPRING_LOOKBACK).min()
    ).to_numpy(dtype=float)

    # cond4: mean of the MOST RECENT SPRING_PRIOR_LEN events (causal, current event included)
    prior_mean: np.ndarray = pd.Series(
        _lr_series.rolling(SPRING_PRIOR_LEN, min_periods=SPRING_PRIOR_LEN).mean()
    ).to_numpy(dtype=float)

    cond1 = min_ret < -SPRING_SIGMA_MULT * roll_std
    cond2 = effort_vs_result > 1.0
    cond3 = is_open > 0.5
    cond4 = prior_mean > 0.0
    spring = cond1 & cond2 & cond3 & cond4
    out["spring"] = np.nan_to_num(spring.astype(float), nan=0.0).astype(np.int8)

    # ------------------------------------------------------------------
    # absorption — high effort, low result (price impact below median)
    # ------------------------------------------------------------------
    evr_pct = _rolling_percentile(effort_vs_result, ROLLING_WINDOW, 0.90)
    ret_pct = _rolling_percentile(np.abs(log_return), ROLLING_WINDOW, 0.50)
    out["absorption"] = (
        (effort_vs_result > evr_pct) & (np.abs(log_return) < ret_pct)
    ).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rolling_percentile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Causal rolling q-th quantile.  NaN positions (insufficient history)
    are filled with +inf so that < comparisons always return False there —
    preventing spurious label fires before sufficient data is available.
    """
    s = pd.Series(arr)
    return (
        s.rolling(window, min_periods=max(10, window // 10))
        .quantile(q)
        .fillna(np.inf)
        .to_numpy(dtype=float)
    )

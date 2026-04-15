# tape/features_trade.py
"""9 trade-side features per order event.

Every feature is causal (no lookahead). Normalisation uses rolling statistics
over the last `ROLLING_WINDOW` events (gotcha #4).

Spec: docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
      §Input Representation (9 trade features numbered 1-9).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tape.constants import ROLLING_WINDOW, WINDOW_LEN

_EPS_RETURN: float = 1e-6  # gotcha #5 — was 1e-4, too coarse for BTC
_EPS_SPREAD_MID: float = 1e-8  # gotcha #10

# Sigma floor for rolling-std normalisers (Fix 2 — cold-start spike guard).
# We use a fixed value of 0.1 because:
#   - log(qty) for gamma(2,1)-distributed qty has std ~0.7 in practice.
#   - A floor of 0.1 keeps |z| < 10 in the worst realistic early-event case
#     while still letting genuine climax events (|z| > 2) score high once the
#     rolling window is warm (≥ ~10 events), without producing the ~qty/1e-10
#     blow-up that min_periods=10 + fillna(1e-10) causes for events 1-9.
_SIGMA_FLOOR: float = 0.1


def compute_trade_features(
    events: pd.DataFrame, *, spread: np.ndarray, mid: np.ndarray
) -> pd.DataFrame:
    """Compute the 9 trade features.

    Parameters
    ----------
    events : DataFrame from `tape.events.group_to_events`
    spread : np.ndarray of aligned OB spread per event (same length as events)
    mid    : np.ndarray of aligned OB mid price per event

    Returns
    -------
    DataFrame with the 9 feature columns in FEATURE order.
    """
    n = len(events)
    ts_ms = events["ts_ms"].to_numpy(dtype=np.int64)
    vwap = events["vwap"].to_numpy(dtype=float)
    total_qty = events["total_qty"].to_numpy(dtype=float)
    is_open_frac = events["is_open_frac"].to_numpy(dtype=float)
    n_fills = events["n_fills"].to_numpy(dtype=float)
    book_walk_abs = events["book_walk_abs"].to_numpy(dtype=float)

    # 1. log_return
    log_return = np.zeros(n, dtype=float)
    log_return[1:] = np.log(vwap[1:] / np.maximum(vwap[:-1], _EPS_RETURN))

    # 2. log_total_qty — normalised by rolling median (gotcha #4, #5)
    _med: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=1).median()
    roll_med_qty: np.ndarray = np.maximum(_med.to_numpy(dtype=float), _EPS_RETURN)
    log_total_qty = np.log(total_qty / roll_med_qty)

    # 3. is_open — passthrough in [0, 1]
    is_open = is_open_frac

    # 4. time_delta = log(Δt_ms + 1)
    dt = np.zeros(n, dtype=float)
    dt[1:] = ts_ms[1:] - ts_ms[:-1]
    time_delta = np.log(dt + 1.0)

    # 5. num_fills = log(count)
    num_fills = np.log(n_fills)

    # 6. book_walk = |last - first| / max(spread, eps * mid), spread/mid aligned
    eps_spread: np.ndarray = np.maximum(spread, _EPS_SPREAD_MID * mid)
    book_walk = book_walk_abs / eps_spread

    # 7. effort_vs_result — both qty and |return| terms median-normalised (gotcha #5)
    #
    #    Old formula: log_total_qty - log(|return| + eps)
    #    Problem: typical BTC tick returns ~1e-3 → log(1e-3 + 1e-6) ≈ -6.9 →
    #    effort_vs_result ≈ 6.9 → clips to +5 for ~98% of events.
    #
    #    Fix: normalise |return| by its rolling 1000-event median (same as qty).
    #    Both terms are now on a relative scale → feature is discriminative.
    #
    #    Event 0: log_return[0] = 0 (undefined) → zero out (option (a) from spec).
    abs_ret = np.abs(log_return)
    _mret: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=1).median()
    roll_med_abs_ret: np.ndarray = np.maximum(_mret.to_numpy(dtype=float), 1e-10)
    normalized_abs_ret = abs_ret / roll_med_abs_ret
    effort_vs_result = np.clip(
        log_total_qty - np.log(normalized_abs_ret + _EPS_RETURN),
        -5.0,
        5.0,
    )
    effort_vs_result[0] = 0.0

    # 8. climax_score — rolling-1000 σ on log_total_qty_raw, clipped [0, 5]  (gotcha #6)
    #
    #    Fix: z-score log_total_qty_raw (pre-median-normalisation log qty) instead
    #    of raw total_qty. log(qty) is approximately symmetric (log-normal fits
    #    gamma well) → well-behaved z-scores; raw qty is heavy-tailed gamma →
    #    rolling σ is dominated by outliers → unreliable z-scores.
    #
    #    Cold-start: min_periods=1 + sigma floor (_SIGMA_FLOOR=0.1) prevents
    #    z ≈ qty/1e-10 blow-up for events 1-9.
    log_total_qty_raw = pd.Series(np.log(np.maximum(total_qty, _EPS_RETURN)))
    _smq: Any = log_total_qty_raw.rolling(ROLLING_WINDOW, min_periods=1).mean()
    roll_mean_log_qty: np.ndarray = _smq.fillna(0.0).to_numpy(dtype=float)
    _ssq: Any = log_total_qty_raw.rolling(ROLLING_WINDOW, min_periods=1).std()
    roll_std_log_qty: np.ndarray = np.maximum(
        _ssq.fillna(0.0).to_numpy(dtype=float), _SIGMA_FLOOR
    )
    _ssr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=1).std()
    roll_std_ret: np.ndarray = np.maximum(
        _ssr.fillna(0.0).to_numpy(dtype=float), _SIGMA_FLOOR
    )
    _smr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=1).mean()
    roll_mean_ret: np.ndarray = _smr.fillna(0.0).to_numpy(dtype=float)
    z_qty = (log_total_qty_raw.to_numpy() - roll_mean_log_qty) / roll_std_log_qty
    z_ret = (abs_ret - roll_mean_ret) / roll_std_ret
    climax_score = np.clip(np.minimum(z_qty, z_ret), 0.0, 5.0)

    # 9. prev_seq_time_span — log(ts_ms[i-1] - ts_ms[i-WINDOW_LEN] + 1) for
    #    each event i >= WINDOW_LEN (gotcha #8).  Zero for the first WINDOW_LEN
    #    events (no prior window exists).
    #
    #    Fix 1: sliding window (not block-averaged).  For event i the feature
    #    captures the time span of the 200 events immediately preceding it:
    #      span[i] = ts_ms[i-1] - ts_ms[i-WINDOW_LEN]
    #    which equals ts_ms[WINDOW_LEN-1..n-2] - ts_ms[0..n-WINDOW_LEN-1].
    #
    #    Causality: event i references ts_ms[i-WINDOW_LEN] through ts_ms[i-1]
    #    — no self-reference and no future data.
    prev_seq_time_span = np.zeros(n, dtype=np.float64)
    if n > WINDOW_LEN:
        # ts_ms[WINDOW_LEN-1 : n-1]  →  last timestamp of the prior window for each i
        # ts_ms[0 : n-WINDOW_LEN]    →  first timestamp of the prior window for each i
        span = ts_ms[WINDOW_LEN - 1 : n - 1] - ts_ms[: n - WINDOW_LEN]
        prev_seq_time_span[WINDOW_LEN:] = np.log(span.astype(np.float64) + 1.0)

    return pd.DataFrame(
        {
            "log_return": log_return,
            "log_total_qty": log_total_qty,
            "is_open": is_open,
            "time_delta": time_delta,
            "num_fills": num_fills,
            "book_walk": book_walk,
            "effort_vs_result": effort_vs_result,
            "climax_score": climax_score,
            "prev_seq_time_span": prev_seq_time_span,
        }
    )

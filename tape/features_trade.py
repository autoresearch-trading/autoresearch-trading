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

    # 7. effort_vs_result — clip(log_total_qty - log(|return|+eps), -5, 5)  (gotcha #5)
    abs_ret = np.abs(log_return)
    effort_vs_result = np.clip(log_total_qty - np.log(abs_ret + _EPS_RETURN), -5.0, 5.0)

    # 8. climax_score — rolling-1000 σ, clipped [0, 5]  (gotcha #6)
    _ssq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).std()
    roll_std_qty: np.ndarray = np.maximum(
        _ssq.fillna(1e-10).to_numpy(dtype=float), 1e-10
    )
    _ssr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).std()
    roll_std_ret: np.ndarray = np.maximum(
        _ssr.fillna(1e-10).to_numpy(dtype=float), 1e-10
    )
    _smq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_mean_qty: np.ndarray = _smq.fillna(0.0).to_numpy(dtype=float)
    _smr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_mean_ret: np.ndarray = _smr.fillna(0.0).to_numpy(dtype=float)
    z_qty = (total_qty - roll_mean_qty) / roll_std_qty
    z_ret = (abs_ret - roll_mean_ret) / roll_std_ret
    climax_score = np.clip(np.minimum(z_qty, z_ret), 0.0, 5.0)

    # 9. prev_seq_time_span — log(last_ts - first_ts + 1) of the PRIOR 200-event
    #    window (gotcha #8). Zero for the first 200 events (no prior window).
    #
    #    The "prior window" is the non-overlapping block that precedes the
    #    current block. Events [0, 200) have no prior block → 0. Events
    #    [200, 400) share the same prior block [0, 200) → constant value.
    #    Events [400, 600) share prior block [200, 400), etc.
    #    This is a step function over WINDOW_LEN-sized blocks, not a per-event
    #    sliding window (which would be a lookahead into the current window).
    prev_seq_time_span = np.zeros(n, dtype=float)
    if n > WINDOW_LEN:
        # Number of complete prior blocks we can reference
        n_blocks = n // WINDOW_LEN
        for block_idx in range(1, n_blocks + 1):
            # Prior block: [prev_start, prev_end)
            prev_start = (block_idx - 1) * WINDOW_LEN
            prev_end = block_idx * WINDOW_LEN  # exclusive
            # Current block: [prev_end, next_end)
            curr_start = prev_end
            curr_end = min((block_idx + 1) * WINDOW_LEN, n)
            if curr_start >= n:
                break
            span_val = float(ts_ms[prev_end - 1] - ts_ms[prev_start])
            log_span = float(np.log(span_val + 1.0))
            prev_seq_time_span[curr_start:curr_end] = log_span

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

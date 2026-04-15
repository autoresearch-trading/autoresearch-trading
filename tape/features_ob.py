# tape/features_ob.py
"""8 orderbook features per event, derived from 10-level snapshots.

Two stages:
  1. `compute_snapshot_features(ob_df)` — per-snapshot features + supports
     (mid, spread, kyle_lambda, cum_ofi_5). kyle_lambda and cum_ofi_5 are
     per-snapshot (gotcha #13, #15).
  2. `align_ob_features_to_events(snap_df, event_ts)` — for each event,
     forward-fill the latest prior snapshot's features. Events before the
     first snapshot get NaN and must be dropped by the caller.

Feature coverage:
  Computed here (7 pure-OB features): log_spread, imbalance_L1, imbalance_L5,
  depth_ratio, delta_imbalance_L1, kyle_lambda, cum_ofi_5.
  Deferred to integration layer (Task 7): trade_vs_mid — requires per-event
  vwap from the trade pipeline. The integration layer computes it as
  clip((vwap - mid) / max(spread, 1e-8*mid), -5, 5) using the aux columns
  `mid` and `spread` returned here.

Gotchas addressed:
  #9  (depth_ratio log(0))  — epsilon-guarded with 1e-6
  #10 (trade_vs_mid)        — computed in integration layer, not here
  #13 (kyle_lambda per-snapshot, not per-event) — rolling 50-snapshot window
  #14 (notional for cross-symbol comparability) — all formulas use qty × price
  #15 (piecewise Cont 2014 OFI) — _piecewise_cont_ofi handles three price cases
  #20 (10 OB levels per side) — depth_ratio sums all 10 levels
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tape.constants import KYLE_LAMBDA_WINDOW, OFI_WINDOW
from tape.ob_align import align_events_to_ob

_EPS: float = 1e-10


def compute_snapshot_features(ob: pd.DataFrame) -> pd.DataFrame:
    """Compute per-snapshot OB features.

    Parameters
    ----------
    ob : pd.DataFrame
        Must have columns: ts_ms, bid{1..10}_price, bid{1..10}_qty,
        ask{1..10}_price, ask{1..10}_qty.

    Returns
    -------
    pd.DataFrame with columns:
        ts_ms, mid, spread, log_spread, imbalance_L1, imbalance_L5,
        depth_ratio, delta_imbalance_L1, kyle_lambda, cum_ofi_5.
    """
    ts = ob["ts_ms"].to_numpy(dtype=np.int64)
    bid1_p = ob["bid1_price"].to_numpy(dtype=float)
    ask1_p = ob["ask1_price"].to_numpy(dtype=float)
    bid1_q = ob["bid1_qty"].to_numpy(dtype=float)
    ask1_q = ob["ask1_qty"].to_numpy(dtype=float)

    mid = (bid1_p + ask1_p) / 2.0
    spread = np.maximum(ask1_p - bid1_p, _EPS)

    # Feature 10: log_spread = log(spread / mid)
    log_spread = np.log(spread / np.maximum(mid, _EPS))

    # Feature 11: imbalance_L1 — notional at best bid/ask (gotcha #14)
    bid1_not = bid1_p * bid1_q
    ask1_not = ask1_p * ask1_q
    imb_l1 = (bid1_not - ask1_not) / np.maximum(bid1_not + ask1_not, _EPS)

    # Feature 12: imbalance_L5 — inverse-level-weighted notional imbalance L1:5
    num = np.zeros_like(mid)
    den = np.zeros_like(mid)
    for lvl in range(1, 6):
        w = 1.0 / lvl
        b = ob[f"bid{lvl}_price"].to_numpy(dtype=float) * ob[f"bid{lvl}_qty"].to_numpy(
            dtype=float
        )
        a = ob[f"ask{lvl}_price"].to_numpy(dtype=float) * ob[f"ask{lvl}_qty"].to_numpy(
            dtype=float
        )
        num += w * (b - a)
        den += w * (b + a)
    imb_l5 = num / np.maximum(den, _EPS)

    # Feature 13: depth_ratio — log(bid_notional_10L / ask_notional_10L), epsilon-guarded
    # Uses full 10 levels per side (gotcha #9, #14, #20)
    bid_not_total = np.zeros_like(mid)
    ask_not_total = np.zeros_like(mid)
    for lvl in range(1, 11):
        bid_not_total += ob[f"bid{lvl}_price"].to_numpy(dtype=float) * ob[
            f"bid{lvl}_qty"
        ].to_numpy(dtype=float)
        ask_not_total += ob[f"ask{lvl}_price"].to_numpy(dtype=float) * ob[
            f"ask{lvl}_qty"
        ].to_numpy(dtype=float)
    depth_ratio = np.log(
        np.maximum(bid_not_total, 1e-6) / np.maximum(ask_not_total, 1e-6)
    )

    # Feature 15: delta_imbalance_L1 — change since previous snapshot
    # First snapshot = 0 (gotcha #11; full day-boundary warm-up is handled by caller).
    delta_imb_l1 = np.concatenate([[0.0], np.diff(imb_l1)])

    # Feature 16: kyle_lambda — per-SNAPSHOT rolling 50-snapshot window (gotcha #13)
    # Proxy: uses (bid_notional - ask_notional) as signed notional. The integration
    # layer (Task 7) can substitute trade-attributed signed notional if desired.
    kyle_lambda = _rolling_kyle_lambda(
        mid, signed_notional=bid_not_total - ask_not_total
    )

    # Feature 17: cum_ofi_5 — piecewise Cont (2014) OFI over last 5 snapshots,
    # normalised by rolling total notional (gotcha #15)
    ofi = _piecewise_cont_ofi(bid1_p, bid1_q, ask1_p, ask1_q)
    ofi_s: Any = pd.Series(ofi).rolling(OFI_WINDOW, min_periods=1).sum()
    cum_ofi_5_num = ofi_s.to_numpy(dtype=float)
    not_s: Any = (
        pd.Series(bid_not_total + ask_not_total)
        .rolling(OFI_WINDOW, min_periods=1)
        .sum()
    )
    cum_ofi_5_den = np.maximum(not_s.to_numpy(dtype=float), _EPS)
    cum_ofi_5 = cum_ofi_5_num / cum_ofi_5_den

    return pd.DataFrame(
        {
            "ts_ms": ts,
            "mid": mid,
            "spread": spread,
            "log_spread": log_spread,
            "imbalance_L1": imb_l1,
            "imbalance_L5": imb_l5,
            "depth_ratio": depth_ratio,
            "delta_imbalance_L1": delta_imb_l1,
            "kyle_lambda": kyle_lambda,
            "cum_ofi_5": cum_ofi_5,
        }
    )


def align_ob_features_to_events(
    snap: pd.DataFrame, event_ts: np.ndarray
) -> pd.DataFrame:
    """Forward-fill per-snapshot OB features onto per-event timestamps.

    Parameters
    ----------
    snap : pd.DataFrame
        Output of `compute_snapshot_features`. Must contain ts_ms.
    event_ts : np.ndarray
        int64 array of event timestamps, shape (n_events,).

    Returns
    -------
    pd.DataFrame with shape (n_events, ...) containing the OB feature columns
    plus aux columns `mid` and `spread`. Events that precede the first snapshot
    receive NaN values — the caller must mask or drop those events.

    Note: `trade_vs_mid` is NOT computed here (requires per-event vwap from
    the trade pipeline). It is computed in the integration layer (Task 7).
    """
    ob_ts = snap["ts_ms"].to_numpy(dtype=np.int64)
    idx = align_events_to_ob(event_ts, ob_ts)

    feature_cols = (
        "log_spread",
        "imbalance_L1",
        "imbalance_L5",
        "depth_ratio",
        "delta_imbalance_L1",
        "kyle_lambda",
        "cum_ofi_5",
        "mid",
        "spread",
    )
    out_cols: dict[str, np.ndarray] = {}
    for col in feature_cols:
        vals = snap[col].to_numpy(dtype=float)
        clipped_idx = np.clip(idx, 0, len(vals) - 1)
        aligned = np.where(idx >= 0, vals[clipped_idx], np.nan)
        out_cols[col] = aligned

    return pd.DataFrame(out_cols)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _piecewise_cont_ofi(
    bid_p: np.ndarray,
    bid_q: np.ndarray,
    ask_p: np.ndarray,
    ask_q: np.ndarray,
) -> np.ndarray:
    """Cont (2014) order-flow imbalance, piecewise on best bid/ask price change.

    For each snapshot transition t-1 → t:
      Bid side:
        bid_p[t] > bid_p[t-1]  → OFI_bid = +bid_q[t]  * bid_p[t]       (new level, fresh size)
        bid_p[t] == bid_p[t-1] → OFI_bid = (bid_q[t] - bid_q[t-1]) * bid_p[t]  (same level, size change)
        bid_p[t] < bid_p[t-1]  → OFI_bid = -bid_q[t-1] * bid_p[t-1]     (old level pulled)
      Ask side (sign-inverted by convention — ask pressure is negative):
        ask_p[t] > ask_p[t-1]  → OFI_ask = -ask_q[t-1] * ask_p[t-1]     (old level pulled)
        ask_p[t] == ask_p[t-1] → OFI_ask = -(ask_q[t] - ask_q[t-1]) * ask_p[t]
        ask_p[t] < ask_p[t-1]  → OFI_ask = +ask_q[t]  * ask_p[t]

    OFI = OFI_bid + OFI_ask (positive = net buy pressure).
    Returns notional-scale OFI, length n (index 0 is always 0).
    """
    n = len(bid_p)
    ofi = np.zeros(n, dtype=float)
    if n < 2:
        return ofi

    dbid = np.sign(bid_p[1:] - bid_p[:-1])  # +1, 0, -1
    dask = np.sign(ask_p[1:] - ask_p[:-1])

    # --- Bid contribution ---
    bid_up = bid_q[1:] * bid_p[1:]  # price rose: fresh size at new level
    bid_same = (bid_q[1:] - bid_q[:-1]) * bid_p[1:]  # same price: delta qty
    bid_down = -bid_q[:-1] * bid_p[:-1]  # price fell: old level removed
    bid_ofi = np.where(dbid > 0, bid_up, np.where(dbid < 0, bid_down, bid_same))

    # --- Ask contribution (sign-inverted) ---
    ask_up = -ask_q[:-1] * ask_p[:-1]  # ask price rose: old level pulled (buy pressure)
    ask_same = -(ask_q[1:] - ask_q[:-1]) * ask_p[1:]  # same price: delta qty, negated
    ask_down = (
        ask_q[1:] * ask_p[1:]
    )  # ask price fell: new ask size (sell pressure, positive ask_ofi = more selling → net negative... but we add to OFI)
    ask_ofi = np.where(dask > 0, ask_up, np.where(dask < 0, ask_down, ask_same))

    ofi[1:] = bid_ofi + ask_ofi
    return ofi


def _rolling_kyle_lambda(mid: np.ndarray, signed_notional: np.ndarray) -> np.ndarray:
    """Per-snapshot Kyle λ: Cov(Δmid, signed_notional) / Var(signed_notional)
    over a rolling `KYLE_LAMBDA_WINDOW`-snapshot window.

    Gotcha #13: per-snapshot, not per-event. Events are aligned to snapshots
    by `align_ob_features_to_events`, so the per-event value equals the value
    at the aligned snapshot (pure forward-fill — no event-level computation).

    Uses `Δmid` (NOT Δvwap). The proxy here is OB-derived signed notional
    (bid_not - ask_not). The integration layer can substitute trade-attributed
    signed notional via an override if needed.

    First KYLE_LAMBDA_WINDOW rows are set to 0 (insufficient history).
    """
    n = len(mid)
    out = np.zeros(n, dtype=float)
    if n < KYLE_LAMBDA_WINDOW:
        return out

    dmid = np.concatenate([[0.0], np.diff(mid)])
    sn_series: Any = pd.Series(signed_notional)
    dmid_series: Any = pd.Series(dmid)

    cov: Any = sn_series.rolling(KYLE_LAMBDA_WINDOW).cov(dmid_series)
    var: Any = sn_series.rolling(KYLE_LAMBDA_WINDOW).var()
    # Avoid division by zero when signed_notional is constant
    lam_series: Any = (cov / var.replace(0.0, np.nan)).fillna(0.0)
    lam = lam_series.to_numpy(dtype=float)

    # Zero out rows with insufficient history; forward-fill the rest
    out[KYLE_LAMBDA_WINDOW:] = lam[KYLE_LAMBDA_WINDOW:]
    return out

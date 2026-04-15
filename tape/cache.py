# tape/cache.py
"""Build the feature tensor + labels for a single symbol-day and cache to .npz.

Output schema (per shard):
  features        : float32, shape (n_events, 17), channel order = FEATURE_NAMES
  directions      : int8 dict  with keys h10/h50/h100/h500 and mask_h{h}
                    saved as dir_{key}
  wyckoff         : int8 dict  with keys stress/informed_flow/climax/spring/absorption
                    saved as wy_{key}
  event_ts        : int64, shape (n_events,)
  symbol          : str
  date            : str
  schema_version  : int (CACHE_SCHEMA_VERSION)

Integration tasks performed here (not in prior pipeline steps):
  - compute_trade_vs_mid: clip((vwap - mid) / max(spread, 1e-8*mid), -5, 5) (gotcha #10)
  - compute_real_kyle_lambda: trade-attributed signed notional per snapshot,
    rolling 50-snapshot Cov(Δmid, cum_signed_notional)/Var(cum_signed_notional) (gotcha #13)
  - Dedup dispatch: pre-April vs April+ routing (gotchas #3, #19)
  - April hold-out guard: date >= 2026-04-14 is hard-gated (gotcha #17)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tape.constants import (
    APRIL_HELDOUT_START,
    APRIL_START,
    CACHE_SCHEMA_VERSION,
    FEATURE_NAMES,
    KYLE_LAMBDA_WINDOW,
    OB_FEATURES,
    ROLLING_WINDOW,
    TRADE_FEATURES,
)
from tape.dedup import dedup_trades_pre_april, filter_trades_april
from tape.events import group_to_events
from tape.features_ob import align_ob_features_to_events, compute_snapshot_features
from tape.features_trade import compute_trade_features
from tape.io_parquet import load_ob_day, load_trades_day
from tape.labels import compute_direction_labels, compute_wyckoff_labels

# Buyer-initiated sides (gotcha #2 sign convention for Kyle λ)
_BUY_SIDES: frozenset[str] = frozenset({"open_long", "close_short"})


# ---------------------------------------------------------------------------
# Integration-layer helpers (exposed for tests)
# ---------------------------------------------------------------------------


def compute_trade_vs_mid(
    vwap: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
) -> np.ndarray:
    """Clip((vwap - mid) / max(spread, 1e-8*mid), -5, 5) — gotcha #10.

    Parameters
    ----------
    vwap   : float array, per-event VWAP
    mid    : float array, aligned OB mid price per event
    spread : float array, aligned OB spread per event

    Returns
    -------
    float32 array, same length as inputs
    """
    eps: np.ndarray = np.maximum(spread, 1e-8 * mid)
    result = np.clip((vwap - mid) / eps, -5.0, 5.0)
    return result.astype(np.float32)


def compute_real_kyle_lambda(
    snap_ts: np.ndarray,
    snap_mid: np.ndarray,
    trades: pd.DataFrame,
) -> np.ndarray:
    """True trade-attributed Kyle's λ per snapshot (gotcha #13).

    Computed per snapshot over a rolling KYLE_LAMBDA_WINDOW-snapshot window
    (~20 min at 24s cadence) using Cov(Δmid, cum_signed_notional) /
    Var(cum_signed_notional).

    Algorithm:
      1. For each consecutive snapshot pair [s, s+1), assign all trades in
         that interval to snapshot s. Compute the interval's signed notional:
             signed_notional[s] = Σ sign(side) × qty × price
         where sign(side) = +1 for buyer-initiated (open_long/close_short),
         -1 for seller-initiated (open_short/close_long).
         Intervals with zero trades → signed_notional = 0 (not skipped).
      2. cum_signed_notional[s] = cumsum of signed_notional[0..s].
      3. For each snapshot s >= KYLE_LAMBDA_WINDOW - 1:
             lambda[s] = Cov(Δmid[s-W+1..s], cum_sn[s-W+1..s])
                       / Var(cum_sn[s-W+1..s])
         Zero variance in cum_sn → lambda = 0 (not NaN).
      4. First KYLE_LAMBDA_WINDOW - 1 snapshots → 0 (insufficient history).

    Parameters
    ----------
    snap_ts  : int64 array of snapshot timestamps, shape (n_snaps,), non-decreasing
    snap_mid : float array of snapshot mid prices, shape (n_snaps,)
    trades   : DataFrame with columns ts_ms, qty, price, side (deduped, filtered)

    Returns
    -------
    float64 array, shape (n_snaps,)
    """
    n = len(snap_ts)
    signed_notional = np.zeros(n, dtype=float)

    if len(trades) > 0:
        trade_ts = trades["ts_ms"].to_numpy(dtype=np.int64)
        trade_qty = trades["qty"].to_numpy(dtype=float)
        trade_price = trades["price"].to_numpy(dtype=float)
        trade_side = trades["side"].to_numpy(dtype=str)

        # sign: +1 for buy, -1 for sell
        sign = np.where(np.isin(trade_side, list(_BUY_SIDES)), 1.0, -1.0)
        notional = sign * trade_qty * trade_price

        # Assign each trade to a snapshot interval: use searchsorted so that
        # a trade at time t is placed in interval s where snap_ts[s] <= t < snap_ts[s+1].
        # searchsorted(snap_ts, trade_ts, side='right') - 1 gives the latest
        # snapshot at or before the trade. That is the interval index.
        interval_idx = np.searchsorted(snap_ts, trade_ts, side="right") - 1
        # Trades before the first snapshot or after the last are clamped — they
        # contribute to interval 0 or n-1 respectively.
        interval_idx = np.clip(interval_idx, 0, n - 1)

        # Accumulate per-interval signed notional (vectorised bincount)
        signed_notional = np.bincount(interval_idx, weights=notional, minlength=n)

    # Cumulative signed notional over all intervals up to and including s
    cum_sn = np.cumsum(signed_notional)

    # Δmid: mid[s] - mid[s-1]; mid[0] has no predecessor → 0
    dmid = np.concatenate([[0.0], np.diff(snap_mid.astype(float))])

    # Rolling Kyle lambda using pandas rolling cov/var
    out = np.zeros(n, dtype=float)
    if n < KYLE_LAMBDA_WINDOW:
        return out

    sn_series: Any = pd.Series(cum_sn)
    dm_series: Any = pd.Series(dmid)

    cov: Any = sn_series.rolling(KYLE_LAMBDA_WINDOW).cov(dm_series)
    var: Any = sn_series.rolling(KYLE_LAMBDA_WINDOW).var()
    # Zero variance → lambda = 0 (not NaN); gotcha: fillna(0) after division
    lam_series: Any = (cov / var.replace(0.0, np.nan)).fillna(0.0)
    lam = lam_series.to_numpy(dtype=float)

    # First KYLE_LAMBDA_WINDOW - 1 entries stay 0 (insufficient history)
    out[KYLE_LAMBDA_WINDOW - 1 :] = lam[KYLE_LAMBDA_WINDOW - 1 :]
    return out


# ---------------------------------------------------------------------------
# Main per-symbol-day builder
# ---------------------------------------------------------------------------


def build_symbol_day(symbol: str, date_str: str) -> dict | None:
    """Build feature tensor + labels for a single symbol-day.

    Returns None if:
    - date is in the April hold-out (>= 2026-04-14) — hard gate, gotcha #17
    - no data on disk for this symbol-day
    - fewer than 400 events after dedup + OB alignment (too few for windowing)
    - fewer than 2 OB snapshots

    Returns a dict suitable for save_shard().
    """
    # Hard gate: April 14+ is untouched (gotcha #17)
    if date_str >= APRIL_HELDOUT_START:
        import sys

        print(
            f"[{symbol} {date_str}] SKIP — date is in the April hold-out "
            f"(>= {APRIL_HELDOUT_START})",
            file=sys.stderr,
        )
        return None

    # Load raw trades
    trades = load_trades_day(symbol, date_str)
    if trades is None or len(trades) == 0:
        return None

    # Dedup dispatch (gotchas #3, #19)
    if date_str >= APRIL_START:
        trades = filter_trades_april(trades)
    else:
        trades = dedup_trades_pre_april(trades)

    # Group to order events
    events = group_to_events(trades)
    if len(events) < 400:
        return None

    # Load orderbook
    ob = load_ob_day(symbol, date_str)
    if ob is None or len(ob) < 2:
        return None

    # Snapshot features (placeholder kyle_lambda; we replace it below)
    snap = compute_snapshot_features(ob)

    # Overwrite kyle_lambda with the real trade-attributed version (gotcha #13)
    snap_ts = snap["ts_ms"].to_numpy(dtype=np.int64)
    snap_mid_vals = snap["mid"].to_numpy(dtype=float)
    real_kl = compute_real_kyle_lambda(snap_ts, snap_mid_vals, trades)
    snap = snap.copy()
    snap["kyle_lambda"] = real_kl

    # Align OB features → event timestamps
    event_ts_arr = events["ts_ms"].to_numpy(dtype=np.int64)
    ob_aligned = align_ob_features_to_events(snap, event_ts_arr)

    # Drop events before the first OB snapshot (NaN mid/spread)
    valid = np.isfinite(ob_aligned["mid"].to_numpy(dtype=float))
    if valid.sum() < 400:
        return None
    events = events.loc[valid].reset_index(drop=True)
    ob_aligned = ob_aligned.loc[valid].reset_index(drop=True)

    # 9 trade features (use aligned spread/mid for book_walk denominator)
    trade_feats = compute_trade_features(
        events,
        spread=ob_aligned["spread"].to_numpy(dtype=float),
        mid=ob_aligned["mid"].to_numpy(dtype=float),
    )

    # trade_vs_mid — integration-layer computation (gotcha #10)
    vwap = events["vwap"].to_numpy(dtype=float)
    mid = ob_aligned["mid"].to_numpy(dtype=float)
    spread = ob_aligned["spread"].to_numpy(dtype=float)
    trade_vs_mid = compute_trade_vs_mid(vwap, mid, spread)

    # Assemble 17-feature matrix in FEATURE_NAMES order
    n = len(events)
    features = np.zeros((n, 17), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        if name in TRADE_FEATURES:
            features[:, i] = trade_feats[name].to_numpy(dtype=np.float32)
        elif name == "trade_vs_mid":
            features[:, i] = trade_vs_mid
        elif name in OB_FEATURES:
            features[:, i] = ob_aligned[name].to_numpy(dtype=np.float32)

    # Validate shape before computing labels (save GPU time if buggy)
    assert features.shape == (
        n,
        17,
    ), f"Feature shape mismatch: expected ({n}, 17), got {features.shape}"

    # Rolling z-scores for Wyckoff labels (recomputed from events — see plan note)
    total_qty = events["total_qty"].to_numpy(dtype=float)
    log_return = trade_feats["log_return"].to_numpy(dtype=float)
    abs_ret = np.abs(log_return)

    _mq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).mean()
    mean_qty: np.ndarray = _mq.fillna(0.0).to_numpy(dtype=float)
    _sq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).std()
    std_qty: np.ndarray = np.maximum(_sq.fillna(1e-10).to_numpy(dtype=float), 1e-10)

    _mr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).mean()
    mean_ret: np.ndarray = _mr.fillna(0.0).to_numpy(dtype=float)
    _sr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).std()
    std_ret: np.ndarray = np.maximum(_sr.fillna(1e-10).to_numpy(dtype=float), 1e-10)
    z_qty = (total_qty - mean_qty) / std_qty
    z_ret = (abs_ret - mean_ret) / std_ret

    directions = compute_direction_labels(vwap)
    wyckoff = compute_wyckoff_labels(
        log_return=log_return,
        effort_vs_result=trade_feats["effort_vs_result"].to_numpy(dtype=float),
        is_open=trade_feats["is_open"].to_numpy(dtype=float),
        climax_score=trade_feats["climax_score"].to_numpy(dtype=float),
        z_qty=z_qty,
        z_ret=z_ret,
        log_spread=ob_aligned["log_spread"].to_numpy(dtype=float),
        depth_ratio=ob_aligned["depth_ratio"].to_numpy(dtype=float),
        kyle_lambda=ob_aligned["kyle_lambda"].to_numpy(dtype=float),
        cum_ofi_5=ob_aligned["cum_ofi_5"].to_numpy(dtype=float),
    )

    dir_arrays: dict[str, np.ndarray] = {}
    for k, v in directions.items():
        arr: np.ndarray = np.asarray(v)
        dir_arrays[k] = arr.astype(np.int8) if arr.dtype != np.dtype(bool) else arr

    return {
        "features": features,
        "event_ts": events["ts_ms"].to_numpy(dtype=np.int64),
        "directions": dir_arrays,
        "wyckoff": wyckoff,
        "symbol": symbol,
        "date": date_str,
        "schema_version": CACHE_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Shard persistence
# ---------------------------------------------------------------------------


def save_shard(shard: dict, out_dir: Path) -> Path:
    """Serialise a shard dict to a compressed .npz file.

    Keys layout in .npz:
      features, event_ts, schema_version
      dir_{key}  for each direction key (h10, mask_h10, ...)
      wy_{key}   for each wyckoff key (stress, absorption, ...)
      symbol, date  (scalar string arrays)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{shard['symbol']}__{shard['date']}.npz"

    payload: dict[str, Any] = {
        "features": shard["features"],
        "event_ts": shard["event_ts"],
        "schema_version": np.array(int(shard["schema_version"]), dtype=np.int32),
    }
    for k, v in shard["directions"].items():
        payload[f"dir_{k}"] = v
    for k, v in shard["wyckoff"].items():
        payload[f"wy_{k}"] = v

    np.savez_compressed(
        path,
        symbol=np.array(shard["symbol"]),
        date=np.array(shard["date"]),
        **payload,
    )
    return path


def load_shard(path: Path) -> dict:
    """Load a .npz shard and return a plain dict of arrays."""
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    return payload

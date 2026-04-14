# tape/events.py
"""Group same-timestamp trades into order events.

An "order event" is the aggregate of all fills that share a millisecond
timestamp. Per the spec, same-timestamp trades are fragments of one order after
dedup.

Output columns (all numeric):
    ts_ms          : int64 — the shared timestamp
    total_qty      : float — sum of qty
    vwap           : float — qty-weighted mean price
    is_open_frac   : float in [0, 1] — fraction of fills whose side is open_*
    n_fills        : int   — number of fills in the event
    book_walk_abs  : float — |last_fill_price - first_fill_price| (unsigned;
                             the spread-normalised version lives in
                             features_trade.book_walk)
    first_ts, last_ts : int64 — currently equal to ts_ms; kept for forward
                                compatibility if we widen grouping later
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

_OPEN_SIDES: frozenset[str] = frozenset({"open_long", "open_short"})


def group_to_events(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deduped trades into order events keyed by ts_ms.

    Assumes `trades` is already sorted by ts_ms ascending; this is the case
    after reading a single day's parquet. The function preserves that ordering.
    """
    if len(trades) == 0:
        return pd.DataFrame(
            {
                "ts_ms": pd.Series([], dtype=np.int64),
                "total_qty": pd.Series([], dtype=float),
                "vwap": pd.Series([], dtype=float),
                "is_open_frac": pd.Series([], dtype=float),
                "n_fills": pd.Series([], dtype=np.int64),
                "book_walk_abs": pd.Series([], dtype=float),
                "first_ts": pd.Series([], dtype=np.int64),
                "last_ts": pd.Series([], dtype=np.int64),
            }
        )

    # Per-fill notional = qty * price for vwap computation.
    qty = trades["qty"].to_numpy(dtype=float)
    price = trades["price"].to_numpy(dtype=float)
    notional = qty * price
    is_open = trades["side"].isin(list(_OPEN_SIDES)).to_numpy(dtype=float)

    df = trades[["ts_ms"]].copy()
    df["_qty"] = qty
    df["_notional"] = notional
    df["_is_open"] = is_open
    df["_price_first"] = price
    df["_price_last"] = price

    agg = df.groupby("ts_ms", sort=True).agg(
        total_qty=("_qty", "sum"),
        _notional_sum=("_notional", "sum"),
        _is_open_sum=("_is_open", "sum"),
        n_fills=("_qty", "size"),
        _price_first=("_price_first", "first"),
        _price_last=("_price_last", "last"),
    )

    ts_ms_arr = np.array(agg.index, dtype=np.int64)
    total_qty_arr = np.array(cast(pd.Series, agg["total_qty"]), dtype=float)
    notional_sum_arr = np.array(cast(pd.Series, agg["_notional_sum"]), dtype=float)
    is_open_sum_arr = np.array(cast(pd.Series, agg["_is_open_sum"]), dtype=float)
    n_fills_arr = np.array(cast(pd.Series, agg["n_fills"]), dtype=np.int64)
    price_first_arr = np.array(cast(pd.Series, agg["_price_first"]), dtype=float)
    price_last_arr = np.array(cast(pd.Series, agg["_price_last"]), dtype=float)

    out = pd.DataFrame(
        {
            "ts_ms": ts_ms_arr,
            "total_qty": total_qty_arr,
            "vwap": notional_sum_arr / total_qty_arr,
            "is_open_frac": is_open_sum_arr / n_fills_arr,
            "n_fills": n_fills_arr,
            "book_walk_abs": np.abs(price_last_arr - price_first_arr),
        }
    )
    out["first_ts"] = ts_ms_arr
    out["last_ts"] = ts_ms_arr
    return out.reset_index(drop=True)

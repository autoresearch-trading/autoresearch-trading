# tape/io_parquet.py
"""Per-symbol-day loaders for trades and orderbook parquet.

The raw orderbook parquet stores the 10-level book as nested arrays in two
columns: `bids` and `asks`, each a list of dicts with keys `price` and `qty`.
`load_ob_day` calls `expand_ob_levels` to flatten these into the flat column
layout expected by `tape.features_ob.compute_snapshot_features`:
    bid1_price, bid1_qty, ..., bid10_price, bid10_qty
    ask1_price, ask1_qty, ..., ask10_price, ask10_qty
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from tape.constants import OB_GLOB, TRADES_GLOB

_N_LEVELS: int = 10


def expand_ob_levels(ob: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested bids/asks arrays into bid{n}_price/qty columns.

    Expects `ob` to have columns `ts_ms`, `bids`, `asks` where each element
    of `bids`/`asks` is a numpy array of dicts with keys `price` and `qty`
    (length _N_LEVELS = 10).

    Returns a new DataFrame with ts_ms plus 40 flat level columns.
    Rows where the nested array is shorter than _N_LEVELS have NaN in
    the missing levels.
    """
    n_rows = len(ob)
    ts_ms = ob["ts_ms"].to_numpy(dtype=np.int64)

    bid_prices = np.full((n_rows, _N_LEVELS), np.nan, dtype=float)
    bid_qtys = np.full((n_rows, _N_LEVELS), np.nan, dtype=float)
    ask_prices = np.full((n_rows, _N_LEVELS), np.nan, dtype=float)
    ask_qtys = np.full((n_rows, _N_LEVELS), np.nan, dtype=float)

    bids_col = ob["bids"].to_numpy()
    asks_col = ob["asks"].to_numpy()

    for i in range(n_rows):
        bids = bids_col[i]
        asks = asks_col[i]
        for lvl_idx in range(min(_N_LEVELS, len(bids))):
            bid_prices[i, lvl_idx] = float(bids[lvl_idx]["price"])
            bid_qtys[i, lvl_idx] = float(bids[lvl_idx]["qty"])
        for lvl_idx in range(min(_N_LEVELS, len(asks))):
            ask_prices[i, lvl_idx] = float(asks[lvl_idx]["price"])
            ask_qtys[i, lvl_idx] = float(asks[lvl_idx]["qty"])

    data: dict[str, object] = {"ts_ms": ts_ms}
    for lvl in range(1, _N_LEVELS + 1):
        data[f"bid{lvl}_price"] = bid_prices[:, lvl - 1]
        data[f"bid{lvl}_qty"] = bid_qtys[:, lvl - 1]
        data[f"ask{lvl}_price"] = ask_prices[:, lvl - 1]
        data[f"ask{lvl}_qty"] = ask_qtys[:, lvl - 1]

    return pd.DataFrame(data)


def load_trades_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    """Load all trades for a symbol-day, sorted by ts_ms.

    Returns None if the directory does not exist (missing data).
    """
    path = Path(TRADES_GLOB.format(sym=symbol, date=date_str)).parent
    if not path.exists():
        return None
    q = f"SELECT * FROM read_parquet('{path}/*.parquet') ORDER BY ts_ms"
    return duckdb.query(q).to_df()


def load_ob_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    """Load all orderbook snapshots for a symbol-day, sorted by ts_ms.

    Returns a flat DataFrame (bid{n}_price/qty, ask{n}_price/qty) ready for
    `compute_snapshot_features`. Returns None if the directory does not exist.
    """
    path = Path(OB_GLOB.format(sym=symbol, date=date_str)).parent
    if not path.exists():
        return None
    q = f"SELECT * FROM read_parquet('{path}/*.parquet') ORDER BY ts_ms"
    raw = duckdb.query(q).to_df()
    if len(raw) == 0:
        return None
    # If the raw data already has flat columns (e.g. in tests), pass through.
    if "bid1_price" in raw.columns:
        return raw
    return expand_ob_levels(raw)

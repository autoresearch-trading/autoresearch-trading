# tape/dedup.py
"""Dedup raw trade rows into one row per fill.

Pre-April (2025-10-16 to 2026-03-31): the API returns both counterparty
perspectives for every fill. Rows share `(ts_ms, qty, price)` but differ on
`side`. Dedup by those three columns ONLY — gotchas #3 and #19.

April 1+: the API includes `event_type`. Filter to `fulfill_taker`; every fill
appears exactly once. Gotcha #3.
"""

from __future__ import annotations

import pandas as pd

_PRE_APRIL_DEDUP_KEYS: tuple[str, str, str] = ("ts_ms", "qty", "price")


def dedup_trades_pre_april(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse buyer/seller rows into one row per fill."""
    return df.drop_duplicates(
        subset=list(_PRE_APRIL_DEDUP_KEYS), keep="first"
    ).reset_index(drop=True)


def filter_trades_april(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fulfill_taker rows (one row per fill)."""
    if "event_type" not in df.columns:
        raise ValueError(
            "April+ data must have event_type column; got pre-April schema"
        )
    return df.loc[df["event_type"] == "fulfill_taker"].reset_index(drop=True)

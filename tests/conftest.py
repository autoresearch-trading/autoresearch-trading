"""Shared fixtures for feature engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_trades():
    """Factory for synthetic trade DataFrames."""

    def _make(
        n: int = 200,
        base_price: float = 100.0,
        base_qty: float = 1.0,
        sides: list[str] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        if sides is None:
            sides = ["open_long", "close_long", "open_short", "close_short"]
        return pd.DataFrame(
            {
                "ts_ms": np.arange(n) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(n)],
                "side": rng.choice(sides, size=n),
                "qty": rng.exponential(base_qty, size=n),
                "price": base_price + rng.normal(0, 0.1, size=n).cumsum(),
                "recv_ms": np.arange(n) * 1000 + 1_000_010,
            }
        )

    return _make


@pytest.fixture
def make_orderbook():
    """Factory for synthetic orderbook DataFrames."""

    def _make(
        n: int = 50,
        best_bid: float = 99.5,
        best_ask: float = 100.5,
        levels: int = 10,
        bid_qty: float = 2.0,
        ask_qty: float = 3.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            bids = np.array(
                [
                    {"price": best_bid - lvl * 0.1, "qty": bid_qty + rng.normal(0, 0.1)}
                    for lvl in range(levels)
                ],
                dtype=object,
            )
            asks = np.array(
                [
                    {"price": best_ask + lvl * 0.1, "qty": ask_qty + rng.normal(0, 0.1)}
                    for lvl in range(levels)
                ],
                dtype=object,
            )
            rows.append(
                {
                    "ts_ms": i * 4000 + 1_000_000,  # ~every 4 trades
                    "symbol": "TEST",
                    "bids": bids,
                    "asks": asks,
                    "recv_ms": i * 4000 + 1_000_010,
                    "agg_level": 1,
                }
            )
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def make_funding():
    """Factory for synthetic funding DataFrames."""

    def _make(
        n: int = 5,
        base_rate: float = 0.0001,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "ts_ms": np.arange(n) * 40000 + 1_000_000,
                "symbol": "TEST",
                "rate": base_rate + rng.normal(0, 0.00001, size=n).cumsum(),
                "interval_sec": 1,
                "recv_ms": np.arange(n) * 40000 + 1_000_010,
            }
        )

    return _make


@pytest.fixture
def empty_df():
    """Empty DataFrame for testing edge cases."""
    return pd.DataFrame()

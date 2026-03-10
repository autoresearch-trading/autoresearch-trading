"""Tests for feature engineering in prepare.py."""

from __future__ import annotations

import numpy as np

from prepare import compute_features


class TestComputeFeaturesBaseline:
    """Verify existing features still work after modifications."""

    def test_output_shape(self, make_trades, make_orderbook, make_funding):
        trades = make_trades(n=200)
        ob = make_orderbook(n=50)
        funding = make_funding(n=5)
        features, timestamps, prices = compute_features(
            trades, ob, funding, trade_batch=100
        )
        assert features.shape[0] == 2  # 200 trades / 100 batch
        assert features.shape[1] == 26  # 8 trade + 16 OB + 2 funding
        assert len(timestamps) == 2
        assert len(prices) == 2

    def test_empty_trades(self, empty_df, make_orderbook, make_funding):
        features, timestamps, prices = compute_features(
            empty_df, make_orderbook(), make_funding()
        )
        assert len(features) == 0

    def test_empty_orderbook(self, make_trades, empty_df, make_funding):
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        # OB features should be zeros
        assert features.shape[1] == 26
        assert np.all(features[:, 8:24] == 0)  # OB columns are indices 8-23

    def test_empty_funding(self, make_trades, make_orderbook, empty_df):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert features.shape[1] == 26
        assert np.all(features[:, 24:26] == 0)  # funding columns are indices 24-25


class TestMicroprice:
    """Test microprice feature computation."""

    def test_microprice_computed(self, make_trades, make_orderbook, make_funding):
        """Microprice should be in the feature output."""
        trades = make_trades(n=200)
        ob = make_orderbook(
            n=50, best_bid=100.0, best_ask=102.0, bid_qty=2.0, ask_qty=3.0
        )
        funding = make_funding(n=5)
        features, _, _ = compute_features(trades, ob, funding, trade_batch=100)
        microprice_col = 22  # first new column after existing 22 OB features
        assert features.shape[1] >= 24  # at least has new features
        assert features[0, microprice_col] != 0.0

    def test_microprice_value_exact(self):
        """Test microprice formula with exact known values."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(100) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(100)],
                "side": ["open_long"] * 100,
                "qty": [1.0] * 100,
                "price": [100.0] * 100,
                "recv_ms": np.arange(100) * 1000 + 1_000_010,
            }
        )
        bids = np.array(
            [{"price": 100.0, "qty": 2.0}]
            + [{"price": 100.0 - i * 0.1, "qty": 1.0} for i in range(1, 10)],
            dtype=object,
        )
        asks = np.array(
            [{"price": 102.0, "qty": 3.0}]
            + [{"price": 102.0 + i * 0.1, "qty": 1.0} for i in range(1, 10)],
            dtype=object,
        )
        ob = pd.DataFrame(
            [
                {
                    "ts_ms": 1_000_000,
                    "symbol": "TEST",
                    "bids": bids,
                    "asks": asks,
                    "recv_ms": 1_000_010,
                    "agg_level": 1,
                }
            ]
        )

        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        microprice_col = 22
        microprice_dev_col = 23
        # microprice = (100*3 + 102*2) / (2+3) = 100.8
        assert abs(features[0, microprice_col] - 100.8) < 1e-6
        # mid = (100 + 102) / 2 = 101, deviation = 100.8 - 101 = -0.2
        assert abs(features[0, microprice_dev_col] - (-0.2)) < 1e-6

    def test_microprice_zero_when_no_ob(self, make_trades, empty_df, make_funding):
        """Microprice should be 0 when no orderbook data."""
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        microprice_col = 22
        microprice_dev_col = 23
        assert np.all(features[:, microprice_col] == 0.0)
        assert np.all(features[:, microprice_dev_col] == 0.0)

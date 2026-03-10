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
        assert features.shape[1] == 24  # current feature count
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
        assert features.shape[1] == 24
        assert np.all(features[:, 8:22] == 0)  # OB columns are indices 8-21

    def test_empty_funding(self, make_trades, make_orderbook, empty_df):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert features.shape[1] == 24
        assert np.all(features[:, 22:24] == 0)  # funding columns are indices 22-23

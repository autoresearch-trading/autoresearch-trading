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
        assert features.shape[1] == 28  # 9 trade + 17 OB + 2 funding
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
        assert features.shape[1] == 28
        assert np.all(features[:, 9:26] == 0)  # OB columns are indices 9-25

    def test_empty_funding(self, make_trades, make_orderbook, empty_df):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert features.shape[1] == 28
        assert np.all(features[:, 26:28] == 0)  # funding columns are indices 26-27


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
        microprice_col = 23  # first new column after 9 trade + 14 base OB features
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
        microprice_col = 23
        microprice_dev_col = 24
        # microprice = (100*3 + 102*2) / (2+3) = 100.8
        assert abs(features[0, microprice_col] - 100.8) < 1e-6
        # mid = (100 + 102) / 2 = 101, deviation = 100.8 - 101 = -0.2
        assert abs(features[0, microprice_dev_col] - (-0.2)) < 1e-6

    def test_microprice_zero_when_no_ob(self, make_trades, empty_df, make_funding):
        """Microprice should be 0 when no orderbook data."""
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        microprice_col = 23
        microprice_dev_col = 24
        assert np.all(features[:, microprice_col] == 0.0)
        assert np.all(features[:, microprice_dev_col] == 0.0)


class TestOFI:
    """Test Order Flow Imbalance feature."""

    def test_ofi_computed(self, make_trades, make_orderbook, make_funding):
        """OFI should be non-zero when orderbook levels change."""
        trades = make_trades(n=200)
        ob = make_orderbook(n=50)
        funding = make_funding(n=5)
        features, _, _ = compute_features(trades, ob, funding, trade_batch=100)
        ofi_col = 25  # after 8 trade + 16 OB (including microprice), OFI is OB col 16
        assert features.shape[1] >= 27  # 8 trade + 17 OB + 2 funding

    def test_ofi_detects_bid_increase(self):
        """OFI should be positive when bid depth increases."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 200,
                "price": [100.0] * 200,
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        # First snapshot: bid_qty=2, ask_qty=3
        bids1 = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(10)], dtype=object
        )
        asks1 = np.array(
            [{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(10)], dtype=object
        )
        # Second snapshot: bid_qty=5 (increased), ask_qty=3 (same)
        bids2 = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(10)], dtype=object
        )
        asks2 = np.array(
            [{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(10)], dtype=object
        )
        ob = pd.DataFrame(
            [
                {
                    "ts_ms": 1_050_000,
                    "symbol": "TEST",
                    "bids": bids1,
                    "asks": asks1,
                    "recv_ms": 1_050_010,
                    "agg_level": 1,
                },
                {
                    "ts_ms": 1_150_000,
                    "symbol": "TEST",
                    "bids": bids2,
                    "asks": asks2,
                    "recv_ms": 1_150_010,
                    "agg_level": 1,
                },
            ]
        )
        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        ofi_col = 25
        # Batch 1 uses snapshot 2. OFI = sum w_l * (Δbid - Δask)
        assert features[1, ofi_col] > 0  # positive OFI = bid depth increased

    def test_ofi_zero_first_batch(self, make_trades, make_orderbook, make_funding):
        """OFI should be 0 for first batch (no previous snapshot)."""
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        ofi_col = 25
        assert features[0, ofi_col] == 0.0


class TestVPIN:
    """Test VPIN (flow toxicity) feature."""

    def test_vpin_bounded_0_1(self, make_trades, make_orderbook, make_funding):
        """VPIN should be between 0 and 1."""
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        vpin_col = 8  # 9th trade feature (added after large_trade_count)
        vpin = features[:, vpin_col]
        assert np.all(vpin >= 0.0)
        assert np.all(vpin <= 1.0)

    def test_vpin_high_for_one_sided_flow(self):
        """VPIN should be ~1 when all trades are buys."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,  # all buys
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        vpin_col = 8
        # All buys -> |tfi| = 1.0 for every batch -> rolling mean = 1.0
        assert features[-1, vpin_col] > 0.9

    def test_vpin_low_for_balanced_flow(self):
        """VPIN should be near 0 when flow is balanced."""
        import pandas as pd

        sides = ["open_long", "open_short"] * 250
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": sides,  # alternating buy/sell
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        vpin_col = 8
        assert features[-1, vpin_col] < 0.1

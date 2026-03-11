"""Tests for v3 feature engineering in prepare.py (20 features)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prepare import compute_features, normalize_features

NUM_FEATURES_V3 = 20


class TestFeatureShape:
    """Output shape and basic structure tests."""

    def test_output_shape_20_features(self, make_trades, make_orderbook, make_funding):
        features, timestamps, prices = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features.shape == (2, NUM_FEATURES_V3)
        assert len(timestamps) == 2
        assert len(prices) == 2

    def test_empty_trades_returns_empty(self, empty_df, make_orderbook, make_funding):
        features, timestamps, prices = compute_features(
            empty_df, make_orderbook(), make_funding()
        )
        assert len(features) == 0

    def test_empty_orderbook_zeros_ob_columns(
        self, make_trades, empty_df, make_funding
    ):
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        assert features.shape[1] == NUM_FEATURES_V3
        # OB features (indices 12-17) should be zero or default
        assert features[0, 12] == 0.0  # spread_bps
        assert features[0, 14] == 0.0  # weighted_imbalance

    def test_empty_funding_zeros_funding_col(
        self, make_trades, make_orderbook, empty_df
    ):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert features.shape[1] == NUM_FEATURES_V3
        assert features[0, 18] == 0.0  # funding_zscore


class TestTrendFeatures:
    """Tests for returns, r_5, r_20, r_100 (indices 0-3)."""

    def test_returns_log_return_of_vwap(self):
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 200,
                "price": [100.0] * 100 + [110.0] * 100,
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        # Batch 0 VWAP=100, Batch 1 VWAP=110
        expected_return = np.log(110.0 / 100.0)
        assert abs(features[1, 0] - expected_return) < 1e-6

    def test_multi_horizon_returns_correct(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        # r_5 at index 1 should equal log(vwap[t]/vwap[t-5])
        # For first 5 batches, r_5 should be 0 (not enough history)
        assert features[0, 1] == 0.0  # r_5 at batch 0

    def test_r_100_zero_for_short_series(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        # Only 5 batches, r_100 needs 100 batches of history -> all zeros
        assert np.all(features[:, 3] == 0.0)


class TestRiskFeatures:
    """Tests for realvol_10 and bipower_var_20 (indices 4-5)."""

    def test_realvol_nonnegative(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 4] >= 0.0)  # realvol_10

    def test_realvol_zero_for_flat_price(self):
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert np.all(features[:, 4] == 0.0)  # realvol

    def test_bipower_variation_nonnegative(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 5] >= 0.0)

    def test_bipower_variation_zero_for_flat_price(self):
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert np.all(features[:, 5] == 0.0)


class TestFlowFeatures:
    """Tests for tfi, volume_spike_ratio, large_trade_share (indices 6-8)."""

    def test_tfi_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        tfi = features[:, 6]
        assert np.all(tfi >= -1.0)
        assert np.all(tfi <= 1.0)

    def test_tfi_one_for_all_buys(self):
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
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert abs(features[0, 6] - 1.0) < 1e-6

    def test_volume_spike_ratio_around_one_for_uniform(self):
        """Uniform volume -> spike ratio ~ 1.0 after warmup."""
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(3000) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(3000)],
                "side": ["open_long"] * 3000,
                "qty": [1.0] * 3000,
                "price": [100.0] * 3000,
                "recv_ms": np.arange(3000) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        # After warmup (20 batches), ratio should be ~1.0
        assert abs(features[-1, 7] - 1.0) < 0.01

    def test_large_trade_share_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        share = features[:, 8]
        assert np.all(share >= 0.0)
        assert np.all(share <= 1.0)


class TestLiquidityFeatures:
    """Tests for kyle_lambda, amihud_illiq, trade_arrival_rate (indices 9-11)."""

    def test_trade_arrival_rate_positive(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        arrival = features[:, 11]
        assert np.all(arrival > 0.0)

    def test_arrival_rate_higher_for_faster_trading(self):
        """Trades arriving faster -> higher arrival rate."""
        slow = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 10000 + 1_000_000,  # 10s apart
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 200,
                "price": [100.0] * 200,
                "recv_ms": np.arange(200) * 10000 + 1_000_010,
            }
        )
        fast = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 100 + 1_000_000,  # 0.1s apart
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 200,
                "price": [100.0] * 200,
                "recv_ms": np.arange(200) * 100 + 1_000_010,
            }
        )
        slow_feat, _, _ = compute_features(
            slow, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        fast_feat, _, _ = compute_features(
            fast, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert fast_feat[0, 11] > slow_feat[0, 11]

    def test_amihud_nonnegative(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500),
            make_orderbook(n=100),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 10] >= 0.0)

    def test_amihud_higher_for_illiquid(self):
        """Higher |return|/notional for small-volume, big-move market."""
        # Liquid: small returns, large notional
        liquid = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [100.0] * 200,
                "price": [100.0 + 0.001 * i for i in range(200)],
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        # Illiquid: large returns, small notional
        illiquid = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [0.01] * 200,
                "price": [100.0 + 0.1 * i for i in range(200)],
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        liq_feat, _, _ = compute_features(
            liquid, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        illiq_feat, _, _ = compute_features(
            illiquid, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert illiq_feat[1, 10] > liq_feat[1, 10]  # Amihud higher for illiquid

    def test_kyle_lambda_sign(self):
        """When buys push price up, Kyle's lambda should be positive."""
        # Create data where buy pressure moves price up
        n = 5000
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0.01, 0.02, n))
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(n) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(n)],
                "side": ["open_long"] * n,  # all buys
                "qty": rng.exponential(1.0, n),
                "price": prices,
                "recv_ms": np.arange(n) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        # After warmup (50 batches), Kyle's lambda should tend positive
        assert features[-1, 9] > 0


class TestOrderbookFeatures:
    """Tests for spread_bps, log_total_depth, weighted_imbalance, microprice_dev, ofi, ob_slope (indices 12-17)."""

    def test_spread_bps_positive(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features[0, 12] > 0.0  # spread should be positive

    def test_weighted_imbalance_bounded(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        imb = features[:, 14]
        assert np.all(imb >= -1.0)
        assert np.all(imb <= 1.0)

    def test_weighted_imbalance_positive_when_bids_dominate(self):
        """More bid depth -> positive imbalance."""
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
            [{"price": 100.0 - i * 0.1, "qty": 10.0} for i in range(5)], dtype=object
        )
        asks = np.array(
            [{"price": 101.0 + i * 0.1, "qty": 1.0} for i in range(5)], dtype=object
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
        assert features[0, 14] > 0.0

    def test_microprice_dev_exact(self):
        """Test microprice deviation with known values."""
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
            + [{"price": 100.0 - i * 0.1, "qty": 1.0} for i in range(1, 5)],
            dtype=object,
        )
        asks = np.array(
            [{"price": 102.0, "qty": 3.0}]
            + [{"price": 102.0 + i * 0.1, "qty": 1.0} for i in range(1, 5)],
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
        # microprice = (100*3 + 102*2) / 5 = 100.8, mid = 101, dev = -0.2
        assert abs(features[0, 15] - (-0.2)) < 1e-6

    def test_ob_slope_asym_with_symmetric_book(self):
        """Symmetric book -> slope asymmetry near zero."""
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
        # Symmetric book: same qty profile on both sides
        bids = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(5)], dtype=object
        )
        asks = np.array(
            [{"price": 100.1 + i * 0.1, "qty": 5.0} for i in range(5)], dtype=object
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
        # Symmetric book -> slope asymmetry close to zero
        assert abs(features[0, 17]) < 0.1

    def test_ob_slope_asym_thin_asks(self):
        """Thin asks (steep slope) -> positive asymmetry."""
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
        # Thick bids, thin asks
        bids = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 100.0} for i in range(5)], dtype=object
        )
        asks = np.array(
            [{"price": 100.1 + i * 0.5, "qty": 1.0} for i in range(5)], dtype=object
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
        # Steep ask slope, flat bid slope -> ask_slope > bid_slope -> positive
        assert features[0, 17] > 0

    def test_ofi_positive_on_bid_increase(self):
        """OFI should be positive when bid depth increases."""
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
        bids1 = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(5)], dtype=object
        )
        asks1 = np.array(
            [{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object
        )
        bids2 = np.array(
            [{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(5)], dtype=object
        )
        asks2 = np.array(
            [{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object
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
        assert features[1, 16] > 0  # positive OFI


class TestFundingAndTimeFeatures:
    """Tests for funding_zscore and utc_hour_linear (indices 18-19)."""

    def test_utc_hour_linear_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        hour = features[:, 19]
        assert np.all(hour >= 0.0)
        assert np.all(hour <= 1.0)

    def test_funding_zscore_zero_when_no_funding(
        self, make_trades, make_orderbook, empty_df
    ):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert np.all(features[:, 18] == 0.0)

    def test_funding_zscore_zero_during_warmup(self):
        """Funding z-score should be 0 until 8+ events are available."""
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
        # Only 5 funding events (less than required 8)
        funding = pd.DataFrame(
            {
                "ts_ms": [1_000_000 + i * 50_000 for i in range(5)],
                "symbol": "TEST",
                "rate": [0.001, 0.002, 0.003, 0.001, 0.005],
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), funding, trade_batch=100
        )
        assert features[0, 18] == 0.0  # Not enough events for z-score

    def test_funding_zscore_constant_between_prints(self):
        """Funding z-score should be forward-filled (constant between funding events)."""
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        # 10 funding events, all before batch 0 ends, spaced apart
        funding = pd.DataFrame(
            {
                "ts_ms": [800_000 + i * 10_000 for i in range(10)],
                "symbol": "TEST",
                "rate": [0.001] * 8 + [0.01, 0.01],  # 8 warmup then spike
            }
        )
        features, _, _ = compute_features(
            trades, pd.DataFrame(), funding, trade_batch=100
        )
        # Batches 0-4 should all have same funding_zscore (same last funding event)
        if features.shape[0] > 1:
            assert features[0, 18] == features[1, 18]  # forward-filled


class TestNormalization:
    """Test that normalization works with new 20-feature layout."""

    def test_normalize_shape_preserved(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=1000),
            make_orderbook(n=200),
            make_funding(n=20),
            trade_batch=100,
        )
        normalized = normalize_features(features)
        assert normalized.shape == features.shape
        assert not np.any(np.isnan(normalized))

    def test_normalize_no_infs(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=1000),
            make_orderbook(n=200),
            make_funding(n=20),
            trade_batch=100,
        )
        normalized = normalize_features(features)
        assert not np.any(np.isinf(normalized))


class TestIntegration:
    """End-to-end integration tests."""

    def test_env_observation_space(self, make_trades, make_orderbook, make_funding):
        from prepare import TradingEnv

        features, _, prices = compute_features(
            make_trades(n=5000),
            make_orderbook(n=1000),
            make_funding(n=100),
            trade_batch=100,
        )
        features = normalize_features(features)
        env = TradingEnv(features, prices, window_size=10)
        assert env.observation_space.shape == (10, NUM_FEATURES_V3)

    def test_batch_prices_are_vwap(self):
        """Verify batch_prices are VWAP-based (training PnL depends on this)."""
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 100 + [2.0] * 100,
                "price": [100.0] * 100 + [110.0] * 100,
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        _, _, prices = compute_features(
            trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100
        )
        assert abs(prices[0] - 100.0) < 1e-6  # VWAP of batch 0
        # Batch 1: VWAP = (2*110*100) / (2*100) = 110
        assert abs(prices[1] - 110.0) < 1e-6

    def test_env_reset_and_step(self, make_trades, make_orderbook, make_funding):
        from prepare import TradingEnv

        features, _, prices = compute_features(
            make_trades(n=5000),
            make_orderbook(n=1000),
            make_funding(n=100),
            trade_batch=100,
        )
        features = normalize_features(features)
        env = TradingEnv(features, prices, window_size=10)
        obs, _ = env.reset()
        assert obs.shape == (10, NUM_FEATURES_V3)
        obs, _, done, truncated, info = env.step(1)
        assert obs.shape == (10, NUM_FEATURES_V3)
        assert "step_pnl" in info

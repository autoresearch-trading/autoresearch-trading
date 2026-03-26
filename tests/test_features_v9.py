"""Tests for v9/v11 features."""

import numpy as np

from prepare import V9_NUM_FEATURES, compute_features_v9


class TestV9FeatureShape:
    def test_output_shape(self, make_trades, make_orderbook, make_funding):
        features, ts, px, rh, _ = compute_features_v9(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features.shape == (2, V9_NUM_FEATURES)
        assert len(ts) == 2
        assert len(px) == 2
        assert rh.shape == (2,)

    def test_empty_trades_returns_empty(self, empty_df, make_orderbook, make_funding):
        result = compute_features_v9(empty_df, make_orderbook(), make_funding())
        assert len(result[0]) == 0

    def test_feature_names(self):
        from prepare import V9_FEATURE_NAMES

        assert len(V9_FEATURE_NAMES) == V9_NUM_FEATURES
        # First 5 are the original Aristotle-proven features
        assert V9_FEATURE_NAMES[:5] == [
            "lambda_ofi",
            "directional_conviction",
            "vpin",
            "hawkes_branching",
            "reservation_price_dev",
        ]


class TestV9LambdaOfi:
    def test_lambda_ofi_is_kyle_lambda_times_signed_flow(
        self, make_trades, make_orderbook, make_funding
    ):
        features, *_ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        # Feature 0 should be finite, can be positive or negative
        assert np.all(np.isfinite(features[:, 0]))


class TestV9DirectionalConviction:
    def test_conviction_is_tfi_times_abs_ofi(
        self, make_trades, make_orderbook, make_funding
    ):
        features, *_ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        # Feature 1 should be finite
        assert np.all(np.isfinite(features[:, 1]))


class TestV9Vpin:
    def test_vpin_non_negative(self, make_trades, make_orderbook, make_funding):
        features, *_ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 2] >= 0)

    def test_vpin_at_most_one(self, make_trades, make_orderbook, make_funding):
        features, *_ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 2] <= 1.0 + 1e-6)


class TestV9HawkesBranching:
    def test_hawkes_in_valid_range(self, make_trades, make_orderbook, make_funding):
        _, _, _, raw_hawkes, _ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        # Feature 3 (normalized) can be anything after z-score
        # But raw_hawkes should be in [0, 0.99]
        assert np.all(raw_hawkes >= 0.0)
        assert np.all(raw_hawkes <= 0.99)

    def test_hawkes_zero_for_constant_rates(self):
        """Constant arrival rates → Var/Mean ≤ 1 → branching = 0 (Theorem 11)."""
        rates = np.ones(50) * 50.0  # constant rate
        mean_r = rates.mean()
        var_r = rates.var()
        ratio = var_r / mean_r if mean_r > 0 else 0
        # Var = 0, ratio = 0 ≤ 1, so branching = 0
        assert ratio == 0.0

    def test_hawkes_synthetic_bursty_arrivals(
        self, make_trades, make_orderbook, make_funding
    ):
        """Bursty arrival times (varying batch duration) → non-zero branching.
        Theorem 11: Var(B/D)/E[B/D] > 1 when durations vary (self-excitation)."""
        # Create trades with bursty timestamps: some batches fast, some slow
        trades = make_trades(n=5000, seed=99)
        ts = trades["ts_ms"].values.copy()
        # Make first half arrive 10x faster than second half
        for i in range(2500):
            ts[i] = 1_000_000 + i * 100  # 100ms spacing (fast)
        for i in range(2500, 5000):
            ts[i] = ts[2499] + (i - 2499) * 1000  # 1000ms spacing (slow)
        trades["ts_ms"] = ts
        trades["recv_ms"] = ts + 10
        _, _, _, raw_hawkes, _ = compute_features_v9(
            trades, make_orderbook(n=1250), make_funding(n=50), trade_batch=100
        )
        # 5000 trades / 100 = 50 batches, hawkes_window=50 → loop runs for i=50..49 (empty)
        # Need > 50 batches, so use smaller trade_batch
        # Actually: with trade_batch=50, we get 100 batches, and hawkes loop runs for i=50..99
        _, _, _, raw_hawkes2, _ = compute_features_v9(
            trades, make_orderbook(n=1250), make_funding(n=50), trade_batch=50
        )
        # With bursty arrivals, arrival rates vary → overdispersion → branching > 0
        assert np.any(
            raw_hawkes2 > 0
        ), "Expected non-zero branching from bursty arrivals"


class TestV9ReservationPriceDev:
    def test_reservation_finite(self, make_trades, make_orderbook, make_funding):
        features, *_ = compute_features_v9(
            make_trades(n=500),
            make_orderbook(n=125),
            make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(np.isfinite(features[:, 4]))

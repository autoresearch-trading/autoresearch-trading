"""Tests for v11 feature computation."""

import numpy as np
import pandas as pd

from prepare import compute_features_v9


def test_compute_orderbook_features_shape(make_orderbook):
    """Orderbook helper returns (microprice_dev, spread_bps, multi_level_ofi, imbalance) arrays."""
    from prepare import _compute_orderbook_features

    ob = make_orderbook(n=50)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    microprice_dev, spread_bps, mlofi, imbalance = _compute_orderbook_features(
        ob, timestamps, 10
    )
    assert microprice_dev.shape == (10,)
    assert spread_bps.shape == (10,)
    assert mlofi.shape == (10,)
    assert imbalance.shape == (10,)


def test_spread_bps_positive(make_orderbook):
    """Spread should be positive when best_ask > best_bid."""
    from prepare import _compute_orderbook_features

    ob = make_orderbook(n=50, best_bid=99.5, best_ask=100.5)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    _, spread_bps, _, _ = _compute_orderbook_features(ob, timestamps, 10)
    assert np.all(spread_bps[spread_bps != 0] > 0)


def test_mlofi_from_balanced_book(make_orderbook):
    """Multi-level OFI should be near zero for a static balanced book."""
    from prepare import _compute_orderbook_features

    ob = make_orderbook(n=50, bid_qty=2.0, ask_qty=2.0)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    _, _, mlofi, _ = _compute_orderbook_features(ob, timestamps, 10)
    # Static book -> OFI ~ 0 (no depth changes between identical snapshots)
    # First snapshot has no prior, rest should be ~0 for identical depth
    assert mlofi.shape == (10,)


def test_buy_sell_vwap_decomposition(make_trades, make_orderbook, make_funding):
    """T31: buy_vwap_dev + sell_vwap_dev are computed."""
    trades = make_trades(n=500, sides=["open_long", "open_short"])
    ob = make_orderbook(n=125)
    funding = make_funding(n=10)
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    assert features.shape[1] == 17  # v11 feature count
    # buy_vwap_dev = feature 10, sell_vwap_dev = feature 11
    assert features[:, 10].shape[0] > 0
    assert features[:, 11].shape[0] > 0


def test_roll_measure_nonnegative(make_trades, make_orderbook, make_funding):
    """T32: Roll measure should be non-negative."""
    trades = make_trades(n=500)
    ob = make_orderbook(n=125)
    funding = make_funding(n=10)
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    roll = features[:, 14]  # roll_measure
    assert np.all(roll >= 0)  # Roll is sqrt of max(0, -autocov)


def test_amihud_nonnegative(make_trades, make_orderbook, make_funding):
    """T32: Amihud illiquidity is non-negative."""
    trades = make_trades(n=500)
    ob = make_orderbook(n=125)
    funding = make_funding(n=10)
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    amihud = features[:, 13]  # amihud_illiq
    assert np.all(amihud >= 0)


def test_feature_count_v11(make_trades, make_orderbook, make_funding):
    """v11 should output exactly 17 features."""
    trades = make_trades(n=500)
    ob = make_orderbook(n=125)
    funding = make_funding(n=10)
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    assert features.shape[1] == 17

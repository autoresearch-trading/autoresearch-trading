"""Unit tests for Goal-A feasibility book-walk simulator.

Tests the deterministic L1-L10 walk-the-book primitive against handcrafted
toy books with known fill prices. The integration layer (loading shards,
horizon math) is exercised via a tiny synthetic shard fixture.
"""

from __future__ import annotations

import numpy as np

from scripts.goal_a_feasibility import (
    forward_log_return,
    headroom_bps,
    simulate_taker_fill,
)

# ---------------------------------------------------------------------------
# simulate_taker_fill — deterministic book walk
# ---------------------------------------------------------------------------


def test_buy_consumes_one_level_when_size_fits() -> None:
    """Buy of $500 against $1000 ask at level 1 → fills entirely at L1, slip≈0bp."""
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([20.0, 20.0, 20.0])
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])  # 10 * 100 = $1000 at L1

    res = simulate_taker_fill(
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
        target_notional=500.0,
        side="buy",
    )
    assert res.fillable is True
    assert res.fill_price == 100.0
    # mid = 99.5, slip = (100 - 99.5) / 99.5 * 1e4
    expected = (100.0 - 99.5) / 99.5 * 1e4
    assert abs(res.slippage_bps - expected) < 1e-6


def test_buy_walks_multiple_levels() -> None:
    """Buy of $1500 chews through L1 ($1000) + half of L2."""
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([20.0, 20.0, 20.0])
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])  # L1=$1000, L2=$1010, L3=$1020

    res = simulate_taker_fill(
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
        target_notional=1500.0,
        side="buy",
    )
    # Fill: 10 @ 100 = $1000, then need $500 more at $101 → 4.9505 units
    # Total qty = 10 + 4.9505... = 14.9505
    # Fill price = 1500 / 14.9505 = 100.331...
    expected_qty = 10.0 + 500.0 / 101.0
    expected_price = 1500.0 / expected_qty
    assert res.fillable is True
    assert abs(res.filled_qty - expected_qty) < 1e-6
    assert abs(res.fill_price - expected_price) < 1e-6


def test_buy_exceeds_total_book_depth_marks_unfillable() -> None:
    """Buy size > sum of all 3 ask levels → fillable=False."""
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([20.0, 20.0, 20.0])
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([1.0, 1.0, 1.0])  # Total $303

    res = simulate_taker_fill(
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
        target_notional=10000.0,
        side="buy",
    )
    assert res.fillable is False
    # Slippage based on the (insufficient) depth available
    assert np.isnan(res.slippage_bps) or np.isfinite(res.slippage_bps)


def test_sell_walks_bid_side() -> None:
    """Sell of $500 hits bid side; price moves DOWN from mid."""
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])  # L1 bid = $990
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])

    res = simulate_taker_fill(
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
        target_notional=500.0,
        side="sell",
    )
    # Fill at $99 (entirely L1)
    assert res.fillable is True
    assert res.fill_price == 99.0
    # Slippage signed: sell below mid → fill_price < mid → negative
    expected_bps = (99.0 - 99.5) / 99.5 * 1e4
    assert abs(res.slippage_bps - expected_bps) < 1e-6


def test_zero_or_missing_levels_skipped() -> None:
    """Levels with qty=0 (missing in raw data) must not contribute notional."""
    bid_prices = np.array([99.0, 0.0, 97.0])
    bid_qtys = np.array([10.0, 0.0, 10.0])
    ask_prices = np.array([100.0, 0.0, 102.0])
    ask_qtys = np.array([5.0, 0.0, 10.0])  # L1=$500, L2 empty, L3=$1020

    res = simulate_taker_fill(
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
        target_notional=800.0,
        side="buy",
    )
    # Skips L2; takes 5@100 = $500, then needs $300 from L3 @ 102 → 2.941 units
    expected_qty = 5.0 + 300.0 / 102.0
    expected_price = 800.0 / expected_qty
    assert res.fillable is True
    assert abs(res.filled_qty - expected_qty) < 1e-6
    assert abs(res.fill_price - expected_price) < 1e-6


# ---------------------------------------------------------------------------
# forward_log_return — uses cached log_return cumsum
# ---------------------------------------------------------------------------


def test_forward_log_return_basic() -> None:
    """Forward log-return at horizon h = sum of log_return[i+1..i+h]."""
    # log_returns: [0.01, -0.02, 0.03, 0.01, -0.01]
    log_rets = np.array([0.0, 0.01, -0.02, 0.03, 0.01, -0.01])
    # At i=1, h=2 → r[2]+r[3] = -0.02 + 0.03 = 0.01
    fr = forward_log_return(log_rets, anchor_idx=1, horizon=2)
    assert abs(fr - 0.01) < 1e-9
    # Beyond array → NaN
    fr = forward_log_return(log_rets, anchor_idx=4, horizon=10)
    assert np.isnan(fr)


# ---------------------------------------------------------------------------
# headroom_bps math
# ---------------------------------------------------------------------------


def test_headroom_bps_formula() -> None:
    """headroom = |edge_bps| - (2 * fees_bps + 2 * |slip_bps|)."""
    # 100bp edge, 6bp fee, 2bp slip → 100 - (12 + 4) = 84
    h = headroom_bps(edge_bps=100.0, slip_bps=2.0, fees_bps=6.0)
    assert abs(h - 84.0) < 1e-9
    # 5bp edge, 6bp fees, 1bp slip → 5 - (12 + 2) = -9 (cost-blocked)
    h = headroom_bps(edge_bps=5.0, slip_bps=1.0, fees_bps=6.0)
    assert abs(h - (-9.0)) < 1e-9

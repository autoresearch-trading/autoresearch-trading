"""Unit tests for Goal-A feasibility book-walk simulator.

Tests the deterministic L1-L10 walk-the-book primitive against handcrafted
toy books with known fill prices. The integration layer (loading shards,
horizon math) is exercised via a tiny synthetic shard fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.goal_a_feasibility import (
    add_accuracy_stress_columns,
    aggregate_cells,
    forward_log_return,
    headroom_at_accuracy_bps,
    headroom_bps,
    simulate_taker_fill,
    survivors_for_accuracy,
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


# ---------------------------------------------------------------------------
# headroom_at_accuracy_bps math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "accuracy, edge, slip, fees, expected",
    [
        # p=0.5 → no signal, headroom = -cost = -(2*6 + 2*2) = -16
        (0.5, 100.0, 2.0, 6.0, -16.0),
        # p=1.0 → recovers oracle headroom: 100 - 16 = 84
        (1.0, 100.0, 2.0, 6.0, 84.0),
        # p=0.55 → 0.10 * 100 - 16 = -6
        (0.55, 100.0, 2.0, 6.0, -6.0),
        # p=0.575 → 0.15 * 100 - 16 = -1
        (0.575, 100.0, 2.0, 6.0, -1.0),
        # p=0.60 → 0.20 * 100 - 16 = 4
        (0.60, 100.0, 2.0, 6.0, 4.0),
        # Negative-edge input: |edge| is taken so sign of edge_bps doesn't matter
        (0.55, -100.0, 2.0, 6.0, -6.0),
        # Slip sign also doesn't matter (cost uses |slip|)
        (0.60, 100.0, -2.0, 6.0, 4.0),
        # Zero-slip, zero-fees, p=0.55, edge=20 → (2*0.55-1)*20 = 2.0
        (0.55, 20.0, 0.0, 0.0, 2.0),
    ],
)
def test_headroom_at_accuracy_bps(
    accuracy: float, edge: float, slip: float, fees: float, expected: float
) -> None:
    h = headroom_at_accuracy_bps(
        edge_bps=edge, slip_bps=slip, fees_bps=fees, accuracy=accuracy
    )
    assert abs(h - expected) < 1e-9, (
        f"acc={accuracy}, edge={edge}, slip={slip}, fees={fees}: "
        f"got {h}, want {expected}"
    )


# ---------------------------------------------------------------------------
# add_accuracy_stress_columns: parquet-level vectorised application
# ---------------------------------------------------------------------------


def test_add_accuracy_stress_columns_round_trip() -> None:
    """Vectorised stress columns must equal scalar headroom_at_accuracy_bps."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC", "ETH", "PUMP"],
            "size_usd": [1000.0, 1000.0, 1000.0],
            "horizon": [500, 500, 500],
            "fillable": [True, True, True],
            "slip_avg_bps": [0.5, 1.0, 3.0],
            "edge_bps": [20.0, 30.0, 100.0],
            "headroom_bps": [7.0, 16.0, 82.0],
        }
    )
    out = add_accuracy_stress_columns(df)
    # 0.55, 0.575, 0.60 columns must be present
    for suffix in ("_55", "_575", "_60"):
        assert f"headroom_acc{suffix}_bps" in out.columns
    # Spot-check PUMP at 0.60: (0.20 * 100) - (12 + 6) = 20 - 18 = 2
    assert abs(float(out.loc[2, "headroom_acc_60_bps"]) - 2.0) < 1e-9
    # ETH at 0.55: (0.10 * 30) - (12 + 2) = 3 - 14 = -11
    assert abs(float(out.loc[1, "headroom_acc_55_bps"]) - (-11.0)) < 1e-9
    # NaN propagation: introduce a NaN edge → all stressed cols NaN
    df2 = df.copy()
    df2.loc[0, "edge_bps"] = np.nan
    out2 = add_accuracy_stress_columns(df2)
    assert np.isnan(out2.loc[0, "headroom_acc_55_bps"])
    assert np.isnan(out2.loc[0, "headroom_acc_575_bps"])
    assert np.isnan(out2.loc[0, "headroom_acc_60_bps"])


# ---------------------------------------------------------------------------
# aggregate_cells: per-accuracy stats appear when stress cols present
# ---------------------------------------------------------------------------


def test_aggregate_cells_emits_per_accuracy_stats() -> None:
    """One symbol/size/horizon cell, 4 windows: verify per-acc stats present."""
    df = pd.DataFrame(
        {
            "symbol": ["X"] * 4,
            "size_usd": [1000.0] * 4,
            "horizon": [500] * 4,
            "fillable": [True, True, True, True],
            "slip_avg_bps": [1.0, 1.0, 1.0, 1.0],
            "edge_bps": [10.0, 50.0, 100.0, 200.0],
            "headroom_bps": [-4.0, 36.0, 86.0, 186.0],  # |edge| - 14
        }
    )
    df = add_accuracy_stress_columns(df)
    summary = aggregate_cells(df)
    assert len(summary) == 1
    # At p=0.60: stressed = 0.20*|edge| - 14 = [-12, -4, 6, 26]
    # median = (-4 + 6)/2 = 1.0; frac_pos = 2/4 = 0.5
    row = summary.iloc[0]
    assert abs(float(row["headroom_60_median_bps"]) - 1.0) < 1e-9
    assert abs(float(row["frac_pos_acc_60"]) - 0.5) < 1e-9
    # At p=0.55: stressed = 0.10*|edge| - 14 = [-13, -9, -4, 6]
    # median = (-9 + -4)/2 = -6.5; frac_pos = 1/4 = 0.25
    assert abs(float(row["headroom_55_median_bps"]) - (-6.5)) < 1e-9
    assert abs(float(row["frac_pos_acc_55"]) - 0.25) < 1e-9
    # Oracle columns must still be present
    assert "headroom_median_bps" in summary.columns
    assert "frac_positive_headroom" in summary.columns


# ---------------------------------------------------------------------------
# survivors_for_accuracy: filter logic
# ---------------------------------------------------------------------------


def test_survivors_for_accuracy_filters_correctly() -> None:
    """Survivor must satisfy frac_pos > 0.55 AND headroom_median > 0."""
    cell_df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "size_usd": [1000.0] * 4,
            "horizon": [500] * 4,
            "n_windows": [100] * 4,
            "n_fillable_with_edge": [100] * 4,
            "fillable_frac": [1.0] * 4,
            "edge_median_bps": [50.0] * 4,
            "slip_median_bps": [0.5] * 4,
            # Survives: frac_pos=0.7, headroom_median=2.5
            # Fails frac_pos: frac_pos=0.4, headroom_median=2.0
            # Fails headroom: frac_pos=0.7, headroom_median=-1.0
            # Fails both: frac_pos=0.3, headroom_median=-5.0
            "headroom_60_median_bps": [2.5, 2.0, -1.0, -5.0],
            "headroom_60_p75_bps": [10.0, 8.0, 1.0, -3.0],
            "headroom_60_p90_bps": [20.0, 15.0, 5.0, 0.0],
            "frac_pos_acc_60": [0.7, 0.4, 0.7, 0.3],
        }
    )
    out = survivors_for_accuracy(cell_df, accuracy=0.60)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "A"


def test_survivors_for_accuracy_handles_missing_columns() -> None:
    """If the stress columns are missing, returns empty df (no crash)."""
    cell_df = pd.DataFrame(
        {
            "symbol": ["A"],
            "size_usd": [1000.0],
            "horizon": [500],
            "n_windows": [100],
            "fillable_frac": [1.0],
        }
    )
    out = survivors_for_accuracy(cell_df, accuracy=0.60)
    assert out.empty

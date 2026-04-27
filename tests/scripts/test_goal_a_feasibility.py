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
    add_maker_headroom_columns,
    aggregate_adverse_selection,
    aggregate_cells,
    aggregate_open_imbalance_cells,
    best_bid_ask_from_levels,
    compute_event_open_flow_qty,
    compute_maker_sensitivity_table,
    compute_open_imbalance_per_event,
    detect_fill_in_range,
    forward_log_return,
    headroom_at_accuracy_bps,
    headroom_bps,
    maker_headroom_at_accuracy_bps,
    model_accuracy_breakeven,
    rolling_quantile_causal,
    simulate_taker_fill,
    survivors_for_accuracy,
)
from tape.dedup import dedup_trades_pre_april

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


# ---------------------------------------------------------------------------
# maker_headroom_at_accuracy_bps math (no slippage; signed maker fee)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "accuracy, edge, maker_fee, expected",
    [
        # zero-fee zero-rebate: cost=0 → headroom = (2p-1)*|edge|
        (0.55, 100.0, 0.0, 10.0),
        (0.575, 100.0, 0.0, 15.0),
        (0.60, 100.0, 0.0, 20.0),
        # pure rebate: maker_fee = -1.0 bp → cost = -2 bp; headroom = gross + 2
        (0.55, 100.0, -1.0, 10.0 + 2.0),
        # pure fee: maker_fee = +3.0 bp → cost = +6 bp; headroom = gross - 6
        (0.60, 100.0, 3.0, 20.0 - 6.0),
        # p=1.0 perfect oracle, zero fee: headroom = |edge|
        (1.0, 50.0, 0.0, 50.0),
        # negative edge magnitude handled (uses |edge|)
        (0.60, -100.0, 3.0, 20.0 - 6.0),
        # rebate larger than gross signal → headroom positive even at p=0.5
        (0.5, 100.0, -5.0, 0.0 + 10.0),
    ],
)
def test_maker_headroom_at_accuracy_bps(
    accuracy: float, edge: float, maker_fee: float, expected: float
) -> None:
    h = maker_headroom_at_accuracy_bps(
        edge_bps=edge, maker_fee_bps=maker_fee, accuracy=accuracy
    )
    assert abs(h - expected) < 1e-9, (
        f"acc={accuracy}, edge={edge}, maker_fee={maker_fee}: "
        f"got {h}, want {expected}"
    )


# ---------------------------------------------------------------------------
# add_maker_headroom_columns: vectorised maker-mode columns
# ---------------------------------------------------------------------------


def test_add_maker_headroom_columns_basic() -> None:
    """Vectorised maker headroom columns must equal the scalar function.

    Also checks the fill_proxy column: True iff |edge_bps| >= 1.0.
    """
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "size_usd": [1000.0] * 3,
            "horizon": [500] * 3,
            "fillable": [True, True, True],
            "slip_avg_bps": [0.5, 1.0, 3.0],
            "edge_bps": [0.4, 30.0, 100.0],  # row 0: edge<1bp → fill_proxy False
        }
    )
    out = add_maker_headroom_columns(
        df, accuracies=(0.55, 0.60), maker_fees_bps=(-1.0, 0.0, 3.0)
    )
    # Required columns
    for fee in ("-1", "0", "3"):
        assert f"maker_fill_proxy_bool" in out.columns
        for acc in ("55", "60"):
            col = f"maker_headroom_acc_{acc}_fee_{fee}_bps"
            assert col in out.columns
    # Row 0: edge=0.4 → |edge| < 1bp → fill_proxy False
    assert bool(out.loc[0, "maker_fill_proxy_bool"]) is False
    # Row 1: edge=30 → fill_proxy True
    assert bool(out.loc[1, "maker_fill_proxy_bool"]) is True
    # Row 2 at acc=0.60, maker_fee=3.0: gross = 0.20*100 = 20, cost=6, headroom=14
    assert abs(float(out.loc[2, "maker_headroom_acc_60_fee_3_bps"]) - 14.0) < 1e-9
    # Row 1 at acc=0.55, maker_fee=-1.0 (rebate): gross=3, cost=-2, headroom=5
    assert abs(float(out.loc[1, "maker_headroom_acc_55_fee_-1_bps"]) - 5.0) < 1e-9


def test_add_maker_headroom_columns_nan_propagates() -> None:
    """NaN edge_bps → NaN maker headroom; fill_proxy is False (not NaN)."""
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "size_usd": [1000.0],
            "horizon": [500],
            "fillable": [True],
            "slip_avg_bps": [1.0],
            "edge_bps": [float("nan")],
        }
    )
    out = add_maker_headroom_columns(df, accuracies=(0.60,), maker_fees_bps=(0.0,))
    assert np.isnan(out.loc[0, "maker_headroom_acc_60_fee_0_bps"])
    # NaN edge → fill_proxy is False (treat as not-fillable)
    assert bool(out.loc[0, "maker_fill_proxy_bool"]) is False


# ---------------------------------------------------------------------------
# compute_maker_sensitivity_table: end-to-end on a tiny per-window df
# ---------------------------------------------------------------------------


def test_compute_maker_sensitivity_table_shape_and_breakeven() -> None:
    """Two cells × two windows each.  At maker_fee=0, accuracy=0.60:

      - cell A: edge_med=100 → gross=20 → cost=0 → headroom=20 (alive)
      - cell B: edge_med=10  → gross=2  → cost=0 → headroom=2  (alive)

    At maker_fee=+3.0, accuracy=0.60:
      - cell A: gross=20 - cost=6 → 14 (alive)
      - cell B: gross=2  - cost=6 → -4 (dead)
    """
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "size_usd": [1000.0] * 4,
            "horizon": [500] * 4,
            "fillable": [True] * 4,
            "slip_avg_bps": [1.0] * 4,
            "edge_bps": [100.0, 100.0, 10.0, 10.0],
        }
    )
    sens = compute_maker_sensitivity_table(
        df,
        maker_fees_bps=(0.0, 3.0),
        accuracies=(0.60,),
    )
    # Shape: 2 fees × 1 accuracy = 2 rows
    assert len(sens) == 2
    # At fee=0: both cells alive
    row_zero = sens.loc[
        (sens["maker_fee_bps"] == 0.0) & (sens["accuracy"] == 0.60)
    ].iloc[0]
    assert int(row_zero["n_cells_alive"]) == 2
    # At fee=3: only cell A alive
    row_three = sens.loc[
        (sens["maker_fee_bps"] == 3.0) & (sens["accuracy"] == 0.60)
    ].iloc[0]
    assert int(row_three["n_cells_alive"]) == 1
    # Both rows must include the with-fill-proxy count (edge=100 and 10 are >1bp)
    assert int(row_zero["n_cells_alive_with_fill_proxy"]) == 2
    assert int(row_three["n_cells_alive_with_fill_proxy"]) == 1
    # top_5_cells_by_median_headroom is a string field with comma-joined cells
    assert isinstance(row_zero["top_5_cells_by_median_headroom"], str)
    assert "A" in row_zero["top_5_cells_by_median_headroom"]


def test_compute_maker_sensitivity_table_fill_proxy_filters() -> None:
    """Cell with edge < 1 bp is alive on raw headroom but pruned by fill proxy."""
    # edge=0.4 → |edge| < 1bp → fill_proxy False on every row
    # At maker_fee = -5 (rebate of 5 bp/leg): cost=-10. gross at p=0.60 = 0.08
    # headroom = 0.08 + 10 = 10.08 (alive on cost)
    df = pd.DataFrame(
        {
            "symbol": ["X"] * 3,
            "size_usd": [1000.0] * 3,
            "horizon": [500] * 3,
            "fillable": [True] * 3,
            "slip_avg_bps": [0.0] * 3,
            "edge_bps": [0.4, 0.4, 0.4],
        }
    )
    sens = compute_maker_sensitivity_table(
        df, maker_fees_bps=(-5.0,), accuracies=(0.60,)
    )
    row = sens.iloc[0]
    # Cell is "alive on headroom" but fill_proxy is False (edge < 1bp every window)
    # So n_cells_alive >= 1, but n_cells_alive_with_fill_proxy == 0
    assert int(row["n_cells_alive"]) == 1
    assert int(row["n_cells_alive_with_fill_proxy"]) == 0


# ---------------------------------------------------------------------------
# Adverse-selection sim — fill detection on a synthetic OB grid
# ---------------------------------------------------------------------------


def test_best_bid_ask_from_levels_basic() -> None:
    """L1 IS the best by parquet convention — confirm the helper picks it."""
    bp = np.array([99.0, 98.5, 98.0])
    bq = np.array([10.0, 10.0, 10.0])
    ap = np.array([100.0, 100.5, 101.0])
    aq = np.array([10.0, 10.0, 10.0])
    bb, ba = best_bid_ask_from_levels(bp, bq, ap, aq)
    assert bb == 99.0
    assert ba == 100.0


def test_best_bid_ask_from_levels_skips_missing_l1() -> None:
    """If L1 has qty=0, fall through to L2."""
    bp = np.array([99.0, 98.5, 98.0])
    bq = np.array([0.0, 10.0, 10.0])  # L1 missing on bid
    ap = np.array([100.0, 100.5, 101.0])
    aq = np.array([10.0, 10.0, 10.0])
    bb, ba = best_bid_ask_from_levels(bp, bq, ap, aq)
    assert bb == 98.5  # falls through to L2
    assert ba == 100.0


def test_best_bid_ask_from_levels_one_sided_book() -> None:
    """Empty bid side → returns NaN."""
    bp = np.array([0.0, 0.0, 0.0])
    bq = np.array([0.0, 0.0, 0.0])
    ap = np.array([100.0, 100.5, 101.0])
    aq = np.array([10.0, 10.0, 10.0])
    bb, ba = best_bid_ask_from_levels(bp, bq, ap, aq)
    assert np.isnan(bb)
    assert np.isnan(ba)


def test_bid_fills_when_ask_crosses_through() -> None:
    """A snapshot in [anchor_ts, horizon_end_ts] with best_ask <= bid_price
    triggers a bid fill. Symmetric for the ask side."""
    # OB grid: 5 snapshots at ts = [0, 100, 200, 300, 400]
    snap_ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    # best_ask drops to 99.5 at ts=200 — this should fill a bid posted at 99.5
    snap_best_bids = np.array([99.0, 99.0, 99.0, 99.0, 99.0])
    snap_best_asks = np.array([100.0, 100.0, 99.5, 100.0, 100.0])

    # Anchor at ts=50, horizon end at ts=350 → range covers ts=100, 200, 300
    bid_filled, ask_filled = detect_fill_in_range(
        snap_best_bids=snap_best_bids,
        snap_best_asks=snap_best_asks,
        snap_ts=snap_ts,
        anchor_ts=50,
        horizon_end_ts=350,
        bid_price=99.5,  # ask drops to 99.5 at ts=200 — should fill
        ask_price=100.5,  # bid never reaches 100.5 — no fill
    )
    assert bid_filled is True
    assert ask_filled is False


def test_no_fill_when_book_stays_above_bid() -> None:
    """If best_ask never <= bid_price in range, bid does not fill."""
    snap_ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    snap_best_bids = np.array([99.0, 99.0, 99.0, 99.0, 99.0])
    snap_best_asks = np.array([100.0, 100.1, 100.2, 100.1, 100.0])  # never <= 99.5

    bid_filled, ask_filled = detect_fill_in_range(
        snap_best_bids=snap_best_bids,
        snap_best_asks=snap_best_asks,
        snap_ts=snap_ts,
        anchor_ts=50,
        horizon_end_ts=350,
        bid_price=99.5,
        ask_price=100.5,
    )
    assert bid_filled is False
    assert ask_filled is False


def test_ask_fills_when_bid_crosses_up() -> None:
    """Symmetric to the bid case: best_bid >= ask_price triggers ask fill."""
    snap_ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    # best_bid spikes to 100.5 at ts=200 — fills an ask posted at 100.5
    snap_best_bids = np.array([99.0, 99.0, 100.5, 99.0, 99.0])
    snap_best_asks = np.array([100.0, 100.0, 101.0, 100.0, 100.0])

    bid_filled, ask_filled = detect_fill_in_range(
        snap_best_bids=snap_best_bids,
        snap_best_asks=snap_best_asks,
        snap_ts=snap_ts,
        anchor_ts=50,
        horizon_end_ts=350,
        bid_price=98.5,  # ask never drops that low → no bid fill
        ask_price=100.5,
    )
    assert bid_filled is False
    assert ask_filled is True


def test_no_fill_when_horizon_window_contains_no_snapshots() -> None:
    """If the snapshot grid has no entries in [anchor, horizon_end], no fill."""
    snap_ts = np.array([0, 1000, 2000], dtype=np.int64)
    snap_best_bids = np.array([99.0, 99.0, 99.0])
    snap_best_asks = np.array([100.0, 100.0, 100.0])
    # Anchor at ts=100, horizon_end at ts=900 → no snapshots in range
    bid_filled, ask_filled = detect_fill_in_range(
        snap_best_bids=snap_best_bids,
        snap_best_asks=snap_best_asks,
        snap_ts=snap_ts,
        anchor_ts=100,
        horizon_end_ts=900,
        bid_price=99.99,  # at this price the L1 snapshot WOULD fill, but
        ask_price=100.01,  # the snapshot at ts=0 is BEFORE the range
    )
    assert bid_filled is False
    assert ask_filled is False


def test_nan_snapshots_treated_as_no_cross() -> None:
    """NaN best bid/ask in the range must not trigger spurious fills."""
    snap_ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    snap_best_bids = np.array([99.0, 99.0, np.nan, 99.0, 99.0])
    snap_best_asks = np.array([100.0, 100.0, np.nan, 100.0, 100.0])

    bid_filled, ask_filled = detect_fill_in_range(
        snap_best_bids=snap_best_bids,
        snap_best_asks=snap_best_asks,
        snap_ts=snap_ts,
        anchor_ts=150,
        horizon_end_ts=250,  # only the NaN snapshot at ts=200 is in range
        bid_price=99.99,
        ask_price=100.01,
    )
    assert bid_filled is False
    assert ask_filled is False


# ---------------------------------------------------------------------------
# model_accuracy_breakeven math
# ---------------------------------------------------------------------------


def test_breakeven_accuracy_zero_realized_pnl_equals_50_percent() -> None:
    """When E[realized | filled] = 0 AND maker_fee = 0, breakeven = 0.5."""
    p = model_accuracy_breakeven(
        expected_realized_pnl_bps=0.0, maker_fee_bps_per_side=0.0
    )
    assert abs(p - 0.5) < 1e-12


def test_breakeven_zero_pnl_nonzero_fee_is_unreachable() -> None:
    """E=0 with positive fee → +inf (no accuracy can break even)."""
    p = model_accuracy_breakeven(
        expected_realized_pnl_bps=0.0, maker_fee_bps_per_side=1.5
    )
    assert p == float("inf")


def test_breakeven_positive_pnl_below_50_when_fee_negative() -> None:
    """A rebate (fee < 0) and positive E should give breakeven < 0.5."""
    # fee = -1, E = 100 → breakeven = 0.5 + (-1)/100 = 0.49
    p = model_accuracy_breakeven(
        expected_realized_pnl_bps=100.0, maker_fee_bps_per_side=-1.0
    )
    assert abs(p - 0.49) < 1e-12


def test_breakeven_positive_pnl_with_pacifica_fee() -> None:
    """At the actual Pacifica maker fee +1.5 bp/side and E=10 bp,
    breakeven = 0.5 + 1.5/10 = 0.65."""
    p = model_accuracy_breakeven(
        expected_realized_pnl_bps=10.0, maker_fee_bps_per_side=1.5
    )
    assert abs(p - 0.65) < 1e-12


def test_breakeven_negative_pnl_below_50_signals_dilemma() -> None:
    """Maker's Dilemma: E < 0 → breakeven < 0.5 (need to be contrarian)."""
    # E = -10, fee = +1.5 → breakeven = 0.5 + 1.5 / -10 = 0.35
    p = model_accuracy_breakeven(
        expected_realized_pnl_bps=-10.0, maker_fee_bps_per_side=1.5
    )
    assert abs(p - 0.35) < 1e-12


# ---------------------------------------------------------------------------
# aggregate_adverse_selection: end-to-end on a tiny per-window DataFrame
# ---------------------------------------------------------------------------


def test_aggregate_adverse_selection_basic_no_dilemma() -> None:
    """One cell, 4 windows, all bids fill with E[realized] = +20 bp.
    With maker_fee = 1.5: breakeven = 0.5 + 1.5/20 = 0.575."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 4,
            "horizon": [100] * 4,
            "offset_bps": [2.0] * 4,
            "bid_filled": [True, True, True, True],
            "ask_filled": [False, False, False, False],
            "realized_bid_pnl_bps": [10.0, 20.0, 30.0, 20.0],
            "realized_ask_pnl_bps": [float("nan")] * 4,
        }
    )
    out = aggregate_adverse_selection(df, maker_fee_bps_per_side=1.5)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["fill_rate_bid"] == 1.0
    assert row["fill_rate_ask"] == 0.0
    assert abs(float(row["mean_pnl_bid_filled"]) - 20.0) < 1e-9
    # mean_pnl_either_filled aggregates bid_pnl + ask_pnl finite-only
    # → only bid fills → mean = 20
    assert abs(float(row["mean_pnl_either_filled"]) - 20.0) < 1e-9
    # breakeven = 0.5 + 1.5 / 20 = 0.575
    assert abs(float(row["model_accuracy_breakeven"]) - 0.575) < 1e-9


def test_aggregate_adverse_selection_dilemma_negative_e() -> None:
    """All filled bids precede a drop → E[realized | filled] < 0 → breakeven < 0.5."""
    df = pd.DataFrame(
        {
            "symbol": ["X"] * 3,
            "horizon": [100] * 3,
            "offset_bps": [2.0] * 3,
            "bid_filled": [True, True, True],
            "ask_filled": [False, False, False],
            "realized_bid_pnl_bps": [-5.0, -10.0, -15.0],
            "realized_ask_pnl_bps": [float("nan")] * 3,
        }
    )
    out = aggregate_adverse_selection(df, maker_fee_bps_per_side=1.5)
    row = out.iloc[0]
    assert float(row["mean_pnl_bid_filled"]) == -10.0
    # Breakeven = 0.5 + 1.5 / -10 = 0.35 → contrarian zone
    assert abs(float(row["model_accuracy_breakeven"]) - 0.35) < 1e-9


# ---------------------------------------------------------------------------
# Open-imbalance feasibility — dedup correctness
# ---------------------------------------------------------------------------


def test_dedup_pre_april_collapses_buyer_seller_pairs() -> None:
    """Pre-April raw data has buyer + seller rows for every fill sharing
    (ts_ms, qty, price) but differing in `side`. Dedup must collapse these
    to one row per fill — gotcha #19. WITHOUT this collapse, every imbalance
    is spurious because both perspectives are counted."""
    raw = pd.DataFrame(
        {
            "ts_ms": [1000, 1000, 2000, 2000, 3000, 3000],
            "qty": [10.0, 10.0, 5.0, 5.0, 3.0, 3.0],
            "price": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "side": [
                "open_long",
                "close_short",
                "open_short",
                "close_long",
                "open_long",
                "close_short",
            ],
        }
    )
    deduped = dedup_trades_pre_april(raw)
    # 6 raw rows → 3 fills (each (ts, qty, price) tuple appears twice)
    assert len(deduped) == 3
    # The first row of each pair is preserved, so we keep the open_* sides
    assert sorted(deduped["side"].tolist()) == ["open_long", "open_long", "open_short"]


# ---------------------------------------------------------------------------
# compute_event_open_flow_qty — per-event aggregation by side
# ---------------------------------------------------------------------------


def test_compute_event_open_flow_qty_groups_by_ts_ms() -> None:
    """Same-ts_ms trades are fragments of one event. Aggregate qty by side.

    Two events at different ts_ms; the second has fragments on both sides.
    """
    deduped = pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 2000, 2000],
            "qty": [10.0, 3.0, 4.0, 2.0],
            "price": [100.0, 100.0, 100.0, 100.0],
            "side": ["open_long", "open_long", "close_long", "open_short"],
        }
    )
    out = compute_event_open_flow_qty(deduped)
    # Sorted by ts_ms; columns: open_long_qty, open_short_qty, close_long_qty,
    # close_short_qty (and ts_ms)
    assert out.shape[0] == 2
    row0 = out.iloc[0]
    assert int(row0["ts_ms"]) == 1000
    assert float(row0["open_long_qty"]) == 10.0
    assert float(row0["open_short_qty"]) == 0.0
    assert float(row0["close_long_qty"]) == 0.0
    assert float(row0["close_short_qty"]) == 0.0
    row1 = out.iloc[1]
    assert int(row1["ts_ms"]) == 2000
    assert float(row1["open_long_qty"]) == 3.0
    assert float(row1["open_short_qty"]) == 2.0
    assert float(row1["close_long_qty"]) == 4.0
    assert float(row1["close_short_qty"]) == 0.0


# ---------------------------------------------------------------------------
# compute_open_imbalance_per_event — open vs flow signal math
# ---------------------------------------------------------------------------


def test_open_imbalance_pure_long_returns_plus_one() -> None:
    """Only open_long fills → open_imbalance = +1."""
    aggr = pd.DataFrame(
        {
            "ts_ms": [1000],
            "open_long_qty": [10.0],
            "open_short_qty": [0.0],
            "close_long_qty": [0.0],
            "close_short_qty": [0.0],
        }
    )
    out = compute_open_imbalance_per_event(aggr)
    assert abs(float(out["open_imbalance"].iloc[0]) - 1.0) < 1e-9
    # No close flow → flow_imbalance also +1 (open_long is buy-side)
    assert abs(float(out["flow_imbalance"].iloc[0]) - 1.0) < 1e-9


def test_open_imbalance_pure_short_returns_minus_one() -> None:
    """Only open_short fills → open_imbalance = -1."""
    aggr = pd.DataFrame(
        {
            "ts_ms": [1000],
            "open_long_qty": [0.0],
            "open_short_qty": [5.0],
            "close_long_qty": [0.0],
            "close_short_qty": [0.0],
        }
    )
    out = compute_open_imbalance_per_event(aggr)
    assert abs(float(out["open_imbalance"].iloc[0]) - (-1.0)) < 1e-9
    assert abs(float(out["flow_imbalance"].iloc[0]) - (-1.0)) < 1e-9


def test_open_imbalance_decouples_from_flow() -> None:
    """Construct a case where open and flow imbalances point in OPPOSITE
    directions: lots of open_long but even more close_long (sells).

    open_imbalance: only opens count → +1 (only open_long present).
    flow_imbalance: buy = open_long, sell = open_short + close_long → strongly
                    negative because close_long sells dominate.
    """
    aggr = pd.DataFrame(
        {
            "ts_ms": [1000],
            "open_long_qty": [3.0],
            "open_short_qty": [0.0],
            "close_long_qty": [10.0],  # this is a sell (closing a long position)
            "close_short_qty": [0.0],
        }
    )
    out = compute_open_imbalance_per_event(aggr)
    # open: (3 - 0) / 3 = +1
    assert abs(float(out["open_imbalance"].iloc[0]) - 1.0) < 1e-9
    # flow: buy = 3 (open_long), sell = 10 (close_long); (3 - 10) / 13 = -0.538
    assert abs(float(out["flow_imbalance"].iloc[0]) - ((3.0 - 10.0) / 13.0)) < 1e-9


def test_open_imbalance_zero_denominator_returns_zero() -> None:
    """No open fills at all (all closes) → denominator zero → imbalance = 0,
    not NaN, not inf. Epsilon-guarded."""
    aggr = pd.DataFrame(
        {
            "ts_ms": [1000],
            "open_long_qty": [0.0],
            "open_short_qty": [0.0],
            "close_long_qty": [5.0],
            "close_short_qty": [3.0],
        }
    )
    out = compute_open_imbalance_per_event(aggr)
    assert float(out["open_imbalance"].iloc[0]) == 0.0
    # flow imbalance: buy = 3 (close_short closes a short = buying back),
    # sell = 5 (close_long); (3 - 5) / 8 = -0.25
    assert abs(float(out["flow_imbalance"].iloc[0]) - ((3.0 - 5.0) / 8.0)) < 1e-9


# ---------------------------------------------------------------------------
# Rolling causal quantile
# ---------------------------------------------------------------------------


def test_rolling_quantile_causal_uses_only_prior_values() -> None:
    """Rolling P95 of |x| at index i uses x[max(0,i-window+1):i+1] only.
    NO peeking at future values."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    out = rolling_quantile_causal(arr, window=5, q=0.95, min_periods=1)
    # At i=0: only 1.0 → P95 = 1.0
    assert abs(out[0] - 1.0) < 1e-9
    # At i=4 (window full = 5 elements: 1..5): P95 of [1,2,3,4,5] = ~4.8
    assert abs(out[4] - np.quantile(np.array([1, 2, 3, 4, 5]), 0.95)) < 1e-9
    # At i=9: window = [6,7,8,9,10] → P95 = 9.8
    assert abs(out[9] - np.quantile(np.array([6, 7, 8, 9, 10]), 0.95)) < 1e-9


def test_rolling_quantile_causal_min_periods_returns_nan() -> None:
    """When fewer than min_periods elements available, return NaN."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_quantile_causal(arr, window=3, q=0.95, min_periods=3)
    # First two indices have <3 elements → NaN
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    # Index 2 onward: 3 elements available → finite
    assert np.isfinite(out[2])


# ---------------------------------------------------------------------------
# Sign convention: aggregate_open_imbalance_cells
# ---------------------------------------------------------------------------


def test_aggregate_open_imbalance_positive_signal_positive_return() -> None:
    """Positive imbalance + positive return → positive signed return.
    Strategy goes long when imbalance > +rolling_p95; pays off when fwd_ret>0.
    """
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 4,
            "horizon": [100] * 4,
            "open_imbalance": [0.8, 0.8, 0.8, 0.8],
            "rolling_p95_open_abs": [0.5, 0.5, 0.5, 0.5],
            "fwd_log_return": [0.001, 0.002, 0.0015, 0.0005],  # all positive
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    assert len(out) == 1
    row = out.iloc[0]
    # All windows extreme (|0.8| > 0.5)
    assert int(row["n_extreme_windows"]) == 4
    # All signed returns positive
    assert float(row["frac_positive_signed_return_extreme"]) == 1.0
    # Mean signed return = mean fwd_return * 1e4 (positive direction)
    expected_mean_bps = float(np.mean([0.001, 0.002, 0.0015, 0.0005])) * 1e4
    assert abs(float(row["mean_signed_return_extreme"]) - expected_mean_bps) < 1e-6


def test_aggregate_open_imbalance_negative_signal_negative_return_positive_pnl() -> (
    None
):
    """When imbalance is in the NEGATIVE tail (signal < -p95), strategy goes
    SHORT. Negative fwd_return × short position = positive signed PnL."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 3,
            "horizon": [100] * 3,
            "open_imbalance": [-0.8, -0.9, -0.7],
            "rolling_p95_open_abs": [0.5, 0.5, 0.5],
            "fwd_log_return": [-0.001, -0.002, -0.0015],  # all negative
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    row = out.iloc[0]
    assert int(row["n_extreme_windows"]) == 3
    # Signed return = sign(imbalance) × fwd_return = (-1) × (-) = +
    assert float(row["frac_positive_signed_return_extreme"]) == 1.0
    # Mean signed return: mean(|fwd_return|) * 1e4
    expected = float(np.mean([0.001, 0.002, 0.0015])) * 1e4
    assert abs(float(row["mean_signed_return_extreme"]) - expected) < 1e-6


def test_aggregate_open_imbalance_below_cutoff_excluded() -> None:
    """Windows where |signal| <= rolling_p95 are NOT in the extreme regime —
    they should be excluded from the aggregation (n_extreme_windows counts
    only the tail)."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 5,
            "horizon": [100] * 5,
            "open_imbalance": [0.1, 0.2, 0.3, 0.8, 0.9],
            "rolling_p95_open_abs": [0.5] * 5,
            "fwd_log_return": [0.001] * 5,
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    row = out.iloc[0]
    # Only windows 3,4 are extreme (|0.8|, |0.9| > 0.5)
    assert int(row["n_extreme_windows"]) == 2
    assert int(row["n_total_windows"]) == 5
    assert abs(float(row["extreme_frequency"]) - 0.4) < 1e-9


def test_aggregate_open_imbalance_extreme_freq_math() -> None:
    """Sanity: 1 of 100 windows extreme → extreme_frequency = 0.01."""
    n = 100
    imb = np.zeros(n)
    imb[0] = 0.99  # one strong-long extreme
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "horizon": [100] * n,
            "open_imbalance": imb,
            "rolling_p95_open_abs": np.full(n, 0.5),
            "fwd_log_return": np.full(n, 0.001),
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    row = out.iloc[0]
    assert int(row["n_extreme_windows"]) == 1
    assert int(row["n_total_windows"]) == 100
    assert abs(float(row["extreme_frequency"]) - 0.01) < 1e-9


def test_aggregate_open_imbalance_headroom_subtracts_round_trip_cost() -> None:
    """headroom_extreme = |mean_signed_return_extreme| - (2 × fee + 2 × slip).

    With fee=4bp, slip=0bp, mean_signed_return=10bp → headroom = 10 - 8 = 2 bp.
    """
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 2,
            "horizon": [100] * 2,
            "open_imbalance": [0.8, 0.8],
            "rolling_p95_open_abs": [0.5, 0.5],
            "fwd_log_return": [0.001, 0.001],  # 10 bp each
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    row = out.iloc[0]
    # mean_signed_return_extreme = 10 bp; headroom = 10 - 8 = 2 bp
    assert abs(float(row["mean_signed_return_extreme"]) - 10.0) < 1e-6
    assert abs(float(row["headroom_extreme_bps"]) - 2.0) < 1e-6


def test_aggregate_open_imbalance_returns_zero_extreme_when_none_in_tail() -> None:
    """If no window crosses the cutoff, returned cell has n_extreme=0 and
    NaN aggregations rather than crashing."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 3,
            "horizon": [100] * 3,
            "open_imbalance": [0.1, 0.2, 0.3],  # all below 0.5
            "rolling_p95_open_abs": [0.5] * 3,
            "fwd_log_return": [0.001, 0.001, 0.001],
        }
    )
    out = aggregate_open_imbalance_cells(
        df,
        signal_kind="open",
        percentile_cutoff=0.95,
        taker_fee_bps_per_side=4.0,
        slip_bps=0.0,
    )
    row = out.iloc[0]
    assert int(row["n_extreme_windows"]) == 0
    assert int(row["n_total_windows"]) == 3
    assert float(row["extreme_frequency"]) == 0.0
    assert np.isnan(float(row["mean_signed_return_extreme"]))
    assert np.isnan(float(row["frac_positive_signed_return_extreme"]))

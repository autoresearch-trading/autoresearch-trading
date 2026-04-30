"""Tests for the post-cascade regime discovery probe.

The probe is intentionally simple: it studies what happens *after* observed
liquidation bursts, so it does not reopen the closed pre-cascade direction or
encoder programs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def test_group_liquidation_bursts_respects_max_gap_and_min_move():
    from scripts.post_cascade_regime_probe import group_liquidation_bursts

    trades = pd.DataFrame(
        {
            "ts_ms": [0, 1_000, 2_000, 120_000, 121_000],
            "price": [100.0, 101.0, 102.0, 105.0, 105.1],
            "qty": [1.0, 2.0, 1.0, 1.0, 1.0],
            "cause": [
                "market_liquidation",
                "market_liquidation",
                "backstop_liquidation",
                "market_liquidation",
                "market_liquidation",
            ],
        }
    )

    bursts = group_liquidation_bursts(
        trades,
        max_gap_ms=10_000,
        min_abs_move_bps=50.0,
    )

    assert len(bursts) == 1
    row = bursts.iloc[0]
    assert int(row["start_ts"]) == 0
    assert int(row["end_ts"]) == 2_000
    assert int(row["n_trades"]) == 3
    assert int(row["sign"]) == 1
    assert float(row["abs_return_bps"]) > 50.0


def test_group_liquidation_bursts_ignores_normal_trades():
    from scripts.post_cascade_regime_probe import group_liquidation_bursts

    trades = pd.DataFrame(
        {
            "ts_ms": [0, 1_000, 2_000],
            "price": [100.0, 99.0, 98.0],
            "qty": [1.0, 1.0, 1.0],
            "cause": ["normal", "market_liquidation", "normal"],
        }
    )

    bursts = group_liquidation_bursts(
        trades,
        max_gap_ms=10_000,
        min_abs_move_bps=1.0,
    )

    assert bursts.empty


def test_reversion_metrics_use_opposite_of_observed_burst_direction():
    from scripts.post_cascade_regime_probe import compute_regime_metrics

    # Up burst followed by -20 bps forward return: +20 bps gross reversion.
    metrics = compute_regime_metrics(
        burst_return_bps=100.0,
        forward_log_return=-0.002,
        round_trip_cost_bps=14.0,
    )

    assert math.isclose(metrics["gross_reversion_bps"], 20.0, abs_tol=1e-9)
    assert math.isclose(metrics["net_reversion_bps"], 6.0, abs_tol=1e-9)
    assert math.isclose(metrics["gross_continuation_bps"], -20.0, abs_tol=1e-9)
    assert bool(metrics["net_reversion_positive"]) is True


def test_reversion_metrics_handle_down_burst_continuation_as_negative_reversion():
    from scripts.post_cascade_regime_probe import compute_regime_metrics

    # Down burst followed by another down move: continuation, not reversion.
    metrics = compute_regime_metrics(
        burst_return_bps=-100.0,
        forward_log_return=-0.003,
        round_trip_cost_bps=14.0,
    )

    assert math.isclose(metrics["gross_reversion_bps"], -30.0, abs_tol=1e-9)
    assert math.isclose(metrics["gross_continuation_bps"], 30.0, abs_tol=1e-9)
    assert bool(metrics["net_reversion_positive"]) is False


def test_forward_log_return_after_delay_uses_anchor_plus_delay():
    from scripts.post_cascade_regime_probe import forward_log_return_after_delay

    log_returns = np.array([0.0, 0.01, 0.02, -0.03, 0.04, 0.05], dtype=float)
    # Anchor=1, delay=1 means enter after event index 2; horizon=2 sums indices 3 and 4.
    out = forward_log_return_after_delay(
        log_returns,
        anchor_idx=1,
        delay_events=1,
        horizon=2,
    )

    assert math.isclose(out, 0.01, abs_tol=1e-12)  # -0.03 + 0.04


def test_forward_log_return_after_delay_returns_nan_if_horizon_overruns():
    from scripts.post_cascade_regime_probe import forward_log_return_after_delay

    out = forward_log_return_after_delay(
        np.ones(5, dtype=float),
        anchor_idx=3,
        delay_events=1,
        horizon=2,
    )

    assert math.isnan(out)


def test_fixed_grid_is_tiny_and_preregistered():
    from scripts.post_cascade_regime_probe import fixed_parameter_grid

    grid = fixed_parameter_grid()

    assert grid["delays"] == (0, 10, 50)
    assert grid["horizons"] == (50, 100, 500)
    assert grid["min_abs_move_bps"] == (10.0,)


def test_markdown_table_does_not_require_tabulate():
    from scripts.post_cascade_regime_probe import dataframe_to_markdown_table

    df = pd.DataFrame({"a": [1], "b": [2.5]})

    out = dataframe_to_markdown_table(df)

    assert "| a | b |" in out
    assert "| 1 | 2.5000 |" in out


def test_summarize_regime_table_reports_core_gate_columns():
    from scripts.post_cascade_regime_probe import summarize_regime_table

    rows = pd.DataFrame(
        {
            "horizon": [50, 50, 50, 100],
            "delay_events": [0, 0, 0, 0],
            "net_reversion_bps": [5.0, 7.0, -1.0, 2.0],
            "gross_reversion_bps": [19.0, 21.0, 13.0, 16.0],
            "date": ["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-02"],
            "symbol": ["BTC", "ETH", "BTC", "BTC"],
        }
    )

    summary = summarize_regime_table(rows)

    h50 = summary[(summary["horizon"] == 50) & (summary["delay_events"] == 0)].iloc[0]
    assert int(h50["n_events"]) == 3
    assert math.isclose(float(h50["median_net_bps"]), 5.0, abs_tol=1e-9)
    assert math.isclose(float(h50["frac_positive"]), 2 / 3, abs_tol=1e-9)
    assert int(h50["n_days"]) == 2
    assert int(h50["n_symbols"]) == 2

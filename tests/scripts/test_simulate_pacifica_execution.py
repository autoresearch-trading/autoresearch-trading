from pathlib import Path

import pandas as pd

from scripts.simulate_pacifica_execution import (
    DEFAULT_ASSUMPTIONS,
    ExecutionAssumptions,
    TradeIntent,
    dataframe_to_markdown_table,
    simulate_round_trip,
    write_execution_simulator_report,
)


def test_taker_round_trip_charges_fees_and_slippage_against_notional() -> None:
    intent = TradeIntent(
        symbol="BTC",
        side="long",
        entry_price=100.0,
        exit_price=101.0,
        quantity=10.0,
        entry_liquidity="taker",
        exit_liquidity="taker",
        slippage_bps_per_side=2.0,
    )

    result = simulate_round_trip(intent, DEFAULT_ASSUMPTIONS)

    assert result.gross_pnl == 10.0
    assert result.fee_bps_total == 8.0
    assert result.fees_paid == 0.8
    assert result.slippage_paid == 0.4
    assert result.funding_paid == 0.0
    assert result.net_pnl == 8.8
    assert result.net_return_bps == 88.0


def test_maker_round_trip_charges_maker_fees_and_adverse_selection() -> None:
    intent = TradeIntent(
        symbol="ETH",
        side="short",
        entry_price=100.0,
        exit_price=99.0,
        quantity=5.0,
        entry_liquidity="maker",
        exit_liquidity="maker",
        adverse_selection_bps_per_side=3.0,
    )

    result = simulate_round_trip(intent, DEFAULT_ASSUMPTIONS)

    assert result.gross_pnl == 5.0
    assert result.fee_bps_total == 3.0
    assert result.fees_paid == 0.15
    assert result.adverse_selection_paid == 0.3
    assert result.net_pnl == 4.55
    assert result.net_return_bps == 91.0


def test_positive_funding_debits_longs_and_credits_shorts() -> None:
    assumptions = ExecutionAssumptions(funding_bps_per_hour=1.0)
    long_intent = TradeIntent(
        symbol="SOL",
        side="long",
        entry_price=100.0,
        exit_price=100.0,
        quantity=10.0,
        hold_hours=3.0,
    )
    short_intent = TradeIntent(
        symbol="SOL",
        side="short",
        entry_price=100.0,
        exit_price=100.0,
        quantity=10.0,
        hold_hours=3.0,
    )

    long_result = simulate_round_trip(long_intent, assumptions)
    short_result = simulate_round_trip(short_intent, assumptions)

    assert long_result.funding_paid == 0.3
    assert short_result.funding_paid == -0.3
    assert long_result.net_pnl < short_result.net_pnl


def test_invalid_side_and_liquidity_raise_clear_errors() -> None:
    bad_side = TradeIntent(
        symbol="BTC", side="flat", entry_price=1, exit_price=1, quantity=1
    )
    bad_liquidity = TradeIntent(
        symbol="BTC",
        side="long",
        entry_price=1,
        exit_price=1,
        quantity=1,
        entry_liquidity="auction",
    )

    for intent, expected in [(bad_side, "side"), (bad_liquidity, "liquidity")]:
        try:
            simulate_round_trip(intent, DEFAULT_ASSUMPTIONS)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - test should fail before this branch
            raise AssertionError("expected ValueError")


def test_write_execution_simulator_report_creates_markdown_and_csvs(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "execution-simulator"

    result = write_execution_simulator_report(out_dir)

    assert result["verdict"] == "DIAGNOSTIC_ACCOUNTING_SPINE"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "assumptions.csv").exists()
    assert (out_dir / "example_round_trips.csv").exists()

    report = (out_dir / "README.md").read_text()
    assert "Pacifica Execution Simulator" in report
    assert "not a strategy" in report
    assert "taker/taker" in report

    examples = pd.read_csv(out_dir / "example_round_trips.csv")
    assert {"symbol", "side", "net_pnl", "net_return_bps"}.issubset(examples.columns)


def test_dataframe_to_markdown_table_does_not_need_tabulate() -> None:
    frame = pd.DataFrame([{"a": "x", "b": 1.25}])

    rendered = dataframe_to_markdown_table(frame)

    assert "| a | b |" in rendered
    assert "| x | 1.2500 |" in rendered

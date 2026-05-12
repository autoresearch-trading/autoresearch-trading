from pathlib import Path

import pandas as pd

from scripts.build_pacifica_paper_ledger import (
    LedgerFill,
    build_paper_ledger,
    max_drawdown,
    write_paper_ledger_report,
)


def test_build_paper_ledger_accounts_for_round_trip_fees_and_realized_pnl() -> None:
    fills = [
        LedgerFill(
            ts_ms=1,
            symbol="BTC",
            side="buy",
            price=100.0,
            quantity=2.0,
            liquidity="taker",
        ),
        LedgerFill(
            ts_ms=2,
            symbol="BTC",
            side="sell",
            price=101.0,
            quantity=2.0,
            liquidity="taker",
        ),
    ]

    result = build_paper_ledger(fills, starting_cash=1_000.0)

    assert result.summary["realized_pnl"] == 2.0
    assert result.summary["fees_paid"] == 0.1608
    assert result.summary["net_pnl"] == 1.8392
    assert result.positions.set_index("symbol").loc["BTC", "quantity"] == 0.0
    assert list(result.fills["position_after_fill"]) == [2.0, 0.0]


def test_build_paper_ledger_includes_funding_debits_in_net_pnl() -> None:
    fills = [
        LedgerFill(
            ts_ms=1,
            symbol="SOL",
            side="buy",
            price=10.0,
            quantity=10.0,
            liquidity="maker",
        ),
        LedgerFill(
            ts_ms=2,
            symbol="SOL",
            side="funding",
            price=10.0,
            quantity=10.0,
            funding_payment=0.25,
        ),
        LedgerFill(
            ts_ms=3,
            symbol="SOL",
            side="sell",
            price=10.5,
            quantity=10.0,
            liquidity="maker",
        ),
    ]

    result = build_paper_ledger(fills, starting_cash=1_000.0)

    assert result.summary["realized_pnl"] == 5.0
    assert result.summary["funding_paid"] == 0.25
    assert result.summary["fees_paid"] == 0.03075
    assert result.summary["net_pnl"] == 4.71925


def test_max_drawdown_uses_chronological_equity_snapshots() -> None:
    equity = pd.Series([100.0, 110.0, 105.0, 90.0, 95.0])

    assert max_drawdown(equity) == -20.0


def test_ledger_refuses_unapproved_symbol_when_not_diagnostic() -> None:
    fills = [LedgerFill(ts_ms=1, symbol="DOGE", side="buy", price=1.0, quantity=1.0)]

    try:
        build_paper_ledger(
            fills,
            eligible_symbols={"BTC"},
            allow_diagnostic_ineligible=False,
        )
    except ValueError as exc:
        assert "ineligible" in str(exc)
    else:  # pragma: no cover - test should fail before this branch
        raise AssertionError("expected ValueError")


def test_write_paper_ledger_report_creates_markdown_and_csvs(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper-ledger"

    result = write_paper_ledger_report(out_dir)

    assert result["verdict"] == "DIAGNOSTIC_LEDGER_SPINE"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "fills.csv").exists()
    assert (out_dir / "positions.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Pacifica Paper Ledger" in report
    assert "strategy-neutral" in report
    assert "does not permit live trading" in report

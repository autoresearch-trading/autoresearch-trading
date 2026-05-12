# scripts/build_pacifica_paper_ledger.py
"""Strategy-neutral Pacifica paper ledger spine.

The ledger is deliberately not a strategy and does not authorize live trading.
It records fills, positions, fees, funding, realized PnL, an equity curve, and
basic drawdown so future strategies have to pass through one accounting path.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulate_pacifica_execution import (
    DEFAULT_ASSUMPTIONS,
    ExecutionAssumptions,
    dataframe_to_markdown_table,
)

DEFAULT_OUT_DIR = Path("docs/experiments/paper-ledger")
VALID_FILL_SIDES = {"buy", "sell", "funding"}


@dataclass(frozen=True)
class LedgerFill:
    """One paper-ledger event.

    `funding` fills use `funding_payment` directly: positive values are debits
    paid by the account, negative values are credits received by the account.
    """

    ts_ms: int
    symbol: str
    side: str
    price: float
    quantity: float
    liquidity: str = "taker"
    funding_payment: float = 0.0


@dataclass(frozen=True)
class LedgerResult:
    fills: pd.DataFrame
    positions: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, Any]


def _fee_bps(liquidity: str, assumptions: ExecutionAssumptions) -> float:
    if liquidity == "maker":
        return assumptions.maker_fee_bps
    if liquidity == "taker":
        return assumptions.taker_fee_bps
    raise ValueError("liquidity must be maker or taker")


def _round_money(value: float) -> float:
    return float(round(value, 10))


def max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown in account currency, as a non-positive number."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    return _round_money(float(drawdown.min()))


def _validate_fill(
    fill: LedgerFill,
    *,
    eligible_symbols: set[str] | None,
    allow_diagnostic_ineligible: bool,
) -> None:
    if fill.side not in VALID_FILL_SIDES:
        raise ValueError(f"side must be one of {sorted(VALID_FILL_SIDES)}")
    if fill.price <= 0:
        raise ValueError("price must be positive")
    if fill.quantity <= 0:
        raise ValueError("quantity must be positive")
    if (
        eligible_symbols is not None
        and fill.symbol not in eligible_symbols
        and not allow_diagnostic_ineligible
    ):
        raise ValueError(f"symbol {fill.symbol} is ineligible for paper ledger fills")


def build_paper_ledger(
    fills: Iterable[LedgerFill],
    *,
    starting_cash: float = 0.0,
    assumptions: ExecutionAssumptions = DEFAULT_ASSUMPTIONS,
    eligible_symbols: set[str] | None = None,
    allow_diagnostic_ineligible: bool = True,
) -> LedgerResult:
    """Build a strategy-neutral paper ledger from chronological fills."""
    ordered = sorted(list(fills), key=lambda item: item.ts_ms)
    positions: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    realized_pnl = 0.0
    fees_paid = 0.0
    funding_paid = 0.0
    fill_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for fill in ordered:
        _validate_fill(
            fill,
            eligible_symbols=eligible_symbols,
            allow_diagnostic_ineligible=allow_diagnostic_ineligible,
        )
        current_qty = positions.get(fill.symbol, 0.0)
        current_avg = avg_cost.get(fill.symbol, 0.0)
        notional = fill.price * fill.quantity
        fill_fee = 0.0
        fill_realized_pnl = 0.0

        if fill.side == "funding":
            funding_paid += fill.funding_payment
        else:
            fill_fee = notional * _fee_bps(fill.liquidity, assumptions) / 10_000.0
            fees_paid += fill_fee
            if fill.side == "buy":
                new_qty = current_qty + fill.quantity
                if new_qty > 0:
                    avg_cost[fill.symbol] = (
                        (current_avg * current_qty) + notional
                    ) / new_qty
                positions[fill.symbol] = new_qty
            elif fill.side == "sell":
                close_qty = min(fill.quantity, max(current_qty, 0.0))
                fill_realized_pnl = (fill.price - current_avg) * close_qty
                realized_pnl += fill_realized_pnl
                new_qty = current_qty - fill.quantity
                positions[fill.symbol] = new_qty
                if new_qty <= 0:
                    avg_cost[fill.symbol] = 0.0

        net_pnl = realized_pnl - fees_paid - funding_paid
        position_after = positions.get(fill.symbol, current_qty)
        fill_rows.append(
            {
                **asdict(fill),
                "notional": _round_money(notional),
                "fee_paid": _round_money(fill_fee),
                "realized_pnl_on_fill": _round_money(fill_realized_pnl),
                "position_after_fill": _round_money(position_after),
                "cumulative_realized_pnl": _round_money(realized_pnl),
                "cumulative_fees_paid": _round_money(fees_paid),
                "cumulative_funding_paid": _round_money(funding_paid),
                "cumulative_net_pnl": _round_money(net_pnl),
            }
        )
        equity_rows.append(
            {
                "ts_ms": fill.ts_ms,
                "equity": _round_money(starting_cash + net_pnl),
                "net_pnl": _round_money(net_pnl),
            }
        )

    position_rows = [
        {
            "symbol": symbol,
            "quantity": _round_money(quantity),
            "avg_cost": _round_money(avg_cost.get(symbol, 0.0)),
        }
        for symbol, quantity in sorted(positions.items())
    ]
    fills_df = pd.DataFrame(fill_rows)
    positions_df = pd.DataFrame(
        position_rows, columns=["symbol", "quantity", "avg_cost"]
    )
    equity_df = pd.DataFrame(equity_rows, columns=["ts_ms", "equity", "net_pnl"])
    final_net_pnl = realized_pnl - fees_paid - funding_paid
    summary = {
        "starting_cash": _round_money(starting_cash),
        "realized_pnl": _round_money(realized_pnl),
        "fees_paid": _round_money(fees_paid),
        "funding_paid": _round_money(funding_paid),
        "net_pnl": _round_money(final_net_pnl),
        "ending_equity": _round_money(starting_cash + final_net_pnl),
        "max_drawdown": (
            max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
        ),
        "fills": int(len(fills_df)),
        "symbols": int(fills_df["symbol"].nunique()) if not fills_df.empty else 0,
    }
    return LedgerResult(
        fills=fills_df,
        positions=positions_df,
        equity_curve=equity_df,
        summary=summary,
    )


def _example_fills() -> list[LedgerFill]:
    return [
        LedgerFill(ts_ms=1, symbol="BTC", side="buy", price=100.0, quantity=2.0),
        LedgerFill(ts_ms=2, symbol="BTC", side="sell", price=101.0, quantity=2.0),
        LedgerFill(
            ts_ms=3,
            symbol="SOL",
            side="buy",
            price=10.0,
            quantity=10.0,
            liquidity="maker",
        ),
        LedgerFill(
            ts_ms=4,
            symbol="SOL",
            side="funding",
            price=10.0,
            quantity=10.0,
            funding_payment=0.25,
        ),
        LedgerFill(
            ts_ms=5,
            symbol="SOL",
            side="sell",
            price=10.5,
            quantity=10.0,
            liquidity="maker",
        ),
    ]


def write_paper_ledger_report(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    starting_cash: float = 1_000.0,
) -> dict[str, Any]:
    """Write a diagnostic paper-ledger example report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_paper_ledger(_example_fills(), starting_cash=starting_cash)
    summary_df = pd.DataFrame([result.summary])
    result.fills.to_csv(out_dir / "fills.csv", index=False)
    result.positions.to_csv(out_dir / "positions.csv", index=False)
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    readme = f"""# Pacifica Paper Ledger

Verdict: `DIAGNOSTIC_LEDGER_SPINE`

This is a strategy-neutral accounting artifact for future Pacifica non-HFT paper
research. It does not permit live trading and does not claim edge. Its job is to
make every future strategy account for fills, fees, funding, realized PnL,
equity, and drawdown in one reusable ledger path.

## Summary

{dataframe_to_markdown_table(summary_df)}

## Example fills

{dataframe_to_markdown_table(result.fills)}

## Open positions after example run

{dataframe_to_markdown_table(result.positions)}

## Required next integrations

- wire strategy candidate outputs into this ledger;
- require eligible-symbol gates before non-diagnostic paper fills;
- add unrealized PnL from mark prices;
- add exposure and concentration reports;
- add random same-frequency controls through the walk-forward validation harness.
"""
    (out_dir / "README.md").write_text(readme)
    return {
        "verdict": "DIAGNOSTIC_LEDGER_SPINE",
        "out_dir": str(out_dir),
        "fills": int(len(result.fills)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=1_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_paper_ledger_report(args.out_dir, starting_cash=args.starting_cash)
    print(f"verdict: {result['verdict']}")
    print(f"wrote report: {Path(result['out_dir']) / 'README.md'}")


if __name__ == "__main__":
    main()

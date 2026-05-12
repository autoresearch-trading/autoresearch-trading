# scripts/simulate_pacifica_execution.py
"""Reusable execution-economics simulator for Pacifica paper-trading research.

This module is intentionally strategy-neutral.  It exists so future backtests,
regime governors, and paper ledgers all charge the same pre-registered costs
before anyone interprets a result as a possible edge.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUT_DIR = Path("docs/experiments/execution-simulator")


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Fixed execution assumptions for diagnostic Pacifica paper accounting."""

    taker_fee_bps: float = 4.0
    maker_fee_bps: float = 1.5
    funding_bps_per_hour: float = 0.0
    default_slippage_bps_per_side: float = 0.0
    default_adverse_selection_bps_per_side: float = 0.0


DEFAULT_ASSUMPTIONS = ExecutionAssumptions()


@dataclass(frozen=True)
class TradeIntent:
    """A strategy-neutral round-trip trade intent.

    Prices and quantity are deliberately explicit.  This avoids hiding account
    state in the execution simulator; account state belongs in the future paper
    ledger.
    """

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_liquidity: str = "taker"
    exit_liquidity: str = "taker"
    hold_hours: float = 0.0
    slippage_bps_per_side: float | None = None
    adverse_selection_bps_per_side: float | None = None


@dataclass(frozen=True)
class SimulatedFill:
    """Post-cost accounting for one simulated round trip."""

    symbol: str
    side: str
    entry_liquidity: str
    exit_liquidity: str
    entry_price: float
    exit_price: float
    quantity: float
    initial_notional: float
    gross_pnl: float
    fee_bps_total: float
    fees_paid: float
    slippage_paid: float
    adverse_selection_paid: float
    funding_paid: float
    net_pnl: float
    net_return_bps: float


VALID_SIDES = {"long", "short"}
VALID_LIQUIDITY = {"maker", "taker"}


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown_table(
    df: pd.DataFrame, *, max_rows: int | None = None
) -> str:
    """Render a simple Markdown table without pandas' optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows) if max_rows is not None else df
    headers = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def _validate_intent(intent: TradeIntent) -> None:
    if intent.side not in VALID_SIDES:
        raise ValueError(f"side must be one of {sorted(VALID_SIDES)}")
    for field_name, value in [
        ("entry_liquidity", intent.entry_liquidity),
        ("exit_liquidity", intent.exit_liquidity),
    ]:
        if value not in VALID_LIQUIDITY:
            raise ValueError(
                f"{field_name} liquidity must be one of {sorted(VALID_LIQUIDITY)}"
            )
    if intent.entry_price <= 0 or intent.exit_price <= 0:
        raise ValueError("entry_price and exit_price must be positive")
    if intent.quantity <= 0:
        raise ValueError("quantity must be positive")
    if intent.hold_hours < 0:
        raise ValueError("hold_hours must be non-negative")


def _fee_bps(liquidity: str, assumptions: ExecutionAssumptions) -> float:
    if liquidity == "maker":
        return assumptions.maker_fee_bps
    if liquidity == "taker":
        return assumptions.taker_fee_bps
    raise ValueError(f"liquidity must be one of {sorted(VALID_LIQUIDITY)}")


def _gross_pnl(intent: TradeIntent) -> float:
    if intent.side == "long":
        return (intent.exit_price - intent.entry_price) * intent.quantity
    if intent.side == "short":
        return (intent.entry_price - intent.exit_price) * intent.quantity
    raise ValueError(f"side must be one of {sorted(VALID_SIDES)}")


def _funding_paid(
    intent: TradeIntent, assumptions: ExecutionAssumptions, initial_notional: float
) -> float:
    signed_payment = (
        initial_notional
        * assumptions.funding_bps_per_hour
        / 10_000.0
        * intent.hold_hours
    )
    if intent.side == "short":
        return -signed_payment
    return signed_payment


def simulate_round_trip(
    intent: TradeIntent, assumptions: ExecutionAssumptions = DEFAULT_ASSUMPTIONS
) -> SimulatedFill:
    """Apply fixed fee/slippage/adverse-selection/funding costs to a round trip.

    Cost bps are charged against initial notional rather than variable fill
    notional.  This keeps the diagnostic accounting deterministic and matches
    the pre-registered baseline assumption that a taker/taker round trip costs
    8 bps before slippage.
    """
    _validate_intent(intent)
    initial_notional = intent.entry_price * intent.quantity
    gross_pnl = _gross_pnl(intent)
    fee_bps_total = _fee_bps(intent.entry_liquidity, assumptions) + _fee_bps(
        intent.exit_liquidity, assumptions
    )
    fees_paid = initial_notional * fee_bps_total / 10_000.0

    slippage_bps = (
        assumptions.default_slippage_bps_per_side
        if intent.slippage_bps_per_side is None
        else intent.slippage_bps_per_side
    )
    adverse_selection_bps = (
        assumptions.default_adverse_selection_bps_per_side
        if intent.adverse_selection_bps_per_side is None
        else intent.adverse_selection_bps_per_side
    )
    slippage_paid = initial_notional * (slippage_bps * 2.0) / 10_000.0
    adverse_selection_paid = initial_notional * (adverse_selection_bps * 2.0) / 10_000.0
    funding_paid = _funding_paid(intent, assumptions, initial_notional)
    net_pnl = (
        gross_pnl - fees_paid - slippage_paid - adverse_selection_paid - funding_paid
    )
    net_return_bps = net_pnl / initial_notional * 10_000.0

    return SimulatedFill(
        symbol=intent.symbol,
        side=intent.side,
        entry_liquidity=intent.entry_liquidity,
        exit_liquidity=intent.exit_liquidity,
        entry_price=float(intent.entry_price),
        exit_price=float(intent.exit_price),
        quantity=float(intent.quantity),
        initial_notional=float(initial_notional),
        gross_pnl=float(round(gross_pnl, 10)),
        fee_bps_total=float(fee_bps_total),
        fees_paid=float(round(fees_paid, 10)),
        slippage_paid=float(round(slippage_paid, 10)),
        adverse_selection_paid=float(round(adverse_selection_paid, 10)),
        funding_paid=float(round(funding_paid, 10)),
        net_pnl=float(round(net_pnl, 10)),
        net_return_bps=float(round(net_return_bps, 10)),
    )


def _example_intents() -> list[TradeIntent]:
    return [
        TradeIntent(
            symbol="BTC",
            side="long",
            entry_price=100.0,
            exit_price=101.0,
            quantity=10.0,
            entry_liquidity="taker",
            exit_liquidity="taker",
            slippage_bps_per_side=2.0,
        ),
        TradeIntent(
            symbol="ETH",
            side="short",
            entry_price=100.0,
            exit_price=99.0,
            quantity=5.0,
            entry_liquidity="maker",
            exit_liquidity="maker",
            adverse_selection_bps_per_side=3.0,
        ),
        TradeIntent(
            symbol="SOL",
            side="long",
            entry_price=100.0,
            exit_price=100.5,
            quantity=20.0,
            entry_liquidity="taker",
            exit_liquidity="maker",
            hold_hours=4.0,
        ),
    ]


def write_execution_simulator_report(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    assumptions: ExecutionAssumptions = DEFAULT_ASSUMPTIONS,
) -> dict[str, Any]:
    """Write diagnostic documentation and example round trips for the simulator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assumption_df = pd.DataFrame([asdict(assumptions)])
    example_df = pd.DataFrame(
        [
            asdict(simulate_round_trip(intent, assumptions))
            for intent in _example_intents()
        ]
    )

    assumption_df.to_csv(out_dir / "assumptions.csv", index=False)
    example_df.to_csv(out_dir / "example_round_trips.csv", index=False)

    readme = f"""# Pacifica Execution Simulator

Verdict: `DIAGNOSTIC_ACCOUNTING_SPINE`

This artifact defines reusable execution-cost accounting for future Pacifica
non-HFT research. It is not a strategy, not a backtest result, and not evidence
of edge. It exists so future probes, baselines, and paper-ledger runs all charge
fees, slippage/adverse selection, and funding consistently.

## Locked default assumptions

{dataframe_to_markdown_table(assumption_df)}

Interpretation:

- taker/taker round trip before slippage: `{assumptions.taker_fee_bps * 2:g}` bps
- maker/maker round trip before adverse selection: `{assumptions.maker_fee_bps * 2:g}` bps
- taker/maker round trip before extra costs: `{assumptions.taker_fee_bps + assumptions.maker_fee_bps:g}` bps

## Example diagnostic round trips

{dataframe_to_markdown_table(example_df)}

## Required use

Any future strategy, backtest, or paper-trading ledger should call this simulator
or preserve equivalent semantics before reporting post-cost PnL, Sortino,
drawdown, or baseline deltas.

## Not yet included

- order-book-depth-aware partial fills;
- live position accounting;
- portfolio exposure caps;
- random same-frequency controls;
- chronological walk-forward validation.

Those belong in the next phases of the system level-up plan.
"""
    (out_dir / "README.md").write_text(readme)
    return {
        "verdict": "DIAGNOSTIC_ACCOUNTING_SPINE",
        "out_dir": str(out_dir),
        "examples": int(len(example_df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_execution_simulator_report(args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"wrote report: {Path(result['out_dir']) / 'README.md'}")


if __name__ == "__main__":
    main()

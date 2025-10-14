#!/usr/bin/env python3
"""Run signal engine backtest over historical QuestDB data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from backtest import BacktestConfig, BacktestEngine, BacktestReporter  # noqa: E402
from config import Settings  # noqa: E402
from db.questdb import QuestDBClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbol",
        help="Symbol to backtest (defaults to first configured symbol).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback window in days (default: 30).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Initial virtual capital for the backtest.",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Fraction of capital to allocate per trade (0-1).",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage per trade (0-1).",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.03,
        help="Take profit percentage per trade (0-1).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum signal confidence required for entry.",
    )
    parser.add_argument(
        "--min-agree",
        type=int,
        default=2,
        help="Minimum number of aligned signals required for entry.",
    )
    parser.add_argument(
        "--no-cvd",
        action="store_true",
        help="Do not require CVD agreement for entries.",
    )
    parser.add_argument(
        "--no-tfi",
        action="store_true",
        help="Do not require TFI agreement for entries.",
    )
    parser.add_argument(
        "--require-ofi",
        action="store_true",
        help="Require OFI agreement for entries.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    console = Console()

    settings = Settings()
    symbol = (args.symbol or settings.symbols[0]).upper()

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    console.print(
        f"[bold blue]Backtest[/bold blue] symbol={symbol} window={start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}"
    )

    client = QuestDBClient(
        host=settings.questdb_host,
        port=settings.questdb_port,
        user=settings.questdb_user,
        password=settings.questdb_password,
    )

    console.print("Loading signals…")
    signals = client.query_signals(
        symbol=symbol,
        start_ts=start_dt,
        end_ts=end_dt,
    )

    if not signals:
        console.print("[yellow]No signals found for selected window.[/yellow]")
        raise SystemExit(1)

    console.print(f"Loaded {len(signals)} signals")

    regimes = client.query_regimes(
        symbol=symbol,
        start_ts=start_dt,
        end_ts=end_dt,
    )

    if regimes:
        console.print(f"Including {len(regimes)} regime updates")
    else:
        console.print("[yellow]No regime data available for window.[/yellow]")

    console.print("Loading price data from trades...")
    price_map = client.get_price_map(
        symbol=symbol,
        start_ts=start_dt,
        end_ts=end_dt,
    )

    if price_map:
        console.print(f"Loaded {len(price_map)} price points")
    else:
        console.print("[yellow]No trade data, using signal prices[/yellow]")

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_confidence=args.min_confidence,
        min_signals_agree=args.min_agree,
        require_cvd=not args.no_cvd,
        require_tfi=not args.no_tfi,
        require_ofi=args.require_ofi,
    )

    engine = BacktestEngine(config)
    results = engine.run(
        signals=signals,
        regimes=regimes,
        price_data={symbol: price_map},
    )

    reporter = BacktestReporter(console)
    reporter.display(results)


if __name__ == "__main__":
    main()

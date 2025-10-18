#!/usr/bin/env python3
"""Run signal engine backtest over historical QuestDB data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from backtest import (BacktestConfig, BacktestEngine,  # noqa: E402
                      BacktestReporter)
from db.questdb import QuestDBClient  # noqa: E402

from config import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        "--symbol",
        help="Single symbol to backtest (deprecated, prefer --symbols).",
    )
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="Space/comma separated symbols to backtest (defaults to configured SYMBOLS).",
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


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    """Parse and normalize a collection of symbols."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        if raw is None:
            continue
        for item in str(raw).split(","):
            symbol = item.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def main() -> None:
    load_dotenv()
    args = parse_args()
    console = Console()

    settings = Settings()
    client = QuestDBClient(
        host=settings.questdb_host,
        port=settings.questdb_port,
        user=settings.questdb_user,
        password=settings.questdb_password,
    )

    if args.symbols:
        symbols = _normalize_symbols(args.symbols)
    elif args.symbol:
        symbols = _normalize_symbols([args.symbol])
    else:
        env_symbols = _normalize_symbols(settings.symbols)
        if env_symbols:
            symbols = env_symbols
        else:
            symbols = client.list_symbols()
            if symbols:
                console.print(
                    "[yellow]SYMBOLS env is empty; using all symbols from QuestDB.[/yellow]"
                )

    if not symbols:
        console.print("[red]No symbols configured or found in QuestDB. Aborting.[/red]")
        raise SystemExit(1)

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

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

    reporter = BacktestReporter(console)

    console.print(
        f"[bold blue]Running backtest[/bold blue] "
        f"symbols={', '.join(symbols)} window={start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}"
    )

    processed = 0
    skipped: list[str] = []
    multi_symbol = len(symbols) > 1

    for index, symbol in enumerate(symbols, start=1):
        if multi_symbol:
            console.rule(f"{index}. {symbol}")

        console.print(f"[bold]Loading signals for {symbol}…[/bold]")
        signals = client.query_signals(
            symbol=symbol,
            start_ts=start_dt,
            end_ts=end_dt,
        )

        if not signals:
            console.print(f"[yellow]No signals found for {symbol}. Skipping.[/yellow]")
            skipped.append(symbol)
            continue

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

        console.print("Loading price data from trades…")
        price_map = client.get_price_map(
            symbol=symbol,
            start_ts=start_dt,
            end_ts=end_dt,
        )

        if price_map:
            console.print(f"Loaded {len(price_map)} price points")
        else:
            console.print("[yellow]No trade data, using signal prices[/yellow]")

        engine = BacktestEngine(config)
        results = engine.run(
            signals=signals,
            regimes=regimes,
            price_data={symbol: price_map} if price_map else None,
        )

        reporter.display(results)
        processed += 1

    if not processed:
        console.print(
            "[red]No symbols produced results for the requested window.[/red]"
        )
        raise SystemExit(1)

    if skipped:
        console.print(
            "[yellow]Skipped symbol(s) with no signals: "
            f"{', '.join(skipped)}[/yellow]"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_PATHS = [
    ROOT / "src",
    REPO_ROOT / "src",  # Optional: provides access to collector package.
]
for path in SRC_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Settings  # noqa: E402
from paper_trading.engine import PaperTradingEngine  # noqa: E402

console = Console()


def parse_symbols(raw: Sequence[str] | None) -> list[str]:
    if not raw:
        return []
    if len(raw) == 1 and "," in raw[0]:
        return [token.strip().upper() for token in raw[0].split(",") if token.strip()]
    return [token.strip().upper() for token in raw if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper trading engine.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to trade (e.g. BTC ETH). Defaults to settings.SYMBOLS.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between trading loop iterations (default: 1.0).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute trades (disable dry-run).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file (default: .env).",
    )
    return parser


async def run_engine(settings: Settings, *, poll_interval: float, dry_run: bool) -> None:
    engine = PaperTradingEngine(settings, dry_run=dry_run, poll_interval=poll_interval)

    try:
        await engine.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt, stopping...[/yellow]")
    finally:
        await engine.stop()
        metrics = engine.get_metrics()
        render_metrics(metrics, dry_run=dry_run)


def render_banner(settings: Settings, *, dry_run: bool, poll_interval: float) -> None:
    mode = "Dry Run" if dry_run else "Execution"
    console.print("[bold green]Pacifica Paper Trading Engine[/bold green]")
    console.print(f"Mode: [cyan]{mode}[/cyan]")
    console.print(f"Symbols: [magenta]{', '.join(settings.symbols)}[/magenta]")
    console.print(f"Poll interval: {poll_interval:.2f}s")
    console.print(f"Initial capital: ${settings.initial_capital:,.2f}")
    console.print("Press Ctrl+C to stop.\n")


def render_metrics(metrics: dict[str, float | int], *, dry_run: bool) -> None:
    table = Table(title=f"Final Metrics ({'Dry Run' if dry_run else 'Execution'})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Capital", f"${metrics.get('capital', 0.0):,.2f}")
    table.add_row("Open Positions", str(metrics.get("open_positions", 0)))
    table.add_row("Total Trades", str(metrics.get("total_trades", 0)))
    table.add_row("Daily P&L", f"${metrics.get('daily_pnl', 0.0):,.2f}")
    table.add_row("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.2f}%")
    console.print()
    console.print(table)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv(args.env_file)
    settings = Settings()

    symbols_override = parse_symbols(args.symbols)
    if symbols_override:
        settings.symbols = symbols_override

    dry_run = not args.execute
    render_banner(settings, dry_run=dry_run, poll_interval=args.poll_interval)

    asyncio.run(run_engine(settings, poll_interval=args.poll_interval, dry_run=dry_run))


if __name__ == "__main__":
    main()

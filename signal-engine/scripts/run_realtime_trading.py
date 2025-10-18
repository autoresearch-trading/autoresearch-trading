#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_PATHS = [
    ROOT / "src",
    REPO_ROOT / "src",
]

for path in SRC_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bytewax.testing import run_main  # noqa: E402
from config import Settings  # noqa: E402
from paper_trading.realtime_engine import RealtimePaperTradingEngine  # noqa: E402
from persistence.async_writer import writer_loop  # noqa: E402
from stream.realtime_dataflow import build_realtime_dataflow  # noqa: E402
from stream.signal_router import SignalRouter  # noqa: E402

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real-time paper trading system."
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Override symbol list defined in environment (comma separated or space separated).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable trade execution (disable dry-run).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file (default: .env).",
    )
    return parser


async def run_system(settings: Settings, *, dry_run: bool) -> None:
    console.print("[bold green]🚀 Starting real-time trading system[/bold green]")
    console.print(f"Symbols: [cyan]{', '.join(settings.symbols)}[/cyan]")
    console.print(f"Mode: {'Dry Run' if dry_run else 'Execution'}")
    console.print(f"Initial capital: ${settings.initial_capital:,.2f}")

    SignalRouter.initialize()
    router_task = asyncio.create_task(
        SignalRouter.dispatch_loop(), name="signal_router_dispatch"
    )
    writer_task = asyncio.create_task(writer_loop(settings), name="questdb_writer")

    engine = RealtimePaperTradingEngine(settings, dry_run=dry_run)
    engine_task = asyncio.create_task(engine.start(), name="paper_trading_engine")

    flow = build_realtime_dataflow(settings)
    dataflow_task = asyncio.create_task(
        asyncio.to_thread(run_main, flow), name="bytewax_flow"
    )

    tasks = [router_task, writer_task, engine_task, dataflow_task]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.stop()
        console.print("[yellow]System shutdown complete[/yellow]")


def parse_symbols(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    if len(raw) == 1 and "," in raw[0]:
        return [token.strip().upper() for token in raw[0].split(",") if token.strip()]
    return [token.strip().upper() for token in raw if token.strip()]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv(args.env_file)
    settings = Settings()

    symbols_override = parse_symbols(args.symbols)
    if symbols_override:
        settings.symbols = symbols_override

    dry_run = not args.execute

    try:
        asyncio.run(run_system(settings, dry_run=dry_run))
    except KeyboardInterrupt:
        console.print("\n[yellow]Manual interrupt received. Stopping...[/yellow]")


if __name__ == "__main__":
    main()

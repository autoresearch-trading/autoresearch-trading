#!/usr/bin/env python3
"""Backfill signals from parquet data into QuestDB via Bytewax."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console

from bytewax.outputs import DynamicSink, StatelessSinkPartition

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config import Settings
from db.questdb import QuestDBSink
from stream.dataflow import build_signal_dataflow
from stream.sources import ParquetOrderbookSource, ParquetTradeSource

console = Console()


class ConsoleSignalSink(DynamicSink):
    """Bytewax sink that prints signals to the terminal."""

    def __init__(self) -> None:
        self._console = console

    def build(
        self, step_id: str, worker_index: int, worker_count: int
    ) -> StatelessSinkPartition:
        return _ConsolePartition(self._console)


class _ConsolePartition(StatelessSinkPartition):
    def __init__(self, console: Console) -> None:
        self._console = console

    def write_batch(self, items):
        for item in items:
            self._console.print(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        help="Date (YYYY-MM-DD) to replay. Defaults to latest available per symbol.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Override symbol list (defaults to SYMBOLS env).",
    )
    parser.add_argument(
        "--skip-orderbook",
        action="store_true",
        help="Skip OFI by omitting orderbook snapshots.",
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Disable regime writes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print signals to console instead of writing to QuestDB.",
    )
    return parser.parse_args()


def discover_files(base: Path, pattern: str, date: str | None) -> List[Path]:
    if not base.exists():
        return []
    if date:
        date_dirs = [base / f"date={date}"]
    else:
        all_dates = sorted(base.glob("date=*"))
        date_dirs = all_dates[-1:] if all_dates else []
    files: list[Path] = []
    for date_dir in date_dirs:
        if date_dir.exists():
            files.extend(sorted(date_dir.glob(pattern)))
    return files


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings()

    symbols = [sym.upper() for sym in (args.symbols or settings.symbols)]

    console.print(f"[bold blue]Signal pipeline replay[/bold blue] symbols={symbols}")

    trade_files: list[Path] = []
    orderbook_files: list[Path] = []

    for symbol in symbols:
        trades_root = settings.data_root / "trades" / f"symbol={symbol}"
        orderbook_root = settings.data_root / "orderbook" / f"symbol={symbol}"

        trade_matches = discover_files(trades_root, "*.parquet", args.date)
        if not trade_matches:
            console.print(f"[yellow]No trades for {symbol} (root={trades_root})[/yellow]")
        trade_files.extend(trade_matches)

        if not args.skip_orderbook:
            orderbook_matches = discover_files(orderbook_root, "*.parquet", args.date)
            if not orderbook_matches:
                console.print(
                    f"[yellow]No orderbook snapshots for {symbol} (root={orderbook_root})[/yellow]"
                )
            orderbook_files.extend(orderbook_matches)

    if not trade_files:
        console.print("[red]No trade data located. Aborting.[/red]")
        raise SystemExit(1)

    console.print(f"Replaying {len(trade_files)} trade files")
    trade_source = ParquetTradeSource(trade_files)

    orderbook_source = None
    if orderbook_files:
        console.print(f"Including {len(orderbook_files)} orderbook files")
        orderbook_source = ParquetOrderbookSource(orderbook_files)
    elif not args.skip_orderbook:
        console.print("[yellow]OFI disabled due to missing orderbook files[/yellow]")

    if args.dry_run:
        signal_sink = ConsoleSignalSink()
        regime_sink = None
        trade_sink = None
        if not args.skip_regime:
            console.print("[yellow]Regime sink disabled in dry-run mode[/yellow]")
    else:
        signal_sink = QuestDBSink.for_signals(
            host=settings.questdb_host,
            port=settings.questdb_port,
            user=settings.questdb_user,
            password=settings.questdb_password,
        )
        regime_sink = None
        if not args.skip_regime:
            regime_sink = QuestDBSink.for_regimes(
                host=settings.questdb_host,
                port=settings.questdb_port,
                user=settings.questdb_user,
                password=settings.questdb_password,
            )
        trade_sink = QuestDBSink.for_trades(
            host=settings.questdb_host,
            port=settings.questdb_port,
            user=settings.questdb_user,
            password=settings.questdb_password,
        )

    flow = build_signal_dataflow(
        trades_source=trade_source,
        signal_sink=signal_sink,
        orderbook_source=orderbook_source,
        regime_sink=regime_sink,
        trade_sink=trade_sink,
        cvd_config=settings.cvd_config(),
        tfi_config=settings.tfi_config(),
        ofi_config=settings.ofi_config(),
        atr_config=settings.atr_config(),
    )

    console.print("[green]Starting replay...[/green]")

    from bytewax.testing import run_main

    run_main(flow)

    console.print("[green]Replay complete[/green]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Manual CVD/TFI sanity check against historical parquet output."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from signals.base import SignalDirection, Trade
from signals.cvd import CVDCalculator
from signals.tfi import TFICalculator

console = Console()


@dataclass
class SignalRecord:
    emitted_at: str
    signal_type: str
    direction: str
    value: float
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[2] / "data"),
        help="Path to data-collector parquet root (default: ../data).",
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Symbol to evaluate (default: BTC).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Trading day (YYYY-MM-DD). Defaults to latest available.",
    )
    return parser.parse_args()


def detect_latest_date(symbol_path: Path) -> str | None:
    dates = sorted(
        (p.name.split("=")[-1] for p in symbol_path.glob("date=*") if p.is_dir())
    )
    return dates[-1] if dates else None


def load_trades(data_root: Path, symbol: str, date: str) -> pd.DataFrame:
    symbol_path = data_root / "trades" / f"symbol={symbol}"
    if not symbol_path.exists():
        raise FileNotFoundError(f"No data for symbol={symbol!r} under {symbol_path}")

    date_path = symbol_path / f"date={date}"
    if not date_path.exists():
        raise FileNotFoundError(f"No data for {symbol=} on {date=}")

    parquet_files = list(date_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {date_path}")

    dfs = [pd.read_parquet(file) for file in parquet_files]
    trades = pd.concat(dfs, ignore_index=True).sort_values("ts_ms")
    return trades


def to_trade(row: pd.Series) -> Trade:
    return Trade(
        ts=pd.to_datetime(row["ts_ms"], unit="ms"),
        recv_ts=pd.to_datetime(row.get("recv_ms", row["ts_ms"]), unit="ms"),
        symbol=row["symbol"],
        trade_id=str(row.get("trade_id", "")),
        side=str(row["side"]).lower(),
        price=float(row["price"]),
        qty=float(row["qty"]),
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    symbol = args.symbol.upper()

    symbol_path = data_root / "trades" / f"symbol={symbol}"
    date = args.date or detect_latest_date(symbol_path)
    if date is None:
        console.print(f"[red]No trade dates found for {symbol}[/red]")
        raise SystemExit(1)

    console.print(
        f"[bold blue]Testing CVD + TFI signals[/bold blue] — symbol: {symbol}, date: {date}"
    )

    try:
        trades_df = load_trades(data_root, symbol, date)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    console.print(f"Loaded {len(trades_df):,} trades\n")

    cvd_calc = CVDCalculator(symbol=symbol)
    tfi_calc = TFICalculator(symbol=symbol)

    emitted: list[SignalRecord] = []
    for _, row in trades_df.iterrows():
        trade = to_trade(row)
        for signal in filter(
            None, (cvd_calc.process_trade(trade), tfi_calc.process_trade(trade))
        ):
            emitted.append(
                SignalRecord(
                    emitted_at=signal.ts.strftime("%H:%M:%S"),
                    signal_type=signal.signal_type,
                    direction=signal.direction,
                    value=signal.value,
                    confidence=signal.confidence,
                )
            )

    if not emitted:
        console.print(
            "[yellow]No signals generated — adjust parameters or date[/yellow]"
        )
        return

    table = Table(title=f"Signals Found: {len(emitted)}")
    table.add_column("Time", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Direction", style="green")
    table.add_column("Value", style="yellow")
    table.add_column("Confidence", style="blue")

    for record in emitted:
        direction_color = {
            SignalDirection.BULLISH: "green",
            SignalDirection.BEARISH: "red",
            SignalDirection.NEUTRAL: "grey",
        }.get(SignalDirection(record.direction), "white")
        table.add_row(
            record.emitted_at,
            record.signal_type,
            f"[{direction_color}]{record.direction}[/{direction_color}]",
            f"{record.value:.4f}",
            f"{record.confidence:.2f}",
        )

    console.print(table)

    total = len(emitted)
    bullish = sum(
        1 for record in emitted if record.direction == SignalDirection.BULLISH
    )
    bearish = sum(
        1 for record in emitted if record.direction == SignalDirection.BEARISH
    )

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"CVD Signals: {sum(1 for r in emitted if r.signal_type == 'cvd')}")
    console.print(f"TFI Signals: {sum(1 for r in emitted if r.signal_type == 'tfi')}")
    console.print(
        f"Bullish: {bullish} ({(bullish / total) * 100:.1f}%)"
        if total
        else "Bullish: 0"
    )
    console.print(
        f"Bearish: {bearish} ({(bearish / total) * 100:.1f}%)"
        if total
        else "Bearish: 0"
    )


if __name__ == "__main__":
    main()

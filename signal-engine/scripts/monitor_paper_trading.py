#!/usr/bin/env python3
"""
Real-time Paper Trading Monitor
Displays live P&L, positions, and performance metrics from QuestDB
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import psycopg
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Settings


class PaperTradingMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()
        self.conn_string = (
            f"host={settings.questdb_host} "
            f"port={settings.questdb_port} "
            f"user={settings.questdb_user} "
            f"password={settings.questdb_password} "
            f"dbname=qdb"
        )

    def get_recent_trades(self, hours: int = 24) -> List[Dict]:
        """Get recent completed trades."""
        try:
            with psycopg.connect(self.conn_string, connect_timeout=3) as conn:
                query = f"""
                    SELECT symbol, side, entry_price, exit_price, pnl, pnl_pct,
                           ts, exit_ts
                    FROM paper_trades
                    WHERE ts > dateadd('h', -{hours}, now())
                    ORDER BY ts DESC
                    LIMIT 50
                """
                result = conn.execute(query)
                rows = result.fetchall()

                trades = []
                for row in rows:
                    trades.append(
                        {
                            "symbol": row[0],
                            "side": row[1],
                            "entry_price": row[2],
                            "exit_price": row[3],
                            "pnl": row[4],
                            "pnl_pct": row[5],
                            "ts": row[6],
                            "exit_ts": row[7],
                        }
                    )
                return trades
        except Exception as e:
            self.console.print(f"[red]Error fetching trades: {e}[/red]")
            return []

    def get_performance_stats(self, hours: int = 24) -> Dict:
        """Calculate performance statistics."""
        try:
            with psycopg.connect(self.conn_string, connect_timeout=3) as conn:
                query = f"""
                    SELECT
                        COUNT(*) as trade_count,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
                    FROM paper_trades
                    WHERE ts > dateadd('h', -{hours}, now())
                """
                result = conn.execute(query)
                row = result.fetchone()

                trade_count = row[0] or 0
                wins = row[5] or 0
                losses = row[6] or 0
                win_rate = (wins / trade_count * 100) if trade_count > 0 else 0

                return {
                    "trade_count": trade_count,
                    "total_pnl": row[1] or 0.0,
                    "avg_pnl": row[2] or 0.0,
                    "best_trade": row[3] or 0.0,
                    "worst_trade": row[4] or 0.0,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
                }
        except Exception as e:
            self.console.print(f"[red]Error fetching stats: {e}[/red]")
            return {}

    def get_recent_signals(self, minutes: int = 5) -> Dict[str, int]:
        """Get signal counts by type in recent period."""
        try:
            with psycopg.connect(self.conn_string, connect_timeout=3) as conn:
                query = f"""
                    SELECT signal_type, COUNT(*) as count
                    FROM signals
                    WHERE ts > dateadd('m', -{minutes}, now())
                    GROUP BY signal_type
                """
                result = conn.execute(query)
                rows = result.fetchall()

                return {row[0]: row[1] for row in rows}
        except Exception:
            return {}

    def create_stats_table(self, stats: Dict) -> Table:
        """Create performance statistics table."""
        table = Table(
            title="Performance Stats (24h)",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        if not stats:
            table.add_row("No data", "N/A")
            return table

        # Color code P&L
        total_pnl = stats.get("total_pnl", 0.0)
        pnl_color = "green" if total_pnl > 0 else "red" if total_pnl < 0 else "yellow"

        table.add_row("Total Trades", str(stats.get("trade_count", 0)))
        table.add_row("Total P&L", f"[{pnl_color}]${total_pnl:.2f}[/{pnl_color}]")
        table.add_row("Average P&L", f"${stats.get('avg_pnl', 0.0):.4f}")
        table.add_row(
            "Best Trade", f"[green]${stats.get('best_trade', 0.0):.2f}[/green]"
        )
        table.add_row("Worst Trade", f"[red]${stats.get('worst_trade', 0.0):.2f}[/red]")
        table.add_row("Win Rate", f"{stats.get('win_rate', 0.0):.1f}%")
        table.add_row(
            "Wins / Losses", f"{stats.get('wins', 0)} / {stats.get('losses', 0)}"
        )

        return table

    def create_trades_table(self, trades: List[Dict], limit: int = 10) -> Table:
        """Create recent trades table."""
        table = Table(
            title=f"Recent Trades (Last {limit})",
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="yellow")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Time", style="dim")

        if not trades:
            table.add_row("No trades", "-", "-", "-", "-", "-", "-")
            return table

        for trade in trades[:limit]:
            pnl = trade["pnl"]
            pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "yellow"

            time_str = trade["ts"].strftime("%H:%M:%S") if trade["ts"] else "N/A"

            table.add_row(
                trade["symbol"],
                trade["side"],
                f"${trade['entry_price']:.2f}",
                f"${trade['exit_price']:.2f}",
                f"[{pnl_color}]${pnl:.2f}[/{pnl_color}]",
                f"[{pnl_color}]{trade['pnl_pct']*100:.3f}%[/{pnl_color}]",
                time_str,
            )

        return table

    def create_signals_panel(self, signals: Dict[str, int]) -> Panel:
        """Create signals activity panel."""
        if not signals:
            content = Text("No recent signals", style="dim")
        else:
            lines = []
            for signal_type, count in sorted(signals.items()):
                lines.append(f"{signal_type.upper()}: {count}")
            content = Text("\n".join(lines), style="green")

        return Panel(content, title="Signals (Last 5m)", border_style="green")

    def generate_dashboard(self, hours: int = 24) -> Layout:
        """Generate complete dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )

        # Header
        header_text = Text(
            f"📊 Paper Trading Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="bold white on blue",
            justify="center",
        )
        layout["header"].update(Panel(header_text))

        # Body
        stats = self.get_performance_stats(hours)
        trades = self.get_recent_trades(hours)
        signals = self.get_recent_signals(5)

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right", ratio=2),
        )

        layout["left"].split_column(
            Layout(self.create_stats_table(stats)),
            Layout(self.create_signals_panel(signals)),
        )

        layout["right"].update(self.create_trades_table(trades, limit=15))

        return layout

    def run(self, refresh_seconds: int = 5, hours: int = 24):
        """Run live monitoring dashboard."""
        self.console.print("[bold green]Starting Paper Trading Monitor...[/bold green]")
        self.console.print(
            f"Refresh interval: {refresh_seconds}s | Time window: {hours}h"
        )
        self.console.print("Press Ctrl+C to exit\n")

        try:
            with Live(
                self.generate_dashboard(hours),
                refresh_per_second=1,
                console=self.console,
            ) as live:
                while True:
                    time.sleep(refresh_seconds)
                    live.update(self.generate_dashboard(hours))
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitoring stopped.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Real-time Paper Trading Monitor")
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Time window for statistics in hours (default: 24)",
    )

    args = parser.parse_args()

    settings = Settings()
    monitor = PaperTradingMonitor(settings)
    monitor.run(refresh_seconds=args.refresh, hours=args.hours)


if __name__ == "__main__":
    main()

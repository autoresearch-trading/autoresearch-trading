from __future__ import annotations

from .metrics import BacktestResults

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Table = None


class BacktestReporter:
    """Helper to render backtest results to the console (requires ``rich``)."""

    def __init__(self, console: "Console | None" = None) -> None:
        if Console is None or Table is None:
            raise ImportError(
                "BacktestReporter requires the 'rich' package. Install it or "
                "instantiate BacktestReporter only in environments where it is available."
            )
        self.console = console or Console()

    def display(self, results: BacktestResults) -> None:
        summary = Table(title="Backtest Results", show_header=True, header_style="bold")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="yellow")

        summary.add_row("Total Trades", str(results.total_trades))
        summary.add_row("Win Rate", f"{results.win_rate:.2%}")
        summary.add_row("Total PnL", f"${results.total_pnl:,.2f}")
        summary.add_row("Total Return", f"{results.total_pnl_pct:.2%}")
        summary.add_row("Profit Factor", f"{results.profit_factor:.2f}")
        summary.add_row("Max Drawdown", f"${results.max_drawdown:,.2f}")
        summary.add_row("Max Drawdown %", f"{results.max_drawdown_pct:.2%}")
        summary.add_row("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
        summary.add_row("Sortino Ratio", f"{results.sortino_ratio:.2f}")
        summary.add_row("Avg Hold (s)", f"{results.avg_hold_time_seconds:.0f}")

        exits = Table(title="Exit Reason Breakdown", show_header=True, header_style="bold")
        exits.add_column("Reason", style="cyan")
        exits.add_column("Count", style="yellow")

        exits.add_row("Take Profit", str(results.exits_take_profit))
        exits.add_row("Stop Loss", str(results.exits_stop_loss))
        exits.add_row("Timeout", str(results.exits_timeout))
        exits.add_row("Regime Change", str(results.exits_regime_change))

        self.console.print(summary)
        self.console.print(exits)

        if results.is_profitable():
            self.console.print(
                "[green bold]✓ Strategy shows edge! Consider paper trading.[/green bold]"
            )
        else:
            self.console.print(
                "[red bold]✗ Strategy did not meet edge criteria. Iterate on parameters.[/red bold]"
            )

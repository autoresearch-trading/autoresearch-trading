from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .engine import Trade


@dataclass
class BacktestResults:
    """Aggregate statistics for a completed backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float

    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    avg_hold_time_seconds: float

    exits_take_profit: int
    exits_stop_loss: int
    exits_timeout: int
    exits_regime_change: int

    def is_profitable(self) -> bool:
        """Determine if the strategy meets minimum profitability thresholds."""
        return (
            self.win_rate >= 0.55
            and self.profit_factor >= 1.5
            and self.max_drawdown_pct <= 0.10
            and self.total_pnl > 0
        )


def calculate_backtest_results(
    trades: Sequence["Trade"],
    *,
    initial_capital: float,
    equity_curve: Sequence[Tuple[datetime, float]] | None,
) -> BacktestResults:
    """Compute summary statistics for a list of executed trades."""
    total_trades = len(trades)
    if total_trades == 0:
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_hold_time_seconds=0.0,
            exits_take_profit=0,
            exits_stop_loss=0,
            exits_timeout=0,
            exits_regime_change=0,
        )

    pnl_values = np.array([trade.pnl for trade in trades], dtype=float)

    winning_mask = pnl_values > 0
    losing_mask = pnl_values < 0

    gross_profit = pnl_values[winning_mask].sum()
    gross_loss = pnl_values[losing_mask].sum()  # negative value

    winning_trades = int(winning_mask.sum())
    losing_trades = int(losing_mask.sum())

    win_rate = winning_trades / total_trades if total_trades else 0.0
    total_pnl = float(pnl_values.sum())
    total_pnl_pct = total_pnl / initial_capital if initial_capital else 0.0

    avg_win = float(pnl_values[winning_mask].mean()) if winning_trades else 0.0
    avg_loss = float(pnl_values[losing_mask].mean()) if losing_trades else 0.0
    largest_win = float(pnl_values.max(initial=0.0)) if winning_trades else 0.0
    largest_loss = float(pnl_values.min(initial=0.0)) if losing_trades else 0.0

    profit_factor = (
        gross_profit / abs(gross_loss)
        if abs(gross_loss) > 1e-12
        else float("inf") if gross_profit > 0 else 0.0
    )

    equity_curve = equity_curve or []
    equity_values = (
        np.array([value for _, value in equity_curve], dtype=float)
        if equity_curve
        else np.array([initial_capital], dtype=float)
    )
    max_drawdown, max_drawdown_pct = _max_drawdown(equity_values)

    returns = _to_returns(equity_values)
    sharpe_ratio = _sharpe_ratio(returns)
    sortino_ratio = _sortino_ratio(returns)

    avg_hold_time_seconds = float(
        np.mean([trade.hold_duration_seconds for trade in trades])
    )

    exits_take_profit = _count_exits(trades, "take_profit")
    exits_stop_loss = _count_exits(trades, "stop_loss")
    exits_timeout = _count_exits(trades, "timeout")
    exits_regime_change = _count_exits(trades, "regime_change")

    return BacktestResults(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        profit_factor=float(profit_factor),
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        avg_hold_time_seconds=avg_hold_time_seconds,
        exits_take_profit=exits_take_profit,
        exits_stop_loss=exits_stop_loss,
        exits_timeout=exits_timeout,
        exits_regime_change=exits_regime_change,
    )


def _count_exits(trades: Iterable["Trade"], reason: str) -> int:
    return sum(1 for trade in trades if trade.exit_reason == reason)


def _to_returns(equity_values: np.ndarray) -> np.ndarray:
    if equity_values.size < 2:
        return np.array([], dtype=float)

    prev = equity_values[:-1]
    prev[prev == 0] = 1e-12  # avoid division by zero
    returns = np.diff(equity_values) / prev
    return returns


def _max_drawdown(equity_values: np.ndarray) -> tuple[float, float]:
    if equity_values.size == 0:
        return 0.0, 0.0

    peak = equity_values[0]
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    for value in equity_values:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = float(drawdown)
            max_drawdown_pct = float(drawdown / peak) if peak > 0 else 0.0

    return max_drawdown, max_drawdown_pct


def _sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=0))
    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(252)


def _sortino_ratio(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0

    downside = returns[returns < 0]
    if downside.size == 0:
        return float("inf")

    downside_std = float(np.std(downside, ddof=0))
    if downside_std == 0:
        return 0.0

    mean_return = float(np.mean(returns))
    return mean_return / downside_std * np.sqrt(252)

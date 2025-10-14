from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from signals.base import MarketRegime

if TYPE_CHECKING:
    from .engine import BacktestConfig, Position, Trade


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str | None = None


class PositionManager:
    """Risk and position sizing utilities for the backtest engine."""

    def __init__(self, config: "BacktestConfig"):
        self.config = config

    def size_position(self, capital: float, price: float) -> float:
        """Determine position quantity based on capital allocation."""
        if price <= 0:
            return 0.0

        allocation = capital * self.config.position_size_pct
        if allocation <= 0:
            return 0.0
        return allocation / price

    def open_position(
        self,
        *,
        entry_time: datetime,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        cvd_value: float | None,
        tfi_value: float | None,
        ofi_value: float | None,
        regime: MarketRegime | None,
    ) -> "Position":
        from .engine import Position

        stop_loss = self._compute_stop_loss(entry_price, side)
        take_profit = self._compute_take_profit(entry_price, side)
        regime_name = regime.regime.value if regime else "unknown"

        return Position(
            entry_time=entry_time,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            cvd_value=cvd_value if cvd_value is not None else 0.0,
            tfi_value=tfi_value if tfi_value is not None else 0.0,
            ofi_value=ofi_value,
            regime=regime_name,
        )

    def close_position(
        self,
        position: "Position",
        *,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> "Trade":
        from .engine import Trade

        pnl = self._realized_pnl(position, exit_price)
        basis = position.entry_price * position.qty
        pnl_pct = pnl / basis if basis else 0.0
        hold_seconds = int((exit_time - position.entry_time).total_seconds())

        return Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            hold_duration_seconds=hold_seconds,
        )

    def unrealized_pnl(
        self, position: "Position", mark_price: float | None
    ) -> float:
        if mark_price is None:
            return 0.0
        return self._realized_pnl(position, mark_price)

    def should_exit(
        self,
        position: "Position",
        *,
        price: float | None,
        now: datetime,
        regime: MarketRegime | None,
    ) -> ExitDecision:
        """Evaluate exit conditions for an open position."""
        if regime and not regime.should_trade:
            return ExitDecision(True, "regime_change")

        if price is None:
            timeout = (
                now - position.entry_time
            ).total_seconds() >= self.config.max_hold_seconds
            return ExitDecision(timeout, "timeout" if timeout else None)

        if position.side == "long":
            if price <= position.stop_loss:
                return ExitDecision(True, "stop_loss")
            if price >= position.take_profit:
                return ExitDecision(True, "take_profit")
        else:
            if price >= position.stop_loss:
                return ExitDecision(True, "stop_loss")
            if price <= position.take_profit:
                return ExitDecision(True, "take_profit")

        if (now - position.entry_time).total_seconds() >= self.config.max_hold_seconds:
            return ExitDecision(True, "timeout")

        return ExitDecision(False, None)

    def _compute_stop_loss(self, entry_price: float, side: str) -> float:
        if side == "long":
            return entry_price * (1.0 - self.config.stop_loss_pct)
        return entry_price * (1.0 + self.config.stop_loss_pct)

    def _compute_take_profit(self, entry_price: float, side: str) -> float:
        if side == "long":
            return entry_price * (1.0 + self.config.take_profit_pct)
        return entry_price * (1.0 - self.config.take_profit_pct)

    def _realized_pnl(self, position: "Position", price: float) -> float:
        if position.side == "long":
            return (price - position.entry_price) * position.qty
        return (position.entry_price - price) * position.qty


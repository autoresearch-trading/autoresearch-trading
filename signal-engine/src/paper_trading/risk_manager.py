from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Tuple

import structlog
from signals.base import PaperTrade

from .models import PaperPosition

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class RiskMetrics:
    """Real-time counters backing circuit breakers."""

    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    max_drawdown_today: float = 0.0
    consecutive_losses: int = 0
    peak_capital_today: float = 0.0
    last_reset: datetime | None = None
    max_daily_loss: float = 0.0
    max_daily_trades: int = 0
    max_consecutive_losses: int = 0


class RiskManager:
    """Guardrails that halt trading when limits are breached."""

    def __init__(self, config: dict) -> None:
        initial_capital = float(config.get("initial_capital", 0.0))
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        max_daily_loss_pct = float(config.get("max_daily_loss_pct", 0.05))

        self.initial_capital = initial_capital
        self.max_position_size_pct = float(config.get("max_position_size_pct", 0.10))
        self.max_total_exposure_pct = float(config.get("max_total_exposure_pct", 0.50))
        self.max_concentration_pct = float(config.get("max_concentration_pct", 0.30))

        now = datetime.now(timezone.utc)
        self.metrics = RiskMetrics(
            daily_pnl=0.0,
            daily_trade_count=0,
            max_drawdown_today=0.0,
            consecutive_losses=0,
            peak_capital_today=initial_capital,
            last_reset=now,
            max_daily_loss=max_daily_loss_pct * initial_capital,
            max_daily_trades=int(config.get("max_daily_trades", 50)),
            max_consecutive_losses=int(config.get("max_consecutive_losses", 5)),
        )

    def can_open_position(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        current_capital: float,
        existing_positions: Iterable[PaperPosition],
    ) -> Tuple[bool, str]:
        """Return (allowed, reason) for a proposed position."""
        if price <= 0 or qty <= 0:
            return False, "invalid_position_size"

        # Check if position already exists for this symbol
        symbol_upper = symbol.upper()
        for pos in existing_positions:
            if pos.symbol == symbol_upper:
                return False, "position_already_open"

        if self.metrics.daily_pnl <= -self.metrics.max_daily_loss:
            return False, "daily_loss_limit_hit"

        if self.metrics.daily_trade_count >= self.metrics.max_daily_trades:
            return False, "daily_trade_limit_hit"

        if self.metrics.consecutive_losses >= self.metrics.max_consecutive_losses:
            return False, "consecutive_loss_limit_hit"

        notional = qty * price
        if notional > current_capital * self.max_position_size_pct:
            return False, "position_too_large"

        exposure_used = sum(pos.entry_price * pos.qty for pos in existing_positions)
        if (exposure_used + notional) > current_capital * self.max_total_exposure_pct:
            return False, "total_exposure_limit_hit"

        symbol_exposure = sum(
            pos.entry_price * pos.qty
            for pos in existing_positions
            if pos.symbol == symbol_upper
        )
        if (symbol_exposure + notional) > current_capital * self.max_concentration_pct:
            return False, "concentration_limit_hit"

        return True, ""

    def record_trade(self, trade: PaperTrade, current_capital: float) -> None:
        """Update metrics after a position is closed."""
        self.metrics.daily_pnl += trade.pnl or 0.0
        self.metrics.daily_trade_count += 1

        if trade.pnl is not None and trade.pnl < 0:
            self.metrics.consecutive_losses += 1
        else:
            self.metrics.consecutive_losses = 0

        if current_capital > self.metrics.peak_capital_today:
            self.metrics.peak_capital_today = current_capital

        drawdown = self.metrics.peak_capital_today - current_capital
        if drawdown > self.metrics.max_drawdown_today:
            self.metrics.max_drawdown_today = drawdown

        log.info(
            "paper_risk_metrics_updated",
            daily_pnl=self.metrics.daily_pnl,
            daily_trade_count=self.metrics.daily_trade_count,
            consecutive_losses=self.metrics.consecutive_losses,
            max_drawdown_today=self.metrics.max_drawdown_today,
        )

    def should_reset_daily(self, now: datetime) -> bool:
        last_reset = self.metrics.last_reset
        if last_reset is None:
            return True
        return now.date() > last_reset.date()

    def reset_daily_limits(self) -> None:
        log.info(
            "paper_risk_reset",
            prev_daily_pnl=self.metrics.daily_pnl,
            prev_trade_count=self.metrics.daily_trade_count,
            prev_drawdown=self.metrics.max_drawdown_today,
        )

        now = datetime.now(timezone.utc)
        self.metrics.daily_pnl = 0.0
        self.metrics.daily_trade_count = 0
        self.metrics.max_drawdown_today = 0.0
        self.metrics.consecutive_losses = 0
        self.metrics.peak_capital_today = self.initial_capital
        self.metrics.last_reset = now

    def get_status(self) -> dict[str, float | int | bool]:
        return {
            "daily_pnl": self.metrics.daily_pnl,
            "daily_pnl_limit": self.metrics.max_daily_loss,
            "daily_trade_count": self.metrics.daily_trade_count,
            "daily_trade_limit": self.metrics.max_daily_trades,
            "consecutive_losses": self.metrics.consecutive_losses,
            "max_consecutive_losses": self.metrics.max_consecutive_losses,
            "max_drawdown_today": self.metrics.max_drawdown_today,
            "is_trading_allowed": (
                self.metrics.daily_pnl > -self.metrics.max_daily_loss
                and self.metrics.consecutive_losses
                < self.metrics.max_consecutive_losses
            ),
        }

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import structlog

from .models import PaperPosition

log = structlog.get_logger(__name__)


class PositionTracker:
    """Manage lifecycle of in-flight paper trading positions."""

    def __init__(self, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self._positions: Dict[str, PaperPosition] = {}

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        cvd_value: float,
        tfi_value: float,
        ofi_value: float | None,
        regime: str,
    ) -> PaperPosition:
        symbol = symbol.upper()
        if symbol in self._positions:
            raise ValueError(f"position already open for {symbol}")
        if qty <= 0:
            raise ValueError("qty must be positive")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if stop_loss <= 0 or take_profit <= 0:
            raise ValueError("stop_loss and take_profit must be positive")

        position = PaperPosition(
            position_id=str(uuid4()),
            symbol=symbol.upper(),
            side=side,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            cvd_value=cvd_value,
            tfi_value=tfi_value,
            ofi_value=ofi_value,
            regime=regime,
            current_price=entry_price,
            unrealized_pnl=0.0,
            last_updated=datetime.now(timezone.utc),
        )

        notional = entry_price * qty
        self.capital -= notional
        self._positions[symbol] = position

        log.info(
            "paper_position_opened",
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            entry_price=position.entry_price,
            notional=notional,
            remaining_capital=self.capital,
        )

        return position

    def close_position(self, symbol: str, exit_price: float) -> PaperPosition:
        symbol = symbol.upper()
        position = self._positions.pop(symbol, None)
        if position is None:
            raise KeyError(f"no open position for {symbol}")
        if exit_price <= 0:
            raise ValueError("exit_price must be positive")

        pnl = self._realized_pnl(position, exit_price)
        margin = position.entry_price * position.qty
        self.capital += margin + pnl

        log.info(
            "paper_position_closed",
            symbol=position.symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl / margin if margin else 0.0,
            capital=self.capital,
        )

        return position

    def update_position_price(self, symbol: str, current_price: float) -> None:
        position = self._positions.get(symbol.upper())
        if position is None:
            return
        if current_price <= 0:
            return

        position.current_price = current_price
        position.unrealized_pnl = self._realized_pnl(position, current_price)
        position.last_updated = datetime.now(timezone.utc)

    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        return self._positions.get(symbol.upper())

    def get_all_positions(self) -> List[PaperPosition]:
        return list(self._positions.values())

    def get_total_exposure(self) -> float:
        return sum(pos.entry_price * pos.qty for pos in self._positions.values())

    def get_equity(self) -> float:
        unrealized = sum(pos.unrealized_pnl for pos in self._positions.values())
        return self.capital + unrealized

    @staticmethod
    def _realized_pnl(position: PaperPosition, price: float) -> float:
        if position.side == "long":
            return (price - position.entry_price) * position.qty
        return (position.entry_price - price) * position.qty

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import structlog
from backtest.strategy import SignalAggregator
from signals.base import MarketRegime, PaperTrade, Signal, SignalType

from .position_tracker import PositionTracker
from .risk_manager import RiskManager

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class ExitDecision:
    should_exit: bool
    reason: Optional[str] = None


class TradeExecutor:
    """Decision engine for opening and closing paper trades."""

    def __init__(
        self,
        *,
        position_tracker: PositionTracker,
        risk_manager: RiskManager,
        db_client,
        config: dict,
        dry_run: bool = False,
    ) -> None:
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.db = db_client
        self.dry_run = dry_run

        self.min_confidence = float(config.get("min_confidence", 0.5))
        self.min_signals_agree = int(config.get("min_signals_agree", 2))
        self.require_cvd = bool(config.get("require_cvd", True))
        self.require_tfi = bool(config.get("require_tfi", True))
        self.require_ofi = bool(config.get("require_ofi", False))
        self.position_size_pct = float(config.get("position_size_pct", 0.10))
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.02))
        self.take_profit_pct = float(config.get("take_profit_pct", 0.03))
        self.max_hold_seconds = int(config.get("max_hold_seconds", 180))

        self.closed_trades: List[PaperTrade] = []

    def evaluate_entry(
        self,
        *,
        symbol: str,
        signals: Iterable[Signal],
        current_price: float,
        regime: Optional[MarketRegime],
        now: datetime,
    ) -> None:
        """Evaluate aggregated signals and attempt to open a position."""
        symbol_signals = list(signals)
        if not symbol_signals:
            return
        if current_price <= 0:
            log.warning("paper_entry_invalid_price", symbol=symbol, price=current_price)
            return

        if regime and not regime.should_trade:
            log.debug("paper_entry_blocked_regime", symbol=symbol, regime=regime.regime)
            return

        long_decision, long_conf = SignalAggregator.should_enter_long(
            symbol_signals,
            min_confidence=self.min_confidence,
            min_signals=self.min_signals_agree,
            require_cvd=self.require_cvd,
            require_tfi=self.require_tfi,
            require_ofi=self.require_ofi,
        )
        short_decision, short_conf = SignalAggregator.should_enter_short(
            symbol_signals,
            min_confidence=self.min_confidence,
            min_signals=self.min_signals_agree,
            require_cvd=self.require_cvd,
            require_tfi=self.require_tfi,
            require_ofi=self.require_ofi,
        )

        side: str | None = None
        if long_decision and short_decision:
            side = "long" if long_conf >= short_conf else "short"
        elif long_decision:
            side = "long"
        elif short_decision:
            side = "short"

        if side is None:
            return

        qty = self._calculate_position_size(
            self.position_tracker.capital, current_price
        )
        if qty <= 0:
            log.warning(
                "paper_entry_insufficient_capital",
                symbol=symbol,
                capital=self.position_tracker.capital,
            )
            return

        allowed, reason = self.risk_manager.can_open_position(
            symbol=symbol,
            side=side,
            qty=qty,
            price=current_price,
            current_capital=self.position_tracker.capital,
            existing_positions=self.position_tracker.get_all_positions(),
        )
        if not allowed:
            log.warning("paper_entry_rejected", symbol=symbol, side=side, reason=reason)
            return

        stop_loss = self._compute_stop_loss(current_price, side)
        take_profit = self._compute_take_profit(current_price, side)

        cvd_value = (
            SignalAggregator.latest_signal_value(symbol_signals, SignalType.CVD) or 0.0
        )
        tfi_value = (
            SignalAggregator.latest_signal_value(symbol_signals, SignalType.TFI) or 0.0
        )
        ofi_value = SignalAggregator.latest_signal_value(symbol_signals, SignalType.OFI)

        if self.dry_run:
            log.info(
                "paper_dry_run_entry",
                symbol=symbol,
                side=side,
                price=current_price,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                cvd_value=cvd_value,
                tfi_value=tfi_value,
                ofi_value=ofi_value,
            )
            return

        position = self.position_tracker.open_position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            cvd_value=cvd_value,
            tfi_value=tfi_value,
            ofi_value=ofi_value,
            regime=(regime.regime.value if regime else "unknown"),
        )

        log.info(
            "paper_position_entered",
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            qty=position.qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def should_exit_position(
        self,
        *,
        position,
        current_price: float | None,
        regime: Optional[MarketRegime],
        now: datetime,
    ) -> ExitDecision:
        if regime and not regime.should_trade:
            return ExitDecision(True, "regime_change")

        if current_price is None or current_price <= 0:
            elapsed = (now - position.entry_time).total_seconds()
            if elapsed >= self.max_hold_seconds:
                return ExitDecision(True, "timeout")
            return ExitDecision(False, None)

        if position.side == "long":
            if current_price <= position.stop_loss:
                return ExitDecision(True, "stop_loss")
            if current_price >= position.take_profit:
                return ExitDecision(True, "take_profit")
        else:
            if current_price >= position.stop_loss:
                return ExitDecision(True, "stop_loss")
            if current_price <= position.take_profit:
                return ExitDecision(True, "take_profit")

        elapsed = (now - position.entry_time).total_seconds()
        if elapsed >= self.max_hold_seconds:
            return ExitDecision(True, "timeout")

        return ExitDecision(False, None)

    def close_position(
        self,
        *,
        position,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        if exit_price <= 0:
            log.error(
                "paper_exit_invalid_price",
                symbol=position.symbol,
                exit_price=exit_price,
            )
            return

        if self.dry_run:
            log.info(
                "paper_dry_run_exit",
                symbol=position.symbol,
                exit_reason=exit_reason,
                exit_price=exit_price,
            )
            return

        self.position_tracker.close_position(position.symbol, exit_price)
        pnl = (
            (exit_price - position.entry_price) * position.qty
            if position.side == "long"
            else (position.entry_price - exit_price) * position.qty
        )
        basis = position.entry_price * position.qty
        pnl_pct = pnl / basis if basis else 0.0

        trade = PaperTrade(
            ts=position.entry_time,
            symbol=position.symbol,
            trade_id=position.position_id,
            side=position.side,
            entry_price=position.entry_price,
            qty=position.qty,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            exit_price=exit_price,
            exit_ts=datetime.now(timezone.utc),
            pnl=pnl,
            pnl_pct=pnl_pct,
            cvd_value=position.cvd_value,
            tfi_value=position.tfi_value,
            ofi_value=position.ofi_value,
            regime=position.regime,
        )

        self.closed_trades.append(trade)
        self.risk_manager.record_trade(trade, self.position_tracker.capital)

        try:
            self.db.write_paper_trade(trade)
        except Exception as exc:  # pragma: no cover - persistence errors logged
            log.error(
                "paper_trade_persist_failed", error=str(exc), trade_id=trade.trade_id
            )

        log.info(
            "paper_position_closed",
            position_id=position.position_id,
            symbol=position.symbol,
            exit_reason=exit_reason,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            capital=self.position_tracker.capital,
        )

    def calculate_win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for trade in self.closed_trades if (trade.pnl or 0.0) > 0)
        return wins / len(self.closed_trades)

    def _calculate_position_size(self, capital: float, price: float) -> float:
        if price <= 0:
            return 0.0
        allocation = capital * self.position_size_pct
        if allocation <= 0:
            return 0.0
        return allocation / price

    def _compute_stop_loss(self, entry_price: float, side: str) -> float:
        if side == "long":
            return entry_price * (1.0 - self.stop_loss_pct)
        return entry_price * (1.0 + self.stop_loss_pct)

    def _compute_take_profit(self, entry_price: float, side: str) -> float:
        if side == "long":
            return entry_price * (1.0 + self.take_profit_pct)
        return entry_price * (1.0 - self.take_profit_pct)

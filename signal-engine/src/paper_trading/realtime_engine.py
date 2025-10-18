from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import structlog

from api.live_data import LiveDataClient
from config import Settings
from persistence.async_writer import enqueue_trade
from signals.base import Signal
from stream.signal_router import SignalRouter

from .position_tracker import PositionTracker
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor

log = structlog.get_logger(__name__)


class RealtimePaperTradingEngine:
    """Paper trading engine that reacts to live signals without database polling."""

    def __init__(self, settings: Settings, *, dry_run: bool = True) -> None:
        self.settings = settings
        self.dry_run = dry_run

        self.position_tracker = PositionTracker(settings.initial_capital)
        self.risk_manager = RiskManager(settings.risk_config())
        self.trade_executor = TradeExecutor(
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            db_client=_AsyncTradeLogger(),
            config=self._trade_config(),
            dry_run=dry_run,
        )

        self.live_data = LiveDataClient(settings)

        self._signal_buffer: Dict[str, List[Signal]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()
        self._latest_price: Dict[str, Tuple[float, datetime]] = {}
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        for symbol in self.settings.symbols:
            SignalRouter.subscribe(symbol, self._on_signal)

        log.info("realtime_paper_trading_started", symbols=self.settings.symbols, dry_run=self.dry_run)

        self._tasks = [
            asyncio.create_task(self._aggregation_loop(), name="signal_aggregation"),
            asyncio.create_task(self._position_monitor_loop(), name="position_monitor"),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            log.debug("realtime_paper_trading_cancelled")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self._running:
            return

        self._stop_event.set()

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for symbol in self.settings.symbols:
            SignalRouter.unsubscribe(symbol, self._on_signal)

        self._running = False
        log.info("realtime_paper_trading_stopped", capital=self.position_tracker.capital)

    async def _on_signal(self, signal: Signal) -> None:
        async with self._buffer_lock:
            self._signal_buffer[signal.symbol].append(signal)
            self._latest_price[signal.symbol] = (signal.price, signal.ts)
        log.debug(
            "realtime_signal_received",
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            price=signal.price,
            confidence=signal.confidence,
        )

    async def _aggregation_loop(self) -> None:
        aggregation_window = 5.0
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=aggregation_window)
                continue
            except asyncio.TimeoutError:
                pass

            now = datetime.now(timezone.utc)
            if self.risk_manager.should_reset_daily(now):
                self.risk_manager.reset_daily_limits()

            symbols_payload = await self._pop_signal_buffers()
            for symbol, signals in symbols_payload.items():
                if not signals:
                    continue

                current_price = await self._get_price(symbol)
                if current_price is None:
                    log.debug("realtime_price_unavailable", symbol=symbol)
                    continue

                self.trade_executor.evaluate_entry(
                    symbol=symbol,
                    signals=signals,
                    current_price=current_price,
                    regime=None,
                    now=now,
                )

    async def _position_monitor_loop(self) -> None:
        interval = 0.5
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                continue
            except asyncio.TimeoutError:
                pass

            now = datetime.now(timezone.utc)
            positions = list(self.position_tracker.get_all_positions())
            for position in positions:
                current_price = await self._get_price(position.symbol)
                decision = self.trade_executor.should_exit_position(
                    position=position,
                    current_price=current_price,
                    regime=None,
                    now=now,
                )
                if decision.should_exit and not self.dry_run:
                    exit_price = current_price or position.entry_price
                    self.trade_executor.close_position(
                        position=position,
                        exit_price=exit_price,
                        exit_reason=decision.reason or "unknown",
                    )
                elif current_price is not None:
                    self.position_tracker.update_position_price(position.symbol, current_price)

    async def _get_price(self, symbol: str) -> float | None:
        now = datetime.now(timezone.utc)
        cached = self._latest_price.get(symbol)
        if cached:
            price, ts = cached
            if (now - ts).total_seconds() <= 2.0:
                return price

        price = await self.live_data.fetch_price(symbol)
        if price:
            self._latest_price[symbol] = (price, now)
            return price
        if cached:
            return cached[0]
        return None

    async def _pop_signal_buffers(self) -> Dict[str, List[Signal]]:
        async with self._buffer_lock:
            payload = dict(self._signal_buffer)
            self._signal_buffer.clear()
            return payload

    def _trade_config(self) -> dict:
        return {
            "min_confidence": self.settings.min_confidence,
            "min_signals_agree": self.settings.min_signals_agree,
            "require_cvd": self.settings.require_cvd,
            "require_tfi": self.settings.require_tfi,
            "require_ofi": self.settings.require_ofi,
            "position_size_pct": self.settings.position_size_pct,
            "stop_loss_pct": self.settings.stop_loss_pct,
            "take_profit_pct": self.settings.take_profit_pct,
            "max_hold_seconds": self.settings.max_hold_seconds,
        }


class _AsyncTradeLogger:
    """Adapter so TradeExecutor can enqueue trades asynchronously."""

    def write_paper_trade(self, trade) -> None:
        enqueue_trade(trade)

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import structlog
from api.live_data import LiveDataClient
from db.questdb import QuestDBClient
from signals.base import MarketRegime, Signal

from config import Settings

from .position_tracker import PositionTracker
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor

log = structlog.get_logger(__name__)


class PaperTradingEngine:
    """Coordinate live signal processing and simulated trading."""

    def __init__(
        self,
        settings: Settings,
        *,
        dry_run: bool = True,
        poll_interval: float = 1.0,
    ) -> None:
        self.settings = settings
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.db = QuestDBClient(
            host=settings.questdb_host,
            port=settings.questdb_port,
            user=settings.questdb_user,
            password=settings.questdb_password,
        )
        self.live_data = LiveDataClient(settings)

        self.position_tracker = PositionTracker(settings.initial_capital)
        self.risk_manager = RiskManager(settings.risk_config())
        self.trade_executor = TradeExecutor(
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            db_client=self.db,
            config=self._trade_config(),
            dry_run=dry_run,
        )

        self.running = False

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        log.info(
            "paper_trading_started",
            dry_run=self.dry_run,
            symbols=self.settings.symbols,
            poll_interval=self.poll_interval,
        )

        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - top-level safeguard
            log.error(
                "paper_trading_unhandled_exception", error=str(exc), exc_info=True
            )
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self.running and not self.position_tracker.get_all_positions():
            return
        log.info("paper_trading_stopping")
        self.running = False

        if not self.dry_run:
            open_positions = list(self.position_tracker.get_all_positions())
            for position in open_positions:
                try:
                    current_price = await self._get_current_price(position.symbol)
                except Exception as exc:  # pragma: no cover - defensive
                    log.error(
                        "paper_close_price_fetch_failed",
                        symbol=position.symbol,
                        error=str(exc),
                    )
                    current_price = None

                if current_price is None:
                    continue

                self.trade_executor.close_position(
                    position=position,
                    exit_price=current_price,
                    exit_reason="shutdown",
                )

        metrics = self.get_metrics()
        log.info(
            "paper_trading_stopped",
            capital=metrics["capital"],
            total_trades=metrics["total_trades"],
            open_positions=metrics["open_positions"],
        )

    async def _trading_cycle(self) -> None:
        now = datetime.now(timezone.utc)

        if self.risk_manager.should_reset_daily(now):
            self.risk_manager.reset_daily_limits()

        for symbol in self.settings.symbols:
            try:
                symbol = symbol.upper()
                signals = await self._fetch_latest_signals(symbol, lookback_seconds=60)
                fresh_signals = self._filter_fresh_signals(
                    signals, now, freshness_window=timedelta(seconds=60)
                )
                regime = await self._fetch_latest_regime(symbol)
                current_price = await self._get_current_price(symbol)

                if current_price is None or current_price <= 0:
                    log.debug("paper_price_unavailable", symbol=symbol)
                    continue

                position = self.position_tracker.get_position(symbol)
                if position:
                    decision = self.trade_executor.should_exit_position(
                        position=position,
                        current_price=current_price,
                        regime=regime,
                        now=now,
                    )
                    if decision.should_exit and not self.dry_run:
                        self.trade_executor.close_position(
                            position=position,
                            exit_price=current_price,
                            exit_reason=decision.reason or "manual",
                        )
                    else:
                        self.position_tracker.update_position_price(
                            symbol, current_price
                        )
                    continue

                if not fresh_signals:
                    if signals:
                        latest_ts = signals[-1].ts
                        log.warning(
                            "paper_signals_stale",
                            symbol=symbol,
                            latest_signal_ts=latest_ts.isoformat(),
                            count=len(signals),
                        )
                    continue

                self.trade_executor.evaluate_entry(
                    symbol=symbol,
                    signals=fresh_signals,
                    current_price=current_price,
                    regime=regime,
                    now=now,
                )
            except Exception as exc:
                log.error(
                    "paper_trading_cycle_error",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )

    async def _fetch_latest_signals(
        self,
        symbol: str,
        *,
        lookback_seconds: int,
    ) -> List[Signal]:
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(seconds=lookback_seconds)

        return await asyncio.to_thread(
            self.db.query_signals,
            symbol,
            start_ts,
            end_ts,
            None,
        )

    async def _fetch_latest_regime(self, symbol: str) -> Optional[MarketRegime]:
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(minutes=5)

        regimes = await asyncio.to_thread(
            self.db.query_regimes,
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return regimes[-1] if regimes else None

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        price = await self.live_data.fetch_price(symbol)
        if price and price > 0:
            return price

        fallback_signals = await self._fetch_latest_signals(symbol, lookback_seconds=5)
        if fallback_signals:
            return fallback_signals[-1].price
        return None

    @staticmethod
    def _filter_fresh_signals(
        signals: Iterable[Signal],
        now: datetime,
        *,
        freshness_window: timedelta,
    ) -> List[Signal]:
        fresh: List[Signal] = []
        for signal in signals:
            if signal.price <= 0:
                continue
            ts = signal.ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            if now - ts > freshness_window:
                continue
            fresh.append(signal)
        return fresh

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

    def get_metrics(self) -> dict[str, float | int]:
        return {
            "capital": self.position_tracker.capital,
            "open_positions": len(self.position_tracker.get_all_positions()),
            "total_trades": len(self.trade_executor.closed_trades),
            "daily_pnl": self.risk_manager.metrics.daily_pnl,
            "win_rate": self.trade_executor.calculate_win_rate(),
        }

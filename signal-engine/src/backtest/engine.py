from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

from signals.base import MarketRegime, Signal, SignalType

from .metrics import BacktestResults, calculate_backtest_results
from .position_manager import ExitDecision, PositionManager
from .strategy import SignalAggregator


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""

    initial_capital: float = 10_000.0
    position_size_pct: float = 0.10
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    max_hold_seconds: int = 180
    min_confidence: float = 0.5

    require_cvd: bool = True
    require_tfi: bool = True
    require_ofi: bool = False
    min_signals_agree: int = 2


@dataclass
class Position:
    """Active trading position."""

    entry_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    cvd_value: float
    tfi_value: float
    ofi_value: Optional[float]
    regime: str


@dataclass
class Trade:
    """Closed trade with P&L."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_duration_seconds: int


class BacktestEngine:
    """Event-driven backtester for signal validation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.current_regime: dict[str, MarketRegime] = {}

        self._position_manager = PositionManager(config)
        self._latest_price: dict[str, float] = {}
        self._equity_curve: list[tuple[datetime, float]] = []

    def run(
        self,
        signals: Sequence[Signal],
        regimes: Sequence[MarketRegime] | None,
        price_data: dict | None,
    ) -> BacktestResults:
        """
        Execute backtest on historical signals.

        Process:
        1. Sort all events by timestamp
        2. Update regime state
        3. Check for position exits (SL/TP/timeout)
        4. Evaluate entry signals
        5. Calculate metrics
        """
        regimes = regimes or []
        signals_sorted = sorted(signals, key=lambda s: s.ts)
        regimes_sorted = sorted(regimes, key=lambda r: r.ts)

        price_map = self._normalize_price_data(price_data, signals_sorted)
        signals_by_ts: dict[datetime, list[Signal]] = defaultdict(list)
        for signal in signals_sorted:
            signals_by_ts[signal.ts].append(signal)

        regimes_by_ts: dict[datetime, list[MarketRegime]] = defaultdict(list)
        for regime in regimes_sorted:
            regimes_by_ts[regime.ts].append(regime)

        all_timestamps = set(signals_by_ts.keys()) | set(regimes_by_ts.keys())
        for symbol_prices in price_map.values():
            all_timestamps.update(symbol_prices.keys())

        timeline = sorted(all_timestamps)
        if not timeline and signals_sorted:
            timeline = sorted({signal.ts for signal in signals_sorted})

        for ts in timeline:
            # Replay events in chronological order to maintain deterministic outcomes.
            self._update_prices_from_data(ts, price_map)
            self._update_prices_from_signals(ts, signals_by_ts)
            self._update_regimes(ts, regimes_by_ts)
            self._evaluate_open_positions(ts)
            self._attempt_entries(ts, signals_by_ts)
            self._record_equity(ts)

        # Close any residual positions at the final observed price.
        if timeline:
            final_ts = timeline[-1]
        elif signals_sorted:
            final_ts = signals_sorted[-1].ts
        else:
            final_ts = datetime.utcnow()

        self._close_remaining_positions(final_ts)
        self._record_equity(final_ts)

        return calculate_backtest_results(
            self.trades,
            initial_capital=self.config.initial_capital,
            equity_curve=self._equity_curve,
        )

    def _update_prices_from_data(
        self,
        ts: datetime,
        price_map: dict[str, dict[datetime, float]],
    ) -> None:
        for symbol, prices in price_map.items():
            price = prices.get(ts)
            if price is not None:
                self._latest_price[symbol] = price

    def _update_prices_from_signals(
        self,
        ts: datetime,
        signals_by_ts: dict[datetime, list[Signal]],
    ) -> None:
        for signal in signals_by_ts.get(ts, []):
            self._latest_price[signal.symbol] = signal.price

    def _update_regimes(
        self,
        ts: datetime,
        regimes_by_ts: dict[datetime, list[MarketRegime]],
    ) -> None:
        for regime in regimes_by_ts.get(ts, []):
            self.current_regime[regime.symbol] = regime

    def _evaluate_open_positions(self, ts: datetime) -> None:
        # Iterate over a snapshot to allow safe removal during the loop.
        for symbol, position in list(self.positions.items()):
            price = self._latest_price.get(symbol)
            regime = self.current_regime.get(symbol)
            decision: ExitDecision = self._position_manager.should_exit(
                position,
                price=price,
                now=ts,
                regime=regime,
            )
            if decision.should_exit and decision.reason:
                exit_price = price if price is not None else position.entry_price
                trade = self._position_manager.close_position(
                    position,
                    exit_price=exit_price,
                    exit_time=ts,
                    exit_reason=decision.reason,
                )
                self._finalize_trade(symbol, position, trade)

    def _attempt_entries(
        self,
        ts: datetime,
        signals_by_ts: dict[datetime, list[Signal]],
    ) -> None:
        signals = signals_by_ts.get(ts, [])
        if not signals:
            return

        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        # Group by symbol so long/short aggregations see the full local context.
        for signal in signals:
            by_symbol[signal.symbol].append(signal)

        for symbol, symbol_signals in by_symbol.items():
            if symbol in self.positions:
                continue

            regime = self.current_regime.get(symbol)
            if regime and not regime.should_trade:
                continue

            long_decision, long_conf = SignalAggregator.should_enter_long(
                symbol_signals,
                min_confidence=self.config.min_confidence,
                min_signals=self.config.min_signals_agree,
                require_cvd=self.config.require_cvd,
                require_tfi=self.config.require_tfi,
                require_ofi=self.config.require_ofi,
            )
            short_decision, short_conf = SignalAggregator.should_enter_short(
                symbol_signals,
                min_confidence=self.config.min_confidence,
                min_signals=self.config.min_signals_agree,
                require_cvd=self.config.require_cvd,
                require_tfi=self.config.require_tfi,
                require_ofi=self.config.require_ofi,
            )

            side: str | None = None
            if long_decision and short_decision:
                side = "long" if long_conf >= short_conf else "short"
            elif long_decision:
                side = "long"
            elif short_decision:
                side = "short"

            if side is None:
                continue

            entry_price = self._latest_price.get(symbol)
            if entry_price is None or entry_price <= 0:
                continue

            qty = self._position_manager.size_position(self.capital, entry_price)
            if qty <= 0:
                continue

            notional = qty * entry_price
            if notional > self.capital:
                continue

            cvd_value = SignalAggregator.latest_signal_value(
                symbol_signals, SignalType.CVD
            )
            tfi_value = SignalAggregator.latest_signal_value(
                symbol_signals, SignalType.TFI
            )
            ofi_value = SignalAggregator.latest_signal_value(
                symbol_signals, SignalType.OFI
            )

            position = self._position_manager.open_position(
                entry_time=ts,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                qty=qty,
                cvd_value=cvd_value,
                tfi_value=tfi_value,
                ofi_value=ofi_value,
                regime=regime,
            )

            self.positions[symbol] = position
            self.capital -= notional

    def _finalize_trade(self, symbol: str, position: Position, trade: Trade) -> None:
        self.trades.append(trade)
        margin_return = position.entry_price * position.qty
        self.capital += margin_return + trade.pnl
        self.positions.pop(symbol, None)

    def _record_equity(self, ts: datetime) -> None:
        equity = self.capital
        for symbol, position in self.positions.items():
            mark_price = self._latest_price.get(symbol, position.entry_price)
            if position.side == "long":
                equity += mark_price * position.qty
            else:
                margin = position.entry_price * position.qty
                pnl = (position.entry_price - mark_price) * position.qty
                equity += margin + pnl
        # Avoid duplicate timestamps so downstream plotting remains stable.
        if not self._equity_curve or self._equity_curve[-1][0] != ts:
            self._equity_curve.append((ts, equity))
        else:
            self._equity_curve[-1] = (ts, equity)

    def _close_remaining_positions(self, ts: datetime) -> None:
        for symbol, position in list(self.positions.items()):
            price = self._latest_price.get(symbol, position.entry_price)
            trade = self._position_manager.close_position(
                position,
                exit_price=price,
                exit_time=ts,
                exit_reason="timeout",
            )
            self._finalize_trade(symbol, position, trade)

    @staticmethod
    def _normalize_price_data(
        price_data: dict | None,
        signals: Sequence[Signal],
    ) -> dict[str, dict[datetime, float]]:
        normalized: dict[str, dict[datetime, float]] = defaultdict(dict)
        if not price_data:
            for signal in signals:
                normalized[signal.symbol][signal.ts] = signal.price
            return normalized

        items = list(price_data.items())
        if not items:
            for signal in signals:
                normalized[signal.symbol][signal.ts] = signal.price
            return normalized

        sample_value = items[0][1]

        if isinstance(sample_value, dict):
            for symbol, series in price_data.items():
                normalized[symbol] = {
                    ts: float(price) for ts, price in series.items()
                }
        else:
            symbol = signals[0].symbol if signals else "symbol"
            normalized[symbol] = {
                ts: float(price) for ts, price in items
            }

        return normalized

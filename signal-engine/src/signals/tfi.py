from __future__ import annotations

from collections import deque
from datetime import timedelta
from typing import Optional

from .base import Signal, SignalDirection, SignalType, Trade


class TFICalculator:
    """Trade flow imbalance calculator over a rolling time window."""

    def __init__(
        self,
        symbol: str,
        window_seconds: int = 60,
        signal_threshold: float = 0.3,
    ) -> None:
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.signal_threshold = signal_threshold
        self.trade_window: deque[Trade] = deque()
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0
        self.window_span = timedelta(seconds=window_seconds)
        # Drift prevention: recalculate volumes periodically
        self._eviction_count: int = 0
        self._recalc_interval: int = 100

    def process_trade(self, trade: Trade) -> Optional[Signal]:
        """Process a trade, returning a signal when imbalance is high."""
        self.trade_window.append(trade)
        if trade.side == "buy":
            self.buy_volume += trade.qty
        else:
            self.sell_volume += trade.qty

        self._evict_old_trades(current_ts=trade.ts)

        total_volume = self.buy_volume + self.sell_volume
        if total_volume <= 0:
            return None

        tfi = (self.buy_volume - self.sell_volume) / total_volume
        if abs(tfi) < self.signal_threshold:
            return None

        direction = SignalDirection.BULLISH if tfi > 0 else SignalDirection.BEARISH
        confidence = min(abs(tfi) / 0.5, 1.0)

        return Signal(
            ts=trade.ts,
            recv_ts=trade.recv_ts,
            symbol=self.symbol,
            signal_type=SignalType.TFI,
            value=tfi,
            confidence=confidence,
            direction=direction,
            price=trade.price,
            spread_bps=0,
            bid_depth=0.0,
            ask_depth=0.0,
            metadata={
                "buy_volume": self.buy_volume,
                "sell_volume": self.sell_volume,
                "trade_count": len(self.trade_window),
                "window_seconds": self.window_seconds,
            },
        )

    def _evict_old_trades(self, current_ts):
        """Remove trades outside the rolling window."""
        cutoff = current_ts - self.window_span
        while self.trade_window and self.trade_window[0].ts < cutoff:
            old_trade = self.trade_window.popleft()
            if old_trade.side == "buy":
                self.buy_volume -= old_trade.qty
            else:
                self.sell_volume -= old_trade.qty
            self._eviction_count += 1

        # Periodic recalculation to prevent floating-point drift
        if self._eviction_count >= self._recalc_interval:
            self._recalculate_volumes()
            self._eviction_count = 0

    def _recalculate_volumes(self):
        """Recalculate volumes from deque to correct floating-point drift."""
        self.buy_volume = sum(t.qty for t in self.trade_window if t.side == "buy")
        self.sell_volume = sum(t.qty for t in self.trade_window if t.side == "sell")

    def reset(self) -> None:
        """Reset internal state, e.g. when switching regimes."""
        self.trade_window.clear()
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self._eviction_count = 0

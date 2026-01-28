from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .base import Signal, SignalDirection, SignalType, Trade


class CVDCalculator:
    """Cumulative volume delta with divergence detection.

    Detects bearish divergence (price higher high, CVD lower high) and
    bullish divergence (price lower low, CVD higher low) to signal
    potential reversals.
    """

    def __init__(
        self,
        symbol: str,
        lookback_periods: int = 100,
        divergence_threshold: float = 0.10,
        min_divergence_denom: float = 1.0,
        price_change_threshold: float = 0.0005,
    ) -> None:
        """Initialize CVD calculator.

        Args:
            symbol: Trading symbol
            lookback_periods: Number of trades to track for divergence (default 100 ~2min)
            divergence_threshold: Minimum CVD divergence ratio to trigger signal
            min_divergence_denom: Floor for denominator to prevent division issues
            price_change_threshold: Minimum price change ratio for HH/LL detection
                                   (0.0005 = 0.05%, ~$43 at $87k BTC)
        """
        self.symbol = symbol
        self.lookback_periods = lookback_periods
        self.divergence_threshold = divergence_threshold
        self.min_divergence_denom = min_divergence_denom
        self.price_change_threshold = price_change_threshold
        self.cvd_cumulative: float = 0.0
        self.price_history: deque[float] = deque(maxlen=lookback_periods)
        self.cvd_history: deque[float] = deque(maxlen=lookback_periods)

    def process_trade(self, trade: Trade) -> Optional[Signal]:
        """Update CVD with the given trade and emit a signal on divergence."""
        volume_delta = trade.qty if trade.side == "buy" else -trade.qty
        self.cvd_cumulative += volume_delta
        self.price_history.append(trade.price)
        self.cvd_history.append(self.cvd_cumulative)

        if len(self.price_history) < self.lookback_periods:
            return None

        direction, confidence = self._detect_divergence()
        if direction is SignalDirection.NEUTRAL:
            return None

        return Signal(
            ts=trade.ts,
            recv_ts=trade.recv_ts,
            symbol=self.symbol,
            signal_type=SignalType.CVD,
            value=self.cvd_cumulative,
            confidence=confidence,
            direction=direction,
            price=trade.price,
            spread_bps=0,
            bid_depth=0.0,
            ask_depth=0.0,
            metadata={
                "volume_delta": volume_delta,
                "lookback_high": max(self.price_history),
                "lookback_low": min(self.price_history),
                "cvd_high": max(self.cvd_history),
                "cvd_low": min(self.cvd_history),
            },
        )

    def _detect_divergence(self) -> tuple[SignalDirection, float]:
        """Check for bullish/bearish divergence between price and CVD."""
        prices = np.fromiter(self.price_history, dtype=float)
        cvds = np.fromiter(self.cvd_history, dtype=float)
        mid_point = len(prices) // 2

        first_high = np.max(prices[:mid_point])
        second_high = np.max(prices[mid_point:])
        first_low = np.min(prices[:mid_point])
        second_low = np.min(prices[mid_point:])

        first_cvd_high = np.max(cvds[:mid_point])
        second_cvd_high = np.max(cvds[mid_point:])
        first_cvd_low = np.min(cvds[:mid_point])
        second_cvd_low = np.min(cvds[mid_point:])

        # Bearish divergence: price makes higher high, but CVD makes lower high
        price_hh_threshold = 1.0 + self.price_change_threshold
        if second_high > first_high * price_hh_threshold:
            denom = max(abs(first_cvd_high), self.min_divergence_denom)
            cvd_divergence = (first_cvd_high - second_cvd_high) / denom
            if cvd_divergence > self.divergence_threshold:
                confidence = min(cvd_divergence / 0.3, 1.0)
                return SignalDirection.BEARISH, confidence

        # Bullish divergence: price makes lower low, but CVD makes higher low
        price_ll_threshold = 1.0 - self.price_change_threshold
        if second_low < first_low * price_ll_threshold:
            denom = max(abs(first_cvd_low), self.min_divergence_denom)
            cvd_divergence = (second_cvd_low - first_cvd_low) / denom
            if cvd_divergence > self.divergence_threshold:
                confidence = min(cvd_divergence / 0.3, 1.0)
                return SignalDirection.BULLISH, confidence

        return SignalDirection.NEUTRAL, 0.0

    def reset(self) -> None:
        """Reset internal state, e.g. when switching regimes."""
        self.cvd_cumulative = 0.0
        self.price_history.clear()
        self.cvd_history.clear()

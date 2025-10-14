from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np
import talib

from signals.base import MarketRegime, RegimeState


class ATRRegimeDetector:
    """ATR-driven regime classification with liquidity and funding filters."""

    def __init__(
        self,
        symbol: str,
        atr_period: int = 14,
        atr_threshold_multiplier: float = 1.5,
        spread_threshold_bps: int = 15,
        min_depth_threshold: float = 10.0,
        extreme_funding_threshold: float = 0.001,
    ) -> None:
        self.symbol = symbol
        self.atr_period = atr_period
        self.atr_threshold_multiplier = atr_threshold_multiplier
        self.spread_threshold_bps = spread_threshold_bps
        self.min_depth_threshold = min_depth_threshold
        self.extreme_funding_threshold = extreme_funding_threshold

        maxlen = atr_period + 1
        self.high_prices: deque[float] = deque(maxlen=maxlen)
        self.low_prices: deque[float] = deque(maxlen=maxlen)
        self.close_prices: deque[float] = deque(maxlen=maxlen)

        self.current_spread_bps: int = 0
        self.current_bid_depth: float = 0.0
        self.current_ask_depth: float = 0.0
        self.current_funding_rate: float = 0.0
        self.last_ts: datetime | None = None

    def update_price(self, ts: datetime, high: float, low: float, close: float) -> None:
        """Append candle values for ATR calculation."""
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.close_prices.append(close)
        self.last_ts = ts

    def update_orderbook_context(
        self,
        spread_bps: int,
        bid_depth: float,
        ask_depth: float,
    ) -> None:
        """Refresh orderbook-driven liquidity metrics."""
        self.current_spread_bps = spread_bps
        self.current_bid_depth = bid_depth
        self.current_ask_depth = ask_depth

    def update_funding_rate(self, funding_rate: float) -> None:
        """Record most recent funding rate."""
        self.current_funding_rate = funding_rate

    def detect_regime(self, ts: datetime | None = None) -> MarketRegime:
        """Return current regime classification."""
        eval_ts = ts or self.last_ts
        if eval_ts is None:
            raise ValueError("Timestamp required before regime detection.")

        if len(self.close_prices) < self.atr_period:
            return MarketRegime(
                ts=eval_ts,
                symbol=self.symbol,
                regime=RegimeState.HIGH_VOL,
                atr=0.0,
                spread_bps=self.current_spread_bps,
                funding_rate=self.current_funding_rate,
                should_trade=False,
            )

        highs = np.fromiter(self.high_prices, dtype=float)
        lows = np.fromiter(self.low_prices, dtype=float)
        closes = np.fromiter(self.close_prices, dtype=float)
        atr = float(
            talib.ATR(highs, lows, closes, timeperiod=self.atr_period)[-1]
        )

        typical_atr = float(np.median(closes) * 0.02)
        is_high_vol = atr > typical_atr * self.atr_threshold_multiplier
        is_wide_spread = self.current_spread_bps > self.spread_threshold_bps
        is_low_liquidity = (
            self.current_bid_depth < self.min_depth_threshold
            or self.current_ask_depth < self.min_depth_threshold
        )
        is_extreme_funding = abs(self.current_funding_rate) > self.extreme_funding_threshold

        if is_extreme_funding:
            regime = RegimeState.RISK_OFF
            should_trade = False
        elif is_low_liquidity or is_wide_spread:
            regime = RegimeState.LOW_LIQUIDITY
            should_trade = False
        elif is_high_vol:
            regime = RegimeState.HIGH_VOL
            should_trade = False
        else:
            regime = RegimeState.LOW_VOL_TRENDING
            should_trade = True

        return MarketRegime(
            ts=eval_ts,
            symbol=self.symbol,
            regime=regime,
            atr=atr,
            spread_bps=self.current_spread_bps,
            funding_rate=self.current_funding_rate,
            should_trade=should_trade,
        )

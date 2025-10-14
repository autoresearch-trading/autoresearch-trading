from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import OrderbookSnapshot, Signal, SignalDirection, SignalType


@dataclass
class _OFIStats:
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        """Welford running variance."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.variance += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.variance / (self.count - 1)))


class OFICalculator:
    """Order Flow Imbalance calculator using top-of-book deltas."""

    def __init__(
        self,
        symbol: str,
        signal_threshold_sigma: float = 2.0,
        history_size: int = 100,
    ) -> None:
        self.symbol = symbol
        self.signal_threshold_sigma = signal_threshold_sigma
        self.history_size = history_size
        self._prev_snapshot: Optional[OrderbookSnapshot] = None
        self._history: list[float] = []
        self._stats = _OFIStats()

    def process_snapshot(self, snapshot: OrderbookSnapshot) -> Optional[Signal]:
        """Process orderbook snapshot and emit an OFI signal if relevant."""
        if self._prev_snapshot is None:
            self._prev_snapshot = snapshot
            return None

        prev = self._prev_snapshot

        curr_bid_price, curr_bid_qty = snapshot.bids[0]
        prev_bid_price, prev_bid_qty = prev.bids[0]
        curr_ask_price, curr_ask_qty = snapshot.asks[0]
        prev_ask_price, prev_ask_qty = prev.asks[0]

        bid_component = self._bid_component(
            curr_bid_price, curr_bid_qty, prev_bid_price, prev_bid_qty
        )
        ask_component = self._ask_component(
            curr_ask_price, curr_ask_qty, prev_ask_price, prev_ask_qty
        )

        ofi = bid_component - ask_component
        self._track(ofi)

        std = self._stats.std
        if self._stats.count < 20 or std == 0.0:
            self._prev_snapshot = snapshot
            return None

        z_score = (ofi - self._stats.mean) / std
        if abs(z_score) < self.signal_threshold_sigma:
            self._prev_snapshot = snapshot
            return None

        direction = SignalDirection.BULLISH if z_score > 0 else SignalDirection.BEARISH
        confidence = min(abs(z_score) / 4.0, 1.0)

        signal = Signal(
            ts=snapshot.ts,
            recv_ts=snapshot.ts,
            symbol=self.symbol,
            signal_type=SignalType.OFI,
            value=ofi,
            confidence=confidence,
            direction=direction,
            price=snapshot.mid_price,
            spread_bps=snapshot.spread_bps,
            bid_depth=sum(qty for _, qty in snapshot.bids),
            ask_depth=sum(qty for _, qty in snapshot.asks),
            metadata={
                "z_score": z_score,
                "mean_ofi": self._stats.mean,
                "std_ofi": std,
                "history_count": self._stats.count,
            },
        )

        self._prev_snapshot = snapshot
        return signal

    def reset(self) -> None:
        """Reset internal statistics."""
        self._prev_snapshot = None
        self._history.clear()
        self._stats = _OFIStats()

    def _track(self, value: float) -> None:
        self._history.append(value)
        self._stats.update(value)
        if len(self._history) > self.history_size:
            self._history.pop(0)

    @staticmethod
    def _bid_component(
        curr_price: float,
        curr_qty: float,
        prev_price: float,
        prev_qty: float,
    ) -> float:
        if curr_price >= prev_price:
            return curr_qty
        if curr_price <= prev_price:
            return -prev_qty
        return 0.0

    @staticmethod
    def _ask_component(
        curr_price: float,
        curr_qty: float,
        prev_price: float,
        prev_qty: float,
    ) -> float:
        if curr_price <= prev_price:
            return curr_qty
        if curr_price >= prev_price:
            return -prev_qty
        return 0.0

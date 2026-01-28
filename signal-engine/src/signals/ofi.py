from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import OrderbookSnapshot, Signal, SignalDirection, SignalType


@dataclass
class _OFIStats:
    """Statistics container for OFI z-score calculation."""

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

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
        if len(self._history) > self.history_size:
            self._history.pop(0)

        # Compute stats from bounded history (not accumulating Welford)
        # This ensures z-score reflects only recent behavior
        if len(self._history) >= 2:
            self._stats.mean = float(np.mean(self._history))
            self._stats.variance = float(np.var(self._history, ddof=1)) * (
                len(self._history) - 1
            )
            self._stats.count = len(self._history)

    @staticmethod
    def _bid_component(
        curr_price: float,
        curr_qty: float,
        prev_price: float,
        prev_qty: float,
    ) -> float:
        """Compute bid-side OFI component.

        Per MASTER_IDEA.md formula:
        I{P^B_n >= P^B_{n-1}} * q^B_n - I{P^B_n <= P^B_{n-1}} * q^B_{n-1}
        """
        if curr_price > prev_price:
            # Price improved: new buying pressure
            return curr_qty
        elif curr_price < prev_price:
            # Price worsened: buying pressure left
            return -prev_qty
        else:
            # Price unchanged: net change in depth (both indicators = 1)
            return curr_qty - prev_qty

    @staticmethod
    def _ask_component(
        curr_price: float,
        curr_qty: float,
        prev_price: float,
        prev_qty: float,
    ) -> float:
        """Compute ask-side OFI component.

        Per MASTER_IDEA.md formula:
        I{P^A_n <= P^A_{n-1}} * q^A_n - I{P^A_n >= P^A_{n-1}} * q^A_{n-1}
        """
        if curr_price < prev_price:
            # Price improved (lower ask): new selling pressure
            return curr_qty
        elif curr_price > prev_price:
            # Price worsened (higher ask): selling pressure left
            return -prev_qty
        else:
            # Price unchanged: net change in depth (both indicators = 1)
            return curr_qty - prev_qty

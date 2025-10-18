from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Tuple

import structlog
from bytewax.inputs import DynamicSource, StatelessSourcePartition

from config import Settings
from live.stream_client import PacificaStreamClient
from signals.base import OrderbookSnapshot, Trade

log = structlog.get_logger(__name__)


class LiveTradeStream(DynamicSource[Trade]):
    """Bytewax source that streams live trades via low-latency polling."""

    def __init__(self, settings: Settings, *, poll_interval: float = 1.0) -> None:
        self.settings = settings
        self.poll_interval = poll_interval

    def build(
        self,
        step_id: str,
        worker_index: int,
        worker_count: int,
    ) -> StatelessSourcePartition[Trade]:
        symbols = self.settings.symbols[worker_index::worker_count]
        return LiveTradePartition(
            settings=self.settings,
            symbols=symbols,
            poll_interval=self.poll_interval,
        )


class LiveTradePartition(StatelessSourcePartition[Trade]):
    """Fetch new trades for the assigned symbols."""

    def __init__(
        self,
        *,
        settings: Settings,
        symbols: List[str],
        poll_interval: float,
    ) -> None:
        self.settings = settings
        self.symbols = [sym.upper() for sym in symbols]
        self.poll_interval = poll_interval
        self._client = PacificaStreamClient(settings)
        self._last_seen: Dict[str, Tuple[datetime, str]] = {}
        self._last_poll = 0.0
        self._idle_sleep = 0.25
        self._buffer: List[Trade] = []

    def next_batch(self) -> List[Trade]:
        # Emit buffered items one at a time
        if self._buffer:
            return [self._buffer.pop(0)]

        if not self.symbols:
            time.sleep(self.poll_interval)
            return []

        self._throttle()

        # Fetch new trades and populate buffer
        for symbol in self.symbols:
            trades = self._client.fetch_trades(symbol)
            if not trades:
                continue
            last_seen = self._last_seen.get(symbol)
            for trade in trades:
                if last_seen:
                    last_ts, last_id = last_seen
                    if trade.ts < last_ts:
                        continue
                    if trade.ts == last_ts and trade.trade_id <= last_id:
                        continue
                self._buffer.append(trade)
                last_seen = (trade.ts, trade.trade_id)

            if last_seen:
                self._last_seen[symbol] = last_seen

        if self._buffer:
            log.debug("live_trade_partition_fetched", symbols=len(self.symbols), trades=len(self._buffer))
            return [self._buffer.pop(0)]

        # No trades available, sleep and return empty to continue polling
        time.sleep(self._idle_sleep)
        return []


    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_poll
        wait = self.poll_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_poll = time.monotonic()


class LiveOrderbookStream(DynamicSource[OrderbookSnapshot]):
    """Bytewax source that streams orderbook snapshots."""

    def __init__(self, settings: Settings, *, poll_interval: float = 3.0, depth: int = 5) -> None:
        self.settings = settings
        self.poll_interval = poll_interval
        self.depth = depth

    def build(
        self,
        step_id: str,
        worker_index: int,
        worker_count: int,
    ) -> StatelessSourcePartition[OrderbookSnapshot]:
        symbols = self.settings.symbols[worker_index::worker_count]
        return LiveOrderbookPartition(
            settings=self.settings,
            symbols=symbols,
            poll_interval=self.poll_interval,
            depth=self.depth,
        )


class LiveOrderbookPartition(StatelessSourcePartition[OrderbookSnapshot]):
    def __init__(
        self,
        *,
        settings: Settings,
        symbols: List[str],
        poll_interval: float,
        depth: int,
    ) -> None:
        self.settings = settings
        self.symbols = [sym.upper() for sym in symbols]
        self.poll_interval = poll_interval
        self.depth = depth
        self._client = PacificaStreamClient(settings)
        self._last_poll = 0.0
        self._idle_sleep = 0.5
        self._last_ts: Dict[str, datetime] = {}
        self._buffer: List[OrderbookSnapshot] = []

    def next_batch(self) -> List[OrderbookSnapshot]:
        # Emit buffered items one at a time
        if self._buffer:
            return [self._buffer.pop(0)]

        if not self.symbols:
            time.sleep(self.poll_interval)
            return []

        self._throttle()

        # Fetch new orderbook snapshots and populate buffer
        for symbol in self.symbols:
            snapshot = self._client.fetch_orderbook(symbol, depth=self.depth)
            if snapshot is None:
                continue

            last_ts = self._last_ts.get(symbol)
            if last_ts and snapshot.ts <= last_ts:
                continue

            self._last_ts[symbol] = snapshot.ts
            self._buffer.append(snapshot)

        if self._buffer:
            log.debug(
                "live_orderbook_partition_fetched",
                symbols=len(self.symbols),
                snapshots=len(self._buffer),
            )
            return [self._buffer.pop(0)]

        # No snapshots available, sleep and return empty to continue polling
        time.sleep(self._idle_sleep)
        return []


    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_poll
        wait = self.poll_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_poll = time.monotonic()

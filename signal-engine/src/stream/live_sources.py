from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Tuple

import structlog
from bytewax.inputs import DynamicSource, StatefulSourcePartition

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
    ) -> StatefulSourcePartition[Trade, dict]:
        symbols = self.settings.symbols[worker_index::worker_count]
        return LiveTradePartition(
            settings=self.settings,
            symbols=symbols,
            poll_interval=self.poll_interval,
        )


class LiveTradePartition(StatefulSourcePartition[Trade, dict]):
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

    def next_batch(self) -> List[Trade]:
        if not self.symbols:
            time.sleep(self.poll_interval)
            return []

        self._throttle()

        batch: List[Trade] = []
        for symbol in self.symbols:
            trades = self._client.fetch_trades(symbol)
            if not trades:
                continue
            last_seen = self._last_seen.get(symbol)
            new_trades: List[Trade] = []
            for trade in trades:
                if last_seen:
                    last_ts, last_id = last_seen
                    if trade.ts < last_ts:
                        continue
                    if trade.ts == last_ts and trade.trade_id <= last_id:
                        continue
                new_trades.append(trade)
                last_seen = (trade.ts, trade.trade_id)

            if last_seen:
                self._last_seen[symbol] = last_seen

            if new_trades:
                batch.extend(new_trades)

        if not batch:
            time.sleep(self._idle_sleep)
        else:
            log.debug("live_trade_partition_batch", symbols=len(self.symbols), trades=len(batch))

        return batch

    def snapshot(self) -> dict:
        serializable: Dict[str, dict[str, str]] = {}
        for symbol, (ts, trade_id) in self._last_seen.items():
            serializable[symbol] = {"ts": ts.isoformat(), "trade_id": trade_id}
        return {"last_seen": serializable}

    def restore(self, state: dict) -> None:  # pragma: no cover - restart recovery path
        last_seen = state.get("last_seen", {})
        restored: Dict[str, Tuple[datetime, str]] = {}
        for symbol, payload in last_seen.items():
            try:
                restored[symbol] = (datetime.fromisoformat(payload["ts"]), payload["trade_id"])
            except Exception:
                continue
        self._last_seen = restored

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
    ) -> StatefulSourcePartition[OrderbookSnapshot, dict]:
        symbols = self.settings.symbols[worker_index::worker_count]
        return LiveOrderbookPartition(
            settings=self.settings,
            symbols=symbols,
            poll_interval=self.poll_interval,
            depth=self.depth,
        )


class LiveOrderbookPartition(StatefulSourcePartition[OrderbookSnapshot, dict]):
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

    def next_batch(self) -> List[OrderbookSnapshot]:
        if not self.symbols:
            time.sleep(self.poll_interval)
            return []

        self._throttle()

        batch: List[OrderbookSnapshot] = []
        for symbol in self.symbols:
            snapshot = self._client.fetch_orderbook(symbol, depth=self.depth)
            if snapshot is None:
                continue

            last_ts = self._last_ts.get(symbol)
            if last_ts and snapshot.ts <= last_ts:
                continue

            self._last_ts[symbol] = snapshot.ts
            batch.append(snapshot)

        if not batch:
            time.sleep(self._idle_sleep)
        else:
            log.debug(
                "live_orderbook_partition_batch",
                symbols=len(self.symbols),
                snapshots=len(batch),
            )

        return batch

    def snapshot(self) -> dict:
        return {"last_ts": {symbol: ts.isoformat() for symbol, ts in self._last_ts.items()}}

    def restore(self, state: dict) -> None:  # pragma: no cover - restart recovery path
        restored: Dict[str, datetime] = {}
        for symbol, value in state.get("last_ts", {}).items():
            try:
                restored[symbol] = datetime.fromisoformat(value)
            except Exception:
                continue
        self._last_ts = restored

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_poll
        wait = self.poll_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_poll = time.monotonic()

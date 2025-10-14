from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Dict, Iterable, List, Optional, Sequence

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .config import APISettings
from .rate import RateController
from .storage import ParquetWriter
from .transform import (
    to_funding_rows,
    to_orderbook_rows,
    to_price_rows,
    to_trade_rows,
)
from .utils import now_ms

PollConfig = Dict[str, float]

log = structlog.get_logger(__name__)


class LiveRunner:
    """Async polling engine that writes Pacifica data to partitioned Parquet files."""

    def __init__(
        self,
        settings: APISettings,
        *,
        max_rps: int,
        out_root: str,
        book_depth: int,
        agg_level: Optional[int],
        per_endpoint_rps: Optional[Dict[str, int]] = None,
    ) -> None:
        self.settings = settings
        self.rate = RateController(max_rps, per_endpoint=per_endpoint_rps)
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout,
            http2=True,
            headers=self._default_headers(),
        )
        self.book_depth = book_depth
        self.agg_level = agg_level
        self._stop_event = asyncio.Event()
        self._writers = {
            "prices": ParquetWriter(out_root, "prices"),
            "trades": ParquetWriter(out_root, "trades"),
            "orderbook": ParquetWriter(out_root, "orderbook"),
            "funding": ParquetWriter(out_root, "funding"),
        }

    def _default_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self, symbols: Sequence[str], poll_config: PollConfig) -> None:
        tasks: List[asyncio.Task] = []
        try:
            if poll_config.get("prices"):
                tasks.append(asyncio.create_task(self._poll_prices(symbols, poll_config["prices"]), name="prices"))
            if poll_config.get("trades"):
                tasks.append(asyncio.create_task(self._poll_trades(symbols, poll_config["trades"]), name="trades"))
            if poll_config.get("orderbook"):
                tasks.append(asyncio.create_task(self._poll_orderbook(symbols, poll_config["orderbook"]), name="orderbook"))
            if poll_config.get("funding"):
                tasks.append(asyncio.create_task(self._poll_funding(symbols, poll_config["funding"]), name="funding"))

            if not tasks:
                log.warning("No polling tasks were scheduled; exiting immediately")
                return

            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("LiveRunner cancelled")
        finally:
            self.request_stop()
            for task in tasks:
                task.cancel()
            await self._drain(tasks)
            await asyncio.gather(*(writer.flush() for writer in self._writers.values()))
            await self._client.aclose()

    async def _drain(self, tasks: Iterable[asyncio.Task]) -> None:
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.5, max=5))
    async def _fetch(self, endpoint: str, *, params: Optional[Dict[str, object]] = None, key: Optional[str] = None) -> Dict[str, object]:
        async with self.rate.throttle(key):
            response = await self._client.get(endpoint, params=params)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and not payload.get("success", True):
            error = payload.get("error") or payload.get("code") or "unknown error"
            raise RuntimeError(f"Pacifica API error: {error}")
        return payload  # type: ignore[return-value]

    async def _poll_prices(self, symbols: Sequence[str], interval: float) -> None:
        symbol_set = {sym.upper() for sym in symbols}
        writer = self._writers["prices"]
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                payload = await self._fetch("/info/prices", key="prices")
                rows = to_price_rows(payload, recv_ms=now_ms(), filter_symbols=symbol_set)
                if rows:
                    await writer.append(rows)
                    log.debug("prices_written", count=len(rows))
            except Exception as exc:
                log.warning("prices_poll_error", error=str(exc))
            await self._sleep_remaining(started, interval)

    async def _poll_trades(self, symbols: Sequence[str], interval: float) -> None:
        writer = self._writers["trades"]
        while not self._stop_event.is_set():
            started = time.perf_counter()
            for symbol in symbols:
                if self._stop_event.is_set():
                    break
                try:
                    payload = await self._fetch("/trades", params={"symbol": symbol}, key="trades")
                    rows = to_trade_rows(payload, recv_ms=now_ms(), symbol=symbol)
                    if rows:
                        await writer.append(rows)
                        log.debug("trades_written", symbol=symbol, count=len(rows))
                except Exception as exc:
                    log.warning("trades_poll_error", symbol=symbol, error=str(exc))
            await self._sleep_remaining(started, interval)

    async def _poll_orderbook(self, symbols: Sequence[str], interval: float) -> None:
        writer = self._writers["orderbook"]
        while not self._stop_event.is_set():
            started = time.perf_counter()
            for symbol in symbols:
                if self._stop_event.is_set():
                    break
                try:
                    params: Dict[str, object] = {"symbol": symbol}
                    if self.agg_level is not None:
                        params["agg_level"] = self.agg_level
                    payload = await self._fetch("/book", params=params, key="orderbook")
                    rows = to_orderbook_rows(
                        payload,
                        symbol=symbol,
                        recv_ms=now_ms(),
                        depth=self.book_depth,
                        agg_level=self.agg_level,
                    )
                    if rows:
                        await writer.append(rows)
                        log.debug("orderbook_written", symbol=symbol, depth=len(rows[0]["bids"]))
                except Exception as exc:
                    log.warning("orderbook_poll_error", symbol=symbol, error=str(exc))
            await self._sleep_remaining(started, interval)

    async def _poll_funding(self, symbols: Sequence[str], interval: float) -> None:
        writer = self._writers["funding"]
        while not self._stop_event.is_set():
            started = time.perf_counter()
            for symbol in symbols:
                if self._stop_event.is_set():
                    break
                try:
                    payload = await self._fetch(
                        "/funding_rate/history", params={"symbol": symbol, "limit": 100}, key="funding"
                    )
                    rows = to_funding_rows(payload, recv_ms=now_ms(), symbol=symbol)
                    if rows:
                        await writer.append(rows)
                        log.debug("funding_written", symbol=symbol, count=len(rows))
                except Exception as exc:
                    log.warning("funding_poll_error", symbol=symbol, error=str(exc))
            await self._sleep_remaining(started, interval)

    async def _sleep_remaining(self, started: float, interval: float) -> None:
        elapsed = time.perf_counter() - started
        remaining = max(0.0, interval - elapsed)
        if remaining:
            await asyncio.sleep(remaining)

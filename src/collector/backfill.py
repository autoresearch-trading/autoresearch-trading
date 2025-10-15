from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Sequence

import structlog

from .pacifica_rest import PacificaREST
from .rate import RateController
from .storage import ParquetWriter
from .transform import to_candle_rows
from .utils import now_ms

log = structlog.get_logger(__name__)

_INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_to_millis(interval: str) -> int:
    normalized = interval.strip().lower()
    if normalized not in _INTERVAL_TO_MS:
        raise ValueError(
            f"Unsupported interval '{interval}'. Expected one of {', '.join(sorted(_INTERVAL_TO_MS))}."
        )
    return _INTERVAL_TO_MS[normalized]


@dataclass(frozen=True)
class BackfillOptions:
    interval: str
    start_ms: int
    end_ms: int
    chunk_size: int
    out_root: str
    max_rps: int
    dataset: str = "candles"

    def validate(self) -> None:
        if self.start_ms >= self.end_ms:
            raise ValueError("start_ms must be earlier than end_ms for backfill.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be at least 1 interval.")
        if self.max_rps <= 0:
            raise ValueError("max_rps must be positive.")


class KlineBackfillRunner:
    """Fetch historical kline data and persist to Parquet partitions."""

    def __init__(self, rest: PacificaREST, options: BackfillOptions) -> None:
        options.validate()
        self.rest = rest
        self.options = options
        self.interval_ms = interval_to_millis(options.interval)
        self.chunk_intervals = options.chunk_size
        self.chunk_ms = self.interval_ms * self.chunk_intervals
        self.rate = RateController(options.max_rps)
        self.writer = ParquetWriter(options.out_root, options.dataset)

    async def run(self, symbols: Sequence[str]) -> None:
        if not symbols:
            raise ValueError("No symbols provided for backfill.")

        for symbol in symbols:
            await self._backfill_symbol(symbol)

        await self.writer.flush()

    async def _backfill_symbol(self, symbol: str) -> None:
        log.info(
            "backfill_start_symbol",
            symbol=symbol,
            interval=self.options.interval,
            start_ms=self.options.start_ms,
            end_ms=self.options.end_ms,
            chunk_ms=self.chunk_ms,
        )
        current_start = self.options.start_ms

        while current_start < self.options.end_ms:
            chunk_end = min(current_start + self.chunk_ms, self.options.end_ms)
            payload = await self._fetch(symbol, current_start, chunk_end)
            rows = to_candle_rows(
                payload,
                recv_ms=now_ms(),
                symbol=symbol,
                interval=self.options.interval,
                interval_ms=self.interval_ms,
            )

            if rows:
                await self.writer.append(rows)
                last_ts = rows[-1]["ts_ms"]
                next_start = last_ts + self.interval_ms
            else:
                next_start = current_start + self.chunk_ms

            if next_start <= current_start:
                # Ensure forward progress even if data gaps occur.
                next_start = current_start + self.interval_ms

            current_start = next_start

        log.info("backfill_complete_symbol", symbol=symbol)

    async def _fetch(self, symbol: str, start_ms: int, end_ms: int) -> Dict[str, object]:
        async with self.rate.throttle("kline"):
            return await asyncio.to_thread(
                self.rest.get_kline,
                symbol=symbol,
                interval=self.options.interval,
                start_time=int(start_ms),
                end_time=int(end_ms),
            )

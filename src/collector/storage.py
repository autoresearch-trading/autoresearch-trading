from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PartitionDescriptor:
    symbol: str
    date: str

    @property
    def path_fragment(self) -> Path:
        return Path(f"symbol={self.symbol}") / f"date={self.date}"


class ParquetWriter:
    """Thin async-friendly wrapper around pyarrow dataset writes with simple buffering."""

    def __init__(self, root: str, dataset: str) -> None:
        self.root = Path(root)
        self.dataset = dataset
        (self.root / dataset).mkdir(parents=True, exist_ok=True)

        self._buffer: List[Dict[str, object]] = []
        self._max_rows = 50_000  # Increased from 5k to reduce file count
        self._max_seconds = 300.0  # 5 minutes instead of 5 seconds
        self._last_flush = time.time()

    async def append(self, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        self._buffer.extend(rows)
        now = time.time()
        due_rows = len(self._buffer) >= self._max_rows
        due_time = (now - self._last_flush) >= self._max_seconds
        if due_rows or due_time:
            await self._flush_buffer()

    async def flush(self) -> None:
        if not self._buffer:
            return
        await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        if not self._buffer:
            self._last_flush = time.time()
            return
        rows = list(self._buffer)
        self._buffer.clear()
        df = pd.DataFrame(rows)
        if df.empty:
            self._last_flush = time.time()
            return
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.strftime(
            "%Y-%m-%d"
        )
        await asyncio.to_thread(self._write_groups, df)
        self._last_flush = time.time()

    def _write_groups(self, df: pd.DataFrame) -> None:
        by_symbol_date = df.groupby(["symbol", "date"], sort=False)
        for (symbol, date), group in by_symbol_date:
            descriptor = PartitionDescriptor(symbol=symbol, date=date)
            self._write_partition(descriptor, group.drop(columns=["date"]))

    def _write_partition(
        self, descriptor: PartitionDescriptor, df: pd.DataFrame
    ) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        dataset_root = self.root / self.dataset
        partition_path = dataset_root / descriptor.path_fragment
        partition_path.mkdir(parents=True, exist_ok=True)

        timestamp = _utc_now().strftime("%Y%m%dT%H%M%S%f")
        filename = f"{self.dataset}-{timestamp}.parquet"
        tmp_path = partition_path / f"{filename}.tmp"
        final_path = partition_path / filename

        pq.write_table(table, tmp_path, compression="snappy")
        tmp_path.rename(final_path)

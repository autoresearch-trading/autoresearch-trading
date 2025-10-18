from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
from bytewax.inputs import (FixedPartitionedSource, StatefulSourcePartition,
                            StatelessSourcePartition)
from signals.base import OrderbookSnapshot, Trade

T = TypeVar("T")


class InMemorySource(FixedPartitionedSource[T, int]):
    """Simple finite source backed by an in-memory iterable."""

    def __init__(self, items: Iterable[T]) -> None:
        self._items = list(items)

    def list_parts(self) -> List[str]:
        return ["items"]

    def build_part(
        self, step_id: str, for_part: str, resume_state: Optional[int]
    ) -> "InMemoryPartition[T]":
        return InMemoryPartition(self._items)


class InMemoryPartition(StatelessSourcePartition[T]):
    def __init__(self, items: Sequence[T]) -> None:
        self._iterator: Iterator[T] = iter(items)

    def next_batch(self) -> List[T]:
        try:
            item = next(self._iterator)
        except StopIteration as exc:
            raise StopIteration from exc
        return [item]


class ParquetTradeSource(FixedPartitionedSource[Trade, int]):
    """Read trades from one or more Parquet files."""

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self._paths = [Path(path) for path in paths]

    def list_parts(self) -> List[str]:
        return [str(idx) for idx in range(len(self._paths))]

    def build_part(
        self, step_id: str, for_part: str, resume_state: Optional[int]
    ) -> "ParquetTradePartition":
        return ParquetTradePartition(self._paths[int(for_part)], resume_state)


class ParquetTradePartition(StatefulSourcePartition[Trade, int]):
    def __init__(self, path: Path, resume_state: Optional[int]) -> None:
        df = pd.read_parquet(path)
        self._rows = df.to_dict(orient="records")
        self._index = 0 if resume_state is None else resume_state

    def next_batch(self) -> List[Trade]:
        if self._index >= len(self._rows):
            raise StopIteration()
        row = self._rows[self._index]
        self._index += 1
        return [_row_to_trade(row)]

    def snapshot(self) -> int:
        return self._index


class ParquetOrderbookSource(FixedPartitionedSource[OrderbookSnapshot, int]):
    """Read top-of-book snapshots from Parquet files."""

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self._paths = [Path(path) for path in paths]

    def list_parts(self) -> List[str]:
        return [str(idx) for idx in range(len(self._paths))]

    def build_part(
        self, step_id: str, for_part: str, resume_state: Optional[int]
    ) -> "ParquetOrderbookPartition":
        return ParquetOrderbookPartition(self._paths[int(for_part)], resume_state)


class ParquetOrderbookPartition(StatefulSourcePartition[OrderbookSnapshot, int]):
    def __init__(self, path: Path, resume_state: Optional[int]) -> None:
        df = pd.read_parquet(path)
        self._rows = df.to_dict(orient="records")
        self._index = 0 if resume_state is None else resume_state

    def next_batch(self) -> List[OrderbookSnapshot]:
        if self._index >= len(self._rows):
            raise StopIteration()
        row = self._rows[self._index]
        self._index += 1
        return [_row_to_snapshot(row)]

    def snapshot(self) -> int:
        return self._index


def _ms_to_datetime(value: int | float) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def _row_to_trade(row: dict) -> Trade:
    ts_ms = row.get("ts_ms") or row.get("ts")
    recv_ms = row.get("recv_ms", ts_ms)
    if ts_ms is None:
        raise ValueError("Trade row missing 'ts_ms'")

    return Trade(
        ts=_ms_to_datetime(ts_ms),
        recv_ts=_ms_to_datetime(recv_ms),
        symbol=row.get("symbol", ""),
        trade_id=str(row.get("trade_id", "")),
        side=str(row.get("side", "")).lower(),
        price=float(row.get("price")),
        qty=float(row.get("qty")),
        is_large=bool(row.get("is_large", False)),
    )


def _row_to_snapshot(row: dict) -> OrderbookSnapshot:
    ts_ms = row.get("ts_ms") or row.get("ts")
    if ts_ms is None:
        raise ValueError("Orderbook row missing 'ts_ms'")
    bids = _normalize_levels(row.get("bids"))
    asks = _normalize_levels(row.get("asks"))

    if not bids or not asks:
        raise ValueError("Orderbook snapshot missing bids or asks")

    mid_price = float(
        row.get(
            "mid_price",
            (bids[0][0] + asks[0][0]) / 2.0,
        )
    )

    spread_bps = row.get("spread_bps")
    if spread_bps is None:
        spread = max(asks[0][0] - bids[0][0], 0.0)
        spread_bps = int((spread / mid_price) * 10_000) if mid_price else 0

    return OrderbookSnapshot(
        ts=_ms_to_datetime(ts_ms),
        symbol=row.get("symbol", ""),
        bids=bids[:5],
        asks=asks[:5],
        mid_price=mid_price,
        spread_bps=int(spread_bps),
    )


def _normalize_levels(levels) -> List[Tuple[float, float]]:
    if levels is None:
        return []
    normalized: List[Tuple[float, float]] = []
    for level in levels:
        if isinstance(level, dict):
            price = level.get("price")
            qty = level.get("qty")
            if price is None or qty is None:
                continue
            normalized.append((float(price), float(qty)))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            normalized.append((float(level[0]), float(level[1])))
    return normalized

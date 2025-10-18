from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from collector.storage import ParquetWriter


@pytest.mark.asyncio
async def test_parquet_writer_creates_partition(tmp_path: Path) -> None:
    writer = ParquetWriter(str(tmp_path), "prices")
    rows = [
        {
            "ts_ms": 1710000000000,
            "symbol": "BTC",
            "price": 50000.0,
            "recv_ms": 1710000001000,
        },
        {
            "ts_ms": 1710000005000,
            "symbol": "BTC",
            "price": 50010.0,
            "recv_ms": 1710000006000,
        },
    ]
    await writer.append(rows)
    await writer.flush()

    dataset_root = tmp_path / "prices"
    partitions = list(dataset_root.glob("symbol=BTC/date=*"))
    assert partitions, "Expected symbol/date partition directory"

    files = list(partitions[0].glob("*.parquet"))
    assert len(files) == 1

    frame = pd.read_parquet(files[0])
    assert len(frame) == 2
    assert set(frame.columns) == {"ts_ms", "symbol", "price", "recv_ms"}


@pytest.mark.asyncio
async def test_parquet_writer_batches_rows(tmp_path: Path) -> None:
    writer = ParquetWriter(str(tmp_path), "prices")
    base_ts = 1710000010000
    for offset in range(3):
        await writer.append(
            [
                {
                    "ts_ms": base_ts + offset,
                    "symbol": "BTC",
                    "price": 50000.0 + offset,
                    "recv_ms": base_ts + offset + 1,
                }
            ]
        )

    await writer.flush()

    files = list((tmp_path / "prices").rglob("*.parquet"))
    assert (
        len(files) == 1
    ), "Buffered appends should coalesce into a single parquet file."

    frame = pd.read_parquet(files[0])
    assert len(frame) == 3

    parquet_file = pq.ParquetFile(files[0])
    compression = parquet_file.metadata.row_group(0).column(0).compression
    assert compression.lower() == "snappy"

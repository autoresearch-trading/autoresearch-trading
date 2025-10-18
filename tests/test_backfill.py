from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from collector.backfill import (BackfillOptions, KlineBackfillRunner,
                                interval_to_millis)


def test_interval_to_millis_known_values() -> None:
    assert interval_to_millis("1m") == 60_000
    assert interval_to_millis("1d") == 86_400_000
    with pytest.raises(ValueError):
        interval_to_millis("7m")


class DummyREST:
    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.calls = []

    def get_kline(self, *, symbol: str, interval: str, start_time: int, end_time: int):
        self.calls.append((symbol, interval, start_time, end_time))
        candles = []
        current = start_time
        while current < end_time:
            candles.append(
                {
                    "symbol": symbol,
                    "start_time": current,
                    "end_time": current + self.interval_ms,
                    "open": "100",
                    "high": "110",
                    "low": "90",
                    "close": "105",
                    "volume": "1.5",
                }
            )
            current += self.interval_ms
        return {"data": candles}


@pytest.mark.asyncio
async def test_backfill_runner_writes_parquet(tmp_path: Path) -> None:
    rest = DummyREST(interval_ms=60_000)
    options = BackfillOptions(
        interval="1m",
        start_ms=0,
        end_ms=180_000,
        chunk_size=1,
        out_root=str(tmp_path),
        max_rps=5,
    )
    runner = KlineBackfillRunner(rest, options)
    await runner.run(["BTC"])

    assert len(rest.calls) == 3

    dataset_root = Path(tmp_path) / "candles" / "symbol=BTC"
    files = list(dataset_root.rglob("*.parquet"))
    assert files

    frames = [pd.read_parquet(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    assert combined["symbol"].unique().tolist() == ["BTC"]
    assert sorted(combined["ts_ms"].tolist()) == [0, 60_000, 120_000]
    assert (combined["close"] == 105.0).all()

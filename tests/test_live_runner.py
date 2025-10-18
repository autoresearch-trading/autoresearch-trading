from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from collector.config import APISettings
from collector.live_runner import LiveRunner


class RecordingWriter:
    def __init__(self) -> None:
        self.rows = []
        self.event = asyncio.Event()

    async def append(self, rows):
        self.rows.extend(rows)
        if rows:
            self.event.set()

    async def flush(self):
        return


@pytest.mark.asyncio
async def test_live_runner_polls_prices(monkeypatch, tmp_path: Path) -> None:
    settings = APISettings(
        base_url="https://example.com", api_key=None, timeout=1.0, network="mainnet"
    )
    runner = LiveRunner(
        settings, max_rps=5, out_root=str(tmp_path), book_depth=5, agg_level=None
    )

    prices_writer = RecordingWriter()
    runner._writers["prices"] = prices_writer

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/info/prices"):
            payload = {"data": {"BTC": {"price": "50000", "ts_ms": 1710000000000}}}
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "not found"})

    await runner._client.aclose()
    runner._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    )

    task = asyncio.create_task(runner._poll_prices(["BTC"], interval=0.05))
    await asyncio.wait_for(prices_writer.event.wait(), timeout=1)
    runner.request_stop()
    await asyncio.wait_for(task, timeout=1)
    assert prices_writer.rows
    assert prices_writer.rows[0]["symbol"] == "BTC"
    await runner._client.aclose()

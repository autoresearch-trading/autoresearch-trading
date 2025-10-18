from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from config import Settings
from paper_trading.realtime_engine import RealtimePaperTradingEngine
from signals.base import Signal, SignalDirection, SignalType


def _build_signal(symbol: str = "BTC", price: float = 100.0) -> Signal:
    now = datetime.now(timezone.utc)
    return Signal(
        ts=now,
        recv_ts=now,
        symbol=symbol,
        signal_type=SignalType.CVD,
        value=1.0,
        confidence=0.8,
        direction=SignalDirection.BULLISH,
        price=price,
        spread_bps=12,
        bid_depth=4.0,
        ask_depth=3.5,
        metadata={},
    )


@pytest.mark.asyncio
async def test_realtime_engine_buffers_signals():
    settings = Settings()
    settings.symbols = ["BTC"]

    engine = RealtimePaperTradingEngine(settings)

    signal = _build_signal("BTC")
    await engine._on_signal(signal)

    buffers = await engine._pop_signal_buffers()
    assert "BTC" in buffers
    assert buffers["BTC"][0] == signal


@pytest.mark.asyncio
async def test_realtime_engine_price_cache(monkeypatch):
    settings = Settings()
    settings.symbols = ["BTC"]

    engine = RealtimePaperTradingEngine(settings)

    now = datetime.now(timezone.utc)
    engine._latest_price["BTC"] = (120.0, now)

    cached = await engine._get_price("BTC")
    assert cached == 120.0

    old = now - timedelta(seconds=10)
    engine._latest_price["BTC"] = (100.0, old)

    monkeypatch.setattr(engine.live_data, "fetch_price", AsyncMock(return_value=130.0))
    refreshed = await engine._get_price("BTC")
    assert refreshed == 130.0

    # When API returns None we should fall back to latest cached value.
    monkeypatch.setattr(engine.live_data, "fetch_price", AsyncMock(return_value=None))
    fallback = await engine._get_price("BTC")
    assert fallback == 130.0

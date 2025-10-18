from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone

import pytest

from signals.base import Signal, SignalDirection, SignalType
from stream.signal_router import SignalRouter


def _make_signal(symbol: str = "BTC") -> Signal:
    now = datetime.now(timezone.utc)
    return Signal(
        ts=now,
        recv_ts=now,
        symbol=symbol,
        signal_type=SignalType.CVD,
        value=1.0,
        confidence=0.9,
        direction=SignalDirection.BULLISH,
        price=100.0,
        spread_bps=10,
        bid_depth=5.0,
        ask_depth=4.0,
        metadata={},
    )


@pytest.mark.asyncio
async def test_signal_router_dispatches_to_subscriber():
    SignalRouter.reset()
    SignalRouter.initialize()

    received: list[Signal] = []

    async def handler(signal: Signal) -> None:
        received.append(signal)

    SignalRouter.subscribe("BTC", handler)

    task = asyncio.create_task(SignalRouter.dispatch_loop())
    try:
        SignalRouter.route_signal("test_step", _make_signal("BTC"))
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].symbol == "BTC"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await SignalRouter.drain()
        SignalRouter.reset()

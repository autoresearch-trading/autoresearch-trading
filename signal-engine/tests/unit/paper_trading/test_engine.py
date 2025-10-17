from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trading.engine import PaperTradingEngine
from signals.base import Signal, SignalDirection, SignalType


def _make_signal(ts: datetime) -> Signal:
    return Signal(
        ts=ts,
        recv_ts=ts,
        symbol="BTC",
        signal_type=SignalType.CVD,
        value=1.0,
        confidence=0.9,
        direction=SignalDirection.BULLISH,
        price=100.0,
        spread_bps=10,
        bid_depth=100.0,
        ask_depth=100.0,
        metadata={},
    )


def test_filter_fresh_signals_excludes_stale():
    now = datetime.now(timezone.utc)
    fresh = _make_signal(now - timedelta(seconds=30))
    stale = _make_signal(now - timedelta(minutes=5))
    filtered = PaperTradingEngine._filter_fresh_signals(
        [fresh, stale],
        now,
        freshness_window=timedelta(seconds=60),
    )

    assert len(filtered) == 1
    assert filtered[0].ts == fresh.ts

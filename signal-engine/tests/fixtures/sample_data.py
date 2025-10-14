from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from signals.base import Trade


def make_trade(
    *,
    ts: datetime,
    side: str,
    price: float,
    qty: float,
    symbol: str = "BTC",
    trade_id: str | None = None,
) -> Trade:
    """Helper for constructing Trade instances in tests."""
    trade_id = trade_id or f"{symbol}-{int(ts.timestamp() * 1000)}"
    return Trade(
        ts=ts,
        recv_ts=ts,
        symbol=symbol,
        trade_id=trade_id,
        side=side,
        price=price,
        qty=qty,
    )


def generate_trades(
    *,
    start: datetime | None = None,
    prices: Iterable[float],
    sides: Iterable[str],
    qtys: Iterable[float],
    step_seconds: int = 1,
    symbol: str = "BTC",
) -> List[Trade]:
    """Bulk create trades with predictable spacing."""
    start = start or datetime(2025, 1, 1, tzinfo=timezone.utc)
    trades: list[Trade] = []
    ts = start
    for price, side, qty in zip(prices, sides, qtys):
        trades.append(make_trade(ts=ts, side=side, price=price, qty=qty, symbol=symbol))
        ts += timedelta(seconds=step_seconds)
    return trades

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, conint, confloat


class PriceRow(BaseModel):
    ts_ms: conint(ge=0)
    symbol: str
    price: confloat(gt=0)
    recv_ms: conint(ge=0)
    source: str = Field(default="pacifica")


class TradeRow(BaseModel):
    ts_ms: conint(ge=0)
    symbol: str
    trade_id: str
    side: str
    qty: confloat(gt=0)
    price: confloat(gt=0)
    recv_ms: conint(ge=0)


class OrderbookLevel(BaseModel):
    price: confloat(gt=0)
    qty: confloat(ge=0)


class OrderbookRow(BaseModel):
    ts_ms: conint(ge=0)
    symbol: str
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    recv_ms: conint(ge=0)
    agg_level: Optional[int] = None


class FundingRow(BaseModel):
    ts_ms: conint(ge=0)
    symbol: str
    rate: float
    interval_sec: conint(gt=0)
    recv_ms: conint(ge=0)


class CandleRow(BaseModel):
    ts_ms: conint(ge=0)
    symbol: str
    interval: str
    open: confloat(gt=0)
    high: confloat(gt=0)
    low: confloat(gt=0)
    close: confloat(gt=0)
    volume: confloat(ge=0)
    start_ms: conint(ge=0)
    end_ms: conint(ge=0)
    recv_ms: conint(ge=0)

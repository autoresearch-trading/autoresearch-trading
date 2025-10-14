from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(str, Enum):
    CVD = "cvd"
    TFI = "tfi"
    OFI = "ofi"
    REGIME = "regime"


class Signal(BaseModel):
    """Canonical representation of a trading signal."""

    model_config = ConfigDict(use_enum_values=True)

    ts: datetime = Field(description="Event timestamp from exchange")
    recv_ts: datetime = Field(description="Receipt timestamp (local clock)")
    symbol: str
    signal_type: SignalType
    value: float
    confidence: float = Field(ge=0, le=1, description="Signal strength on [0,1]")
    direction: SignalDirection
    price: float
    spread_bps: int
    bid_depth: float
    ask_depth: float
    metadata: dict[str, Any] = Field(default_factory=dict)



class OrderbookSnapshot(BaseModel):
    """Top-N orderbook levels at a given timestamp."""

    ts: datetime
    symbol: str
    bids: list[tuple[float, float]] = Field(max_length=5)
    asks: list[tuple[float, float]] = Field(max_length=5)
    mid_price: float
    spread_bps: int


class Trade(BaseModel):
    """Exchange trade enriched by the collector."""

    ts: datetime
    recv_ts: datetime | None = None
    symbol: str
    trade_id: str
    side: str  # 'buy' or 'sell'
    price: float
    qty: float
    is_large: bool = False

    @model_validator(mode="after")
    def _default_recv_ts(self) -> "Trade":
        if self.recv_ts is None:
            self.recv_ts = self.ts
        return self


class RegimeState(str, Enum):
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL = "high_vol"
    LOW_LIQUIDITY = "low_liquidity"
    RISK_OFF = "risk_off"


class MarketRegime(BaseModel):
    """Market regime classification output."""

    ts: datetime
    symbol: str
    regime: RegimeState
    atr: float
    spread_bps: int
    funding_rate: float
    should_trade: bool


class PaperTrade(BaseModel):
    """Paper trading execution record."""

    ts: datetime
    symbol: str
    trade_id: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    exit_price: float | None = None
    exit_ts: datetime | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    cvd_value: float
    tfi_value: float
    ofi_value: float | None = None
    regime: str

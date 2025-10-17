from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class PaperPosition:
    """In-memory representation of an open paper trading position."""

    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    cvd_value: float
    tfi_value: float
    ofi_value: float | None
    regime: str
    current_price: float
    unrealized_pnl: float
    last_updated: datetime

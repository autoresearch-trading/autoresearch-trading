from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import structlog
from config import Settings
from signals.base import OrderbookSnapshot, Trade

try:  # pragma: no cover - optional dependency
    from collector.api_client import APIClient
    from collector.config import APISettings
    from collector.pacifica_rest import PacificaREST
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIClient = None  # type: ignore[assignment]
    APISettings = None  # type: ignore[assignment]
    PacificaREST = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)


class PacificaStreamClient:
    """Thin synchronous client for low-latency polling of Pacifica endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rest: PacificaREST | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if APIClient and APISettings and PacificaREST:
            api_settings = APISettings(
                base_url=self.settings.pacifica_api_base_url.rstrip("/"),
                api_key=self.settings.pacifica_api_key,
                timeout=self.settings.pacifica_api_timeout,
                network=self.settings.pacifica_network,
            )
            self._rest = PacificaREST(APIClient(api_settings))
        else:
            log.warning("pacifica_stream_client_unavailable")

    def fetch_trades(self, symbol: str) -> List[Trade]:
        """Return latest trades for a symbol (sorted oldest → newest)."""
        if not self._rest:
            return []

        try:
            payload = self._rest.get_recent_trades(symbol=symbol)
        except Exception as exc:  # pragma: no cover - network failure path
            log.error("pacifica_fetch_trades_failed", symbol=symbol, error=str(exc))
            return []

        trades = [
            _dict_to_trade(item, default_symbol=symbol)
            for item in _extract_sequence(payload)
        ]
        trades = [trade for trade in trades if trade is not None]
        trades.sort(key=lambda trade: (trade.ts, trade.trade_id))
        return trades  # type: ignore[return-value]

    def fetch_orderbook(
        self, symbol: str, depth: int = 5
    ) -> Optional[OrderbookSnapshot]:
        """Return the latest orderbook snapshot."""
        if not self._rest:
            return None

        try:
            payload = self._rest.get_orderbook(symbol=symbol)
        except Exception as exc:  # pragma: no cover - network failure path
            log.error("pacifica_fetch_orderbook_failed", symbol=symbol, error=str(exc))
            return None

        return _dict_to_orderbook(payload, default_symbol=symbol, depth=depth)


def _extract_sequence(payload: Any) -> Sequence[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        possible = data.get("trades") or data.get("items")
        if isinstance(possible, list):
            data = possible
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return data  # type: ignore[return-value]
    return []


def _to_datetime_ms(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 1e14:  # nanoseconds
        seconds = numeric / 1_000_000_000.0
    elif numeric > 1e12:  # microseconds
        seconds = numeric / 1_000_000.0
    elif numeric > 1e10:  # milliseconds
        seconds = numeric / 1_000.0
    else:
        seconds = numeric
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_to_trade(item: Any, *, default_symbol: str) -> Trade | None:
    if not isinstance(item, dict):
        return None

    ts_value = (
        item.get("ts_ms")
        or item.get("ts")
        or item.get("time")
        or item.get("created_at")
    )
    ts = _to_datetime_ms(ts_value)
    if ts is None:
        return None

    price = _to_float(item.get("price"))
    qty = _to_float(item.get("qty") or item.get("amount"))
    if price is None or qty is None:
        return None

    trade_id = str(
        item.get("trade_id")
        or item.get("id")
        or item.get("tid")
        or f"{int(ts.timestamp()*1000)}"
    )
    side = str(item.get("side") or "").lower() or "buy"
    recv_ts = datetime.now(timezone.utc)

    return Trade(
        ts=ts,
        recv_ts=recv_ts,
        symbol=str(item.get("symbol") or default_symbol).upper(),
        trade_id=trade_id,
        side=side,
        price=price,
        qty=qty,
        is_large=bool(item.get("is_large", False)),
    )


def _dict_to_orderbook(
    payload: Any, *, default_symbol: str, depth: int
) -> OrderbookSnapshot | None:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        snapshot = data
    else:
        snapshot = payload if isinstance(payload, dict) else None

    if not isinstance(snapshot, dict):
        return None

    bids_raw = snapshot.get("bids")
    asks_raw = snapshot.get("asks")

    if isinstance(bids_raw, Sequence) and isinstance(asks_raw, Sequence):
        bids = _normalize_levels(bids_raw, depth)
        asks = _normalize_levels(asks_raw, depth)
        ts_value = snapshot.get("ts_ms") or snapshot.get("ts")
    else:
        ladder = snapshot.get("l") or snapshot.get("L")
        if not (isinstance(ladder, Sequence) and len(ladder) == 2):
            return None
        bids = _normalize_levels(ladder[0], depth)
        asks = _normalize_levels(ladder[1], depth)
        ts_value = snapshot.get("t") or snapshot.get("timestamp")

    if not bids or not asks:
        return None

    ts = _to_datetime_ms(ts_value) or datetime.now(timezone.utc)
    mid_price = _to_float(snapshot.get("mid_price"))
    if mid_price is None and bids and asks:
        mid_price = (bids[0][0] + asks[0][0]) / 2.0

    spread = max(asks[0][0] - bids[0][0], 0.0)
    spread_bps = int((spread / mid_price) * 10_000) if mid_price else 0

    return OrderbookSnapshot(
        ts=ts,
        symbol=str(snapshot.get("symbol") or default_symbol).upper(),
        bids=bids[:depth],
        asks=asks[:depth],
        mid_price=mid_price or bids[0][0],
        spread_bps=spread_bps,
    )


def _normalize_levels(levels: Iterable[Any], depth: int) -> List[Tuple[float, float]]:
    normalized: List[Tuple[float, float]] = []
    for level in levels:
        price: Optional[float] = None
        qty: Optional[float] = None

        if isinstance(level, dict):
            price = _to_float(level.get("price") or level.get("p"))
            qty = _to_float(level.get("qty") or level.get("a"))
        elif (
            isinstance(level, Sequence)
            and not isinstance(level, (str, bytes))
            and len(level) >= 2
        ):
            price = _to_float(level[0])
            qty = _to_float(level[1])

        if price is None or qty is None:
            continue

        normalized.append((price, qty))
        if len(normalized) >= depth:
            break

    return normalized

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set

from .models import FundingRow, OrderbookLevel, OrderbookRow, PriceRow, TradeRow


def _extract_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def to_price_rows(payload: Dict[str, Any], *, recv_ms: int, filter_symbols: Set[str]) -> List[Dict[str, Any]]:
    data = _extract_data(payload)
    if not isinstance(data, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for symbol, item in data.items():
        if filter_symbols and symbol not in filter_symbols:
            continue
        if not isinstance(item, dict):
            continue
        price_raw = item.get("price")
        if price_raw is None:
            continue
        ts_ms = int(item.get("ts_ms") or recv_ms)
        row = PriceRow(
            ts_ms=ts_ms,
            symbol=str(symbol).upper(),
            price=float(price_raw),
            recv_ms=recv_ms,
        )
        rows.append(row.model_dump())
    return rows


def to_trade_rows(payload: Dict[str, Any], *, recv_ms: int, symbol: str = "") -> List[Dict[str, Any]]:
    data = _extract_data(payload)
    if not isinstance(data, Sequence):
        return []
    rows: List[Dict[str, Any]] = []
    for trade in data:
        if not isinstance(trade, dict):
            continue
        
        # Handle missing fields with proper defaults
        qty_raw = trade.get("qty") or trade.get("amount")
        price_raw = trade.get("price")
        ts_raw = trade.get("ts_ms") or trade.get("time") or trade.get("created_at")
        symbol_raw = trade.get("symbol", symbol)
        trade_id_raw = trade.get("id", "")
        side_raw = trade.get("side", "")
        
        # Skip if essential fields are missing
        if qty_raw is None or price_raw is None:
            continue
            
        row = TradeRow(
            ts_ms=int(ts_raw or recv_ms),
            symbol=str(symbol_raw).upper(),
            trade_id=str(trade_id_raw),
            side=str(side_raw).lower(),
            qty=float(qty_raw),
            price=float(price_raw),
            recv_ms=recv_ms,
        )
        rows.append(row.model_dump())
    return rows


def _normalize_levels(levels: Iterable[Sequence[Any]], depth: int) -> List[OrderbookLevel]:
    normalized: List[OrderbookLevel] = []
    for price_raw, qty_raw, *_ in levels:
        normalized.append(OrderbookLevel(price=float(price_raw), qty=float(qty_raw)))
        if len(normalized) >= depth:
            break
    return normalized


def _normalize_levels_kv(levels: Iterable[Dict[str, Any]], depth: int) -> List[OrderbookLevel]:
    normalized: List[OrderbookLevel] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        price_raw = level.get("p") or level.get("price")
        qty_raw = level.get("a") or level.get("qty")
        if price_raw is None or qty_raw is None:
            continue
        normalized.append(OrderbookLevel(price=float(price_raw), qty=float(qty_raw)))
        if len(normalized) >= depth:
            break
    return normalized


def to_orderbook_rows(
    payload: Dict[str, Any],
    *,
    symbol: str,
    recv_ms: int,
    depth: int,
    agg_level: int | None,
) -> List[Dict[str, Any]]:
    book = _extract_data(payload)
    if not isinstance(book, dict):
        return []

    bids_raw = book.get("bids")
    asks_raw = book.get("asks")
    ts_ms = book.get("ts_ms")

    if isinstance(bids_raw, list) and isinstance(asks_raw, list):
        bids = _normalize_levels(bids_raw, depth)
        asks = _normalize_levels(asks_raw, depth)
        derived_ts = int(ts_ms or recv_ms)
    else:
        ladder = book.get("l") or book.get("L")
        if not (isinstance(ladder, list) and len(ladder) == 2):
            return []
        raw_bids, raw_asks = ladder
        bids = _normalize_levels_kv(raw_bids, depth) if isinstance(raw_bids, list) else []
        asks = _normalize_levels_kv(raw_asks, depth) if isinstance(raw_asks, list) else []
        derived_ts = int(book.get("t") or recv_ms)

    row = OrderbookRow(
        ts_ms=derived_ts,
        symbol=str(symbol).upper(),
        bids=bids,
        asks=asks,
        recv_ms=recv_ms,
        agg_level=agg_level,
    )
    return [row.model_dump()]


def to_funding_rows(payload: Dict[str, Any], *, recv_ms: int, symbol: str = "") -> List[Dict[str, Any]]:
    data = _extract_data(payload)
    if not isinstance(data, Sequence):
        return []
    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        
        # Handle missing fields with proper defaults
        rate_raw = item.get("rate") or item.get("funding_rate")
        ts_raw = item.get("timestamp") or item.get("ts_ms") or item.get("created_at")
        symbol_raw = item.get("symbol", symbol)
        interval_raw = item.get("interval_sec") or item.get("interval") or 0
        
        # Skip if essential fields are missing
        if rate_raw is None:
            continue
            
        row = FundingRow(
            ts_ms=int(ts_raw or recv_ms),
            symbol=str(symbol_raw).upper(),
            rate=float(rate_raw),
            interval_sec=int(interval_raw) or 1,
            recv_ms=recv_ms,
        )
        rows.append(row.model_dump())
    return rows

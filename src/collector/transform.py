from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .models import (
    CandleRow,
    FundingRow,
    OrderbookLevel,
    OrderbookRow,
    PriceRow,
    TradeRow,
)


def _extract_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def to_price_rows(
    payload: Dict[str, Any], *, recv_ms: int, filter_symbols: Set[str]
) -> List[Dict[str, Any]]:
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


def to_trade_rows(
    payload: Dict[str, Any], *, recv_ms: int, symbol: str = ""
) -> List[Dict[str, Any]]:
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


def _normalize_levels(
    levels: Iterable[Sequence[Any]], depth: int
) -> List[OrderbookLevel]:
    normalized: List[OrderbookLevel] = []
    for price_raw, qty_raw, *_ in levels:
        normalized.append(OrderbookLevel(price=float(price_raw), qty=float(qty_raw)))
        if len(normalized) >= depth:
            break
    return normalized


def _normalize_levels_kv(
    levels: Iterable[Dict[str, Any]], depth: int
) -> List[OrderbookLevel]:
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
        bids = (
            _normalize_levels_kv(raw_bids, depth) if isinstance(raw_bids, list) else []
        )
        asks = (
            _normalize_levels_kv(raw_asks, depth) if isinstance(raw_asks, list) else []
        )
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


def to_funding_rows(
    payload: Dict[str, Any], *, recv_ms: int, symbol: str = ""
) -> List[Dict[str, Any]]:
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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_ms(value: Any) -> int | None:
    if value is None:
        return None
    numeric: float
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            numeric = float(value)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(numeric):
        return None
    ms = int(round(numeric))
    if ms < 0:
        return None
    return ms


def _normalize_symbol(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return default.upper()


def _pick_first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is not None:
                return value
    return None


def to_candle_rows(
    payload: Dict[str, Any],
    *,
    recv_ms: int,
    symbol: str,
    interval: str,
    interval_ms: int,
) -> List[Dict[str, Any]]:
    data = _extract_data(payload)
    if not isinstance(data, Sequence):
        return []

    rows: List[Dict[str, Any]] = []
    default_symbol = symbol.upper()

    for entry in data:
        entry_symbol = default_symbol
        open_price = high_price = low_price = close_price = None
        volume = None
        start_ms = end_ms = None

        if isinstance(entry, dict):
            entry_symbol = _normalize_symbol(entry.get("symbol"), default_symbol)
            open_price = _to_float(entry.get("open") or entry.get("o"))
            high_price = _to_float(entry.get("high") or entry.get("h"))
            low_price = _to_float(entry.get("low") or entry.get("l"))
            close_price = _to_float(entry.get("close") or entry.get("c"))
            volume = _to_float(
                entry.get("volume")
                or entry.get("v")
                or entry.get("amount")
                or entry.get("qty")
            )
            start_ms = _to_int_ms(
                _pick_first(
                    entry,
                    ("t", "start_time", "open_time", "open_ts", "timestamp", "ts_ms"),
                )
            )
            end_ms = _to_int_ms(
                _pick_first(entry, ("T", "end_time", "close_time", "close_ts"))
            )
        elif isinstance(entry, Sequence):
            # Binance-style tuples: [open_time, open, high, low, close, volume, close_time, ...]
            if len(entry) >= 6:
                start_ms = _to_int_ms(entry[0])
                open_price = _to_float(entry[1])
                high_price = _to_float(entry[2])
                low_price = _to_float(entry[3])
                close_price = _to_float(entry[4])
                volume = _to_float(entry[5]) if len(entry) >= 6 else None
                end_ms = _to_int_ms(entry[6]) if len(entry) >= 7 else None
        else:
            continue

        if (
            open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
        ):
            continue

        if start_ms is None and end_ms is not None:
            start_ms = end_ms - interval_ms
        if start_ms is None:
            continue

        if end_ms is None:
            end_ms = start_ms + interval_ms

        volume_value = volume if volume is not None and not math.isnan(volume) else 0.0
        row = CandleRow(
            ts_ms=start_ms,
            symbol=entry_symbol,
            interval=interval,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=max(volume_value, 0.0),
            start_ms=start_ms,
            end_ms=max(end_ms, start_ms + interval_ms),
            recv_ms=recv_ms,
        )
        rows.append(row.model_dump())

    rows.sort(key=lambda item: item["ts_ms"])
    return rows

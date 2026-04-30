# scripts/build_pacifica_full_fidelity_silver.py
"""Normalize raw Pacifica full-fidelity JSONL.GZ archive into silver tables.

This is deliberately not an alpha model.  It preserves public market-data fields
that earlier lossy tables dropped, then writes queryable parquet tables for the
non-HFT regime-state layer.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RAW_DIR = Path("data/pacifica_full_fidelity")
DEFAULT_OUT_DIR = Path("data/pacifica_silver")
DEFAULT_CHANNELS = ("prices", "trades", "bbo", "book", "candle", "mark_price_candle")

JsonObject = dict[str, Any]


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _base(record: JsonObject) -> JsonObject:
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    symbol = record.get("symbol") or data.get("symbol") or data.get("s")
    return {
        "event_ts_ms": _to_int(
            record.get("event_ts_ms")
            or data.get("timestamp")
            or data.get("t")
            or data.get("T")
        ),
        "recv_ms": _to_int(record.get("recv_ms")),
        "symbol": str(symbol) if symbol is not None else "UNKNOWN",
        "channel": str(record.get("channel") or "unknown"),
    }


def _side_from_direction(direction: str | None) -> int:
    if not direction:
        return 0
    direction = direction.lower()
    # Pacifica trade directions encode position action.  Buy pressure is
    # open_long/close_short; sell pressure is open_short/close_long.
    if direction in {"open_long", "close_short"}:
        return 1
    if direction in {"open_short", "close_long"}:
        return -1
    return 0


def normalize_trade_record(record: JsonObject) -> JsonObject:
    data = record.get("data") or {}
    row = _base(record)
    qty = _to_float(data.get("a"))
    price = _to_float(data.get("p"))
    side = _side_from_direction(data.get("d"))
    row.update(
        {
            "price": price,
            "qty": qty,
            "signed_qty": (qty or 0.0) * side,
            "notional": (qty or 0.0) * (price or 0.0),
            "direction": data.get("d"),
            "trade_class": data.get("tc"),
            "history_id": _to_int(data.get("h")),
            "nonce": _to_int(data.get("li")),
            "is_taker_internal": _to_int(data.get("it")),
        }
    )
    return row


def normalize_bbo_record(record: JsonObject) -> JsonObject:
    data = record.get("data") or {}
    row = _base(record)
    bid_px = _to_float(data.get("b"))
    ask_px = _to_float(data.get("a"))
    bid_qty = _to_float(data.get("B"))
    ask_qty = _to_float(data.get("A"))
    mid = (bid_px + ask_px) / 2 if bid_px is not None and ask_px is not None else None
    spread_bps = ((ask_px - bid_px) / mid * 10_000) if mid else None
    row.update(
        {
            "bid_px": bid_px,
            "bid_qty": bid_qty,
            "ask_px": ask_px,
            "ask_qty": ask_qty,
            "mid": mid,
            "spread_bps": spread_bps,
            "top_bid_notional": (bid_px or 0.0) * (bid_qty or 0.0),
            "top_ask_notional": (ask_px or 0.0) * (ask_qty or 0.0),
            "order_id": _to_int(data.get("i")),
            "last_order_id": _to_int(data.get("li")),
        }
    )
    return row


def _levels(data: JsonObject) -> tuple[list[JsonObject], list[JsonObject]]:
    raw = data.get("l") or [[], []]
    if not isinstance(raw, list) or len(raw) < 2:
        return [], []
    bids = raw[0] if isinstance(raw[0], list) else []
    asks = raw[1] if isinstance(raw[1], list) else []
    return bids, asks


def _sum_level_qty(levels: list[JsonObject], n: int) -> float:
    return sum(_to_float(level.get("a")) or 0.0 for level in levels[:n])


def _sum_level_orders(levels: list[JsonObject], n: int) -> int:
    return sum(_to_int(level.get("n")) or 0 for level in levels[:n])


def normalize_book_record(record: JsonObject) -> JsonObject:
    data = record.get("data") or {}
    row = _base(record)
    bids, asks = _levels(data)
    bid_px = _to_float(bids[0].get("p")) if bids else None
    ask_px = _to_float(asks[0].get("p")) if asks else None
    mid = (bid_px + ask_px) / 2 if bid_px is not None and ask_px is not None else None
    spread_bps = ((ask_px - bid_px) / mid * 10_000) if mid else None
    row.update(
        {
            "bid_px_l1": bid_px,
            "ask_px_l1": ask_px,
            "mid_l1": mid,
            "spread_bps_l1": spread_bps,
            "bid_depth_l1": _sum_level_qty(bids, 1),
            "ask_depth_l1": _sum_level_qty(asks, 1),
            "bid_depth_l5": _sum_level_qty(bids, 5),
            "ask_depth_l5": _sum_level_qty(asks, 5),
            "bid_depth_l10": _sum_level_qty(bids, 10),
            "ask_depth_l10": _sum_level_qty(asks, 10),
            "bid_orders_l1": _sum_level_orders(bids, 1),
            "ask_orders_l1": _sum_level_orders(asks, 1),
            "bid_orders_l5": _sum_level_orders(bids, 5),
            "ask_orders_l5": _sum_level_orders(asks, 5),
            "bid_orders_l10": _sum_level_orders(bids, 10),
            "ask_orders_l10": _sum_level_orders(asks, 10),
            "nonce": _to_int(data.get("li")),
        }
    )
    return row


def normalize_price_record(record: JsonObject) -> JsonObject:
    data = record.get("data") or {}
    row = _base(record)
    mid = _to_float(data.get("mid"))
    mark = _to_float(data.get("mark"))
    oracle = _to_float(data.get("oracle"))
    row.update(
        {
            "mid": mid,
            "mark": mark,
            "oracle": oracle,
            "funding": _to_float(data.get("funding")),
            "next_funding": _to_float(data.get("next_funding")),
            "open_interest": _to_float(data.get("open_interest")),
            "volume_24h": _to_float(data.get("volume_24h")),
            "yesterday_price": _to_float(data.get("yesterday_price")),
            "mark_oracle_basis_bps": (
                ((mark - oracle) / oracle * 10_000)
                if mark is not None and oracle
                else None
            ),
            "mid_oracle_basis_bps": (
                ((mid - oracle) / oracle * 10_000)
                if mid is not None and oracle
                else None
            ),
        }
    )
    return row


def normalize_candle_record(record: JsonObject) -> JsonObject:
    data = record.get("data") or {}
    row = _base(record)
    row.update(
        {
            "interval": data.get("i"),
            "start_ts_ms": _to_int(data.get("t")),
            "end_ts_ms": _to_int(data.get("T")),
            "open": _to_float(data.get("o")),
            "high": _to_float(data.get("h")),
            "low": _to_float(data.get("l")),
            "close": _to_float(data.get("c")),
            "volume": _to_float(data.get("v")),
            "trade_count": _to_int(data.get("n")),
        }
    )
    return row


NORMALIZERS = {
    "prices": normalize_price_record,
    "trades": normalize_trade_record,
    "bbo": normalize_bbo_record,
    "book": normalize_book_record,
    "candle": normalize_candle_record,
    "mark_price_candle": normalize_candle_record,
}


def iter_raw_records(
    raw_dir: Path, *, channels: Sequence[str] | None = None
) -> Iterable[JsonObject]:
    wanted = set(channels or DEFAULT_CHANNELS)
    for path in sorted(raw_dir.glob("channel=*/symbol=*/date=*/*.jsonl.gz")):
        channel = next(
            (
                part.split("=", 1)[1]
                for part in path.parts
                if part.startswith("channel=")
            ),
            None,
        )
        if channel not in wanted:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def normalize_records(
    records: Iterable[JsonObject], *, channels: Sequence[str] | None = None
) -> dict[str, list[JsonObject]]:
    wanted = set(channels or DEFAULT_CHANNELS)
    tables: dict[str, list[JsonObject]] = {channel: [] for channel in wanted}
    for record in records:
        channel = str(record.get("channel") or "unknown")
        if channel not in wanted or channel not in NORMALIZERS:
            continue
        tables.setdefault(channel, []).append(NORMALIZERS[channel](record))
    return tables


def _quality_rows(tables: dict[str, list[JsonObject]]) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for channel, records in sorted(tables.items()):
        symbols = {row.get("symbol") for row in records if row.get("symbol")}
        event_ts = [
            row.get("event_ts_ms")
            for row in records
            if row.get("event_ts_ms") is not None
        ]
        rows.append(
            {
                "channel": channel,
                "rows": len(records),
                "symbols": len(symbols),
                "min_event_ts_ms": min(event_ts) if event_ts else None,
                "max_event_ts_ms": max(event_ts) if event_ts else None,
            }
        )
    return rows


def write_silver_tables(
    raw_dir: Path, out_dir: Path, *, channels: Sequence[str] | None = None
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(channels or DEFAULT_CHANNELS)
    tables = normalize_records(
        iter_raw_records(raw_dir, channels=selected), channels=selected
    )
    written: dict[str, int] = {}
    for channel in selected:
        rows = tables.get(channel, [])
        if not rows:
            written[channel] = 0
            continue
        df = pd.DataFrame(rows).sort_values(
            ["symbol", "event_ts_ms", "recv_ms"], na_position="last"
        )
        df.to_parquet(out_dir / f"{channel}.parquet", index=False)
        written[channel] = len(df)
    pd.DataFrame(_quality_rows(tables)).to_csv(
        out_dir / "quality_summary.csv", index=False
    )
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
    written = write_silver_tables(args.raw_dir, args.out_dir, channels=channels)
    for channel, rows in sorted(written.items()):
        print(f"{channel}: {rows} rows")
    print(f"wrote silver tables to {args.out_dir}")


if __name__ == "__main__":
    main()

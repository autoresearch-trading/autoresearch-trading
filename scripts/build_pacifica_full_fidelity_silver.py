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
import shutil
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pacifica_source_manifest import (
    build_source_manifest,
    plan_changed_sealed_source_objects,
    read_source_manifest,
)

DEFAULT_RAW_DIR = Path("data/pacifica_full_fidelity")
DEFAULT_OUT_DIR = Path("data/pacifica_silver_partitioned")
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
    for path in sorted(raw_dir.rglob("*.jsonl.gz")):
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
        yield from iter_raw_records_from_path(path)


def iter_raw_records_from_path(path: Path) -> Iterable[JsonObject]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    except (EOFError, gzip.BadGzipFile):
        # The live collector may still be appending the newest gzip member.
        # Skip incomplete active files; they will be picked up on a later build.
        return


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


def _event_date(row: JsonObject) -> str:
    ts_ms = _to_int(row.get("event_ts_ms") or row.get("recv_ms"))
    if ts_ms is None:
        return "unknown"
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


def _empty_quality(channels: Sequence[str]) -> dict[str, JsonObject]:
    return {
        channel: {
            "channel": channel,
            "rows": 0,
            "symbols_set": set(),
            "min_event_ts_ms": None,
            "max_event_ts_ms": None,
        }
        for channel in channels
    }


def _update_quality(quality: dict[str, JsonObject], row: JsonObject) -> None:
    channel = str(row.get("channel") or "unknown")
    stats = quality.setdefault(
        channel,
        {
            "channel": channel,
            "rows": 0,
            "symbols_set": set(),
            "min_event_ts_ms": None,
            "max_event_ts_ms": None,
        },
    )
    stats["rows"] += 1
    if row.get("symbol"):
        stats["symbols_set"].add(row["symbol"])
    ts_ms = row.get("event_ts_ms")
    if ts_ms is None:
        return
    stats["min_event_ts_ms"] = (
        ts_ms
        if stats["min_event_ts_ms"] is None
        else min(stats["min_event_ts_ms"], ts_ms)
    )
    stats["max_event_ts_ms"] = (
        ts_ms
        if stats["max_event_ts_ms"] is None
        else max(stats["max_event_ts_ms"], ts_ms)
    )


def _quality_frame(quality: dict[str, JsonObject]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "channel": channel,
                "rows": stats["rows"],
                "symbols": len(stats["symbols_set"]),
                "min_event_ts_ms": stats["min_event_ts_ms"],
                "max_event_ts_ms": stats["max_event_ts_ms"],
            }
            for channel, stats in sorted(quality.items())
        ]
    )


def _flush_partition_buffers(
    out_dir: Path,
    buffers: dict[tuple[str, str, str], list[JsonObject]],
    part_counters: dict[tuple[str, str, str], int],
) -> None:
    for key, rows in list(buffers.items()):
        if not rows:
            continue
        channel, symbol, date = key
        part_idx = part_counters.get(key, 0)
        path = (
            out_dir
            / f"channel={channel}"
            / f"symbol={symbol}"
            / f"date={date}"
            / f"part-{part_idx:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values(
            ["symbol", "event_ts_ms", "recv_ms"], na_position="last"
        ).to_parquet(path, index=False)
        part_counters[key] = part_idx + 1
        buffers[key] = []


def write_partitioned_silver_tables(
    raw_dir: Path,
    out_dir: Path,
    *,
    channels: Sequence[str] | None = None,
    chunk_size: int = 250_000,
) -> dict[str, int]:
    """Stream raw archive into channel/symbol/date parquet partitions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(channels or DEFAULT_CHANNELS)
    wanted = set(selected)
    buffers: dict[tuple[str, str, str], list[JsonObject]] = {}
    part_counters: dict[tuple[str, str, str], int] = {}
    quality = _empty_quality(selected)
    written = {channel: 0 for channel in selected}
    buffered_rows = 0

    for record in iter_raw_records(raw_dir, channels=selected):
        channel = str(record.get("channel") or "unknown")
        if channel not in wanted or channel not in NORMALIZERS:
            continue
        row = NORMALIZERS[channel](record)
        key = (channel, str(row.get("symbol") or "UNKNOWN"), _event_date(row))
        buffers.setdefault(key, []).append(row)
        written[channel] = written.get(channel, 0) + 1
        _update_quality(quality, row)
        buffered_rows += 1
        if buffered_rows >= chunk_size:
            _flush_partition_buffers(out_dir, buffers, part_counters)
            buffered_rows = 0

    _flush_partition_buffers(out_dir, buffers, part_counters)
    _quality_frame(quality).to_csv(out_dir / "quality_summary.csv", index=False)
    return written


def _safe_partition_value(value: Any) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _source_chunk_silver_path(out_dir: Path, manifest_row: pd.Series) -> Path:
    return (
        out_dir
        / f"channel={_safe_partition_value(manifest_row['channel'])}"
        / f"symbol={_safe_partition_value(manifest_row['symbol'])}"
        / f"date={_safe_partition_value(manifest_row['date'])}"
        / f"hour={_safe_partition_value(manifest_row['hour'])}"
        / f"run={_safe_partition_value(manifest_row['run'])}"
        / "part.parquet"
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validated_source_path(raw_dir: Path, source_path: str) -> Path:
    rel = Path(source_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"source_path outside raw_dir: {source_path}")
    candidate = (raw_dir / rel).resolve()
    raw_root = raw_dir.resolve()
    if not candidate.is_relative_to(raw_root):
        raise ValueError(f"source_path outside raw_dir: {source_path}")
    return candidate


def _verified_sealed_rows(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool)
    required = {"status", "sha_verified", "gzip_readable"}
    if not required.issubset(frame.columns):
        return pd.Series([False] * len(frame), index=frame.index)
    return (
        frame["status"].astype(str).eq("sealed")
        & frame["sha_verified"].map(_truthy)
        & frame["gzip_readable"].map(_truthy)
    )


def _flat_silver_seed_files(
    base_silver_dir: Path, channels: Sequence[str]
) -> list[Path]:
    """Return selected flat v1 channel parquet files that cannot seed incremental refreshes."""
    return [
        base_silver_dir / f"{channel}.parquet"
        for channel in channels
        if (base_silver_dir / f"{channel}.parquet").exists()
    ]


def _remove_candidate_partitions(out_dir: Path, plan: pd.DataFrame) -> None:
    if plan.empty:
        return
    for _, row in plan[["channel", "symbol", "date"]].drop_duplicates().iterrows():
        path = (
            out_dir
            / f"channel={_safe_partition_value(row['channel'])}"
            / f"symbol={_safe_partition_value(row['symbol'])}"
            / f"date={_safe_partition_value(row['date'])}"
        )
        if path.exists():
            shutil.rmtree(path)


def _expand_to_affected_verified_partitions(
    current_manifest: pd.DataFrame, plan: pd.DataFrame
) -> pd.DataFrame:
    if plan.empty:
        return plan
    affected = {
        (str(row["channel"]), str(row["symbol"]), str(row["date"]))
        for _, row in plan.iterrows()
    }
    verified = current_manifest[_verified_sealed_rows(current_manifest)].copy()
    expanded = verified[
        verified.apply(
            lambda row: (str(row["channel"]), str(row["symbol"]), str(row["date"]))
            in affected,
            axis=1,
        )
    ].copy()
    return expanded.sort_values("source_key").reset_index(drop=True)


def _write_source_chunk_silver(
    raw_dir: Path,
    out_dir: Path,
    manifest_row: pd.Series,
    *,
    channels: Sequence[str],
) -> int:
    channel = str(manifest_row["channel"])
    if channel not in set(channels) or channel not in NORMALIZERS:
        return 0
    source_path = _validated_source_path(raw_dir, str(manifest_row["source_path"]))
    rows: list[JsonObject] = []
    for record in iter_raw_records_from_path(source_path):
        record_channel = str(record.get("channel") or channel)
        if record_channel != channel:
            continue
        row = NORMALIZERS[channel](record)
        row["source_key"] = str(manifest_row["source_key"])
        row["source_path"] = str(manifest_row["source_path"])
        row["source_sha256"] = str(manifest_row.get("sha256", ""))
        rows.append(row)
    out_path = _source_chunk_silver_path(out_dir, manifest_row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if out_path.exists():
            out_path.unlink()
        return 0
    tmp_path = out_path.with_name("part.tmp.parquet")
    df = pd.DataFrame(rows).drop_duplicates()
    df.sort_values(["symbol", "event_ts_ms", "recv_ms"], na_position="last").to_parquet(
        tmp_path, index=False
    )
    tmp_path.replace(out_path)
    return len(df)


def write_incremental_silver_tables(
    raw_dir: Path,
    out_dir: Path,
    *,
    current_manifest: pd.DataFrame | None = None,
    previous_manifest: pd.DataFrame | None = None,
    previous_manifest_path: Path | None = None,
    channels: Sequence[str] | None = None,
    base_silver_dir: Path | None = None,
    allow_canonical_out_dir: bool = False,
) -> dict[str, Any]:
    """Write only new/changed sealed source chunks into a side-by-side silver dir.

    The output path is deterministic per source object
    (channel/symbol/date/hour/run/part.parquet), so changed chunks replace their
    own candidate output without touching canonical silver.  Missing-sidecar or
    otherwise unsealed chunks are excluded from the plan.
    """
    raw_dir = raw_dir.resolve()
    selected = tuple(channels or DEFAULT_CHANNELS)
    if out_dir.resolve() == DEFAULT_OUT_DIR.resolve() and not allow_canonical_out_dir:
        raise ValueError(
            "incremental silver requires an explicit side-by-side candidate out_dir; "
            f"refusing canonical default {DEFAULT_OUT_DIR}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_silver_dir is not None and base_silver_dir.exists():
        flat_seed_files = _flat_silver_seed_files(base_silver_dir, selected)
        if flat_seed_files:
            rel = ", ".join(path.name for path in flat_seed_files)
            raise ValueError(
                "flat canonical seed is unsafe for incremental silver refresh; "
                f"refusing selected flat channel files: {rel}"
            )
        # Optional side-by-side seed. Existing candidate files are left in place
        # so reruns can remain incremental; canonical base is never mutated.
        for child in base_silver_dir.iterdir():
            dest = out_dir / child.name
            if dest.exists():
                continue
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    if current_manifest is None:
        current_manifest = build_source_manifest(
            raw_dir, channels=selected, verify_sha=True, count_rows=True
        )
    if previous_manifest is None and previous_manifest_path is not None:
        previous_manifest = read_source_manifest(previous_manifest_path)
    plan = plan_changed_sealed_source_objects(previous_manifest, current_manifest)
    if not plan.empty and not _verified_sealed_rows(plan).all():
        raise ValueError(
            "incremental silver requires verified sealed source rows: "
            "build the manifest with --verify-sha --count-rows"
        )
    write_plan = plan
    if base_silver_dir is not None:
        _remove_candidate_partitions(out_dir, plan)
        write_plan = _expand_to_affected_verified_partitions(current_manifest, plan)
    written_by_channel = {channel: 0 for channel in selected}
    processed = 0
    for _, row in write_plan.iterrows():
        channel = str(row["channel"])
        if channel not in written_by_channel:
            continue
        rows_written = _write_source_chunk_silver(
            raw_dir, out_dir, row, channels=selected
        )
        written_by_channel[channel] += rows_written
        processed += 1
    current_manifest.to_csv(
        out_dir / "source_manifest.csv", index=False, lineterminator="\n"
    )
    plan.to_csv(out_dir / "incremental_plan.csv", index=False, lineterminator="\n")
    quality = pd.DataFrame(
        [
            {
                "channel": channel,
                "rows": rows,
                "processed_source_objects": (
                    int((write_plan["channel"].astype(str) == channel).sum())
                    if not write_plan.empty
                    else 0
                ),
            }
            for channel, rows in sorted(written_by_channel.items())
        ]
    )
    quality.to_csv(out_dir / "quality_summary.csv", index=False, lineterminator="\n")
    return {
        "processed_source_objects": processed,
        "planned_source_objects": int(len(plan)),
        "written_rows_by_channel": written_by_channel,
        "manifest_rows": int(len(current_manifest)),
        "out_dir": str(out_dir),
    }


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
    parser.add_argument(
        "--layout",
        choices=("partitioned", "flat", "incremental"),
        default="partitioned",
        help="Output layout. incremental writes only new/changed sealed source chunks into a side-by-side dir.",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Current source-object manifest CSV for incremental layout. Built from --raw-dir if omitted.",
    )
    parser.add_argument(
        "--previous-source-manifest",
        type=Path,
        help="Previous source-object manifest CSV used to plan new/changed sealed chunks.",
    )
    parser.add_argument(
        "--base-silver-dir",
        type=Path,
        help="Optional canonical silver dir to copy into the side-by-side candidate before incremental writes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
    if args.layout == "flat":
        written = write_silver_tables(args.raw_dir, args.out_dir, channels=channels)
        for channel, rows in sorted(written.items()):
            print(f"{channel}: {rows} rows")
    elif args.layout == "incremental":
        current_manifest = (
            read_source_manifest(args.source_manifest) if args.source_manifest else None
        )
        result = write_incremental_silver_tables(
            args.raw_dir,
            args.out_dir,
            current_manifest=current_manifest,
            previous_manifest_path=args.previous_source_manifest,
            channels=channels,
            base_silver_dir=args.base_silver_dir,
        )
        for channel, rows in sorted(result["written_rows_by_channel"].items()):
            print(f"{channel}: {rows} incremental rows")
        print(f"processed_source_objects: {result['processed_source_objects']}")
    else:
        written = write_partitioned_silver_tables(
            args.raw_dir, args.out_dir, channels=channels, chunk_size=args.chunk_size
        )
        for channel, rows in sorted(written.items()):
            print(f"{channel}: {rows} rows")
    print(f"wrote silver tables to {args.out_dir}")


if __name__ == "__main__":
    main()

# scripts/watch_pacifica_realtime_research.py
"""Read-only near-real-time diagnostics for the Pacifica full-fidelity archive.

This monitor is intentionally not a trading engine.  It inventories the raw
append-only JSONL.GZ archive, computes a small latest 1-minute market-quality
snapshot, and writes markdown/CSV diagnostics that can be compared to later
batch recomputations.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.build_pacifica_full_fidelity_silver import (
    normalize_bbo_record,
    normalize_price_record,
    normalize_trade_record,
)

DEFAULT_RAW_DIR = Path("data/pacifica_full_fidelity")
DEFAULT_SILVER_DIR = Path("data/pacifica_silver_partitioned")
DEFAULT_OUT_DIR = Path("data/pacifica_realtime_research")

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class InventoryRow:
    channel: str
    symbol: str
    file_count: int
    row_count: int
    latest_event_ts_ms: int | None
    latest_recv_ms: int | None
    latest_age_s: float | None


@dataclass(frozen=True)
class RawInventory:
    raw_dir: str
    file_count: int
    row_count: int
    symbol_count: int
    latest_recv_ms: int | None
    latest_age_s: float | None
    rows: list[InventoryRow]


@dataclass(frozen=True)
class RealtimeFeature:
    symbol: str
    window_start_ms: int
    window_end_ms: int
    trade_count_1m: int
    trade_volume_1m: float
    trade_notional_1m: float
    signed_trade_volume_1m: float
    last_price: float | None
    return_bps_1m: float | None
    spread_bps: float | None
    top_depth_notional: float | None
    mark_oracle_basis_bps: float | None
    mid_oracle_basis_bps: float | None
    open_interest: float | None
    funding: float | None
    stress_score: float


@dataclass(frozen=True)
class RealtimeReport:
    generated_at_ms: int
    raw_dir: str
    stale_after_s: float
    inventory: RawInventory
    features: list[RealtimeFeature]
    warnings: list[str]


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


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def utc_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def _raw_files(raw_dir: Path, *, max_files: int | None = None) -> list[Path]:
    files = list(raw_dir.glob("channel=*/symbol=*/date=*/*.jsonl.gz"))
    return _limit_recent_files(files, max_files=max_files)


def _silver_files(silver_dir: Path, *, max_files: int | None = None) -> list[Path]:
    files = list(silver_dir.glob("channel=*/symbol=*/date=*/*.parquet"))
    return _limit_recent_files(files, max_files=max_files)


def _limit_recent_files(files: list[Path], *, max_files: int | None) -> list[Path]:
    if max_files is not None and max_files > 0 and len(files) > max_files:
        files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[
            :max_files
        ]
    return sorted(files)


def _path_part(path: Path, prefix: str) -> str | None:
    return next(
        (part.split("=", 1)[1] for part in path.parts if part.startswith(prefix)), None
    )


def iter_raw_records_with_path(
    raw_dir: Path,
    *,
    max_files: int | None = None,
    max_records_per_file: int | None = 5_000,
) -> list[tuple[Path, JsonObject]]:
    rows: list[tuple[Path, JsonObject]] = []
    for path in _raw_files(raw_dir, max_files=max_files):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                if max_records_per_file is not None and index >= max_records_per_file:
                    break
                if not line.strip():
                    continue
                rows.append((path, json.loads(line)))
    return rows


def _event_ts(record: JsonObject) -> int | None:
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    return _to_int(
        record.get("event_ts_ms")
        or data.get("timestamp")
        or data.get("t")
        or data.get("T")
    )


def _recv_ms(record: JsonObject) -> int | None:
    return _to_int(record.get("recv_ms"))


def inventory_raw_archive(
    raw_dir: Path,
    *,
    now_ms: int | None = None,
    max_files: int | None = None,
    max_records_per_file: int | None = 5_000,
) -> RawInventory:
    now = utc_ms() if now_ms is None else now_ms
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    total_rows = 0
    latest_recv: int | None = None
    symbols: set[str] = set()
    files = _raw_files(raw_dir, max_files=max_files)

    for path in files:
        channel = _path_part(path, "channel=") or "unknown"
        symbol = _path_part(path, "symbol=") or "UNKNOWN"
        key = (channel, symbol)
        row = stats.setdefault(
            key,
            {
                "channel": channel,
                "symbol": symbol,
                "file_count": 0,
                "row_count": 0,
                "latest_event_ts_ms": None,
                "latest_recv_ms": None,
            },
        )
        row["file_count"] += 1
        symbols.add(symbol)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                if max_records_per_file is not None and index >= max_records_per_file:
                    break
                if not line.strip():
                    continue
                total_rows += 1
                row["row_count"] += 1
                record = json.loads(line)
                event_ts = _event_ts(record)
                recv = _recv_ms(record)
                if event_ts is not None:
                    row["latest_event_ts_ms"] = (
                        event_ts
                        if row["latest_event_ts_ms"] is None
                        else max(row["latest_event_ts_ms"], event_ts)
                    )
                if recv is not None:
                    row["latest_recv_ms"] = (
                        recv
                        if row["latest_recv_ms"] is None
                        else max(row["latest_recv_ms"], recv)
                    )
                    latest_recv = (
                        recv if latest_recv is None else max(latest_recv, recv)
                    )

    inventory_rows = [
        InventoryRow(
            channel=str(row["channel"]),
            symbol=str(row["symbol"]),
            file_count=int(row["file_count"]),
            row_count=int(row["row_count"]),
            latest_event_ts_ms=row["latest_event_ts_ms"],
            latest_recv_ms=row["latest_recv_ms"],
            latest_age_s=(
                (now - row["latest_recv_ms"]) / 1000
                if row["latest_recv_ms"] is not None
                else None
            ),
        )
        for row in stats.values()
    ]
    inventory_rows.sort(key=lambda row: (row.channel, row.symbol))
    return RawInventory(
        raw_dir=str(raw_dir),
        file_count=len(files),
        row_count=total_rows,
        symbol_count=len(symbols),
        latest_recv_ms=latest_recv,
        latest_age_s=((now - latest_recv) / 1000 if latest_recv is not None else None),
        rows=inventory_rows,
    )


def inventory_silver_archive(
    silver_dir: Path,
    *,
    now_ms: int | None = None,
    max_files: int | None = None,
) -> RawInventory:
    now = utc_ms() if now_ms is None else now_ms
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    total_rows = 0
    latest_recv: int | None = None
    symbols: set[str] = set()
    files = _silver_files(silver_dir, max_files=max_files)

    for path in files:
        df = pd.read_parquet(path, columns=None)
        if df.empty:
            channel = _path_part(path, "channel=") or "unknown"
            symbol = _path_part(path, "symbol=") or "UNKNOWN"
            key = (channel, symbol)
            stats.setdefault(
                key,
                {
                    "channel": channel,
                    "symbol": symbol,
                    "file_count": 0,
                    "row_count": 0,
                    "latest_event_ts_ms": None,
                    "latest_recv_ms": None,
                },
            )["file_count"] += 1
            continue
        channel = (
            str(df["channel"].iloc[0])
            if "channel" in df
            else (_path_part(path, "channel=") or "unknown")
        )
        symbol = (
            str(df["symbol"].iloc[0])
            if "symbol" in df
            else (_path_part(path, "symbol=") or "UNKNOWN")
        )
        key = (channel, symbol)
        row = stats.setdefault(
            key,
            {
                "channel": channel,
                "symbol": symbol,
                "file_count": 0,
                "row_count": 0,
                "latest_event_ts_ms": None,
                "latest_recv_ms": None,
            },
        )
        row["file_count"] += 1
        row["row_count"] += len(df)
        total_rows += len(df)
        symbols.add(symbol)
        event_ts = _to_int(df["event_ts_ms"].max()) if "event_ts_ms" in df else None
        recv = _to_int(df["recv_ms"].max()) if "recv_ms" in df else None
        if event_ts is not None:
            row["latest_event_ts_ms"] = (
                event_ts
                if row["latest_event_ts_ms"] is None
                else max(row["latest_event_ts_ms"], event_ts)
            )
        if recv is not None:
            row["latest_recv_ms"] = (
                recv
                if row["latest_recv_ms"] is None
                else max(row["latest_recv_ms"], recv)
            )
            latest_recv = recv if latest_recv is None else max(latest_recv, recv)

    inventory_rows = [
        InventoryRow(
            channel=str(row["channel"]),
            symbol=str(row["symbol"]),
            file_count=int(row["file_count"]),
            row_count=int(row["row_count"]),
            latest_event_ts_ms=row["latest_event_ts_ms"],
            latest_recv_ms=row["latest_recv_ms"],
            latest_age_s=(
                (now - row["latest_recv_ms"]) / 1000
                if row["latest_recv_ms"] is not None
                else None
            ),
        )
        for row in stats.values()
    ]
    inventory_rows.sort(key=lambda row: (row.channel, row.symbol))
    return RawInventory(
        raw_dir=str(silver_dir),
        file_count=len(files),
        row_count=total_rows,
        symbol_count=len(symbols),
        latest_recv_ms=latest_recv,
        latest_age_s=((now - latest_recv) / 1000 if latest_recv is not None else None),
        rows=inventory_rows,
    )


def _safe_basis(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return (numerator - denominator) / denominator * 10_000


def _latest_by_symbol(rows: list[JsonObject]) -> dict[str, JsonObject]:
    latest: dict[str, JsonObject] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "UNKNOWN")
        ts = _to_int(row.get("event_ts_ms") or row.get("recv_ms"))
        old_ts = _to_int(
            latest.get(symbol, {}).get("event_ts_ms")
            or latest.get(symbol, {}).get("recv_ms")
        )
        if old_ts is None or (ts is not None and ts >= old_ts):
            latest[symbol] = row
    return latest


def _stress_score(
    *,
    trade_count: int,
    spread_bps: float | None,
    return_bps: float | None,
    basis_bps: float | None,
) -> float:
    score = 0.0
    score += min(trade_count / 100.0, 2.0)
    if spread_bps is not None:
        score += min(max(spread_bps, 0.0) / 100.0, 5.0)
    if return_bps is not None:
        score += min(abs(return_bps) / 100.0, 5.0)
    if basis_bps is not None:
        score += min(abs(basis_bps) / 100.0, 5.0)
    return score


def _compute_latest_features_from_normalized(
    normalized_trades: list[JsonObject],
    normalized_bbo: list[JsonObject],
    normalized_prices: list[JsonObject],
) -> list[RealtimeFeature]:
    latest_trade_ts_by_symbol: dict[str, int] = {}
    for trade in normalized_trades:
        symbol = str(trade.get("symbol") or "UNKNOWN")
        ts = _to_int(trade.get("event_ts_ms"))
        if ts is not None:
            latest_trade_ts_by_symbol[symbol] = max(
                latest_trade_ts_by_symbol.get(symbol, ts), ts
            )

    latest_bbo = _latest_by_symbol(normalized_bbo)
    latest_prices = _latest_by_symbol(normalized_prices)
    features: list[RealtimeFeature] = []

    for symbol, latest_ts in sorted(latest_trade_ts_by_symbol.items()):
        window_start = (latest_ts // 60_000) * 60_000
        window_end = window_start + 60_000
        trades = [
            trade
            for trade in normalized_trades
            if trade.get("symbol") == symbol
            and (ts := _to_int(trade.get("event_ts_ms"))) is not None
            and window_start <= ts < window_end
        ]
        trades.sort(
            key=lambda row: (
                _to_int(row.get("event_ts_ms")) or 0,
                _to_int(row.get("recv_ms")) or 0,
            )
        )
        prices = [_to_float(trade.get("price")) for trade in trades]
        prices = [price for price in prices if price is not None]
        first_price = prices[0] if prices else None
        last_price = prices[-1] if prices else None
        return_bps = _safe_basis(last_price, first_price)
        trade_volume = sum(_to_float(trade.get("qty")) or 0.0 for trade in trades)
        trade_notional = sum(
            _to_float(trade.get("notional")) or 0.0 for trade in trades
        )
        signed_volume = sum(
            _to_float(trade.get("signed_qty")) or 0.0 for trade in trades
        )

        bbo = latest_bbo.get(symbol, {})
        price = latest_prices.get(symbol, {})
        top_depth = None
        if bbo:
            bid_notional = _to_float(bbo.get("top_bid_notional")) or 0.0
            ask_notional = _to_float(bbo.get("top_ask_notional")) or 0.0
            top_depth = bid_notional + ask_notional
        mark_basis = _to_float(price.get("mark_oracle_basis_bps"))
        stress = _stress_score(
            trade_count=len(trades),
            spread_bps=_to_float(bbo.get("spread_bps")),
            return_bps=return_bps,
            basis_bps=mark_basis,
        )
        features.append(
            RealtimeFeature(
                symbol=symbol,
                window_start_ms=window_start,
                window_end_ms=window_end,
                trade_count_1m=len(trades),
                trade_volume_1m=trade_volume,
                trade_notional_1m=trade_notional,
                signed_trade_volume_1m=signed_volume,
                last_price=last_price,
                return_bps_1m=return_bps,
                spread_bps=_to_float(bbo.get("spread_bps")),
                top_depth_notional=top_depth,
                mark_oracle_basis_bps=mark_basis,
                mid_oracle_basis_bps=_to_float(price.get("mid_oracle_basis_bps")),
                open_interest=_to_float(price.get("open_interest")),
                funding=_to_float(price.get("funding")),
                stress_score=stress,
            )
        )
    features.sort(key=lambda row: row.stress_score, reverse=True)
    return features


def _compute_latest_features(records: list[JsonObject]) -> list[RealtimeFeature]:
    normalized_trades: list[JsonObject] = []
    normalized_bbo: list[JsonObject] = []
    normalized_prices: list[JsonObject] = []
    for record in records:
        channel = record.get("channel")
        if channel == "trades":
            normalized_trades.append(normalize_trade_record(record))
        elif channel == "bbo":
            normalized_bbo.append(normalize_bbo_record(record))
        elif channel == "prices":
            normalized_prices.append(normalize_price_record(record))
    return _compute_latest_features_from_normalized(
        normalized_trades, normalized_bbo, normalized_prices
    )


def _read_silver_rows(
    silver_dir: Path, *, max_files: int | None = None
) -> dict[str, list[JsonObject]]:
    tables = {"trades": [], "bbo": [], "prices": []}
    wanted = set(tables)
    for path in _silver_files(silver_dir, max_files=max_files):
        channel = _path_part(path, "channel=") or "unknown"
        if channel not in wanted:
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        tables[channel].extend(df.to_dict(orient="records"))
    return tables


def _compute_latest_features_from_silver(
    silver_dir: Path, *, max_files: int | None = None
) -> list[RealtimeFeature]:
    tables = _read_silver_rows(silver_dir, max_files=max_files)
    return _compute_latest_features_from_normalized(
        tables["trades"], tables["bbo"], tables["prices"]
    )


def build_realtime_report(
    raw_dir: Path,
    *,
    source: str = "raw",
    now_ms: int | None = None,
    stale_after_s: float = 300.0,
    max_files: int | None = 200,
    max_records_per_file: int | None = 5_000,
) -> RealtimeReport:
    if source not in {"raw", "silver"}:
        raise ValueError("source must be 'raw' or 'silver'")
    now = utc_ms() if now_ms is None else now_ms
    if source == "silver":
        inventory = inventory_silver_archive(raw_dir, now_ms=now, max_files=max_files)
        features = _compute_latest_features_from_silver(raw_dir, max_files=max_files)
    else:
        inventory = inventory_raw_archive(
            raw_dir,
            now_ms=now,
            max_files=max_files,
            max_records_per_file=max_records_per_file,
        )
        records = [
            record
            for _, record in iter_raw_records_with_path(
                raw_dir, max_files=max_files, max_records_per_file=max_records_per_file
            )
        ]
        features = _compute_latest_features(records)
    warnings: list[str] = []
    if inventory.latest_age_s is None:
        warnings.append(f"{source} archive has no readable rows")
    elif inventory.latest_age_s > stale_after_s:
        warnings.append(
            f"{source} archive stale: latest row age {inventory.latest_age_s:.1f}s exceeds {stale_after_s:.1f}s"
        )
    present_channels = {row.channel for row in inventory.rows}
    for channel in ("trades", "bbo", "prices"):
        if channel not in present_channels:
            warnings.append(f"missing channel in {source} archive: {channel}")
    return RealtimeReport(
        generated_at_ms=now,
        raw_dir=str(raw_dir),
        stale_after_s=stale_after_s,
        inventory=inventory,
        features=features,
        warnings=warnings,
    )


def _fmt_ts(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "n/a"
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def render_markdown(report: RealtimeReport) -> str:
    top_features = [asdict(row) for row in report.features[:20]]
    top_inventory = [
        asdict(row)
        for row in sorted(
            report.inventory.rows,
            key=lambda row: (
                row.latest_age_s if row.latest_age_s is not None else float("inf")
            ),
            reverse=True,
        )[:30]
    ]
    return "\n".join(
        [
            "# Pacifica realtime research monitor",
            "",
            "This is read-only diagnostics. It does not place trades, tune thresholds, or claim edge.",
            "",
            f"Generated UTC: {_fmt_ts(report.generated_at_ms)}",
            f"Source dir: `{report.raw_dir}`",
            f"Files: {report.inventory.file_count}",
            f"Rows: {report.inventory.row_count}",
            f"Symbols: {report.inventory.symbol_count}",
            f"Latest source row age seconds: {report.inventory.latest_age_s}",
            "",
            "## Warnings",
            "",
            (
                "\n".join(f"- {warning}" for warning in report.warnings)
                if report.warnings
                else "_none_"
            ),
            "",
            "## Top stress features",
            "",
            _markdown_table(
                top_features,
                [
                    "symbol",
                    "trade_count_1m",
                    "trade_volume_1m",
                    "return_bps_1m",
                    "spread_bps",
                    "top_depth_notional",
                    "mark_oracle_basis_bps",
                    "stress_score",
                ],
            ),
            "",
            "## Stalest channel/symbol inventory rows",
            "",
            _markdown_table(
                top_inventory,
                [
                    "channel",
                    "symbol",
                    "file_count",
                    "row_count",
                    "latest_event_ts_ms",
                    "latest_recv_ms",
                    "latest_age_s",
                ],
            ),
            "",
        ]
    )


def write_realtime_outputs(report: RealtimeReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out_dir / "README.md",
        "features_csv": out_dir / "latest_features.csv",
        "inventory_csv": out_dir / "raw_inventory.csv",
        "warnings_json": out_dir / "warnings.json",
    }
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    pd.DataFrame([asdict(row) for row in report.features]).to_csv(
        paths["features_csv"], index=False
    )
    pd.DataFrame([asdict(row) for row in report.inventory.rows]).to_csv(
        paths["inventory_csv"], index=False
    )
    paths["warnings_json"].write_text(
        json.dumps(report.warnings, indent=2), encoding="utf-8"
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("raw", "silver"),
        default="silver",
        help="Read from normalized silver parquet by default; use raw for direct JSONL.GZ diagnostics.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw JSONL.GZ directory. Used when --source raw, or as fallback if --input-dir is omitted.",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=DEFAULT_SILVER_DIR,
        help="Partitioned silver parquet directory. Used when --source silver.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Explicit source directory override for either source mode.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stale-after-s", type=float, default=300.0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Selected source files to scan. Default: all silver files, or 200 raw files. Use 0 for all files.",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=5_000,
        help="Scan at most this many JSONL rows per selected gzip file. Use 0 for all rows.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_files == -1:
        max_files = None if args.source == "silver" else 200
    else:
        max_files = None if args.max_files == 0 else args.max_files
    max_records_per_file = (
        None if args.max_records_per_file == 0 else args.max_records_per_file
    )
    if args.input_dir is not None:
        source_dir = args.input_dir
    elif args.source == "silver":
        source_dir = args.silver_dir
    else:
        source_dir = args.raw_dir or DEFAULT_RAW_DIR
    report = build_realtime_report(
        source_dir,
        source=args.source,
        stale_after_s=args.stale_after_s,
        max_files=max_files,
        max_records_per_file=max_records_per_file,
    )
    paths = write_realtime_outputs(report, args.out_dir)
    print(f"wrote realtime research diagnostics to {args.out_dir}")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    if report.warnings:
        print("warnings:")
        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

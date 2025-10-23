#!/usr/bin/env python3
"""Quick health-check for the synced Pacifica parquet datasets.

The script scans the `trades`, `orderbook`, `prices`, and `funding` folders
under a given data root. For each dataset it reports:
  * number of files discovered and inspected
  * aggregated row counts from parquet metadata
  * date range inferred from `ts_ms`
  * per-symbol file counts
  * any schema or ordering issues detected

Usage:
    python scripts/validate_cloud_dataset.py --data-root ./cloud-data --max-files 200

By default all files are inspected; use --max-files to cap the per-dataset
inspection when working with very large archives.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

EXPECTED_COLUMNS = {
    "trades": {"ts_ms", "recv_ms", "symbol", "trade_id", "side", "qty", "price"},
    "orderbook": {"ts_ms", "recv_ms", "symbol", "bids", "asks"},
    "prices": {"ts_ms", "recv_ms", "symbol", "price"},
    "funding": {"ts_ms", "recv_ms", "symbol", "rate", "interval_sec"},
}


def _format_ts(ts_ms: int | float | None) -> str:
    if ts_ms is None or (isinstance(ts_ms, float) and math.isnan(ts_ms)):
        return "n/a"
    dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    return dt.isoformat()


def _iter_parquet_files(base: Path) -> Iterable[Path]:
    return base.glob("symbol=*/date=*/*.parquet")


def inspect_dataset(dataset: str, root: Path, max_files: int | None) -> dict:
    path = root / dataset
    files = sorted(_iter_parquet_files(path))
    total_files = len(files)
    if total_files == 0:
        return {
            "dataset": dataset,
            "total_files": 0,
            "inspected": 0,
            "total_rows": 0,
            "min_ts": None,
            "max_ts": None,
            "dates": set(),
            "symbol_counts": Counter(),
            "issues": ["No files found"],
        }

    issues: list[str] = []
    symbol_counts: Counter[str] = Counter()
    dates: set[str] = set()
    min_ts: int | None = None
    max_ts: int | None = None
    total_rows = 0

    files_to_process = files if max_files is None else files[:max_files]

    for file_path in files_to_process:
        try:
            metadata = pq.read_metadata(file_path)
        except Exception as exc:  # pragma: no cover - corrupt file
            issues.append(f"{file_path}: failed to read metadata ({exc})")
            continue

        dataset_schema = set(metadata.schema.names)
        missing = EXPECTED_COLUMNS[dataset] - dataset_schema
        if missing:
            issues.append(f"{file_path}: missing columns {sorted(missing)}")

        total_rows += metadata.num_rows

        symbol_part = file_path.parent.parent.name  # e.g. symbol=BTC
        date_part = file_path.parent.name  # e.g. date=2025-10-18
        symbol = symbol_part.split("=", 1)[-1]
        date_value = date_part.split("=", 1)[-1]
        symbol_counts[symbol] += 1
        dates.add(date_value)

        # Check ordering and duplicates inside the file.
        try:
            table_ts = pq.read_table(file_path, columns=["ts_ms"])
        except Exception as exc:
            issues.append(f"{file_path}: failed to read ts_ms ({exc})")
            continue

        ts_values = table_ts.column(0).to_numpy(zero_copy_only=False)
        if ts_values.size == 0:
            continue

        file_min = int(ts_values.min())
        file_max = int(ts_values.max())
        min_ts = file_min if min_ts is None else min(min_ts, file_min)
        max_ts = file_max if max_ts is None else max(max_ts, file_max)

        if np.any(np.diff(ts_values) < 0):
            issues.append(f"{file_path}: ts_ms not sorted")

        if dataset == "trades":
            ids_table = pq.read_table(file_path, columns=["trade_id"])
            ids = ids_table.column(0).to_numpy(zero_copy_only=False)
            unique = np.unique(ids)
            if unique.size != ids.size:
                issues.append(f"{file_path}: duplicate trade_id values detected")

        if dataset == "funding" and "interval_sec" not in dataset_schema:
            issues.append(f"{file_path}: missing interval_sec column")

    return {
        "dataset": dataset,
        "total_files": total_files,
        "inspected": len(files_to_process),
        "total_rows": total_rows,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "dates": dates,
        "symbol_counts": symbol_counts,
        "issues": issues,
    }


def print_summary(report: dict) -> None:
    dataset = report["dataset"]
    print(f"\n=== {dataset.upper()} ===")
    print(f"Files: {report['inspected']} inspected / {report['total_files']} total")
    print(f"Rows (metadata): {report['total_rows']:,}")
    print(
        "Date range:",
        _format_ts(report["min_ts"]),
        "→",
        _format_ts(report["max_ts"]),
    )

    dates = sorted(report["dates"])
    if dates:
        preview = ", ".join(dates[:5])
        if len(dates) > 5:
            preview += f" … ({len(dates)} days)"
        print("Dates covered:", preview)

    symbols = report["symbol_counts"]
    if symbols:
        top = symbols.most_common(10)
        formatted = ", ".join(f"{sym}:{count}" for sym, count in top)
        if len(symbols) > len(top):
            formatted += f" … ({len(symbols)} symbols total)"
        print("Files per symbol:", formatted)

    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("Issues: none detected ✅")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate synced Pacifica parquet datasets"
    )
    parser.add_argument(
        "--data-root",
        default="./cloud-data",
        help="Path where the S3 archive was synced (default: ./cloud-data)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on files inspected per dataset.",
    )
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(
            f"Data root '{root}' does not exist. Run fetch_cloud_data.sh first."
        )

    print(f"Inspecting parquet datasets under {root}")
    for dataset in EXPECTED_COLUMNS:
        summary = inspect_dataset(dataset, root, args.max_files)
        print_summary(summary)


if __name__ == "__main__":
    main()

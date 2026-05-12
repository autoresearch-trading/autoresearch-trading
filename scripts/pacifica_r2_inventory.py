#!/usr/bin/env python3
"""Convert `rclone lsjson` output into the CSV inventory used by R2 planners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sorted_inventory(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["key", "size_bytes", "mod_time"]).sort_values(
        "key", ignore_index=True
    )


def rclone_lsjson_to_inventory(payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload:
        if item.get("IsDir"):
            continue
        path = item.get("Path")
        if not path:
            continue
        rows.append(
            {
                "key": str(path),
                "size_bytes": int(item.get("Size") or 0),
                "mod_time": str(item.get("ModTime") or ""),
            }
        )
    return _sorted_inventory(rows)


def rclone_lsf_to_inventory(listing: str) -> pd.DataFrame:
    """Parse `rclone lsf --format pst --separator ';'` output into inventory rows."""

    rows: list[dict[str, Any]] = []
    for raw_line in listing.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        parts = line.split(";", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid rclone lsf inventory line: {raw_line!r}")
        path, size, mod_time = parts
        if not path:
            continue
        rows.append(
            {
                "key": path,
                "size_bytes": int(size or 0),
                "mod_time": mod_time,
            }
        )
    return _sorted_inventory(rows)


def _write_inventory_frame(inventory: pd.DataFrame, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out_csv, index=False, lineterminator="\n")
    return out_csv


def write_inventory_csv(lsjson_path: Path, out_csv: Path) -> Path:
    payload = json.loads(lsjson_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("rclone lsjson payload must be a JSON list")
    inventory = rclone_lsjson_to_inventory(payload)
    return _write_inventory_frame(inventory, out_csv)


def write_inventory_csv_from_lsf(lsf_path: Path, out_csv: Path) -> Path:
    inventory = rclone_lsf_to_inventory(lsf_path.read_text())
    return _write_inventory_frame(inventory, out_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsjson", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = write_inventory_csv(args.lsjson, args.out_csv)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

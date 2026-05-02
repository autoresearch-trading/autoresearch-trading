#!/usr/bin/env python3
"""Convert `rclone lsjson` output into the CSV inventory used by R2 planners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    return pd.DataFrame(rows, columns=["key", "size_bytes", "mod_time"]).sort_values(
        "key", ignore_index=True
    )


def write_inventory_csv(lsjson_path: Path, out_csv: Path) -> Path:
    payload = json.loads(lsjson_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("rclone lsjson payload must be a JSON list")
    inventory = rclone_lsjson_to_inventory(payload)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out_csv, index=False)
    return out_csv


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

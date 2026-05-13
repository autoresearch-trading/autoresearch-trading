#!/usr/bin/env python3
"""Convert rclone object listings into the CSV inventory used by R2 planners."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def _sorted_inventory(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["key", "size_bytes", "mod_time"]).sort_values(
        "key", ignore_index=True
    )


def _prefix_key(path: str, key_prefix: str = "") -> str:
    prefix = key_prefix.strip("/")
    clean_path = path.strip("/")
    if not prefix:
        return clean_path
    if clean_path == prefix or clean_path.startswith(f"{prefix}/"):
        return clean_path
    return f"{prefix}/{clean_path}"


def _parse_lsf_line(raw_line: str, *, key_prefix: str = "") -> dict[str, Any] | None:
    line = raw_line.rstrip("\n").rstrip("\r")
    if not line:
        return None
    parts = line.split(";", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid rclone lsf inventory line: {raw_line!r}")
    path, size, mod_time = parts
    if not path:
        return None
    return {
        "key": _prefix_key(path, key_prefix),
        "size_bytes": int(size or 0),
        "mod_time": mod_time,
    }


def iter_rclone_lsf_inventory_rows(
    lines: Iterable[str], *, key_prefix: str = ""
) -> Iterable[dict[str, Any]]:
    """Yield inventory rows from line-oriented ``rclone lsf`` output.

    The caller can pass an open file handle so large inventories are processed
    line-by-line instead of through a captured stdout/status field that can be
    truncated by terminals or agent tooling.
    """

    for raw_line in lines:
        row = _parse_lsf_line(raw_line, key_prefix=key_prefix)
        if row is not None:
            yield row


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


def rclone_lsf_to_inventory(listing: str, *, key_prefix: str = "") -> pd.DataFrame:
    """Parse `rclone lsf --format pst --separator ';'` output into inventory rows."""

    return _sorted_inventory(
        list(
            iter_rclone_lsf_inventory_rows(listing.splitlines(), key_prefix=key_prefix)
        )
    )


def _write_inventory_frame(inventory: pd.DataFrame, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out_csv, index=False, lineterminator="\n")
    return out_csv


def write_inventory_csv_from_lsf_stream(
    lsf_path: Path, out_csv: Path, *, key_prefix: str = ""
) -> Path:
    """Convert a saved line-oriented rclone listing to CSV without loading it all.

    Unlike ``rclone_lsjson_to_inventory`` and the small-listing convenience
    parser, this preserves input order so it can write rows as it reads them.
    """

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with (
        lsf_path.open("r", encoding="utf-8") as src,
        out_csv.open("w", encoding="utf-8", newline="") as dst,
    ):
        writer = csv.DictWriter(
            dst, fieldnames=["key", "size_bytes", "mod_time"], lineterminator="\n"
        )
        writer.writeheader()
        for row in iter_rclone_lsf_inventory_rows(src, key_prefix=key_prefix):
            writer.writerow(row)
    return out_csv


def write_inventory_csv(lsjson_path: Path, out_csv: Path) -> Path:
    payload = json.loads(lsjson_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("rclone lsjson payload must be a JSON list")
    inventory = rclone_lsjson_to_inventory(payload)
    return _write_inventory_frame(inventory, out_csv)


def write_inventory_csv_from_lsf(
    lsf_path: Path, out_csv: Path, *, key_prefix: str = ""
) -> Path:
    inventory = rclone_lsf_to_inventory(lsf_path.read_text(), key_prefix=key_prefix)
    return _write_inventory_frame(inventory, out_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lsjson", type=Path)
    source.add_argument("--lsf", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--key-prefix",
        default="",
        help="Prefix to prepend to relative rclone lsf paths, e.g. raw/pacifica/full_fidelity/.",
    )
    parser.add_argument(
        "--stream-lsf",
        action="store_true",
        help="Process --lsf line-by-line without sorting/loading the whole listing.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.lsjson is not None:
        out = write_inventory_csv(args.lsjson, args.out_csv)
    elif args.stream_lsf:
        out = write_inventory_csv_from_lsf_stream(
            args.lsf, args.out_csv, key_prefix=args.key_prefix
        )
    else:
        out = write_inventory_csv_from_lsf(
            args.lsf, args.out_csv, key_prefix=args.key_prefix
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Health check for the Pacifica full-fidelity always-on collector host.

Prints a compact JSON status and exits non-zero when core safety checks fail.
No credentials or environment dumps are printed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def newest_file(root: Path) -> tuple[str | None, float | None]:
    newest_path: Path | None = None
    newest_mtime: float | None = None
    if not root.exists():
        return None, None
    for path in root.rglob("*.jsonl.gz"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest_mtime is None or mtime > newest_mtime:
            newest_path = path
            newest_mtime = mtime
    return (str(newest_path) if newest_path else None), newest_mtime


def db_counts(state_db: Path) -> dict[str, dict[str, int]]:
    if not state_db.exists():
        return {}
    con = sqlite3.connect(state_db)
    try:
        rows = con.execute(
            "select status, count(*), coalesce(sum(size_bytes), 0) "
            "from archive_files group by status"
        ).fetchall()
    finally:
        con.close()
    return {
        status: {"files": int(count), "bytes": int(size)}
        for status, count, size in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/pacifica_full_fidelity")
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path("data/pacifica_full_fidelity_storage.sqlite"),
    )
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--max-newest-age-min", type=float, default=20.0)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    usage = shutil.disk_usage(args.root if args.root.exists() else args.root.parent)
    free_gb = usage.free / 1024**3
    newest_path, newest_mtime = newest_file(args.root)
    newest_age_min = None
    if newest_mtime is not None:
        newest_age_min = (now.timestamp() - newest_mtime) / 60.0

    counts = db_counts(args.state_db)
    failures: list[str] = []
    if free_gb < args.min_free_gb:
        failures.append(
            f"free disk {free_gb:.2f} GiB below floor {args.min_free_gb:.2f} GiB"
        )
    if newest_age_min is None:
        failures.append("no raw jsonl.gz files found")
    elif newest_age_min > args.max_newest_age_min:
        failures.append(
            f"newest raw file age {newest_age_min:.1f} min above limit "
            f"{args.max_newest_age_min:.1f} min"
        )

    sealed = counts.get("sealed", {}).get("bytes", 0)
    uploaded = counts.get("uploaded", {}).get("bytes", 0)
    unverified_gb = (sealed + uploaded) / 1024**3

    status = {
        "ok": not failures,
        "checked_at": now.isoformat(),
        "root": str(args.root),
        "state_db": str(args.state_db),
        "free_gb": round(free_gb, 2),
        "newest_raw_file": newest_path,
        "newest_raw_age_min": (
            None if newest_age_min is None else round(newest_age_min, 2)
        ),
        "db_counts": counts,
        "unverified_gb": round(unverified_gb, 2),
        "failures": failures,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

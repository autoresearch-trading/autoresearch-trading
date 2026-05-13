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
import time
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


def _connect_readonly(state_db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{state_db.resolve()}?mode=ro", uri=True)


def _empty_counter() -> dict[str, int]:
    return {"files": 0, "bytes": 0}


def db_counts(state_db: Path) -> dict[str, dict[str, int]]:
    if not state_db.exists():
        return {}
    con = _connect_readonly(state_db)
    try:
        rows = con.execute(
            "select status, count(*), coalesce(sum(size_bytes), 0) "
            "from archive_files group by status order by status"
        ).fetchall()
    finally:
        con.close()
    return {
        status: {"files": int(count), "bytes": int(size)}
        for status, count, size in rows
    }


def db_error_counts(state_db: Path, *, top_limit: int = 20) -> dict[str, object]:
    if not state_db.exists():
        return {
            "rows_with_errors": 0,
            "bytes_with_errors": 0,
            "by_status": {},
            "top_errors": [],
        }
    con = _connect_readonly(state_db)
    try:
        total = con.execute(
            """
            select count(*), coalesce(sum(size_bytes), 0)
            from archive_files
            where error is not null
            """
        ).fetchone()
        by_status_rows = con.execute(
            """
            select status, count(*), coalesce(sum(size_bytes), 0)
            from archive_files
            where error is not null
            group by status
            order by status
            """
        ).fetchall()
        top_rows = con.execute(
            """
            select status, error, count(*), coalesce(sum(size_bytes), 0)
            from archive_files
            where error is not null
            group by status, error
            order by count(*) desc, status, error
            limit ?
            """,
            (top_limit,),
        ).fetchall()
    finally:
        con.close()
    return {
        "rows_with_errors": int(total[0]),
        "bytes_with_errors": int(total[1]),
        "by_status": {
            status: {"files": int(count), "bytes": int(size)}
            for status, count, size in by_status_rows
        },
        "top_errors": [
            {
                "status": status,
                "error": error,
                "files": int(count),
                "bytes": int(size),
            }
            for status, error, count, size in top_rows
        ],
    }


def _recent_counter(
    con: sqlite3.Connection, column: str, *, cutoff_epoch: float
) -> dict[str, int]:
    count, size = con.execute(
        f"""
        select count(*), coalesce(sum(size_bytes), 0)
        from archive_files
        where {column} is not null and {column} >= ?
        """,
        (cutoff_epoch,),
    ).fetchone()
    return {"files": int(count), "bytes": int(size)}


def db_recent_activity(
    state_db: Path, *, window_seconds: int, now_epoch: float | None = None
) -> dict[str, object]:
    if now_epoch is None:
        now_epoch = time.time()
    window_seconds = max(0, int(window_seconds))
    cutoff = float(now_epoch) - window_seconds
    if not state_db.exists():
        return {
            "window_seconds": window_seconds,
            "first_seen": _empty_counter(),
            "last_seen": _empty_counter(),
            "uploaded": _empty_counter(),
            "verified": _empty_counter(),
            "pruned": _empty_counter(),
        }
    con = _connect_readonly(state_db)
    try:
        return {
            "window_seconds": window_seconds,
            "first_seen": _recent_counter(con, "first_seen_at", cutoff_epoch=cutoff),
            "last_seen": _recent_counter(con, "last_seen_at", cutoff_epoch=cutoff),
            "uploaded": _recent_counter(con, "uploaded_at", cutoff_epoch=cutoff),
            "verified": _recent_counter(con, "remote_verified_at", cutoff_epoch=cutoff),
            "pruned": _recent_counter(con, "pruned_at", cutoff_epoch=cutoff),
        }
    finally:
        con.close()


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
    parser.add_argument(
        "--recent-window-min",
        type=float,
        default=60.0,
        help="window for recent upload/verify/prune throughput counts",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    usage = shutil.disk_usage(args.root if args.root.exists() else args.root.parent)
    free_gb = usage.free / 1024**3
    newest_path, newest_mtime = newest_file(args.root)
    newest_age_min = None
    if newest_mtime is not None:
        newest_age_min = (now.timestamp() - newest_mtime) / 60.0

    counts = db_counts(args.state_db)
    errors = db_error_counts(args.state_db)
    recent_activity = db_recent_activity(
        args.state_db,
        window_seconds=int(args.recent_window_min * 60),
        now_epoch=now.timestamp(),
    )
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
    rows_with_errors = int(errors["rows_with_errors"])
    if rows_with_errors:
        failures.append(f"archive lifecycle DB has {rows_with_errors} error rows")

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
        "db_error_counts": errors,
        "recent_activity": recent_activity,
        "unverified_gb": round(unverified_gb, 2),
        "failures": failures,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

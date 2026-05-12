#!/usr/bin/env python3
"""Manage Pacifica full-fidelity raw archive local/R2 lifecycle.

This script intentionally does not delete raw archive files unless they are already
recorded as remote-verified in the local SQLite state DB. It is the storage
control-plane companion to ``collect_pacifica_full_fidelity.py``:

1. scan sealed local archive files;
2. record deterministic R2 object keys, sizes, and checksums;
3. let an upload/verification step mark rows verified;
4. prune only verified local copies after a retention window.

Credentials are deliberately out-of-scope here; use rclone config, environment
variables, or a secret manager for the actual R2 upload command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ARCHIVE_EXTENSIONS = (".jsonl.gz", ".jsonl.zst", ".parquet")
STATE_SCHEMA_VERSION = 1
RcloneRunner = Callable[[list[str]], str]


def remote_path_for(remote_base: str, object_key: str) -> str:
    """Join an rclone remote/base path and object key without dropping prefixes."""
    return f"{remote_base.rstrip('/')}/{object_key.lstrip('/')}"


def run_rclone(args: list[str], *, input_text: str | None = None) -> str:
    """Run rclone and return stdout, keeping credentials out of logs."""
    proc = subprocess.run(
        ["rclone", *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"rclone {' '.join(args[:2])} failed with exit {proc.returncode}: {stderr}"
        )
    return proc.stdout


def _sidecar_text(sha256: str, local_path: str) -> str:
    return f"{sha256}  {Path(local_path).name}\n"


def _set_error(conn: sqlite3.Connection, local_path: str, error: str | None) -> None:
    conn.execute(
        "update archive_files set error=? where local_path=?", (error, local_path)
    )


def _coerce_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _is_current_hour_partition(path: Path, *, now: datetime) -> bool:
    date_part = f"date={now.strftime('%Y-%m-%d')}"
    hour_part = f"hour={now.strftime('%H')}"
    parts = set(path.parts)
    return date_part in parts and hour_part in parts


def _is_archive_payload(path: Path) -> bool:
    name = path.name
    if name.endswith(".tmp") or name.endswith(".partial") or name.endswith(".lock"):
        return False
    if name.endswith(".sqlite") or name.endswith(".sqlite3") or name.endswith(".db"):
        return False
    return any(name.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def iter_archive_files(
    root: Path, *, skip_current_hour: bool = False, now: datetime | None = None
) -> Iterator[Path]:
    """Yield sealed archive payload files under ``root`` in stable order.

    Active temp files and SQLite state files are skipped so upload/prune jobs do
    not race a writer or accidentally treat control-plane state as market data.
    When ``skip_current_hour`` is true, also skip the current UTC hour partition;
    the live collector can still be appending to those gzip files.
    """
    root = root.resolve()
    if not root.exists():
        return
    now = _coerce_utc(now)
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if skip_current_hour and _is_current_hour_partition(path, now=now):
            continue
        if _is_archive_payload(path):
            files.append(path)
    yield from sorted(files)


def iter_recent_archive_files(
    root: Path,
    *,
    recent_hours: int,
    skip_current_hour: bool = False,
    now: datetime | None = None,
) -> Iterator[Path]:
    """Yield archive payloads from only recent date/hour partitions.

    The full archive has tens of thousands of sealed chunks. For freshness-lane
    cycles, scanning ``channel=*/symbol=*/date=.../hour=...`` for a bounded set
    of recent UTC hours avoids rewalking and restatting historical backlog while
    still discovering newly closed chunks that are eligible for upload.
    """
    root = root.resolve()
    if not root.exists() or recent_hours <= 0:
        return
    now = _coerce_utc(now)
    hours: list[datetime] = []
    for offset in range(recent_hours + 1):
        hour = (now - timedelta(hours=offset)).replace(
            minute=0, second=0, microsecond=0
        )
        if skip_current_hour and hour == now.replace(minute=0, second=0, microsecond=0):
            continue
        hours.append(hour)
    files: list[Path] = []
    seen: set[Path] = set()
    for hour in sorted(hours):
        date_part = f"date={hour:%Y-%m-%d}"
        hour_part = f"hour={hour:%H}"
        pattern = f"channel=*/symbol=*/{date_part}/{hour_part}/*"
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if skip_current_hour and _is_current_hour_partition(path, now=now):
                continue
            if _is_archive_payload(path):
                seen.add(path)
                files.append(path)
    yield from sorted(files)


def object_key_for(root: Path, file_path: Path, r2_prefix: str) -> str:
    """Return deterministic object key preserving partition path under prefix."""
    rel = file_path.resolve().relative_to(root.resolve())
    prefix = r2_prefix.strip("/")
    return f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def connect_state(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute(
        """
        create table if not exists archive_files (
            local_path text primary key,
            object_key text not null,
            size_bytes integer not null,
            modified_at real,
            sha256 text not null,
            status text not null check(status in ('sealed','uploaded','verified','pruned','missing')),
            first_seen_at real not null,
            last_seen_at real not null,
            uploaded_at real,
            remote_verified_at real,
            pruned_at real,
            error text
        )
        """
    )
    existing_columns = {
        row[1] for row in conn.execute("pragma table_info(archive_files)").fetchall()
    }
    if "modified_at" not in existing_columns:
        conn.execute("alter table archive_files add column modified_at real")
    conn.execute(
        """
        create table if not exists storage_metadata (
            key text primary key,
            value text not null
        )
        """
    )
    conn.execute(
        "insert or replace into storage_metadata(key, value) values ('schema_version', ?)",
        (str(STATE_SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def scan_archive_files(
    root: Path,
    db_path: Path,
    *,
    r2_prefix: str,
    skip_current_hour: bool = False,
    recent_hours: int | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Record sealed local archive files and return their state rows."""
    root = root.resolve()
    seen_at = time.time()
    rows: list[dict[str, Any]] = []
    iterator = (
        iter_recent_archive_files(
            root,
            recent_hours=recent_hours,
            skip_current_hour=skip_current_hour,
            now=now,
        )
        if recent_hours is not None
        else iter_archive_files(root, skip_current_hour=skip_current_hour, now=now)
    )
    with connect_state(db_path) as conn:
        for path in iterator:
            stat = path.stat()
            modified_at = float(stat.st_mtime)
            object_key = object_key_for(root, path, r2_prefix)
            existing = conn.execute(
                "select status, size_bytes, modified_at, sha256, error from archive_files where local_path=?",
                (str(path),),
            ).fetchone()
            digest = None
            if existing and existing["modified_at"] is not None:
                unchanged_stat = (
                    int(existing["size_bytes"]) == int(stat.st_size)
                    and float(existing["modified_at"]) == modified_at
                )
                if unchanged_stat:
                    digest = existing["sha256"]
            if digest is None:
                digest = sha256_file(path)
            status = "sealed"
            error = None
            if existing:
                status = existing["status"]
                unchanged = (
                    int(existing["size_bytes"]) == int(stat.st_size)
                    and existing["sha256"] == digest
                )
                if status in {"pruned", "missing"}:
                    status = "sealed"
                elif status in {"uploaded", "verified"} and not unchanged:
                    status = "sealed"
                if status == "uploaded" and unchanged:
                    error = existing["error"]
            conn.execute(
                """
                insert into archive_files(
                    local_path, object_key, size_bytes, modified_at, sha256, status,
                    first_seen_at, last_seen_at, error
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(local_path) do update set
                    object_key=excluded.object_key,
                    size_bytes=excluded.size_bytes,
                    modified_at=excluded.modified_at,
                    sha256=excluded.sha256,
                    status=excluded.status,
                    last_seen_at=excluded.last_seen_at,
                    error=excluded.error
                """,
                (
                    str(path),
                    object_key,
                    stat.st_size,
                    modified_at,
                    digest,
                    status,
                    seen_at,
                    seen_at,
                    error,
                ),
            )
            saved = conn.execute(
                "select * from archive_files where local_path=?", (str(path),)
            ).fetchone()
            rows.append(_row_to_dict(saved))
        conn.commit()
    return rows


def mark_uploaded(db_path: Path, local_path: Path, *, verified: bool = False) -> None:
    now = time.time()
    status = "verified" if verified else "uploaded"
    with connect_state(db_path) as conn:
        conn.execute(
            """
            update archive_files
            set status=?, uploaded_at=coalesce(uploaded_at, ?),
                remote_verified_at=case when ? then ? else remote_verified_at end
            where local_path=?
            """,
            (status, now, 1 if verified else 0, now, str(local_path)),
        )
        conn.commit()


def write_manifest(
    db_path: Path,
    manifest_path: Path,
    *,
    statuses: tuple[str, ...] = ("sealed", "uploaded", "verified"),
) -> Path:
    """Write JSONL manifest rows for audit/upload tooling."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    placeholders = ",".join("?" for _ in statuses)
    with (
        connect_state(db_path) as conn,
        manifest_path.open("w", encoding="utf-8") as fh,
    ):
        rows = conn.execute(
            f"select * from archive_files where status in ({placeholders}) order by object_key",
            statuses,
        )
        for row in rows:
            fh.write(
                json.dumps(_row_to_dict(row), sort_keys=True, separators=(",", ":"))
            )
            fh.write("\n")
    return manifest_path


def _local_file_too_recent(path: Path, *, min_age_seconds: float) -> bool:
    if min_age_seconds <= 0:
        return False
    return time.time() - path.stat().st_mtime < min_age_seconds


def _upload_order_clause(order: str) -> str:
    if order == "object-key":
        return "object_key"
    if order == "newest-first":
        return "coalesce(modified_at, last_seen_at) desc, object_key desc"
    if order == "oldest-first":
        return "coalesce(modified_at, last_seen_at) asc, object_key asc"
    raise ValueError(f"unknown upload order: {order}")


def _upload_candidate_rows(
    conn: sqlite3.Connection,
    *,
    min_upload_age_seconds: float,
    order: str,
    limit: int | None,
) -> list[sqlite3.Row]:
    order_clause = _upload_order_clause(order)
    params: list[float] = []
    sealed_predicate = "status='sealed'"
    if min_upload_age_seconds > 0:
        cutoff = time.time() - min_upload_age_seconds
        sealed_predicate = (
            "(status='sealed' and (modified_at is null or modified_at <= ?))"
        )
        params.append(cutoff)
    query = f"""
        select * from archive_files
        where {sealed_predicate}
           or (status='uploaded' and error is not null)
        order by case
            when status='uploaded' and error is not null then 0
            else 1
        end, {order_clause}
    """
    if limit is not None:
        query += f" limit {int(limit)}"
    return conn.execute(query, params).fetchall()


def _remote_base_with_prefix(remote_base: str, r2_prefix: str) -> str:
    prefix = r2_prefix.strip("/")
    if not prefix:
        return remote_base.rstrip("/")
    return f"{remote_base.rstrip('/')}/{prefix}"


def _expected_object_key(root: Path, local_path: Path, r2_prefix: str) -> str:
    rel = local_path.resolve().relative_to(root.resolve())
    prefix = r2_prefix.strip("/")
    return f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()


def upload_pending_files(
    db_path: Path,
    *,
    remote_base: str,
    runner: Callable[..., str] = run_rclone,
    limit: int | None = None,
    min_upload_age_seconds: float = 0.0,
    order: str = "object-key",
) -> dict[str, int]:
    """Copy sealed files and SHA-256 sidecars to R2, then mark rows uploaded.

    This uses rclone copy semantics (`copyto` for data, `rcat` for the tiny
    checksum sidecar). It never deletes remote or local objects.
    """
    result = {"uploaded": 0, "skipped": 0, "failed": 0}
    with connect_state(db_path) as conn:
        rows = _upload_candidate_rows(
            conn,
            min_upload_age_seconds=min_upload_age_seconds,
            order=order,
            limit=limit,
        )
        for row in rows:
            local_path = Path(row["local_path"])
            if not local_path.exists():
                result["skipped"] += 1
                conn.execute(
                    "update archive_files set status='missing', error=? where local_path=?",
                    ("local file missing during upload", row["local_path"]),
                )
                conn.commit()
                continue
            if _local_file_too_recent(
                local_path, min_age_seconds=min_upload_age_seconds
            ):
                result["skipped"] += 1
                if not (row["status"] == "uploaded" and row["error"] is not None):
                    _set_error(conn, row["local_path"], None)
                conn.commit()
                continue
            remote_path = remote_path_for(remote_base, row["object_key"])
            try:
                runner(["copyto", "--s3-no-check-bucket", str(local_path), remote_path])
                runner(
                    ["rcat", "--s3-no-check-bucket", remote_path + ".sha256"],
                    input_text=_sidecar_text(row["sha256"], row["local_path"]),
                )
                now = time.time()
                conn.execute(
                    """
                    update archive_files
                    set status='uploaded', uploaded_at=?, error=null
                    where local_path=?
                    """,
                    (now, row["local_path"]),
                )
                conn.commit()
                result["uploaded"] += 1
            except (
                Exception
            ) as exc:  # noqa: BLE001 - keep batch moving and record row error
                result["failed"] += 1
                _set_error(conn, row["local_path"], str(exc))
                conn.commit()
        conn.commit()
    return result


def upload_pending_files_batch(
    db_path: Path,
    *,
    root: Path,
    remote_base: str,
    r2_prefix: str,
    runner: Callable[..., str] = run_rclone,
    limit: int | None = None,
    min_upload_age_seconds: float = 0.0,
    order: str = "object-key",
    sidecar_work_dir: Path | None = None,
    transfers: int = 16,
    checkers: int = 32,
) -> dict[str, int]:
    """Upload a batch of payloads and SHA-256 sidecars with parallel rclone copy.

    The per-file uploader is robust but slow because it starts rclone once per
    payload and once per sidecar. This batch uploader keeps the same lifecycle DB
    semantics but uses two bounded rclone copy calls: one payload copy with
    ``--files-from`` and one copy of a temporary sidecar mirror. It never deletes
    remote or local archive objects.
    """
    result = {"uploaded": 0, "skipped": 0, "failed": 0}
    root = root.resolve()
    owned_tmp: tempfile.TemporaryDirectory[str] | None = None
    if sidecar_work_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="pacifica-r2-batch-upload-")
        work_dir = Path(owned_tmp.name)
    else:
        work_dir = sidecar_work_dir
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    sidecar_root = work_dir / "sidecars"
    payload_list = work_dir / "payload_files.txt"
    try:
        selected: list[tuple[sqlite3.Row, Path, Path]] = []
        with connect_state(db_path) as conn:
            rows = _upload_candidate_rows(
                conn,
                min_upload_age_seconds=min_upload_age_seconds,
                order=order,
                limit=limit,
            )
            for row in rows:
                local_path = Path(row["local_path"])
                if not local_path.exists():
                    result["skipped"] += 1
                    conn.execute(
                        "update archive_files set status='missing', error=? where local_path=?",
                        ("local file missing during batch upload", row["local_path"]),
                    )
                    conn.commit()
                    continue
                if _local_file_too_recent(
                    local_path, min_age_seconds=min_upload_age_seconds
                ):
                    result["skipped"] += 1
                    if not (row["status"] == "uploaded" and row["error"] is not None):
                        _set_error(conn, row["local_path"], None)
                    conn.commit()
                    continue
                try:
                    relative_path = local_path.resolve().relative_to(root)
                    expected_key = _expected_object_key(root, local_path, r2_prefix)
                except ValueError:
                    result["failed"] += 1
                    _set_error(conn, row["local_path"], "local path outside batch root")
                    conn.commit()
                    continue
                if row["object_key"] != expected_key:
                    result["failed"] += 1
                    _set_error(
                        conn,
                        row["local_path"],
                        f"object key mismatch expected={expected_key} saved={row['object_key']}",
                    )
                    conn.commit()
                    continue
                selected.append((row, local_path, relative_path))

            if not selected:
                conn.commit()
                return result

            sidecar_root.mkdir(parents=True, exist_ok=True)
            payload_list.write_text(
                "".join(
                    f"{relative_path.as_posix()}\n" for _, _, relative_path in selected
                ),
                encoding="utf-8",
            )
            for row, local_path, relative_path in selected:
                sidecar_path = sidecar_root / f"{relative_path.as_posix()}.sha256"
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                sidecar_path.write_text(
                    _sidecar_text(row["sha256"], row["local_path"]),
                    encoding="utf-8",
                )

            remote_dest = _remote_base_with_prefix(remote_base, r2_prefix)
            try:
                runner(
                    [
                        "copy",
                        "--s3-no-check-bucket",
                        "--files-from",
                        str(payload_list),
                        "--transfers",
                        str(int(transfers)),
                        "--checkers",
                        str(int(checkers)),
                        str(root),
                        remote_dest,
                    ]
                )
                runner(
                    [
                        "copy",
                        "--s3-no-check-bucket",
                        "--transfers",
                        str(int(transfers)),
                        "--checkers",
                        str(int(checkers)),
                        str(sidecar_root),
                        remote_dest,
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - record row errors, keep DB safe
                message = str(exc)
                for row, _, _ in selected:
                    result["failed"] += 1
                    _set_error(conn, row["local_path"], message)
                conn.commit()
                return result

            now = time.time()
            for row, _, _ in selected:
                conn.execute(
                    """
                    update archive_files
                    set status='uploaded', uploaded_at=?, error=null
                    where local_path=?
                    """,
                    (now, row["local_path"]),
                )
            result["uploaded"] += len(selected)
            conn.commit()
            return result
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()


def _remote_size_bytes(remote_path: str, *, runner: Callable[..., str]) -> int:
    raw = runner(["size", "--json", remote_path])
    payload = json.loads(raw)
    if int(payload.get("count", 0)) != 1:
        raise ValueError(f"remote object count is {payload.get('count')}, expected 1")
    return int(payload.get("bytes", 0))


def verify_uploaded_files(
    db_path: Path,
    *,
    remote_base: str,
    runner: Callable[..., str] = run_rclone,
    limit: int | None = None,
    min_verify_age_seconds: float = 0.0,
) -> dict[str, int]:
    """Verify uploaded R2 objects by remote size and SHA-256 sidecar.

    Cloudflare R2/S3 ETags are not reliable full-file hashes for multipart
    uploads, so this verifier checks object byte size and a sibling `.sha256`
    sidecar written by `upload_pending_files`. The sidecar is an audit guard and
    must match the local manifest checksum before the local file can be pruned.
    """
    result = {"verified": 0, "failed": 0, "skipped": 0}
    with connect_state(db_path) as conn:
        query = """
            select * from archive_files
            where status='uploaded' and error is null
            order by object_key
        """
        if limit is not None:
            query += f" limit {int(limit)}"
        rows = conn.execute(query).fetchall()
        for row in rows:
            local_path = Path(row["local_path"])
            if not local_path.exists():
                result["skipped"] += 1
                _set_error(
                    conn, row["local_path"], "local file missing during verification"
                )
                conn.commit()
                continue
            if _local_file_too_recent(
                local_path, min_age_seconds=min_verify_age_seconds
            ):
                result["skipped"] += 1
                _set_error(conn, row["local_path"], None)
                conn.commit()
                continue
            remote_path = remote_path_for(remote_base, row["object_key"])
            try:
                size = _remote_size_bytes(remote_path, runner=runner)
                if size != int(row["size_bytes"]):
                    raise ValueError(
                        f"size mismatch remote={size} local={row['size_bytes']}"
                    )
                sidecar = runner(["cat", remote_path + ".sha256"]).strip().split()
                if not sidecar or sidecar[0] != row["sha256"]:
                    found = sidecar[0] if sidecar else "<empty>"
                    raise ValueError(
                        f"sha256 sidecar mismatch remote={found} local={row['sha256']}"
                    )
                now = time.time()
                conn.execute(
                    """
                    update archive_files
                    set status='verified', uploaded_at=coalesce(uploaded_at, ?),
                        remote_verified_at=?, error=null
                    where local_path=?
                    """,
                    (now, now, row["local_path"]),
                )
                conn.commit()
                result["verified"] += 1
            except (
                Exception
            ) as exc:  # noqa: BLE001 - keep batch moving and record row error
                result["failed"] += 1
                _set_error(conn, row["local_path"], str(exc))
                conn.commit()
        conn.commit()
    return result


def upload_then_verify(
    db_path: Path,
    *,
    remote_base: str,
    runner: Callable[..., str] = run_rclone,
    upload_limit: int | None = None,
    verify_limit: int | None = None,
    min_upload_age_seconds: float = 0.0,
    upload_order: str = "object-key",
) -> dict[str, dict[str, int]]:
    """Run upload and verification with independently bounded batch sizes."""
    upload_result = upload_pending_files(
        db_path,
        remote_base=remote_base,
        runner=runner,
        limit=upload_limit,
        min_upload_age_seconds=min_upload_age_seconds,
        order=upload_order,
    )
    verify_result = verify_uploaded_files(
        db_path,
        remote_base=remote_base,
        runner=runner,
        limit=verify_limit,
        min_verify_age_seconds=min_upload_age_seconds,
    )
    return {"upload": upload_result, "verify": verify_result}


def reset_mismatch_errors_to_sealed(
    db_path: Path,
    *,
    min_age_seconds: float = 0.0,
    limit: int | None = None,
    dry_run: bool = True,
) -> dict[str, int]:
    """Reset stable historical uploaded mismatch rows to sealed for re-upload.

    This is intentionally narrow and non-destructive: it only targets rows that
    are already marked uploaded with size/hash verification mismatch errors. It
    never touches R2 objects and only clears the DB error when the local file
    exists and is old enough to be considered stable.
    """
    result = {"reset": 0, "skipped_recent": 0, "skipped_missing": 0}
    with connect_state(db_path) as conn:
        query = """
            select local_path from archive_files
            where status='uploaded'
              and (
                error like 'size mismatch%'
                or error like 'sha256 sidecar mismatch%'
              )
            order by object_key
        """
        if limit is not None:
            query += f" limit {int(limit)}"
        rows = conn.execute(query).fetchall()
        for row in rows:
            local_path = Path(row["local_path"])
            if not local_path.exists():
                result["skipped_missing"] += 1
                continue
            if _local_file_too_recent(local_path, min_age_seconds=min_age_seconds):
                result["skipped_recent"] += 1
                continue
            result["reset"] += 1
            if not dry_run:
                conn.execute(
                    """
                    update archive_files
                    set status='sealed', error=null, uploaded_at=null,
                        remote_verified_at=null
                    where local_path=?
                    """,
                    (row["local_path"],),
                )
        conn.commit()
    return result


def prune_verified_files(
    db_path: Path, *, older_than_epoch: float, dry_run: bool = True
) -> list[Path]:
    """Delete only local files marked remote-verified before ``older_than_epoch``."""
    deleted: list[Path] = []
    now = time.time()
    with connect_state(db_path) as conn:
        rows = conn.execute(
            """
            select local_path from archive_files
            where status='verified'
              and remote_verified_at is not null
              and remote_verified_at < ?
            order by remote_verified_at, local_path
            """,
            (older_than_epoch,),
        ).fetchall()
        for row in rows:
            path = Path(row["local_path"])
            if path.exists():
                if not dry_run:
                    path.unlink()
                deleted.append(path)
            if not dry_run:
                conn.execute(
                    "update archive_files set status='pruned', pruned_at=? where local_path=?",
                    (now, str(path)),
                )
        conn.commit()
    return deleted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/pacifica_full_fidelity")
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path("data/pacifica_full_fidelity_storage.sqlite"),
    )
    parser.add_argument("--r2-prefix", default="raw/pacifica/full_fidelity")
    parser.add_argument(
        "--remote-base",
        default="r2:pacifica-trading-data",
        help="rclone remote/bucket base, e.g. r2:bucket",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional fallback max rows for upload/verify batches",
    )
    parser.add_argument(
        "--upload-limit",
        type=int,
        default=None,
        help="optional max rows for upload batches; overrides --limit for upload steps",
    )
    parser.add_argument(
        "--verify-limit",
        type=int,
        default=None,
        help="optional max rows for verify batches; overrides --limit for verify steps",
    )
    parser.add_argument(
        "--min-upload-age-seconds",
        type=float,
        default=0.0,
        help="skip local payloads modified more recently than this many seconds before upload",
    )
    parser.add_argument(
        "--upload-order",
        choices=("object-key", "newest-first", "oldest-first"),
        default="object-key",
        help="ordering for upload candidates; use newest-first as a freshness lane during backlog catch-up",
    )
    parser.add_argument(
        "--transfers",
        type=int,
        default=16,
        help="parallel rclone transfers for upload-batch payload/sidecar copies",
    )
    parser.add_argument(
        "--checkers",
        type=int,
        default=32,
        help="parallel rclone checkers for upload-batch payload/sidecar copies",
    )
    parser.add_argument(
        "--sidecar-work-dir",
        type=Path,
        default=None,
        help="optional work directory for upload-batch generated .sha256 sidecars",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan sealed local archive files into state DB")
    scan.add_argument(
        "--skip-current-hour",
        action="store_true",
        help="skip current UTC hour partitions that the live collector may still be appending",
    )
    scan.add_argument(
        "--recent-hours",
        type=int,
        default=None,
        help="only scan bounded recent UTC hour partitions instead of the full archive tree",
    )
    manifest = sub.add_parser("manifest", help="write JSONL manifest from state DB")
    manifest.add_argument("--out", type=Path, required=True)
    prune = sub.add_parser(
        "prune", help="prune verified local files older than retention"
    )
    prune.add_argument("--retention-days", type=float, default=3.0)
    prune.add_argument(
        "--execute",
        action="store_true",
        help="actually delete files; default is dry-run",
    )
    uploaded = sub.add_parser(
        "mark-uploaded", help="mark one local file uploaded but not verified"
    )
    uploaded.add_argument("--local-path", type=Path, required=True)
    verified = sub.add_parser(
        "mark-verified", help="mark one local file remote-verified"
    )
    verified.add_argument("--local-path", type=Path, required=True)
    sub.add_parser(
        "upload",
        help="upload sealed files plus .sha256 sidecars via rclone copy semantics",
    )
    sub.add_parser(
        "upload-batch",
        help="upload a bounded batch using parallel rclone copy and generated .sha256 sidecars",
    )
    sub.add_parser(
        "verify", help="verify remote size and .sha256 sidecar, then mark rows verified"
    )
    sub.add_parser(
        "upload-verify", help="run upload and then verification in one batch"
    )
    repair = sub.add_parser(
        "reset-mismatch-errors",
        help="reset stable uploaded size/hash mismatch rows to sealed for non-destructive reupload",
    )
    repair.add_argument(
        "--execute",
        action="store_true",
        help="actually reset rows; default is dry-run",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "scan":
        rows = scan_archive_files(
            args.root,
            args.state_db,
            r2_prefix=args.r2_prefix,
            skip_current_hour=args.skip_current_hour,
            recent_hours=args.recent_hours,
        )
        print(
            json.dumps(
                {"scanned": len(rows), "state_db": str(args.state_db)}, sort_keys=True
            )
        )
    elif args.command == "manifest":
        path = write_manifest(args.state_db, args.out)
        print(json.dumps({"manifest": str(path)}, sort_keys=True))
    elif args.command == "prune":
        cutoff = time.time() - args.retention_days * 24 * 3600
        deleted = prune_verified_files(
            args.state_db, older_than_epoch=cutoff, dry_run=not args.execute
        )
        print(
            json.dumps(
                {
                    "dry_run": not args.execute,
                    "candidates": len(deleted),
                    "paths": [str(p) for p in deleted],
                },
                sort_keys=True,
            )
        )
    elif args.command == "mark-uploaded":
        mark_uploaded(args.state_db, args.local_path, verified=False)
        print(
            json.dumps(
                {"local_path": str(args.local_path), "status": "uploaded"},
                sort_keys=True,
            )
        )
    elif args.command == "mark-verified":
        mark_uploaded(args.state_db, args.local_path, verified=True)
        print(
            json.dumps(
                {"local_path": str(args.local_path), "status": "verified"},
                sort_keys=True,
            )
        )
    elif args.command == "upload":
        result = upload_pending_files(
            args.state_db,
            remote_base=args.remote_base,
            limit=args.upload_limit if args.upload_limit is not None else args.limit,
            min_upload_age_seconds=args.min_upload_age_seconds,
            order=args.upload_order,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "upload-batch":
        result = upload_pending_files_batch(
            args.state_db,
            root=args.root,
            remote_base=args.remote_base,
            r2_prefix=args.r2_prefix,
            limit=args.upload_limit if args.upload_limit is not None else args.limit,
            min_upload_age_seconds=args.min_upload_age_seconds,
            order=args.upload_order,
            sidecar_work_dir=args.sidecar_work_dir,
            transfers=args.transfers,
            checkers=args.checkers,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "verify":
        result = verify_uploaded_files(
            args.state_db,
            remote_base=args.remote_base,
            limit=args.verify_limit if args.verify_limit is not None else args.limit,
            min_verify_age_seconds=args.min_upload_age_seconds,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "reset-mismatch-errors":
        result = reset_mismatch_errors_to_sealed(
            args.state_db,
            min_age_seconds=args.min_upload_age_seconds,
            limit=args.limit,
            dry_run=not args.execute,
        )
        print(
            json.dumps(
                {"dry_run": not args.execute, **result},
                sort_keys=True,
            )
        )
    elif args.command == "upload-verify":
        result = upload_then_verify(
            args.state_db,
            remote_base=args.remote_base,
            upload_limit=(
                args.upload_limit if args.upload_limit is not None else args.limit
            ),
            verify_limit=(
                args.verify_limit if args.verify_limit is not None else args.limit
            ),
            min_upload_age_seconds=args.min_upload_age_seconds,
            upload_order=args.upload_order,
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

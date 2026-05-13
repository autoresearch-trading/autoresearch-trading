import sqlite3
import time
from pathlib import Path

from scripts.check_pacifica_full_fidelity_health import (
    db_counts,
    db_error_counts,
    db_recent_activity,
)
from scripts.pacifica_full_fidelity_storage import connect_state


def _insert_archive_row(
    db: Path,
    *,
    local_path: str,
    status: str,
    size_bytes: int,
    now: float,
    uploaded_at: float | None = None,
    remote_verified_at: float | None = None,
    pruned_at: float | None = None,
    error: str | None = None,
) -> None:
    with connect_state(db) as conn:
        conn.execute(
            """
            insert into archive_files(
                local_path, object_key, size_bytes, modified_at, sha256, status,
                first_seen_at, last_seen_at, uploaded_at, remote_verified_at,
                pruned_at, error
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                local_path,
                f"raw/{local_path}",
                size_bytes,
                now - 7200,
                "a" * 64,
                status,
                now - 3600,
                now - 60,
                uploaded_at,
                remote_verified_at,
                pruned_at,
                error,
            ),
        )
        conn.commit()


def test_db_error_counts_groups_rows_by_status_and_error(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    now = time.time()
    _insert_archive_row(
        db,
        local_path="a.jsonl.gz",
        status="uploaded",
        size_bytes=100,
        now=now,
        uploaded_at=now - 120,
        error="size mismatch remote=1 local=2",
    )
    _insert_archive_row(
        db,
        local_path="b.jsonl.gz",
        status="sealed",
        size_bytes=200,
        now=now,
        error="local file missing during upload",
    )
    _insert_archive_row(
        db,
        local_path="c.jsonl.gz",
        status="verified",
        size_bytes=300,
        now=now,
        remote_verified_at=now - 60,
    )

    assert db_counts(db)["verified"] == {"files": 1, "bytes": 300}

    errors = db_error_counts(db)

    assert errors["rows_with_errors"] == 2
    assert errors["bytes_with_errors"] == 300
    assert errors["by_status"] == {
        "sealed": {"files": 1, "bytes": 200},
        "uploaded": {"files": 1, "bytes": 100},
    }
    assert errors["top_errors"] == [
        {
            "status": "sealed",
            "error": "local file missing during upload",
            "files": 1,
            "bytes": 200,
        },
        {
            "status": "uploaded",
            "error": "size mismatch remote=1 local=2",
            "files": 1,
            "bytes": 100,
        },
    ]


def test_db_recent_activity_counts_upload_verify_and_prune_throughput(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.sqlite"
    now = 1_800_000_000.0
    _insert_archive_row(
        db,
        local_path="uploaded-recent.jsonl.gz",
        status="uploaded",
        size_bytes=111,
        now=now,
        uploaded_at=now - 300,
    )
    _insert_archive_row(
        db,
        local_path="verified-recent.jsonl.gz",
        status="verified",
        size_bytes=222,
        now=now,
        uploaded_at=now - 400,
        remote_verified_at=now - 200,
    )
    _insert_archive_row(
        db,
        local_path="pruned-recent.jsonl.gz",
        status="pruned",
        size_bytes=333,
        now=now,
        uploaded_at=now - 500,
        remote_verified_at=now - 400,
        pruned_at=now - 100,
    )
    _insert_archive_row(
        db,
        local_path="old-upload.jsonl.gz",
        status="uploaded",
        size_bytes=444,
        now=now,
        uploaded_at=now - 7200,
    )

    activity = db_recent_activity(db, window_seconds=3600, now_epoch=now)

    assert activity == {
        "window_seconds": 3600,
        "first_seen": {"files": 4, "bytes": 1110},
        "last_seen": {"files": 4, "bytes": 1110},
        "uploaded": {"files": 3, "bytes": 666},
        "verified": {"files": 2, "bytes": 555},
        "pruned": {"files": 1, "bytes": 333},
    }

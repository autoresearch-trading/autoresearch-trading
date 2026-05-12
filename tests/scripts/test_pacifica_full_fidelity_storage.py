import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.pacifica_full_fidelity_storage import (
    connect_state,
    iter_archive_files,
    mark_uploaded,
    object_key_for,
    prune_verified_files,
    repair_uploaded_sidecars_batch,
    scan_archive_files,
    upload_pending_files,
    upload_pending_files_batch,
    upload_then_verify,
    verify_uploaded_files,
)


def _write(path: Path, content: bytes = b"payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_scan_archive_files_records_deterministic_r2_keys_and_checksums(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-01"
        / "hour=13"
        / "run-1.jsonl.gz",
        b"one\ntwo\n",
    )
    _write(raw_file.with_suffix(raw_file.suffix + ".tmp"), b"do not upload")

    rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    assert len(rows) == 1
    row = rows[0]
    assert row["local_path"] == str(raw_file)
    assert row["object_key"] == (
        "raw/pacifica/full_fidelity/channel=trades/symbol=BTC/"
        "date=2026-05-01/hour=13/run-1.jsonl.gz"
    )
    assert row["size_bytes"] == len(b"one\ntwo\n")
    assert len(row["sha256"]) == 64
    assert row["status"] == "sealed"

    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select object_key, status, size_bytes, sha256 from archive_files"
        ).fetchone()
    assert saved == (row["object_key"], "sealed", row["size_bytes"], row["sha256"])


def test_object_key_for_preserves_partition_path_under_prefix(tmp_path):
    root = tmp_path / "raw"
    file_path = (
        root / "channel=book" / "symbol=ETH" / "date=2026-05-01" / "run.jsonl.gz"
    )

    assert object_key_for(root, file_path, "raw/pacifica/") == (
        "raw/pacifica/channel=book/symbol=ETH/date=2026-05-01/run.jsonl.gz"
    )


def test_iter_archive_files_skips_active_tmp_and_state_files(tmp_path):
    root = tmp_path / "raw"
    keep = _write(
        root / "channel=book" / "symbol=BTC" / "date=2026-05-01" / "run.jsonl.gz"
    )
    _write(
        root / "channel=book" / "symbol=BTC" / "date=2026-05-01" / "run.jsonl.gz.tmp"
    )
    _write(root / "state.sqlite")

    assert list(iter_archive_files(root)) == [keep]


def test_iter_archive_files_can_skip_current_hour_partition(tmp_path):
    root = tmp_path / "raw"
    active_hour = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=13"
        / "run-live.jsonl.gz"
    )
    closed_hour = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=12"
        / "run-live.jsonl.gz"
    )
    now = datetime(2026, 5, 2, 13, 30, tzinfo=timezone.utc)

    assert active_hour in list(iter_archive_files(root))
    assert list(iter_archive_files(root, skip_current_hour=True, now=now)) == [
        closed_hour
    ]


def test_prune_verified_files_deletes_only_verified_remote_durable_files(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    old_verified = _write(
        root / "channel=trades" / "symbol=BTC" / "date=2026-05-01" / "old.jsonl.gz"
    )
    new_verified = _write(
        root / "channel=trades" / "symbol=BTC" / "date=2026-05-01" / "new.jsonl.gz"
    )
    sealed = _write(
        root / "channel=trades" / "symbol=BTC" / "date=2026-05-01" / "sealed.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica")

    cutoff = time.time() - 8 * 24 * 3600
    old_mtime = time.time() - 10 * 24 * 3600
    os.utime(old_verified, (old_mtime, old_mtime))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "update archive_files set status='verified', remote_verified_at=? where local_path=?",
            (old_mtime, str(old_verified)),
        )
        conn.execute(
            "update archive_files set status='verified', remote_verified_at=? where local_path=?",
            (time.time(), str(new_verified)),
        )
        conn.commit()

    deleted = prune_verified_files(db, older_than_epoch=cutoff, dry_run=False)

    assert deleted == [old_verified]
    assert not old_verified.exists()
    assert new_verified.exists()
    assert sealed.exists()
    with sqlite3.connect(db) as conn:
        statuses = dict(conn.execute("select local_path, status from archive_files"))
    assert statuses[str(old_verified)] == "pruned"
    assert statuses[str(new_verified)] == "verified"
    assert statuses[str(sealed)] == "sealed"


class FakeRcloneRunner:
    def __init__(self):
        self.commands = []
        self.remote_sizes = {}
        self.remote_text = {}

    def __call__(self, args, *, input_text=None):
        self.commands.append((tuple(args), input_text))
        if args[0] == "copyto":
            src, dst = args[2], args[3]
            self.remote_sizes[dst] = Path(src).stat().st_size
            return ""
        if args[0] == "rcat":
            dst = args[2]
            self.remote_text[dst] = input_text or ""
            self.remote_sizes[dst] = len((input_text or "").encode("utf-8"))
            return ""
        if args[0] == "size" and args[1] == "--json":
            size = self.remote_sizes.get(args[2], 0)
            count = 1 if args[2] in self.remote_sizes else 0
            return f'{{"bytes":{size},"count":{count}}}'
        if args[0] == "cat":
            return self.remote_text[args[1]]
        raise AssertionError(f"unexpected rclone command: {args}")


def test_upload_pending_files_ignores_recently_modified_sealed_chunks(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    recent_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=18"
        / "run-live.jsonl.gz",
        b"still-changing",
    )
    now = time.time()
    os.utime(recent_file, (now, now))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db,
        remote_base="r2:pacifica-trading-data",
        runner=runner,
        min_upload_age_seconds=3600,
    )

    assert result == {"uploaded": 0, "skipped": 0, "failed": 0}
    assert runner.commands == []
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(recent_file),),
        ).fetchone()
    assert saved == ("sealed", None)


def test_upload_pending_files_preserves_recent_uploaded_error_until_reupload(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    recent_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=18"
        / "run-live.jsonl.gz",
        b"still-changing",
    )
    now = time.time()
    os.utime(recent_file, (now, now))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            update archive_files
            set status='uploaded', uploaded_at=?, error='size mismatch remote=3 local=14'
            where local_path=?
            """,
            (now - 7200, str(recent_file)),
        )
        conn.commit()
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db,
        remote_base="r2:pacifica-trading-data",
        runner=runner,
        min_upload_age_seconds=3600,
    )

    assert result == {"uploaded": 0, "skipped": 1, "failed": 0}
    assert runner.commands == []
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(recent_file),),
        ).fetchone()
    assert saved == ("uploaded", "size mismatch remote=3 local=14")


def test_upload_pending_files_copies_data_and_sha256_sidecar_then_marks_uploaded(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-01"
        / "hour=13"
        / "run.jsonl.gz",
        b"abc",
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db, remote_base="r2:pacifica-trading-data", runner=runner
    )

    assert result == {"uploaded": 1, "skipped": 0, "failed": 0}
    object_path = "r2:pacifica-trading-data/raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=13/run.jsonl.gz"
    assert ("copyto", "--s3-no-check-bucket", str(raw_file), object_path) in [
        cmd for cmd, _ in runner.commands
    ]
    assert ("rcat", "--s3-no-check-bucket", object_path + ".sha256") in [
        cmd for cmd, _ in runner.commands
    ]
    sidecar = runner.remote_text[object_path + ".sha256"]
    assert sidecar.endswith("  run.jsonl.gz\n")
    assert len(sidecar.split()[0]) == 64
    with sqlite3.connect(db) as conn:
        status = conn.execute(
            "select status from archive_files where local_path=?", (str(raw_file),)
        ).fetchone()[0]
    assert status == "uploaded"


def test_upload_pending_files_reuploads_uploaded_rows_with_verification_errors(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=12"
        / "run-live.jsonl.gz",
        b"complete-hour",
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            update archive_files
            set status='uploaded', uploaded_at=?, error='size mismatch remote=3 local=13'
            where local_path=?
            """,
            (time.time(), str(raw_file)),
        )
        conn.commit()
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db, remote_base="r2:pacifica-trading-data", runner=runner
    )

    assert result == {"uploaded": 1, "skipped": 0, "failed": 0}
    copy_commands = [cmd for cmd, _ in runner.commands if cmd[0] == "copyto"]
    assert copy_commands == [
        (
            "copyto",
            "--s3-no-check-bucket",
            str(raw_file),
            "r2:pacifica-trading-data/raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-02/hour=12/run-live.jsonl.gz",
        )
    ]
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(raw_file),),
        ).fetchone()
    assert saved == ("uploaded", None)


def test_scan_archive_files_resets_mutated_uploaded_rows_to_sealed(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    path = _write(
        root
        / "channel=book"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=17"
        / "chunk.jsonl.gz",
        b"before",
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    mark_uploaded(db, path)
    path.write_bytes(b"after-append")

    rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    assert len(rows) == 1
    assert rows[0]["status"] == "sealed"
    assert rows[0]["error"] is None
    assert rows[0]["size_bytes"] == len(b"after-append")


def test_scan_archive_files_preserves_unchanged_uploaded_error_for_reupload(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    path = _write(
        root
        / "channel=book"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=17"
        / "chunk.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    mark_uploaded(db, path)
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set error='size mismatch remote=1 local=2' where local_path=?",
            (str(path),),
        )
        conn.commit()

    rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    assert len(rows) == 1
    assert rows[0]["status"] == "uploaded"
    assert rows[0]["error"] == "size mismatch remote=1 local=2"


def test_upload_pending_files_prioritizes_errored_uploaded_rows_before_new_sealed_rows(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    sealed_file = _write(
        root
        / "channel=book"
        / "symbol=AAA"
        / "date=2026-05-02"
        / "hour=17"
        / "sealed.jsonl.gz"
    )
    errored_uploaded_file = _write(
        root
        / "channel=book"
        / "symbol=ZZZ"
        / "date=2026-05-02"
        / "hour=17"
        / "errored.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    mark_uploaded(db, errored_uploaded_file)
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set error='size mismatch remote=1 local=2' where local_path=?",
            (str(errored_uploaded_file),),
        )
        conn.commit()

    copied: list[str] = []

    def runner(args, *, input_text=None):
        if args[0] == "copyto":
            copied.append(Path(args[2]).name)
        return ""

    result = upload_pending_files(db, remote_base="r2:bucket", runner=runner, limit=1)

    assert result == {"uploaded": 1, "skipped": 0, "failed": 0}
    assert copied == [errored_uploaded_file.name]
    with connect_state(db) as conn:
        rows = conn.execute(
            "select local_path,status,error from archive_files order by object_key"
        ).fetchall()
    by_name = {
        Path(row["local_path"]).name: (row["status"], row["error"]) for row in rows
    }
    assert by_name[errored_uploaded_file.name] == ("uploaded", None)
    assert by_name[sealed_file.name] == ("sealed", None)


def test_upload_pending_files_can_prioritize_newest_sealed_chunks_for_freshness_lane(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    old_file = _write(
        root
        / "channel=bbo"
        / "symbol=ZZZ"
        / "date=2026-05-01"
        / "hour=01"
        / "old.jsonl.gz",
        b"old",
    )
    new_file = _write(
        root
        / "channel=bbo"
        / "symbol=AAA"
        / "date=2026-05-03"
        / "hour=22"
        / "new.jsonl.gz",
        b"new",
    )
    older = time.time() - 3 * 24 * 3600
    newer = time.time() - 3 * 3600
    os.utime(old_file, (older, older))
    os.utime(new_file, (newer, newer))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db,
        remote_base="r2:bucket",
        runner=runner,
        limit=1,
        order="newest-first",
        min_upload_age_seconds=7200,
    )

    assert result == {"uploaded": 1, "skipped": 0, "failed": 0}
    copy_commands = [cmd for cmd, _ in runner.commands if cmd[0] == "copyto"]
    assert copy_commands[0][2] == str(new_file)
    with connect_state(db) as conn:
        statuses = dict(conn.execute("select local_path, status from archive_files"))
    assert statuses[str(new_file)] == "uploaded"
    assert statuses[str(old_file)] == "sealed"


def test_scan_archive_files_reuses_checksum_for_unchanged_files(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    path = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-03"
        / "hour=22"
        / "chunk.jsonl.gz",
        b"stable",
    )
    first_rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    import scripts.pacifica_full_fidelity_storage as storage

    def fail_if_rehashed(_path):
        raise AssertionError("unchanged files should not be rehashed")

    monkeypatch.setattr(storage, "sha256_file", fail_if_rehashed)
    second_rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    assert second_rows[0]["sha256"] == first_rows[0]["sha256"]
    assert second_rows[0]["size_bytes"] == len(b"stable")
    assert second_rows[0]["status"] == "sealed"


def test_scan_archive_files_can_scan_only_recent_hour_partitions(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    now = datetime(2026, 5, 4, 12, 30, tzinfo=timezone.utc)
    old_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-03"
        / "hour=06"
        / "old.jsonl.gz",
        b"old",
    )
    recent_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=10"
        / "recent.jsonl.gz",
        b"recent",
    )
    current_hour_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=12"
        / "current.jsonl.gz",
        b"current",
    )
    for path in (old_file, recent_file, current_hour_file):
        mtime = now.timestamp() - 3 * 3600
        os.utime(path, (mtime, mtime))

    rows = scan_archive_files(
        root,
        db,
        r2_prefix="raw/pacifica/full_fidelity",
        skip_current_hour=True,
        recent_hours=4,
        now=now,
    )

    assert [Path(row["local_path"]).name for row in rows] == ["recent.jsonl.gz"]
    with connect_state(db) as conn:
        saved_names = [
            Path(row[0]).name
            for row in conn.execute(
                "select local_path from archive_files order by local_path"
            )
        ]
    assert saved_names == ["recent.jsonl.gz"]


def test_upload_pending_files_does_not_spend_limit_on_too_recent_sealed_rows(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    old_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=06"
        / "eligible.jsonl.gz",
        b"eligible",
    )
    too_recent_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=11"
        / "too-recent.jsonl.gz",
        b"too-recent",
    )
    old_mtime = time.time() - 4 * 3600
    new_mtime = time.time()
    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(too_recent_file, (new_mtime, new_mtime))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    runner = FakeRcloneRunner()

    result = upload_pending_files(
        db,
        remote_base="r2:bucket",
        runner=runner,
        limit=1,
        order="newest-first",
        min_upload_age_seconds=7200,
    )

    assert result == {"uploaded": 1, "skipped": 0, "failed": 0}
    copy_commands = [cmd for cmd, _ in runner.commands if cmd[0] == "copyto"]
    assert copy_commands == [
        (
            "copyto",
            "--s3-no-check-bucket",
            str(old_file),
            "r2:bucket/raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-04/hour=06/eligible.jsonl.gz",
        )
    ]
    with connect_state(db) as conn:
        statuses = dict(conn.execute("select local_path, status from archive_files"))
    assert statuses[str(old_file)] == "uploaded"
    assert statuses[str(too_recent_file)] == "sealed"


def test_reset_mismatch_errors_to_sealed_repairs_only_stable_uploaded_mismatch_rows(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    stable_mismatch = _write(
        root / "channel=bbo" / "symbol=BTC" / "date=2026-05-02" / "stable.jsonl.gz"
    )
    recent_mismatch = _write(
        root / "channel=bbo" / "symbol=ETH" / "date=2026-05-02" / "recent.jsonl.gz"
    )
    stable_non_mismatch = _write(
        root / "channel=bbo" / "symbol=SOL" / "date=2026-05-02" / "other.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    now = time.time()
    old = now - 7200
    os.utime(stable_mismatch, (old, old))
    os.utime(stable_non_mismatch, (old, old))
    os.utime(recent_mismatch, (now, now))
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', error='size mismatch remote=1 local=2' where local_path=?",
            (str(stable_mismatch),),
        )
        conn.execute(
            "update archive_files set status='uploaded', error='sha256 sidecar mismatch remote=a local=b' where local_path=?",
            (str(recent_mismatch),),
        )
        conn.execute(
            "update archive_files set status='uploaded', error='network timeout' where local_path=?",
            (str(stable_non_mismatch),),
        )
        conn.commit()

    from scripts.pacifica_full_fidelity_storage import reset_mismatch_errors_to_sealed

    result = reset_mismatch_errors_to_sealed(
        db,
        min_age_seconds=3600,
        limit=None,
        dry_run=False,
    )

    assert result == {"reset": 1, "skipped_recent": 1, "skipped_missing": 0}
    with connect_state(db) as conn:
        rows = conn.execute(
            "select local_path,status,error from archive_files order by local_path"
        ).fetchall()
    by_name = {
        Path(row["local_path"]).name: (row["status"], row["error"]) for row in rows
    }
    assert by_name["stable.jsonl.gz"] == ("sealed", None)
    assert by_name["recent.jsonl.gz"] == (
        "uploaded",
        "sha256 sidecar mismatch remote=a local=b",
    )
    assert by_name["other.jsonl.gz"] == ("uploaded", "network timeout")


def test_reset_mismatch_errors_to_sealed_dry_run_does_not_mutate_rows(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    path = _write(
        root / "channel=bbo" / "symbol=BTC" / "date=2026-05-02" / "stable.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    old = time.time() - 7200
    os.utime(path, (old, old))
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', error='size mismatch remote=1 local=2' where local_path=?",
            (str(path),),
        )
        conn.commit()

    from scripts.pacifica_full_fidelity_storage import reset_mismatch_errors_to_sealed

    result = reset_mismatch_errors_to_sealed(
        db,
        min_age_seconds=3600,
        limit=None,
        dry_run=True,
    )

    assert result == {"reset": 1, "skipped_recent": 0, "skipped_missing": 0}
    with connect_state(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?", (str(path),)
        ).fetchone()
    assert tuple(saved) == ("uploaded", "size mismatch remote=1 local=2")


def test_verify_uploaded_files_defers_errored_rows_until_reupload(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    path = _write(
        root
        / "channel=book"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=17"
        / "chunk.jsonl.gz"
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    mark_uploaded(db, path)
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set error='size mismatch remote=1 local=2' where local_path=?",
            (str(path),),
        )
        conn.commit()

    def runner(args, *, input_text=None):
        raise AssertionError(f"unexpected rclone command: {args}")

    result = verify_uploaded_files(db, remote_base="r2:bucket", runner=runner)

    assert result == {"verified": 0, "failed": 0, "skipped": 0}
    with connect_state(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?", (str(path),)
        ).fetchone()
    assert tuple(saved) == ("uploaded", "size mismatch remote=1 local=2")


def test_verify_uploaded_files_skips_recently_modified_uploaded_rows(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    recent_file = _write(
        root
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2026-05-02"
        / "hour=18"
        / "run-live.jsonl.gz",
        b"still-changing",
    )
    now = time.time()
    os.utime(recent_file, (now, now))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', uploaded_at=?, error=null where local_path=?",
            (now, str(recent_file)),
        )
        conn.commit()
    runner = FakeRcloneRunner()

    result = verify_uploaded_files(
        db,
        remote_base="r2:pacifica-trading-data",
        runner=runner,
        min_verify_age_seconds=3600,
    )

    assert result == {"verified": 0, "failed": 0, "skipped": 1}
    assert runner.commands == []
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(recent_file),),
        ).fetchone()
    assert saved == ("uploaded", None)


def test_verify_uploaded_files_checks_remote_size_and_sha256_sidecar_before_marking_verified(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-01"
        / "hour=13"
        / "run.jsonl.gz",
        b"abc",
    )
    rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    row = rows[0]
    object_path = "r2:pacifica-trading-data/" + row["object_key"]
    runner = FakeRcloneRunner()
    runner.remote_sizes[object_path] = row["size_bytes"]
    runner.remote_text[object_path + ".sha256"] = f'{row["sha256"]}  run.jsonl.gz\n'
    with sqlite3.connect(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', uploaded_at=? where local_path=?",
            (time.time(), str(raw_file)),
        )
        conn.commit()

    result = verify_uploaded_files(
        db, remote_base="r2:pacifica-trading-data", runner=runner
    )

    assert result == {"verified": 1, "failed": 0, "skipped": 0}
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, remote_verified_at, error from archive_files where local_path=?",
            (str(raw_file),),
        ).fetchone()
    assert saved[0] == "verified"
    assert saved[1] is not None
    assert saved[2] is None


def test_verify_uploaded_files_refuses_mismatched_remote_size(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-01"
        / "hour=13"
        / "run.jsonl.gz",
        b"abc",
    )
    rows = scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    row = rows[0]
    object_path = "r2:pacifica-trading-data/" + row["object_key"]
    runner = FakeRcloneRunner()
    runner.remote_sizes[object_path] = row["size_bytes"] + 1
    runner.remote_text[object_path + ".sha256"] = f'{row["sha256"]}  run.jsonl.gz\n'
    with sqlite3.connect(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', uploaded_at=? where local_path=?",
            (time.time(), str(raw_file)),
        )
        conn.commit()

    result = verify_uploaded_files(
        db, remote_base="r2:pacifica-trading-data", runner=runner
    )

    assert result == {"verified": 0, "failed": 1, "skipped": 0}
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(raw_file),),
        ).fetchone()
    assert saved[0] == "uploaded"
    assert "size mismatch" in saved[1]


def test_upload_then_verify_uses_separate_upload_and_verify_limits(tmp_path):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    files = []
    for idx in range(3):
        path = _write(
            root
            / "channel=bbo"
            / f"symbol=S{idx}"
            / "date=2026-05-03"
            / "hour=22"
            / f"chunk-{idx}.jsonl.gz",
            f"payload-{idx}".encode(),
        )
        files.append(path)
        old = time.time() - (idx + 3) * 3600
        os.utime(path, (old, old))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    runner = FakeRcloneRunner()

    result = upload_then_verify(
        db,
        remote_base="r2:bucket",
        runner=runner,
        upload_limit=3,
        verify_limit=1,
        min_upload_age_seconds=7200,
        upload_order="newest-first",
    )

    assert result == {
        "upload": {"uploaded": 3, "skipped": 0, "failed": 0},
        "verify": {"verified": 1, "failed": 0, "skipped": 0},
    }
    copy_commands = [cmd for cmd, _ in runner.commands if cmd[0] == "copyto"]
    size_commands = [cmd for cmd, _ in runner.commands if cmd[0] == "size"]
    assert len(copy_commands) == 3
    assert len(size_commands) == 1
    with connect_state(db) as conn:
        statuses = [
            row[0]
            for row in conn.execute(
                "select status from archive_files order by object_key"
            ).fetchall()
        ]
    assert statuses.count("verified") == 1
    assert statuses.count("uploaded") == 2


def test_upload_pending_files_batch_copies_payloads_and_sidecars_with_one_rclone_copy_per_kind(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    files = []
    for symbol in ("BTC", "ETH"):
        path = _write(
            root
            / "channel=book"
            / f"symbol={symbol}"
            / "date=2026-05-04"
            / "hour=10"
            / f"{symbol}.jsonl.gz",
            f"payload-{symbol}".encode(),
        )
        old = time.time() - 4 * 3600
        os.utime(path, (old, old))
        files.append(path)
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    commands = []

    def runner(args, *, input_text=None):
        commands.append(tuple(args))
        assert input_text is None
        return ""

    sidecars = tmp_path / "sidecars"
    result = upload_pending_files_batch(
        db,
        root=root,
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        runner=runner,
        sidecar_work_dir=sidecars,
        limit=2,
        min_upload_age_seconds=7200,
        order="object-key",
        transfers=8,
        checkers=16,
    )

    assert result == {"uploaded": 2, "skipped": 0, "failed": 0}
    assert commands == [
        (
            "copy",
            "--s3-no-check-bucket",
            "--files-from",
            str(sidecars / "payload_files.txt"),
            "--transfers",
            "8",
            "--checkers",
            "16",
            str(root),
            "r2:bucket/raw/pacifica/full_fidelity",
        ),
        (
            "copy",
            "--s3-no-check-bucket",
            "--transfers",
            "8",
            "--checkers",
            "16",
            str(sidecars / "sidecars"),
            "r2:bucket/raw/pacifica/full_fidelity",
        ),
    ]
    listed = (sidecars / "payload_files.txt").read_text().splitlines()
    assert listed == [str(path.relative_to(root)) for path in files]
    for path in files:
        sidecar = (
            sidecars
            / "sidecars"
            / path.relative_to(root).with_name(path.name + ".sha256")
        )
        assert sidecar.exists()
        assert sidecar.read_text().endswith(f"  {path.name}\n")
    with connect_state(db) as conn:
        statuses = dict(conn.execute("select local_path, status from archive_files"))
    assert {statuses[str(path)] for path in files} == {"uploaded"}


def test_repair_uploaded_sidecars_batch_copies_only_sidecars_for_uploaded_rows(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    uploaded_file = _write(
        root
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=20"
        / "uploaded.jsonl.gz",
        b"uploaded-payload",
    )
    sealed_file = _write(
        root
        / "channel=trades"
        / "symbol=ETH"
        / "date=2026-05-04"
        / "hour=20"
        / "sealed.jsonl.gz",
        b"sealed-payload",
    )
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")
    with connect_state(db) as conn:
        conn.execute(
            "update archive_files set status='uploaded', uploaded_at=? where local_path=?",
            (time.time(), str(uploaded_file)),
        )
        conn.commit()

    commands = []

    def runner(args, *, input_text=None):
        commands.append(tuple(args))
        assert input_text is None
        return ""

    work_dir = tmp_path / "repair-sidecars"
    result = repair_uploaded_sidecars_batch(
        db,
        root=root,
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        runner=runner,
        sidecar_work_dir=work_dir,
        limit=10,
        order="newest-first",
        transfers=4,
        checkers=8,
    )

    assert result == {"sidecars_uploaded": 1, "skipped": 0, "failed": 0}
    assert commands == [
        (
            "copy",
            "--s3-no-check-bucket",
            "--transfers",
            "4",
            "--checkers",
            "8",
            str(work_dir / "sidecars"),
            "r2:bucket/raw/pacifica/full_fidelity",
        )
    ]
    uploaded_sidecar = (
        work_dir
        / "sidecars"
        / uploaded_file.relative_to(root).with_name(uploaded_file.name + ".sha256")
    )
    sealed_sidecar = (
        work_dir
        / "sidecars"
        / sealed_file.relative_to(root).with_name(sealed_file.name + ".sha256")
    )
    assert uploaded_sidecar.exists()
    assert uploaded_sidecar.read_text().endswith("  uploaded.jsonl.gz\n")
    assert not sealed_sidecar.exists()
    with connect_state(db) as conn:
        statuses = dict(conn.execute("select local_path, status from archive_files"))
    assert statuses[str(uploaded_file)] == "uploaded"
    assert statuses[str(sealed_file)] == "sealed"


def test_upload_pending_files_batch_marks_candidates_failed_when_batch_copy_fails(
    tmp_path,
):
    root = tmp_path / "raw"
    db = tmp_path / "state.sqlite"
    raw_file = _write(
        root
        / "channel=book"
        / "symbol=BTC"
        / "date=2026-05-04"
        / "hour=10"
        / "BTC.jsonl.gz",
        b"payload",
    )
    old = time.time() - 4 * 3600
    os.utime(raw_file, (old, old))
    scan_archive_files(root, db, r2_prefix="raw/pacifica/full_fidelity")

    def runner(args, *, input_text=None):
        raise RuntimeError("rclone batch failed")

    result = upload_pending_files_batch(
        db,
        root=root,
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        runner=runner,
        sidecar_work_dir=tmp_path / "sidecars",
        limit=1,
        min_upload_age_seconds=7200,
    )

    assert result == {"uploaded": 0, "skipped": 0, "failed": 1}
    with connect_state(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(raw_file),),
        ).fetchone()
    assert tuple(saved) == ("sealed", "rclone batch failed")

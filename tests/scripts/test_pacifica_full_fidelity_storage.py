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
    scan_archive_files,
    upload_pending_files,
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


def test_upload_pending_files_skips_recently_modified_chunks(tmp_path):
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

    assert result == {"uploaded": 0, "skipped": 1, "failed": 0}
    assert runner.commands == []
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            "select status, error from archive_files where local_path=?",
            (str(recent_file),),
        ).fetchone()
    assert saved == ("sealed", None)


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

import gzip
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.check_pacifica_r2_archive_health import (
    analyze_r2_archive_inventory,
    parse_raw_object_key,
    write_r2_archive_health_report,
)


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-02/hour=13/run-a.jsonl.gz",
                "size_bytes": 100,
                "mod_time": "2026-05-02T13:59:00Z",
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-02/hour=13/run-a.jsonl.gz.sha256",
                "size_bytes": 64,
                "mod_time": "2026-05-02T14:00:00Z",
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-02/hour=14/run-active.jsonl.gz",
                "size_bytes": 200,
                "mod_time": "2026-05-02T14:05:00Z",
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=trades/symbol=SOL/date=2026-05-01/hour=23/run-missing-sidecar.jsonl.gz",
                "size_bytes": 300,
                "mod_time": "2026-05-01T23:30:00Z",
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=prices/symbol=BP/date=2026-05-01/run-orphan.jsonl.gz.sha256",
                "size_bytes": 64,
                "mod_time": "2026-05-01T12:00:00Z",
            },
            {
                "key": "cold/pacifica/full_fidelity/manifest.csv",
                "size_bytes": 999,
                "mod_time": "2026-05-02T00:00:00Z",
            },
        ]
    )


def test_parse_raw_object_key_extracts_partitions_with_optional_hour() -> None:
    with_hour = parse_raw_object_key(
        "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-02/hour=13/run.jsonl.gz"
    )
    assert with_hour == {
        "channel": "bbo",
        "symbol": "BTC",
        "date": "2026-05-02",
        "hour": "13",
    }

    without_hour = parse_raw_object_key(
        "raw/pacifica/full_fidelity/channel=prices/symbol=BP/date=2026-05-01/run.jsonl.gz"
    )
    assert without_hour == {
        "channel": "prices",
        "symbol": "BP",
        "date": "2026-05-01",
        "hour": "",
    }


def test_analyze_r2_archive_inventory_finds_pairing_and_active_hour_issues() -> None:
    now = datetime(2026, 5, 2, 14, 30, tzinfo=UTC)

    result = analyze_r2_archive_inventory(_inventory(), now=now)

    assert result["objects_total"] == 5
    assert result["payload_objects"] == 3
    assert result["sidecar_objects"] == 2
    assert result["missing_sidecar_count"] == 2
    assert result["orphan_sidecar_count"] == 1
    assert result["active_hour_object_count"] == 1
    assert result["latest_payload_mod_time"] == "2026-05-02T14:05:00+00:00"

    missing_keys = set(result["missing_sidecars"]["key"])
    assert (
        "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-02/hour=14/run-active.jsonl.gz"
        in missing_keys
    )
    assert (
        "raw/pacifica/full_fidelity/channel=trades/symbol=SOL/date=2026-05-01/hour=23/run-missing-sidecar.jsonl.gz"
        in missing_keys
    )

    orphan_keys = set(result["orphan_sidecars"]["key"])
    assert (
        "raw/pacifica/full_fidelity/channel=prices/symbol=BP/date=2026-05-01/run-orphan.jsonl.gz.sha256"
        in orphan_keys
    )

    active_keys = set(result["active_hour_objects"]["key"])
    assert (
        "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-02/hour=14/run-active.jsonl.gz"
        in active_keys
    )


def test_write_r2_archive_health_report_is_read_only_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.csv"
    _inventory().to_csv(inventory_path, index=False)
    out_dir = tmp_path / "r2-health"
    now = datetime(2026, 5, 2, 14, 30, tzinfo=UTC)

    result = write_r2_archive_health_report(inventory_path, out_dir, now=now)

    assert result["write_or_delete_executed"] is False
    assert result["missing_sidecar_count"] == 2
    assert result["orphan_sidecar_count"] == 1
    assert (out_dir / "README.md").exists()
    assert (out_dir / "prefix_summary.csv").exists()
    assert (out_dir / "missing_sidecars.csv").exists()
    assert (out_dir / "orphan_sidecars.csv").exists()
    assert (out_dir / "active_hour_objects.csv").exists()
    assert (out_dir / "latest_remote_objects.csv").exists()
    readme = (out_dir / "README.md").read_text()
    assert "No R2 writes or deletes were executed" in readme
    assert "Missing payload sidecars: 2" in readme
    assert "Active current-hour payload objects: 1" in readme


def test_write_r2_archive_health_report_can_audit_local_gzip_integrity(
    tmp_path: Path,
) -> None:
    good_key = "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-02/hour=13/run-good.jsonl.gz"
    bad_key = "raw/pacifica/full_fidelity/channel=trades/symbol=ETH/date=2026-05-02/hour=13/run-bad.jsonl.gz"
    missing_key = "raw/pacifica/full_fidelity/channel=book/symbol=SOL/date=2026-05-02/hour=13/run-missing.jsonl.gz"
    inventory = pd.DataFrame(
        [
            {"key": good_key, "size_bytes": 50, "mod_time": "2026-05-02T13:10:00Z"},
            {"key": bad_key, "size_bytes": 50, "mod_time": "2026-05-02T13:11:00Z"},
            {"key": missing_key, "size_bytes": 50, "mod_time": "2026-05-02T13:12:00Z"},
        ]
    )
    inventory_path = tmp_path / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    raw_root = tmp_path / "raw"
    good_path = raw_root / good_key.removeprefix("raw/pacifica/full_fidelity/")
    bad_path = raw_root / bad_key.removeprefix("raw/pacifica/full_fidelity/")
    good_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(good_path, "wt") as fh:
        fh.write('{"ok": true}\n')
    bad_path.write_bytes(b"not-a-gzip")

    out_dir = tmp_path / "r2-health"
    result = write_r2_archive_health_report(
        inventory_path,
        out_dir,
        now=datetime(2026, 5, 2, 14, 30, tzinfo=UTC),
        local_raw_root=raw_root,
    )

    assert result["gzip_audit_total"] == 3
    assert result["gzip_audit_ok_count"] == 1
    assert result["gzip_audit_bad_count"] == 1
    assert result["gzip_audit_missing_count"] == 1
    audit = pd.read_csv(out_dir / "gzip_integrity_audit.csv")
    assert set(audit["status"]) == {"ok", "bad_gzip", "missing_local_file"}
    readme = (out_dir / "README.md").read_text()
    assert "Gzip integrity audit" in readme
    assert "Gzip-readable local payloads: 1" in readme

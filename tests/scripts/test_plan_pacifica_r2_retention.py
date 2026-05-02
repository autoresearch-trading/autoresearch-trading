from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.plan_pacifica_r2_retention import (
    R2RetentionPolicy,
    classify_r2_objects,
    write_retention_report,
)


def _inventory(now: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz",
                "size_bytes": 100,
                "mod_time": (now - timedelta(days=5)).isoformat(),
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-03-01/hour=00/b.jsonl.gz",
                "size_bytes": 200,
                "mod_time": (now - timedelta(days=70)).isoformat(),
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-03-01/hour=00/b.jsonl.gz.sha256",
                "size_bytes": 64,
                "mod_time": (now - timedelta(days=70)).isoformat(),
            },
            {
                "key": "raw/pacifica/full_fidelity/channel=trades/symbol=SOL/date=2026-01-01/hour=00/c.jsonl.gz",
                "size_bytes": 300,
                "mod_time": (now - timedelta(days=130)).isoformat(),
                "compacted_verified": True,
                "manifest_verified": True,
            },
        ]
    )


def test_classify_r2_objects_keeps_recent_raw_and_sidecars() -> None:
    now = datetime(2026, 5, 10, tzinfo=UTC)

    out = classify_r2_objects(_inventory(now), now=now, policy=R2RetentionPolicy())

    recent = out[out["key"].str.endswith("a.jsonl.gz")].iloc[0]
    assert recent["retention_action"] == "retain_recent_raw"
    assert bool(recent["remote_delete_allowed"]) is False

    sidecar = out[out["key"].str.endswith(".sha256")].iloc[0]
    assert sidecar["retention_action"] == "retain_checksum_sidecar"
    assert bool(sidecar["remote_delete_allowed"]) is False


def test_classify_r2_objects_requires_compaction_and_manifest_before_delete() -> None:
    now = datetime(2026, 5, 10, tzinfo=UTC)

    out = classify_r2_objects(_inventory(now), now=now, policy=R2RetentionPolicy())

    uncompacted = out[out["key"].str.contains("b.jsonl.gz$")].iloc[0]
    assert uncompacted["retention_action"] == "needs_compaction_before_raw_expiry"
    assert bool(uncompacted["remote_delete_allowed"]) is False

    compacted = out[out["key"].str.contains("c.jsonl.gz$")].iloc[0]
    assert compacted["retention_action"] == "eligible_for_remote_expiry_after_review"
    assert bool(compacted["remote_delete_allowed"]) is True


def test_write_retention_report_is_non_destructive_and_summarizes_bytes(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 5, 10, tzinfo=UTC)
    inventory_path = tmp_path / "inventory.csv"
    _inventory(now).to_csv(inventory_path, index=False)
    out_dir = tmp_path / "r2-retention"

    result = write_retention_report(inventory_path, out_dir, now=now)

    assert result["delete_command_written"] is False
    assert (out_dir / "README.md").exists()
    assert (out_dir / "r2_retention_plan.csv").exists()
    assert (out_dir / "r2_retention_summary.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "No remote deletion was executed" in report
    assert "0-60 days" in report
    assert "60+ days" in report

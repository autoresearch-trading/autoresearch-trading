import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_pacifica_source_manifest import (
    SourceObjectKey,
    build_source_manifest,
    diff_source_manifests,
    plan_changed_sealed_source_objects,
    read_source_manifest,
    write_source_manifest,
)


def _write_raw(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def test_build_source_manifest_keys_sealed_objects_by_channel_symbol_date_hour_run(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    path = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-12"
        / "hour=14"
        / "run-20260512T111943Z.jsonl.gz"
    )
    expected_sha = _write_raw(path, [{"channel": "trades", "symbol": "BTC"}])

    manifest = build_source_manifest(raw, channels=["trades"], count_rows=True)

    assert list(manifest["source_key"]) == [
        "channel=trades/symbol=BTC/date=2026-05-12/hour=14/run=run-20260512T111943Z"
    ]
    row = manifest.iloc[0]
    assert SourceObjectKey.from_manifest_row(row).as_tuple() == (
        "trades",
        "BTC",
        "2026-05-12",
        "14",
        "run-20260512T111943Z",
    )
    assert row["status"] == "sealed"
    assert row["sha256"] == expected_sha
    assert row["row_count"] == 1
    assert row["source_path"] == str(path.relative_to(raw))


def test_build_source_manifest_marks_missing_sidecar_as_unsealed_and_excludes_from_plan(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    unsealed = (
        raw
        / "channel=bbo"
        / "symbol=ETH"
        / "date=2026-05-12"
        / "hour=15"
        / "run-20260512T150000Z.jsonl.gz"
    )
    unsealed.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(unsealed, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"channel": "bbo", "symbol": "ETH"}) + "\n")

    manifest = build_source_manifest(raw, channels=["bbo"], count_rows=True)
    plan = plan_changed_sealed_source_objects(None, manifest)

    assert manifest.iloc[0]["status"] == "unsealed_missing_sidecar"
    assert manifest.iloc[0]["row_count"] == 1
    assert plan.empty


def test_diff_source_manifests_detects_new_changed_removed_and_unchanged_rows(
    tmp_path: Path,
) -> None:
    previous = pd.DataFrame(
        [
            {
                "source_key": "channel=trades/symbol=BTC/date=2026-05-12/hour=14/run=old",
                "channel": "trades",
                "symbol": "BTC",
                "date": "2026-05-12",
                "hour": "14",
                "run": "old",
                "source_path": "a.jsonl.gz",
                "size_bytes": 10,
                "mtime_ns": 1,
                "sha256": "same",
                "status": "sealed",
            },
            {
                "source_key": "channel=trades/symbol=ETH/date=2026-05-12/hour=14/run=changed",
                "channel": "trades",
                "symbol": "ETH",
                "date": "2026-05-12",
                "hour": "14",
                "run": "changed",
                "source_path": "b.jsonl.gz",
                "size_bytes": 10,
                "mtime_ns": 1,
                "sha256": "before",
                "status": "sealed",
            },
            {
                "source_key": "channel=bbo/symbol=SOL/date=2026-05-12/hour=14/run=removed",
                "channel": "bbo",
                "symbol": "SOL",
                "date": "2026-05-12",
                "hour": "14",
                "run": "removed",
                "source_path": "c.jsonl.gz",
                "size_bytes": 1,
                "mtime_ns": 1,
                "sha256": "gone",
                "status": "sealed",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "source_key": "channel=trades/symbol=BTC/date=2026-05-12/hour=14/run=old",
                "channel": "trades",
                "symbol": "BTC",
                "date": "2026-05-12",
                "hour": "14",
                "run": "old",
                "source_path": "a.jsonl.gz",
                "size_bytes": 10,
                "mtime_ns": 1,
                "sha256": "same",
                "status": "sealed",
            },
            {
                "source_key": "channel=trades/symbol=ETH/date=2026-05-12/hour=14/run=changed",
                "channel": "trades",
                "symbol": "ETH",
                "date": "2026-05-12",
                "hour": "14",
                "run": "changed",
                "source_path": "b.jsonl.gz",
                "size_bytes": 11,
                "mtime_ns": 2,
                "sha256": "after",
                "status": "sealed",
            },
            {
                "source_key": "channel=prices/symbol=BTC/date=2026-05-12/hour=14/run=new",
                "channel": "prices",
                "symbol": "BTC",
                "date": "2026-05-12",
                "hour": "14",
                "run": "new",
                "source_path": "d.jsonl.gz",
                "size_bytes": 5,
                "mtime_ns": 2,
                "sha256": "new",
                "status": "sealed",
            },
        ]
    )

    diff = diff_source_manifests(previous, current)

    assert diff.set_index("source_key")["change_status"].to_dict() == {
        "channel=trades/symbol=BTC/date=2026-05-12/hour=14/run=old": "unchanged",
        "channel=trades/symbol=ETH/date=2026-05-12/hour=14/run=changed": "changed",
        "channel=prices/symbol=BTC/date=2026-05-12/hour=14/run=new": "new",
        "channel=bbo/symbol=SOL/date=2026-05-12/hour=14/run=removed": "removed",
    }
    changed_plan = plan_changed_sealed_source_objects(previous, current)
    assert set(changed_plan["change_status"]) == {"changed", "new"}
    assert set(changed_plan["run"]) == {"changed", "new"}


def test_write_and_read_source_manifest_round_trips_csv_with_stable_sort(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    btc = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-05-12"
        / "hour=14"
        / "run-b.jsonl.gz"
    )
    eth = (
        raw
        / "channel=trades"
        / "symbol=ETH"
        / "date=2026-05-12"
        / "hour=13"
        / "run-a.jsonl.gz"
    )
    _write_raw(btc, [{"channel": "trades", "symbol": "BTC"}])
    _write_raw(eth, [{"channel": "trades", "symbol": "ETH"}])

    manifest_path = tmp_path / "manifest.csv"
    written = write_source_manifest(
        raw, manifest_path, channels=["trades"], count_rows=False
    )
    read_back = read_source_manifest(manifest_path)

    assert list(written["source_key"]) == sorted(written["source_key"])
    assert list(read_back["source_key"]) == list(written["source_key"])
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")

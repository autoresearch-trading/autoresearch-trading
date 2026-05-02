import gzip
import json
from pathlib import Path

import pandas as pd

from scripts.build_pacifica_cold_archive import (
    build_cold_archive,
    iter_jsonl_gzip_lines,
    parse_raw_partition_path,
    restore_raw_cache_from_cold_archive,
    restore_sample_from_cold_archive,
    verify_cold_archive_manifest,
)


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def test_parse_raw_partition_path_extracts_channel_symbol_date_hour() -> None:
    path = Path(
        "data/pacifica_full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=13/run.jsonl.gz"
    )

    partition = parse_raw_partition_path(path)

    assert partition == {
        "channel": "trades",
        "symbol": "BTC",
        "date": "2026-05-01",
        "hour": "13",
    }


def test_build_cold_archive_writes_lossless_parquet_and_manifest(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    first = raw_root / "channel=trades/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz"
    second = raw_root / "channel=trades/symbol=ETH/date=2026-05-01/hour=00/b.jsonl.gz"
    _write_gzip_jsonl(first, [{"px": 1}, {"px": 2}])
    _write_gzip_jsonl(second, [{"px": 3}])
    out_dir = tmp_path / "cold"

    result = build_cold_archive(raw_root, out_dir)

    assert result["source_files"] == 2
    assert result["rows"] == 3
    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.exists()
    manifest = pd.read_csv(manifest_path)
    assert set(manifest["status"]) == {"verified"}
    assert set(manifest["rows"]) == {1, 2}
    assert (out_dir / "README.md").exists()
    parquet_files = sorted(out_dir.glob("archive_part-*.parquet"))
    assert len(parquet_files) == 1
    archive = pd.read_parquet(parquet_files[0])
    assert list(archive["raw_json"]) == [
        json.dumps({"px": 1}, sort_keys=True),
        json.dumps({"px": 2}, sort_keys=True),
        json.dumps({"px": 3}, sort_keys=True),
    ]
    assert set(archive["source_key"]) == {
        str(first.relative_to(raw_root)),
        str(second.relative_to(raw_root)),
    }


def test_verify_cold_archive_manifest_detects_missing_archive_file(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "channel=bbo/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz"
    _write_gzip_jsonl(source, [{"bid": 1}])
    out_dir = tmp_path / "cold"
    build_cold_archive(raw_root, out_dir)
    for archive_file in out_dir.glob("archive_part-*.parquet"):
        archive_file.unlink()

    result = verify_cold_archive_manifest(out_dir / "manifest.csv", raw_root=raw_root)

    assert result["ok"] is False
    assert result["missing_archive_files"] == 1
    assert result["verified_sources"] == 0


def test_restore_sample_from_cold_archive_reconstructs_raw_line_sequences(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    first = raw_root / "channel=bbo/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz"
    second = raw_root / "channel=bbo/symbol=ETH/date=2026-05-01/hour=00/b.jsonl.gz"
    _write_gzip_jsonl(first, [{"bid": 1}, {"bid": 2}])
    _write_gzip_jsonl(second, [{"bid": 3}])
    out_dir = tmp_path / "cold"
    build_cold_archive(raw_root, out_dir)

    result = restore_sample_from_cold_archive(
        out_dir / "manifest.csv",
        raw_root=raw_root,
        out_dir=out_dir / "restore-sample",
        max_sources=1,
    )

    assert result["ok"] is True
    assert result["sampled_sources"] == 1
    assert result["matched_sources"] == 1
    assert result["mismatched_sources"] == 0
    report_path = out_dir / "restore-sample" / "restore_sample_report.csv"
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert report.to_dict("records") == [
        {
            "source_key": "channel=bbo/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz",
            "expected_rows": 2,
            "restored_rows": 2,
            "raw_rows": 2,
            "match": True,
        }
    ]
    assert (
        "No R2 writes or deletes were executed"
        in (out_dir / "restore-sample" / "README.md").read_text()
    )


def test_restore_raw_cache_from_cold_archive_recreates_jsonl_gzip_files(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    first = raw_root / "channel=trades/symbol=BTC/date=2026-05-01/hour=00/a.jsonl.gz"
    second = raw_root / "channel=bbo/symbol=ETH/date=2026-05-01/hour=00/b.jsonl.gz"
    _write_gzip_jsonl(first, [{"trade": 1}, {"trade": 2}])
    _write_gzip_jsonl(second, [{"bid": 3}])
    cold_dir = tmp_path / "cold"
    build_cold_archive(raw_root, cold_dir)
    restored_root = tmp_path / "restored-raw"

    result = restore_raw_cache_from_cold_archive(
        cold_dir / "manifest.csv",
        out_raw_root=restored_root,
        original_raw_root=raw_root,
    )

    assert result["ok"] is True
    assert result["restored_sources"] == 2
    assert result["line_mismatches"] == 0
    assert result["write_or_delete_executed"] is False
    restored_first = restored_root / first.relative_to(raw_root)
    restored_second = restored_root / second.relative_to(raw_root)
    assert list(iter_jsonl_gzip_lines(restored_first)) == list(
        iter_jsonl_gzip_lines(first)
    )
    assert list(iter_jsonl_gzip_lines(restored_second)) == list(
        iter_jsonl_gzip_lines(second)
    )
    report = pd.read_csv(restored_root / "restore_raw_cache_report.csv")
    assert set(report["source_key"]) == {
        str(first.relative_to(raw_root)),
        str(second.relative_to(raw_root)),
    }
    assert set(report["line_match"]) == {True}
    assert (
        "No R2 writes or deletes were executed"
        in (restored_root / "README.md").read_text()
    )

#!/usr/bin/env python3
"""Build and verify a lossless cold archive from Pacifica raw JSONL.GZ chunks.

This tool is intentionally local/non-destructive. It reads a bounded raw cache
or restored R2 prefix, writes compressed parquet archive parts plus a manifest,
and verifies that each source object is represented before any future raw-expiry
planner can treat compaction gates as satisfied.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_OUT_DIR = Path("docs/ops/pacifica-cold-archive")
RAW_SUFFIX = ".jsonl.gz"


@dataclass(frozen=True)
class SourceSummary:
    source_key: str
    source_path: str
    channel: str
    symbol: str
    date: str
    hour: str
    size_bytes: int
    sha256: str
    rows: int
    archive_file: str
    archive_size_bytes: int
    archive_sha256: str
    status: str


def parse_raw_partition_path(path: Path) -> dict[str, str]:
    """Extract channel/symbol/date/hour partition values from a raw path."""
    values = {"channel": "", "symbol": "", "date": "", "hour": ""}
    for part in path.parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key in values:
            values[key] = value
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_raw_files(raw_root: Path) -> list[Path]:
    return sorted(p for p in raw_root.rglob(f"*{RAW_SUFFIX}") if p.is_file())


def iter_jsonl_gzip_lines(path: Path) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped


def build_cold_archive(
    raw_root: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    part_name: str = "archive_part-00000.parquet",
) -> dict[str, Any]:
    """Write a lossless parquet archive and source manifest from raw JSONL.GZ."""
    raw_root = raw_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / part_name
    source_files = iter_raw_files(raw_root)

    rows: list[dict[str, Any]] = []
    source_row_counts: dict[str, int] = {}
    source_partitions: dict[str, dict[str, str]] = {}
    source_abs_paths: dict[str, Path] = {}

    for source_path in source_files:
        source_key = str(source_path.relative_to(raw_root))
        source_abs_paths[source_key] = source_path
        source_partitions[source_key] = parse_raw_partition_path(source_path)
        line_count = 0
        for line_number, raw_line in enumerate(
            iter_jsonl_gzip_lines(source_path), start=1
        ):
            partition = source_partitions[source_key]
            rows.append(
                {
                    "source_key": source_key,
                    "line_number": line_number,
                    "channel": partition["channel"],
                    "symbol": partition["symbol"],
                    "date": partition["date"],
                    "hour": partition["hour"],
                    "raw_json": raw_line,
                }
            )
            line_count += 1
        source_row_counts[source_key] = line_count

    archive_df = pd.DataFrame(
        rows,
        columns=[
            "source_key",
            "line_number",
            "channel",
            "symbol",
            "date",
            "hour",
            "raw_json",
        ],
    )
    archive_df.to_parquet(archive_path, index=False, compression="zstd")
    archive_size = archive_path.stat().st_size
    archive_sha = sha256_file(archive_path)

    manifest_rows: list[SourceSummary] = []
    for source_key in sorted(source_row_counts):
        source_path = source_abs_paths[source_key]
        partition = source_partitions[source_key]
        manifest_rows.append(
            SourceSummary(
                source_key=source_key,
                source_path=str(source_path),
                channel=partition["channel"],
                symbol=partition["symbol"],
                date=partition["date"],
                hour=partition["hour"],
                size_bytes=source_path.stat().st_size,
                sha256=sha256_file(source_path),
                rows=source_row_counts[source_key],
                archive_file=archive_path.name,
                archive_size_bytes=archive_size,
                archive_sha256=archive_sha,
                status="verified",
            )
        )

    manifest_path = out_dir / "manifest.csv"
    manifest = pd.DataFrame([asdict(row) for row in manifest_rows])
    manifest.to_csv(manifest_path, index=False)
    verification = verify_cold_archive_manifest(manifest_path, raw_root=raw_root)
    _write_report(out_dir, raw_root=raw_root, result=verification)
    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "archive_file": str(archive_path),
        "source_files": len(source_files),
        "rows": len(rows),
        "ok": verification["ok"],
    }


def verify_cold_archive_manifest(
    manifest_path: Path,
    *,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    """Verify manifest source checksums and archive-file checksums/row coverage."""
    manifest_path = manifest_path.resolve()
    out_dir = manifest_path.parent
    if not manifest_path.exists():
        return {"ok": False, "reason": "manifest_missing", "verified_sources": 0}
    manifest = pd.read_csv(manifest_path)
    missing_archive_files = 0
    archive_checksum_failures = 0
    source_checksum_failures = 0
    row_count_failures = 0
    verified_sources = 0

    archive_cache: dict[str, pd.DataFrame | None] = {}
    archive_sha_cache: dict[str, str | None] = {}
    for _, row in manifest.iterrows():
        archive_file = str(row["archive_file"])
        archive_path = out_dir / archive_file
        if archive_file not in archive_cache:
            if not archive_path.exists():
                archive_cache[archive_file] = None
                archive_sha_cache[archive_file] = None
            else:
                archive_cache[archive_file] = pd.read_parquet(archive_path)
                archive_sha_cache[archive_file] = sha256_file(archive_path)
        archive_df = archive_cache[archive_file]
        if archive_df is None:
            missing_archive_files += 1
            continue
        if archive_sha_cache[archive_file] != str(row["archive_sha256"]):
            archive_checksum_failures += 1
            continue

        source_key = str(row["source_key"])
        expected_rows = int(row["rows"])
        actual_rows = int((archive_df["source_key"] == source_key).sum())
        if actual_rows != expected_rows:
            row_count_failures += 1
            continue

        if raw_root is not None:
            source_path = Path(raw_root) / source_key
            if not source_path.exists() or sha256_file(source_path) != str(
                row["sha256"]
            ):
                source_checksum_failures += 1
                continue
        verified_sources += 1

    ok = bool(
        len(manifest) == verified_sources
        and missing_archive_files == 0
        and archive_checksum_failures == 0
        and source_checksum_failures == 0
        and row_count_failures == 0
    )
    return {
        "ok": ok,
        "manifest": str(manifest_path),
        "sources": int(len(manifest)),
        "verified_sources": verified_sources,
        "missing_archive_files": missing_archive_files,
        "archive_checksum_failures": archive_checksum_failures,
        "source_checksum_failures": source_checksum_failures,
        "row_count_failures": row_count_failures,
    }


def _write_report(out_dir: Path, *, raw_root: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Pacifica Cold Archive Manifest Verification",
        "",
        "This is a local, non-destructive cold-archive artifact. It does not delete R2 raw objects.",
        "",
        f"Raw root: `{raw_root}`",
        f"Manifest: `manifest.csv`",
        f"Verified: `{result['ok']}`",
        "",
        "## Verification summary",
        "",
        f"- Sources: {result.get('sources', 0)}",
        f"- Verified sources: {result.get('verified_sources', 0)}",
        f"- Missing archive files: {result.get('missing_archive_files', 0)}",
        f"- Archive checksum failures: {result.get('archive_checksum_failures', 0)}",
        f"- Source checksum failures: {result.get('source_checksum_failures', 0)}",
        f"- Row-count failures: {result.get('row_count_failures', 0)}",
        "",
        "Remote R2 raw expiry remains blocked until restore and downstream rebuild gates pass and Diego approves a separate destructive apply step.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def restore_sample_from_cold_archive(
    manifest_path: Path,
    *,
    raw_root: Path,
    out_dir: Path,
    max_sources: int = 5,
) -> dict[str, Any]:
    """Sample cold parquet rows and verify they reconstruct source JSONL sequences."""
    manifest_path = manifest_path.resolve()
    raw_root = raw_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path).sort_values("source_key", ignore_index=True)
    sampled = manifest.head(max_sources).copy()

    archive_cache: dict[str, pd.DataFrame] = {}
    report_rows: list[dict[str, Any]] = []
    for _, row in sampled.iterrows():
        archive_file = str(row["archive_file"])
        archive_path = manifest_path.parent / archive_file
        if archive_file not in archive_cache:
            archive_cache[archive_file] = pd.read_parquet(archive_path)
        archive_df = archive_cache[archive_file]
        source_key = str(row["source_key"])
        restored_lines = (
            archive_df[archive_df["source_key"] == source_key]
            .sort_values("line_number")["raw_json"]
            .astype(str)
            .tolist()
        )
        raw_path = raw_root / source_key
        raw_lines = list(iter_jsonl_gzip_lines(raw_path)) if raw_path.exists() else []
        match = restored_lines == raw_lines
        report_rows.append(
            {
                "source_key": source_key,
                "expected_rows": int(row["rows"]),
                "restored_rows": len(restored_lines),
                "raw_rows": len(raw_lines),
                "match": bool(match),
            }
        )

    report = pd.DataFrame(
        report_rows,
        columns=["source_key", "expected_rows", "restored_rows", "raw_rows", "match"],
    )
    report_path = out_dir / "restore_sample_report.csv"
    report.to_csv(report_path, index=False)

    sampled_sources = int(len(report))
    matched_sources = int(report["match"].sum()) if not report.empty else 0
    mismatched_sources = sampled_sources - matched_sources
    ok = mismatched_sources == 0
    lines = [
        "# Pacifica Cold Archive Restore Sample",
        "",
        "No R2 writes or deletes were executed. This is a local restore-sampling diagnostic.",
        "",
        f"Manifest: `{manifest_path}`",
        f"Raw root: `{raw_root}`",
        f"Sampled sources: {sampled_sources}",
        f"Matched sources: {matched_sources}",
        f"Mismatched sources: {mismatched_sources}",
        f"OK: `{ok}`",
        "",
        "This checks that ordered `raw_json` rows in the cold parquet archive reconstruct the original source JSONL line sequence for sampled raw chunks.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    return {
        "ok": ok,
        "report": str(report_path),
        "sampled_sources": sampled_sources,
        "matched_sources": matched_sources,
        "mismatched_sources": mismatched_sources,
        "write_or_delete_executed": False,
    }


def restore_raw_cache_from_cold_archive(
    manifest_path: Path,
    *,
    out_raw_root: Path,
    original_raw_root: Path | None = None,
) -> dict[str, Any]:
    """Recreate partitioned `.jsonl.gz` raw files from cold parquet rows."""
    manifest_path = manifest_path.resolve()
    out_raw_root = out_raw_root.resolve()
    out_raw_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path).sort_values("source_key", ignore_index=True)

    archive_cache: dict[str, pd.DataFrame] = {}
    report_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        archive_file = str(row["archive_file"])
        archive_path = manifest_path.parent / archive_file
        if archive_file not in archive_cache:
            archive_cache[archive_file] = pd.read_parquet(archive_path)
        archive_df = archive_cache[archive_file]
        source_key = str(row["source_key"])
        restored_lines = (
            archive_df[archive_df["source_key"] == source_key]
            .sort_values("line_number")["raw_json"]
            .astype(str)
            .tolist()
        )
        out_path = out_raw_root / source_key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            for line in restored_lines:
                fh.write(line + "\n")

        expected_rows = int(row["rows"])
        row_match = len(restored_lines) == expected_rows
        line_match = True
        raw_rows = ""
        if original_raw_root is not None:
            original_path = Path(original_raw_root) / source_key
            original_lines = (
                list(iter_jsonl_gzip_lines(original_path))
                if original_path.exists()
                else []
            )
            raw_rows = len(original_lines)
            line_match = restored_lines == original_lines
        report_rows.append(
            {
                "source_key": source_key,
                "expected_rows": expected_rows,
                "restored_rows": len(restored_lines),
                "raw_rows": raw_rows,
                "row_match": bool(row_match),
                "line_match": bool(line_match),
                "restored_path": str(out_path),
            }
        )

    report = pd.DataFrame(
        report_rows,
        columns=[
            "source_key",
            "expected_rows",
            "restored_rows",
            "raw_rows",
            "row_match",
            "line_match",
            "restored_path",
        ],
    )
    report_path = out_raw_root / "restore_raw_cache_report.csv"
    report.to_csv(report_path, index=False)
    restored_sources = int(len(report))
    row_mismatches = int((~report["row_match"]).sum()) if not report.empty else 0
    line_mismatches = int((~report["line_match"]).sum()) if not report.empty else 0
    ok = row_mismatches == 0 and line_mismatches == 0
    lines = [
        "# Pacifica Cold Archive Restored Raw Cache",
        "",
        "No R2 writes or deletes were executed. This is a local cold-to-raw restore artifact.",
        "",
        f"Manifest: `{manifest_path}`",
        f"Restored raw root: `{out_raw_root}`",
        f"Original raw root: `{original_raw_root or ''}`",
        f"Restored sources: {restored_sources}",
        f"Row mismatches: {row_mismatches}",
        f"Line mismatches: {line_mismatches}",
        f"OK: `{ok}`",
        "",
        "The restored gzip container bytes are not expected to match the original gzip bytes. The gate is ordered JSONL line equality plus row counts.",
    ]
    (out_raw_root / "README.md").write_text("\n".join(lines) + "\n")
    return {
        "ok": ok,
        "report": str(report_path),
        "restored_raw_root": str(out_raw_root),
        "restored_sources": restored_sources,
        "row_mismatches": row_mismatches,
        "line_mismatches": line_mismatches,
        "write_or_delete_executed": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="build a cold archive from local raw JSONL.GZ files"
    )
    build_parser.add_argument("--raw-root", type=Path, required=True)
    build_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a cold archive manifest"
    )
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--raw-root", type=Path)

    restore_parser = subparsers.add_parser(
        "restore-sample", help="sample cold archive rows against raw source JSONL lines"
    )
    restore_parser.add_argument("--manifest", type=Path, required=True)
    restore_parser.add_argument("--raw-root", type=Path, required=True)
    restore_parser.add_argument("--out-dir", type=Path, required=True)
    restore_parser.add_argument("--max-sources", type=int, default=5)

    restore_raw_parser = subparsers.add_parser(
        "restore-raw-cache",
        help="recreate partitioned JSONL.GZ raw files from cold parquet",
    )
    restore_raw_parser.add_argument("--manifest", type=Path, required=True)
    restore_raw_parser.add_argument("--out-raw-root", type=Path, required=True)
    restore_raw_parser.add_argument("--original-raw-root", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "build":
        result = build_cold_archive(args.raw_root, args.out_dir)
        print(f"wrote archive: {result['archive_file']}")
        print(f"manifest: {result['manifest']}")
        print(f"source_files: {result['source_files']}")
        print(f"rows: {result['rows']}")
        print(f"ok: {result['ok']}")
    elif args.command == "verify":
        result = verify_cold_archive_manifest(args.manifest, raw_root=args.raw_root)
        for key, value in result.items():
            print(f"{key}: {value}")
    elif args.command == "restore-sample":
        result = restore_sample_from_cold_archive(
            args.manifest,
            raw_root=args.raw_root,
            out_dir=args.out_dir,
            max_sources=args.max_sources,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
    elif args.command == "restore-raw-cache":
        result = restore_raw_cache_from_cold_archive(
            args.manifest,
            out_raw_root=args.out_raw_root,
            original_raw_root=args.original_raw_root,
        )
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

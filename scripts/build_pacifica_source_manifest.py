# scripts/build_pacifica_source_manifest.py
"""Build a sealed source-object manifest for Pacifica raw JSONL.GZ chunks.

The manifest is keyed at the immutable raw-object level:
channel/symbol/date/hour/run.  Downstream silver and regime refreshes can diff
this CSV against the previous run and process only new or changed sealed chunks,
while leaving canonical data untouched until a side-by-side verification passes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RAW_DIR = Path("data/pacifica_full_fidelity")
DEFAULT_MANIFEST_PATH = Path("data/ops/pacifica-source-manifest/source_manifest.csv")
DEFAULT_CHANNELS = ("prices", "trades", "bbo", "book", "candle", "mark_price_candle")
MANIFEST_COLUMNS = [
    "source_key",
    "channel",
    "symbol",
    "date",
    "hour",
    "run",
    "source_path",
    "sidecar_path",
    "size_bytes",
    "mtime_ns",
    "modified_at_utc",
    "sha256",
    "sha_verified",
    "gzip_readable",
    "row_count",
    "status",
]
FINGERPRINT_COLUMNS = (
    "size_bytes",
    "mtime_ns",
    "sha256",
    "sha_verified",
    "gzip_readable",
    "status",
)
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


@dataclass(frozen=True, order=True)
class SourceObjectKey:
    channel: str
    symbol: str
    date: str
    hour: str
    run: str

    @property
    def source_key(self) -> str:
        return (
            f"channel={self.channel}/symbol={self.symbol}/date={self.date}/"
            f"hour={self.hour}/run={self.run}"
        )

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (self.channel, self.symbol, self.date, self.hour, self.run)

    @classmethod
    def from_manifest_row(cls, row: pd.Series | dict[str, Any]) -> "SourceObjectKey":
        return cls(
            channel=str(row["channel"]),
            symbol=str(row["symbol"]),
            date=str(row["date"]),
            hour=(
                str(row["hour"]).zfill(2)
                if str(row["hour"]).isdigit()
                else str(row["hour"])
            ),
            run=str(row["run"]),
        )


def _partition_value(
    parts: tuple[str, ...], name: str, default: str = "unknown"
) -> str:
    prefix = f"{name}="
    for part in parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return default


def _run_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl.gz"):
        return name[: -len(".jsonl.gz")]
    return path.stem


def parse_source_object_key(path: Path, raw_dir: Path) -> SourceObjectKey:
    rel = path.relative_to(raw_dir)
    parts = tuple(rel.parts)
    return SourceObjectKey(
        channel=_partition_value(parts, "channel"),
        symbol=_partition_value(parts, "symbol"),
        date=_partition_value(parts, "date"),
        hour=_partition_value(parts, "hour"),
        run=_run_name(path),
    )


def _read_sidecar_sha256(sidecar_path: Path) -> tuple[str | None, str]:
    if not sidecar_path.exists():
        return None, "unsealed_missing_sidecar"
    text = sidecar_path.read_text(encoding="utf-8").strip()
    token = text.split()[0] if text.split() else ""
    if not _SHA256_RE.match(token):
        return None, "unsealed_invalid_sidecar"
    return token.lower(), "sealed"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _manifest_row(
    path: Path,
    raw_dir: Path,
    *,
    verify_sha: bool = False,
    count_rows: bool = False,
) -> dict[str, Any]:
    key = parse_source_object_key(path, raw_dir)
    stat = path.stat()
    sidecar = path.with_name(path.name + ".sha256")
    sha, status = _read_sidecar_sha256(sidecar)
    sha_verified = False
    if sha and verify_sha:
        actual = _sha256_file(path)
        sha_verified = actual == sha
        if not sha_verified:
            status = "unsealed_sha_mismatch"
    row_count: int | None = None
    gzip_readable = False
    if count_rows:
        try:
            row_count = _count_gzip_rows(path)
            gzip_readable = True
        except (EOFError, gzip.BadGzipFile, OSError, UnicodeDecodeError):
            row_count = None
            gzip_readable = False
            if status == "sealed":
                status = "unsealed_gzip_unreadable"
    rel_path = path.relative_to(raw_dir)
    rel_sidecar = sidecar.relative_to(raw_dir)
    return {
        "source_key": key.source_key,
        "channel": key.channel,
        "symbol": key.symbol,
        "date": key.date,
        "hour": key.hour,
        "run": key.run,
        "source_path": str(rel_path),
        "sidecar_path": str(rel_sidecar),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": sha or "",
        "sha_verified": bool(sha_verified),
        "gzip_readable": bool(gzip_readable),
        "row_count": row_count if row_count is not None else pd.NA,
        "status": status,
    }


def build_source_manifest(
    raw_dir: Path,
    *,
    channels: list[str] | tuple[str, ...] | None = None,
    verify_sha: bool = False,
    count_rows: bool = False,
) -> pd.DataFrame:
    """Return a stable source-object manifest for local raw JSONL.GZ chunks."""
    raw_dir = raw_dir.resolve()
    selected = set(channels or DEFAULT_CHANNELS)
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*.jsonl.gz")):
        key = parse_source_object_key(path, raw_dir)
        if key.channel not in selected:
            continue
        rows.append(
            _manifest_row(
                path,
                raw_dir,
                verify_sha=verify_sha,
                count_rows=count_rows,
            )
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        return manifest
    return manifest.sort_values("source_key").reset_index(drop=True)


def read_source_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    dtype = {
        "source_key": "string",
        "channel": "string",
        "symbol": "string",
        "date": "string",
        "hour": "string",
        "run": "string",
        "source_path": "string",
        "sidecar_path": "string",
        "sha256": "string",
        "status": "string",
    }
    frame = pd.read_csv(path, dtype=dtype)
    for col in MANIFEST_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[MANIFEST_COLUMNS].sort_values("source_key").reset_index(drop=True)


def write_source_manifest(
    raw_dir: Path,
    manifest_path: Path,
    *,
    channels: list[str] | tuple[str, ...] | None = None,
    verify_sha: bool = False,
    count_rows: bool = False,
) -> pd.DataFrame:
    manifest = build_source_manifest(
        raw_dir,
        channels=channels,
        verify_sha=verify_sha,
        count_rows=count_rows,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest


def _fingerprint_changed(previous_row: pd.Series, current_row: pd.Series) -> bool:
    for col in FINGERPRINT_COLUMNS:
        prev = previous_row.get(col, pd.NA)
        cur = current_row.get(col, pd.NA)
        if pd.isna(prev) and pd.isna(cur):
            continue
        if str(prev) != str(cur):
            return True
    return False


def diff_source_manifests(
    previous: pd.DataFrame | None, current: pd.DataFrame
) -> pd.DataFrame:
    """Classify current/previous manifest rows as new/changed/unchanged/removed."""
    current = current.copy()
    previous = (
        previous.copy()
        if previous is not None
        else pd.DataFrame(columns=current.columns)
    )
    if current.empty and previous.empty:
        out = pd.DataFrame(columns=[*MANIFEST_COLUMNS, "change_status"])
        return out
    current_by_key = {str(row["source_key"]): row for _, row in current.iterrows()}
    previous_by_key = {str(row["source_key"]): row for _, row in previous.iterrows()}
    rows: list[dict[str, Any]] = []
    for source_key in sorted(set(current_by_key) | set(previous_by_key)):
        if source_key not in previous_by_key:
            row = current_by_key[source_key].to_dict()
            row["change_status"] = "new"
        elif source_key not in current_by_key:
            row = previous_by_key[source_key].to_dict()
            row["change_status"] = "removed"
        else:
            cur = current_by_key[source_key]
            prev = previous_by_key[source_key]
            row = cur.to_dict()
            row["change_status"] = (
                "changed" if _fingerprint_changed(prev, cur) else "unchanged"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plan_changed_sealed_source_objects(
    previous: pd.DataFrame | None, current: pd.DataFrame
) -> pd.DataFrame:
    diff = diff_source_manifests(previous, current)
    if diff.empty:
        return diff
    plan = diff[
        diff["change_status"].isin(["new", "changed"])
        & diff["status"].astype(str).eq("sealed")
    ].copy()
    return plan.sort_values("source_key").reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--count-rows", action="store_true")
    parser.add_argument(
        "--plan-out",
        type=Path,
        help="Optional CSV path for new/changed sealed source-object plan.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
    current = write_source_manifest(
        args.raw_dir,
        args.out,
        channels=channels,
        verify_sha=args.verify_sha,
        count_rows=args.count_rows,
    )
    print(f"wrote {len(current)} source-object manifest rows to {args.out}")
    if args.plan_out:
        previous = read_source_manifest(args.previous) if args.previous else None
        plan = plan_changed_sealed_source_objects(previous, current)
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(args.plan_out, index=False, lineterminator="\n")
        print(f"wrote {len(plan)} changed sealed source-object rows to {args.plan_out}")


if __name__ == "__main__":
    main()

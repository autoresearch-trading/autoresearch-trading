#!/usr/bin/env python3
"""Read-only health checks for Pacifica full-fidelity raw objects in R2.

This script consumes an object inventory CSV, usually generated from Cloudflare R2
via MCP, rclone, or Wrangler, and writes local diagnostic reports. It never writes
or deletes remote objects.
"""

from __future__ import annotations

import argparse
import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RAW_PREFIX = "raw/pacifica/full_fidelity/"
DEFAULT_OUT_DIR = Path("docs/ops/pacifica-r2-archive-health")


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown_table(
    df: pd.DataFrame, *, max_rows: int | None = None
) -> str:
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows) if max_rows is not None else df
    headers = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def parse_raw_object_key(
    key: str, *, raw_prefix: str = DEFAULT_RAW_PREFIX
) -> dict[str, str]:
    """Extract channel/symbol/date/hour partition values from a raw R2 object key."""
    values = {"channel": "", "symbol": "", "date": "", "hour": ""}
    suffix = str(key)
    if suffix.startswith(raw_prefix):
        suffix = suffix[len(raw_prefix) :]
    for part in suffix.split("/"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name in values:
            values[name] = value
    return values


def _normalize_inventory(inventory: pd.DataFrame, *, raw_prefix: str) -> pd.DataFrame:
    required = {"key", "size_bytes", "mod_time"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")

    out = inventory.copy()
    out["key"] = out["key"].astype(str)
    out = out[out["key"].str.startswith(raw_prefix)].copy()
    out["size_bytes"] = (
        pd.to_numeric(out["size_bytes"], errors="coerce").fillna(0).astype(int)
    )
    out["mod_time_ts"] = pd.to_datetime(out["mod_time"], utc=True, errors="coerce")
    out["is_sidecar"] = out["key"].str.endswith(".sha256")
    out["payload_key"] = out["key"].str.removesuffix(".sha256")
    out["is_payload"] = out["key"].str.endswith((".jsonl.gz", ".jsonl.zst"))
    partitions = out["key"].apply(
        lambda key: parse_raw_object_key(key, raw_prefix=raw_prefix)
    )
    for column in ["channel", "symbol", "date", "hour"]:
        out[column] = partitions.apply(lambda row, col=column: row[col])
    return out.sort_values("key", ignore_index=True)


def _current_hour_key_parts(now: datetime) -> tuple[str, str]:
    now_utc = now.astimezone(UTC)
    return now_utc.strftime("%Y-%m-%d"), now_utc.strftime("%H")


def analyze_r2_archive_inventory(
    inventory: pd.DataFrame,
    *,
    now: datetime | None = None,
    raw_prefix: str = DEFAULT_RAW_PREFIX,
) -> dict[str, Any]:
    """Analyze a raw R2 object inventory for sidecar pairing and freshness issues."""
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    objects = _normalize_inventory(inventory, raw_prefix=raw_prefix)
    payloads = objects[objects["is_payload"]].copy()
    sidecars = objects[objects["is_sidecar"]].copy()
    payload_keys = set(payloads["key"])
    sidecar_payload_keys = set(sidecars["payload_key"])

    missing_sidecars = payloads[~payloads["key"].isin(sidecar_payload_keys)].copy()
    orphan_sidecars = sidecars[~sidecars["payload_key"].isin(payload_keys)].copy()

    current_date, current_hour = _current_hour_key_parts(now)
    active_hour_objects = payloads[
        (payloads["date"] == current_date) & (payloads["hour"] == current_hour)
    ].copy()

    prefix_summary = (
        payloads.groupby(["channel", "date"], as_index=False)
        .agg(
            payload_objects=("key", "size"),
            payload_bytes=("size_bytes", "sum"),
            symbols=("symbol", "nunique"),
            latest_mod_time=("mod_time_ts", "max"),
        )
        .sort_values(["date", "channel"], ascending=[False, True], ignore_index=True)
    )
    if not prefix_summary.empty:
        prefix_summary["latest_mod_time"] = prefix_summary[
            "latest_mod_time"
        ].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    latest_remote_objects = (
        payloads.sort_values("mod_time_ts", ascending=False).head(50).copy()
    )
    latest_payload_mod_time = ""
    if not payloads.empty and payloads["mod_time_ts"].notna().any():
        latest_payload_mod_time = payloads["mod_time_ts"].max().isoformat()

    return {
        "objects": objects.drop(columns=["mod_time_ts"]),
        "prefix_summary": prefix_summary,
        "missing_sidecars": missing_sidecars.drop(columns=["mod_time_ts"]),
        "orphan_sidecars": orphan_sidecars.drop(columns=["mod_time_ts"]),
        "active_hour_objects": active_hour_objects.drop(columns=["mod_time_ts"]),
        "latest_remote_objects": latest_remote_objects.drop(columns=["mod_time_ts"]),
        "objects_total": int(len(objects)),
        "payload_objects": int(len(payloads)),
        "sidecar_objects": int(len(sidecars)),
        "payload_bytes": int(payloads["size_bytes"].sum()) if not payloads.empty else 0,
        "missing_sidecar_count": int(len(missing_sidecars)),
        "orphan_sidecar_count": int(len(orphan_sidecars)),
        "active_hour_object_count": int(len(active_hour_objects)),
        "latest_payload_mod_time": latest_payload_mod_time,
        "write_or_delete_executed": False,
    }


def _local_payload_path(key: str, *, raw_prefix: str, local_raw_root: Path) -> Path:
    suffix = key.removeprefix(raw_prefix)
    return local_raw_root / suffix


def audit_local_gzip_integrity(
    payloads: pd.DataFrame,
    *,
    local_raw_root: Path,
    raw_prefix: str = DEFAULT_RAW_PREFIX,
) -> pd.DataFrame:
    """Read local rehydrated payloads and report gzip decompression health."""
    rows: list[dict[str, Any]] = []
    for _, payload in payloads.iterrows():
        key = str(payload["key"])
        local_path = _local_payload_path(
            key, raw_prefix=raw_prefix, local_raw_root=local_raw_root
        )
        status = "ok"
        error = ""
        rows_read = 0
        if not local_path.exists():
            status = "missing_local_file"
        elif not key.endswith(".jsonl.gz"):
            status = "not_gzip_payload"
        else:
            try:
                with gzip.open(local_path, "rt") as fh:
                    for _ in fh:
                        rows_read += 1
            except (EOFError, OSError, gzip.BadGzipFile) as exc:
                status = "bad_gzip"
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "key": key,
                "local_path": str(local_path),
                "status": status,
                "rows_read": rows_read,
                "error": error,
            }
        )
    return pd.DataFrame(
        rows, columns=["key", "local_path", "status", "rows_read", "error"]
    )


def write_r2_archive_health_report(
    inventory_path: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    now: datetime | None = None,
    raw_prefix: str = DEFAULT_RAW_PREFIX,
    local_raw_root: Path | None = None,
) -> dict[str, Any]:
    inventory = pd.read_csv(inventory_path)
    result = analyze_r2_archive_inventory(inventory, now=now, raw_prefix=raw_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    if local_raw_root is not None:
        gzip_audit = audit_local_gzip_integrity(
            result["objects"][result["objects"]["is_payload"]].copy(),
            local_raw_root=local_raw_root,
            raw_prefix=raw_prefix,
        )
        result["gzip_integrity_audit"] = gzip_audit
        result["gzip_audit_total"] = int(len(gzip_audit))
        result["gzip_audit_ok_count"] = int((gzip_audit["status"] == "ok").sum())
        result["gzip_audit_bad_count"] = int((gzip_audit["status"] == "bad_gzip").sum())
        result["gzip_audit_missing_count"] = int(
            (gzip_audit["status"] == "missing_local_file").sum()
        )
    else:
        result["gzip_integrity_audit"] = pd.DataFrame(
            columns=["key", "local_path", "status", "rows_read", "error"]
        )
        result["gzip_audit_total"] = 0
        result["gzip_audit_ok_count"] = 0
        result["gzip_audit_bad_count"] = 0
        result["gzip_audit_missing_count"] = 0

    artifact_map = {
        "prefix_summary": out_dir / "prefix_summary.csv",
        "missing_sidecars": out_dir / "missing_sidecars.csv",
        "orphan_sidecars": out_dir / "orphan_sidecars.csv",
        "active_hour_objects": out_dir / "active_hour_objects.csv",
        "latest_remote_objects": out_dir / "latest_remote_objects.csv",
        "gzip_integrity_audit": out_dir / "gzip_integrity_audit.csv",
    }
    for key, path in artifact_map.items():
        result[key].to_csv(path, index=False)

    generated_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    lines = [
        "# Pacifica R2 Raw Archive Health",
        "",
        "No R2 writes or deletes were executed. This is a read-only diagnostic report from an object inventory snapshot.",
        "",
        f"Generated at: `{generated_at}`",
        f"Raw prefix: `{raw_prefix}`",
        "",
        "## Summary",
        "",
        f"- Raw-prefix objects: {result['objects_total']}",
        f"- Payload objects: {result['payload_objects']}",
        f"- Sidecar objects: {result['sidecar_objects']}",
        f"- Payload bytes: {result['payload_bytes']}",
        f"- Latest payload mod time: {result['latest_payload_mod_time'] or 'n/a'}",
        f"- Missing payload sidecars: {result['missing_sidecar_count']}",
        f"- Orphan sidecars: {result['orphan_sidecar_count']}",
        f"- Active current-hour payload objects: {result['active_hour_object_count']}",
        f"- Gzip audit local root: `{str(local_raw_root) if local_raw_root is not None else 'not requested'}`",
        f"- Gzip-readable local payloads: {result['gzip_audit_ok_count']} / {result['gzip_audit_total']}",
        f"- Bad gzip local payloads: {result['gzip_audit_bad_count']}",
        f"- Missing local payloads for gzip audit: {result['gzip_audit_missing_count']}",
        "",
        "## Prefix summary",
        "",
        dataframe_to_markdown_table(result["prefix_summary"], max_rows=25),
        "",
        "## Latest payload objects",
        "",
        dataframe_to_markdown_table(
            result["latest_remote_objects"][
                ["key", "size_bytes", "mod_time", "channel", "symbol", "date", "hour"]
            ],
            max_rows=20,
        ),
        "",
        "## Gzip integrity audit",
        "",
        "This optional audit is local-only and only checks rehydrated payloads under `--local-raw-root`; it does not read, write, or delete remote R2 objects.",
        "",
        dataframe_to_markdown_table(result["gzip_integrity_audit"], max_rows=25),
        "",
        "## Output files",
        "",
        "- `prefix_summary.csv` — payload counts/bytes by channel/date.",
        "- `missing_sidecars.csv` — payloads without matching `.sha256` sidecars.",
        "- `orphan_sidecars.csv` — `.sha256` sidecars without matching payloads.",
        "- `active_hour_objects.csv` — current UTC hour payloads; these should normally be absent for sealed-chunk uploads.",
        "- `latest_remote_objects.csv` — newest payload objects in this inventory snapshot.",
        "- `gzip_integrity_audit.csv` — optional local gzip decompression status for rehydrated payloads.",
    ]
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")

    out = dict(result)
    for key, path in artifact_map.items():
        out[f"{key}_path"] = str(path)
    out["readme"] = str(readme_path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument(
        "--local-raw-root",
        type=Path,
        help="Optional local rehydrated raw root for read-only gzip decompression audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_r2_archive_health_report(
        args.inventory_csv,
        args.out_dir,
        raw_prefix=args.raw_prefix,
        local_raw_root=args.local_raw_root,
    )
    print(f"wrote report: {result['readme']}")
    print(f"payload_objects: {result['payload_objects']}")
    print(f"missing_sidecars: {result['missing_sidecar_count']}")
    print(f"orphan_sidecars: {result['orphan_sidecar_count']}")
    print(f"active_hour_objects: {result['active_hour_object_count']}")
    print(
        f"gzip_audit_ok: {result['gzip_audit_ok_count']} / {result['gzip_audit_total']}"
    )
    print(f"gzip_audit_bad: {result['gzip_audit_bad_count']}")
    print(f"gzip_audit_missing: {result['gzip_audit_missing_count']}")
    print("write_or_delete_executed: False")


if __name__ == "__main__":
    main()

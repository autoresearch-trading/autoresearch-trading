#!/usr/bin/env python3
"""Read-only health checks for Pacifica full-fidelity raw objects in R2.

This script consumes an object inventory CSV, usually generated from Cloudflare R2
via MCP, rclone, or Wrangler, and writes local diagnostic reports. It never writes
or deletes remote objects.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

DEFAULT_RAW_PREFIX = "raw/pacifica/full_fidelity/"
DEFAULT_REMOTE_BASE = "r2:pacifica-trading-data"
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


def parse_inventory_mod_time(value: Any) -> datetime | None:
    """Parse inventory timestamps from lsjson or line-oriented lsf listings.

    `rclone lsjson` emits offset-aware ISO timestamps. `rclone lsf --format t`
    emits process-local timestamps without an offset, so naive values must be
    interpreted as local time before converting to UTC.
    """

    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z") or "+" in text[10:] or "-" in text[10:]:
            return pd.Timestamp(text).to_pydatetime().astimezone(UTC)
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").astimezone(UTC)
    except (ValueError, TypeError):
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().astimezone(UTC)


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
    out["mod_time_ts"] = out["mod_time"].apply(parse_inventory_mod_time)
    out["mod_time_ts"] = pd.to_datetime(out["mod_time_ts"], utc=True, errors="coerce")
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
    stale_after_min: float = 180.0,
) -> dict[str, Any]:
    """Analyze a raw R2 object inventory for sidecar pairing and freshness issues."""
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)

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
    channel_coverage = (
        payloads.groupby(["channel"], as_index=False)
        .agg(
            payload_objects=("key", "size"),
            payload_bytes=("size_bytes", "sum"),
            dates=("date", "nunique"),
            symbols=("symbol", "nunique"),
            latest_mod_time=("mod_time_ts", "max"),
        )
        .sort_values(
            ["payload_objects", "channel"], ascending=[False, True], ignore_index=True
        )
    )
    date_coverage = (
        payloads.groupby(["date"], as_index=False)
        .agg(
            payload_objects=("key", "size"),
            payload_bytes=("size_bytes", "sum"),
            channels=("channel", "nunique"),
            symbols=("symbol", "nunique"),
            latest_mod_time=("mod_time_ts", "max"),
        )
        .sort_values(["date"], ascending=[False], ignore_index=True)
    )
    channel_date_symbol_coverage = (
        payloads.groupby(["channel", "date", "symbol"], as_index=False)
        .agg(
            payload_objects=("key", "size"),
            payload_bytes=("size_bytes", "sum"),
            hours=("hour", "nunique"),
            latest_mod_time=("mod_time_ts", "max"),
        )
        .sort_values(
            ["date", "channel", "symbol"],
            ascending=[False, True, True],
            ignore_index=True,
        )
    )
    for frame in [
        prefix_summary,
        channel_coverage,
        date_coverage,
        channel_date_symbol_coverage,
    ]:
        if not frame.empty and "latest_mod_time" in frame.columns:
            frame["latest_mod_time"] = frame["latest_mod_time"].dt.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            )

    latest_remote_objects = (
        payloads.sort_values("mod_time_ts", ascending=False).head(50).copy()
    )
    latest_payload_mod_time = ""
    latest_payload_age_min: float | None = None
    latest_payload_freshness_ok = False
    if not payloads.empty and payloads["mod_time_ts"].notna().any():
        latest_ts = payloads["mod_time_ts"].max()
        latest_payload_mod_time = latest_ts.isoformat()
        latest_payload_age_min = round(
            (now - latest_ts.to_pydatetime()).total_seconds() / 60.0, 2
        )
        latest_payload_freshness_ok = latest_payload_age_min <= stale_after_min

    failures: list[str] = []
    if payloads.empty:
        failures.append("R2_RAW_PAYLOAD_INVENTORY_EMPTY")
    if missing_sidecars.empty is False:
        failures.append("R2_SIDECAR_MISSING")
    if orphan_sidecars.empty is False:
        failures.append("R2_ORPHAN_SIDECAR")
    if active_hour_objects.empty is False:
        failures.append("R2_CURRENT_HOUR_PAYLOAD_LEAKAGE")
    if latest_payload_age_min is not None and not latest_payload_freshness_ok:
        failures.append("R2_REMOTE_FRESHNESS_STALE")

    return {
        "objects": objects.drop(columns=["mod_time_ts"], errors="ignore"),
        "prefix_summary": prefix_summary,
        "channel_coverage": channel_coverage,
        "date_coverage": date_coverage,
        "channel_date_symbol_coverage": channel_date_symbol_coverage,
        "missing_sidecars": missing_sidecars.drop(
            columns=["mod_time_ts"], errors="ignore"
        ),
        "orphan_sidecars": orphan_sidecars.drop(
            columns=["mod_time_ts"], errors="ignore"
        ),
        "active_hour_objects": active_hour_objects.drop(
            columns=["mod_time_ts"], errors="ignore"
        ),
        "latest_remote_objects": latest_remote_objects.drop(
            columns=["mod_time_ts"], errors="ignore"
        ),
        "objects_total": int(len(objects)),
        "payload_objects": int(len(payloads)),
        "sidecar_objects": int(len(sidecars)),
        "payload_bytes": int(payloads["size_bytes"].sum()) if not payloads.empty else 0,
        "missing_sidecar_count": int(len(missing_sidecars)),
        "orphan_sidecar_count": int(len(orphan_sidecars)),
        "active_hour_object_count": int(len(active_hour_objects)),
        "latest_payload_mod_time": latest_payload_mod_time,
        "latest_payload_age_min": latest_payload_age_min,
        "latest_payload_freshness_ok": latest_payload_freshness_ok,
        "stale_after_min": stale_after_min,
        "distinct_channels": (
            int(payloads["channel"].nunique()) if not payloads.empty else 0
        ),
        "distinct_dates": int(payloads["date"].nunique()) if not payloads.empty else 0,
        "distinct_symbols": (
            int(payloads["symbol"].nunique()) if not payloads.empty else 0
        ),
        "failures": failures,
        "ok": not failures,
        "write_or_delete_executed": False,
    }


def _sidecar_digest(sidecar_text: str | None) -> str:
    if not sidecar_text:
        return ""
    first_token = sidecar_text.strip().split()[0] if sidecar_text.strip() else ""
    if len(first_token) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in first_token
    ):
        return first_token.lower()
    return ""


def verify_gzip_payload_bytes(
    key: str, payload_bytes: bytes, sidecar_text: str | None
) -> dict[str, Any]:
    """Verify one sampled payload's SHA-256 sidecar and gzip readability."""

    digest = hashlib.sha256(payload_bytes).hexdigest()
    expected_digest = _sidecar_digest(sidecar_text)
    rows_read = 0
    gzip_status = "ok"
    error = ""
    if not key.endswith(".jsonl.gz"):
        gzip_status = "not_gzip_payload"
    else:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload_bytes), mode="rb") as fh:
                for _ in fh:
                    rows_read += 1
        except (EOFError, OSError, gzip.BadGzipFile) as exc:
            gzip_status = "bad_gzip"
            error = f"{type(exc).__name__}: {exc}"

    if gzip_status != "ok":
        status = gzip_status
    elif expected_digest and digest != expected_digest:
        status = "sha256_mismatch"
    elif not expected_digest:
        status = "sidecar_missing_or_invalid"
    else:
        status = "ok"

    return {
        "key": key,
        "status": status,
        "rows_read": rows_read,
        "size_bytes": len(payload_bytes),
        "sha256": digest,
        "sidecar_sha256": expected_digest,
        "sha256_matches_sidecar": bool(expected_digest and digest == expected_digest),
        "error": error,
    }


def _remote_object(remote_base: str, key: str) -> str:
    return f"{remote_base.rstrip('/')}/{key.lstrip('/')}"


def _rclone_cat_bytes(remote: str, *, timeout_s: int) -> tuple[int, bytes, str]:
    try:
        proc = subprocess.run(
            ["rclone", "cat", remote],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return (
            proc.returncode,
            proc.stdout,
            proc.stderr.decode("utf-8", errors="replace"),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.encode("utf-8", errors="replace")
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return (124, stdout, stderr + f"\ntimeout after {timeout_s}s")


def select_gzip_sample_payloads(
    payloads: pd.DataFrame, *, max_samples: int, max_payload_bytes: int
) -> pd.DataFrame:
    """Pick recent gzip payloads with channel diversity, capped by payload size."""

    if payloads.empty or max_samples <= 0:
        return payloads.head(0).copy()
    candidates = payloads[
        payloads["key"].astype(str).str.endswith(".jsonl.gz")
        & (pd.to_numeric(payloads["size_bytes"], errors="coerce") <= max_payload_bytes)
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values("mod_time_ts", ascending=False)
    selected_indices: list[Any] = []
    seen_channels: set[str] = set()
    for idx, row in candidates.iterrows():
        channel = str(row.get("channel", ""))
        if channel in seen_channels:
            continue
        selected_indices.append(idx)
        seen_channels.add(channel)
        if len(selected_indices) >= max_samples:
            break
    if len(selected_indices) < max_samples:
        for idx in candidates.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= max_samples:
                break
    return candidates.loc[selected_indices].copy()


def audit_remote_gzip_samples(
    payloads_with_timestamps: pd.DataFrame,
    *,
    remote_base: str = DEFAULT_REMOTE_BASE,
    max_samples: int = 0,
    max_payload_bytes: int = 50_000_000,
    timeout_s: int = 60,
    fetch_bytes: Callable[[str], tuple[int, bytes, str]] | None = None,
) -> pd.DataFrame:
    """Read-only remote sample verifier: payload SHA sidecar + gzip decode."""

    if max_samples <= 0:
        return pd.DataFrame(
            columns=[
                "key",
                "status",
                "rows_read",
                "size_bytes",
                "sha256",
                "sidecar_sha256",
                "sha256_matches_sidecar",
                "error",
            ]
        )

    fetch = fetch_bytes or (
        lambda remote: _rclone_cat_bytes(remote, timeout_s=timeout_s)
    )
    sample = select_gzip_sample_payloads(
        payloads_with_timestamps,
        max_samples=max_samples,
        max_payload_bytes=max_payload_bytes,
    )
    columns = [
        "key",
        "status",
        "rows_read",
        "size_bytes",
        "sha256",
        "sidecar_sha256",
        "sha256_matches_sidecar",
        "error",
    ]
    if sample.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, payload in sample.iterrows():
        key = str(payload["key"])
        payload_rc, payload_bytes, payload_err = fetch(_remote_object(remote_base, key))
        sidecar_rc, sidecar_bytes, sidecar_err = fetch(
            _remote_object(remote_base, f"{key}.sha256")
        )
        if payload_rc != 0:
            rows.append(
                {
                    "key": key,
                    "status": "payload_fetch_failed",
                    "rows_read": 0,
                    "size_bytes": int(payload.get("size_bytes") or 0),
                    "sha256": "",
                    "sidecar_sha256": "",
                    "sha256_matches_sidecar": False,
                    "error": payload_err[-500:],
                }
            )
            continue
        sidecar_text = None
        if sidecar_rc == 0:
            sidecar_text = sidecar_bytes.decode("utf-8", errors="replace")
        result = verify_gzip_payload_bytes(key, payload_bytes, sidecar_text)
        if sidecar_rc != 0 and result["status"] == "sidecar_missing_or_invalid":
            result["error"] = sidecar_err[-500:]
        rows.append(result)
    return pd.DataFrame(rows)


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
    stale_after_min: float = 180.0,
    remote_base: str = DEFAULT_REMOTE_BASE,
    gzip_sample_size: int = 0,
    gzip_sample_max_bytes: int = 50_000_000,
    gzip_sample_timeout_s: int = 60,
) -> dict[str, Any]:
    inventory = pd.read_csv(inventory_path)
    result = analyze_r2_archive_inventory(
        inventory, now=now, raw_prefix=raw_prefix, stale_after_min=stale_after_min
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_with_timestamps = _normalize_inventory(inventory, raw_prefix=raw_prefix)
    payloads_with_timestamps = objects_with_timestamps[
        objects_with_timestamps["is_payload"]
    ].copy()
    remote_gzip_audit = audit_remote_gzip_samples(
        payloads_with_timestamps,
        remote_base=remote_base,
        max_samples=gzip_sample_size,
        max_payload_bytes=gzip_sample_max_bytes,
        timeout_s=gzip_sample_timeout_s,
    )
    result["remote_gzip_sample_verification"] = remote_gzip_audit
    result["remote_gzip_sample_total"] = int(len(remote_gzip_audit))
    result["remote_gzip_sample_ok_count"] = (
        int((remote_gzip_audit["status"] == "ok").sum())
        if not remote_gzip_audit.empty
        else 0
    )
    result["remote_gzip_sample_bad_count"] = (
        int((remote_gzip_audit["status"] != "ok").sum())
        if not remote_gzip_audit.empty
        else 0
    )
    if result["remote_gzip_sample_bad_count"]:
        result["failures"] = sorted(
            set([*result["failures"], "R2_GZIP_SAMPLE_VERIFICATION_FAILED"])
        )
        result["ok"] = False

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
        "channel_coverage": out_dir / "channel_coverage.csv",
        "date_coverage": out_dir / "date_coverage.csv",
        "channel_date_symbol_coverage": out_dir / "channel_date_symbol_coverage.csv",
        "missing_sidecars": out_dir / "missing_sidecars.csv",
        "orphan_sidecars": out_dir / "orphan_sidecars.csv",
        "active_hour_objects": out_dir / "active_hour_objects.csv",
        "latest_remote_objects": out_dir / "latest_remote_objects.csv",
        "remote_gzip_sample_verification": out_dir
        / "remote_gzip_sample_verification.csv",
        "gzip_integrity_audit": out_dir / "gzip_integrity_audit.csv",
    }
    for key, path in artifact_map.items():
        result[key].to_csv(path, index=False, lineterminator="\n")

    generated_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    lines = [
        "# Pacifica R2 Raw Archive Health",
        "",
        "No R2 writes or deletes were executed. This is a read-only diagnostic report from an object inventory snapshot plus optional read-only sample downloads.",
        "",
        f"Generated at: `{generated_at}`",
        f"Inventory CSV: `{inventory_path}`",
        f"Raw prefix: `{raw_prefix}`",
        f"Remote base for sample reads: `{remote_base}`",
        "",
        "## Summary",
        "",
        f"- OK: {result['ok']}",
        f"- Failures: {result['failures']}",
        f"- Raw-prefix objects: {result['objects_total']}",
        f"- Payload objects: {result['payload_objects']}",
        f"- Sidecar objects: {result['sidecar_objects']}",
        f"- Payload bytes: {result['payload_bytes']}",
        f"- Latest payload mod time: {result['latest_payload_mod_time'] or 'n/a'}",
        f"- Latest payload age minutes: {result['latest_payload_age_min'] if result['latest_payload_age_min'] is not None else 'n/a'}",
        f"- Stale threshold minutes: {result['stale_after_min']}",
        f"- Latest payload freshness OK: {result['latest_payload_freshness_ok']}",
        f"- Missing payload sidecars: {result['missing_sidecar_count']}",
        f"- Orphan sidecars: {result['orphan_sidecar_count']}",
        f"- Active current-hour payload objects: {result['active_hour_object_count']}",
        f"- Distinct channels: {result['distinct_channels']}",
        f"- Distinct dates: {result['distinct_dates']}",
        f"- Distinct symbols: {result['distinct_symbols']}",
        f"- Remote gzip sample verification: {result['remote_gzip_sample_ok_count']} OK / {result['remote_gzip_sample_total']} sampled",
        f"- Remote gzip sample failures: {result['remote_gzip_sample_bad_count']}",
        f"- Gzip audit local root: `{str(local_raw_root) if local_raw_root is not None else 'not requested'}`",
        f"- Gzip-readable local payloads: {result['gzip_audit_ok_count']} / {result['gzip_audit_total']}",
        f"- Bad gzip local payloads: {result['gzip_audit_bad_count']}",
        f"- Missing local payloads for gzip audit: {result['gzip_audit_missing_count']}",
        "",
        "## Prefix summary",
        "",
        dataframe_to_markdown_table(result["prefix_summary"], max_rows=25),
        "",
        "## Channel coverage",
        "",
        dataframe_to_markdown_table(result["channel_coverage"], max_rows=25),
        "",
        "## Date coverage",
        "",
        dataframe_to_markdown_table(result["date_coverage"], max_rows=25),
        "",
        "## Channel/date/symbol coverage",
        "",
        "Full coverage is written to `channel_date_symbol_coverage.csv`; the table below shows the newest 25 rows.",
        "",
        dataframe_to_markdown_table(
            result["channel_date_symbol_coverage"], max_rows=25
        ),
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
        "## Remote gzip sample verification",
        "",
        "This optional sample uses read-only `rclone cat` calls for payloads and sibling `.sha256` sidecars, then verifies SHA-256 and gzip decompression locally. It does not write or delete R2 objects.",
        "",
        dataframe_to_markdown_table(
            result["remote_gzip_sample_verification"], max_rows=25
        ),
        "",
        "## Local Gzip integrity audit",
        "",
        "This optional audit is local-only and only checks rehydrated payloads under `--local-raw-root`; it does not read, write, or delete remote R2 objects.",
        "",
        dataframe_to_markdown_table(result["gzip_integrity_audit"], max_rows=25),
        "",
        "## Output files",
        "",
        "- `prefix_summary.csv` — payload counts/bytes by channel/date.",
        "- `channel_coverage.csv` — payload counts/bytes/dates/symbols by channel.",
        "- `date_coverage.csv` — payload counts/bytes/channels/symbols by date.",
        "- `channel_date_symbol_coverage.csv` — payload counts/bytes/hours by channel/date/symbol.",
        "- `missing_sidecars.csv` — payloads without matching `.sha256` sidecars.",
        "- `orphan_sidecars.csv` — `.sha256` sidecars without matching payloads.",
        "- `active_hour_objects.csv` — current UTC hour payloads; these should normally be absent for sealed-chunk uploads.",
        "- `latest_remote_objects.csv` — newest payload objects in this inventory snapshot.",
        "- `remote_gzip_sample_verification.csv` — optional read-only remote payload+sidecar SHA/gzip sample verification.",
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
    parser.add_argument("--stale-after-min", type=float, default=180.0)
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE)
    parser.add_argument(
        "--gzip-sample-size",
        type=int,
        default=0,
        help="Number of remote payload samples to read with rclone cat for SHA/gzip verification.",
    )
    parser.add_argument(
        "--gzip-sample-max-bytes",
        type=int,
        default=50_000_000,
        help="Skip remote gzip samples larger than this many bytes.",
    )
    parser.add_argument("--gzip-sample-timeout-s", type=int, default=60)
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
        stale_after_min=args.stale_after_min,
        remote_base=args.remote_base,
        gzip_sample_size=args.gzip_sample_size,
        gzip_sample_max_bytes=args.gzip_sample_max_bytes,
        gzip_sample_timeout_s=args.gzip_sample_timeout_s,
    )
    print(f"wrote report: {result['readme']}")
    print(f"ok: {result['ok']}")
    print(f"failures: {result['failures']}")
    print(f"payload_objects: {result['payload_objects']}")
    print(f"sidecar_objects: {result['sidecar_objects']}")
    print(f"latest_payload_mod_time: {result['latest_payload_mod_time'] or 'n/a'}")
    print(f"latest_payload_age_min: {result['latest_payload_age_min']}")
    print(f"missing_sidecars: {result['missing_sidecar_count']}")
    print(f"orphan_sidecars: {result['orphan_sidecar_count']}")
    print(f"active_hour_objects: {result['active_hour_object_count']}")
    print(f"distinct_channels: {result['distinct_channels']}")
    print(f"distinct_dates: {result['distinct_dates']}")
    print(f"distinct_symbols: {result['distinct_symbols']}")
    print(
        "remote_gzip_sample_ok: "
        f"{result['remote_gzip_sample_ok_count']} / {result['remote_gzip_sample_total']}"
    )
    print(f"remote_gzip_sample_bad: {result['remote_gzip_sample_bad_count']}")
    print(
        f"gzip_audit_ok: {result['gzip_audit_ok_count']} / {result['gzip_audit_total']}"
    )
    print(f"gzip_audit_bad: {result['gzip_audit_bad_count']}")
    print(f"gzip_audit_missing: {result['gzip_audit_missing_count']}")
    print("write_or_delete_executed: False")


if __name__ == "__main__":
    main()

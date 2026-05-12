#!/usr/bin/env python3
"""Bounded read-only R2 freshness smoke check for Pacifica raw archive.

This is intentionally cheaper than a full bucket inventory. It samples a small,
configured set of high-signal channel/symbol prefixes for today and yesterday,
checks that payloads have sibling `.sha256` sidecars, and reports latest sampled
remote freshness. It never deletes or mutates R2 objects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_PREFIXES = (
    "channel=bbo/symbol=BTC",
    "channel=book/symbol=ETH",
    "channel=trades/symbol=BTC",
    "channel=mark_price_candle/symbol=ICP",
)


def parse_rclone_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_lsf_listing(text: str, *, sample_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue
        rel, size_raw, mtime_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not (rel.endswith(".jsonl.gz") or rel.endswith(".jsonl.gz.sha256")):
            continue
        mtime = parse_rclone_time(mtime_raw)
        if mtime is None:
            continue
        try:
            size = int(size_raw)
        except ValueError:
            size = 0
        rows.append(
            {
                "sample_prefix": sample_prefix,
                "relative_path": rel,
                "size_bytes": size,
                "modified_at": mtime.isoformat(),
            }
        )
    return rows


def _row_mtime(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(
        str(row["modified_at"]).replace("Z", "+00:00")
    ).astimezone(UTC)


def analyze_lsf_samples(
    rows: list[dict[str, Any]], *, now: datetime, stale_after_min: float
) -> dict[str, Any]:
    now = now.astimezone(UTC)
    payloads = [
        row for row in rows if str(row.get("relative_path", "")).endswith(".jsonl.gz")
    ]
    sidecars = {
        (row["sample_prefix"], str(row["relative_path"])[: -len(".sha256")])
        for row in rows
        if str(row.get("relative_path", "")).endswith(".jsonl.gz.sha256")
    }
    missing_sidecars = [
        row
        for row in payloads
        if (row["sample_prefix"], row["relative_path"]) not in sidecars
    ]

    failures: list[str] = []
    latest_payload: dict[str, Any] | None = None
    latest_age_min: float | None = None
    if not payloads:
        failures.append("R2_RAW_SAMPLE_EMPTY")
    else:
        latest_payload = max(payloads, key=_row_mtime)
        latest_age_min = round(
            (now - _row_mtime(latest_payload)).total_seconds() / 60.0, 2
        )
        if latest_age_min > stale_after_min:
            failures.append("R2_REMOTE_FRESHNESS_STALE")
    if missing_sidecars:
        failures.append("R2_SIDECAR_MISSING")

    return {
        "ok": not failures,
        "checked_at": now.isoformat(),
        "stale_after_min": stale_after_min,
        "sampled_rows": len(rows),
        "payload_count": len(payloads),
        "sidecar_count": len(sidecars),
        "sidecar_missing_count": len(missing_sidecars),
        "latest_payload_age_min": latest_age_min,
        "latest_payload": latest_payload,
        "missing_sidecar_samples": missing_sidecars[:20],
        "failures": failures,
    }


def run_rclone_lsf(remote_prefix: str, *, timeout_s: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [
                "rclone",
                "lsf",
                remote_prefix,
                "--recursive",
                "--files-only",
                "--format",
                "pst",
                "--separator",
                ";",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return 124, stdout, stderr + f"\ntimeout after {timeout_s}s"


def collect_samples(
    *,
    remote_base: str,
    r2_prefix: str,
    sample_prefixes: list[str],
    now: datetime,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = [now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sample in sample_prefixes:
        sample = sample.strip().strip("/")
        if not sample:
            continue
        for date in dates:
            sample_with_date = f"{sample}/date={date}"
            remote = (
                f"{remote_base.rstrip('/')}/{r2_prefix.strip('/')}/{sample_with_date}"
            )
            returncode, stdout, stderr = run_rclone_lsf(remote, timeout_s=timeout_s)
            if returncode != 0:
                errors.append(
                    {
                        "sample_prefix": sample_with_date,
                        "returncode": returncode,
                        "stderr_tail": stderr[-1000:],
                    }
                )
                continue
            rows.extend(parse_lsf_listing(stdout, sample_prefix=sample_with_date))
    return rows, errors


def _parse_sample_prefixes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SAMPLE_PREFIXES)
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-base", default="r2:pacifica-trading-data")
    parser.add_argument("--r2-prefix", default="raw/pacifica/full_fidelity")
    parser.add_argument("--sample-prefixes", default=None)
    parser.add_argument("--stale-after-min", type=float, default=180.0)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    now = datetime.now(UTC)
    rows, errors = collect_samples(
        remote_base=args.remote_base,
        r2_prefix=args.r2_prefix,
        sample_prefixes=_parse_sample_prefixes(args.sample_prefixes),
        now=now,
        timeout_s=args.timeout_s,
    )
    status = analyze_lsf_samples(rows, now=now, stale_after_min=args.stale_after_min)
    status["listing_errors"] = errors[:20]
    if errors and not rows:
        status["ok"] = False
        status["failures"] = sorted(
            set([*status["failures"], "R2_SAMPLE_LISTING_FAILED"])
        )
    text = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0 if status["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

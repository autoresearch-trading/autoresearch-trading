#!/usr/bin/env python3
"""Operationally independent Fly-side watchdogs for Pacifica collection support.

Runs read-only/non-destructive operational checks on the always-on host:
- public API/docs surface watcher;
- R2 inventory + raw retention/cold-compaction policy planner.

The script is safe to run in a loop. It uses marker files under --state-dir to avoid
running expensive checks more often than their configured intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WatchdogConfig:
    root: Path = Path("/data/ops")
    state_dir: Path = Path("/data/ops/.state")
    remote_base: str = "r2:pacifica-trading-data"
    r2_prefix: str = "raw/pacifica/full_fidelity"
    reports_prefix: str = "ops/pacifica/full_fidelity"
    api_surface_interval_s: int = 86_400
    r2_retention_interval_s: int = 86_400
    command_timeout_s: int = 600
    upload_reports: bool = True


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_due(marker: Path, *, interval_s: int, now: datetime | None = None) -> bool:
    if interval_s <= 0:
        return False
    if now is None:
        now = utc_now()
    if not marker.exists():
        return True
    try:
        last = datetime.fromisoformat(marker.read_text().strip())
    except ValueError:
        return True
    return (now - last).total_seconds() >= interval_s


def mark_run(marker: Path, *, now: datetime | None = None) -> None:
    if now is None:
        now = utc_now()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(now.isoformat())


def _subprocess_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def operation_status_row(
    operation: str,
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "ok": returncode == 0,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command or [],
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def run_command(
    operation: str, command: list[str], *, timeout_s: int
) -> dict[str, Any]:
    started_at = utc_now().isoformat()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return operation_status_row(
            operation,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at=started_at,
            finished_at=utc_now().isoformat(),
            command=command,
        )
    except subprocess.TimeoutExpired as exc:
        return operation_status_row(
            operation,
            returncode=124,
            stdout=_subprocess_output_to_text(exc.stdout),
            stderr=_subprocess_output_to_text(exc.stderr)
            + f"\ntimeout after {timeout_s}s",
            started_at=started_at,
            finished_at=utc_now().isoformat(),
            command=command,
        )


def run_command_stdout_to_file(
    operation: str, command: list[str], *, stdout_path: Path, timeout_s: int
) -> dict[str, Any]:
    """Run a command whose stdout may be large, preserving stdout in a file."""

    started_at = utc_now().isoformat()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr = ""
    try:
        with stdout_path.open("w") as out:
            completed = subprocess.run(
                command,
                check=False,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        stdout_summary = f"wrote {stdout_path.stat().st_size} bytes to {stdout_path}"
        return operation_status_row(
            operation,
            returncode=completed.returncode,
            stdout=stdout_summary,
            stderr=completed.stderr,
            started_at=started_at,
            finished_at=utc_now().isoformat(),
            command=command,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _subprocess_output_to_text(exc.stderr)
        return operation_status_row(
            operation,
            returncode=124,
            stdout=f"partial stdout retained at {stdout_path}",
            stderr=stderr + f"\ntimeout after {timeout_s}s",
            started_at=started_at,
            finished_at=utc_now().isoformat(),
            command=command,
        )


def write_watchdog_summary(
    out_dir: Path, rows: list[dict[str, Any]], *, config: WatchdogConfig
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = all(row["ok"] for row in rows)
    status = {
        "ok": ok,
        "checked_at": utc_now().isoformat(),
        "config": {k: str(v) for k, v in asdict(config).items()},
        "operations": rows,
    }
    (out_dir / "latest_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Pacifica Fly Ops Watchdogs",
        "",
        f"Checked at: {status['checked_at']}",
        f"Verdict: {'OK' if ok else 'CHECK_FAILURE'}",
        "",
        "These are Fly-side operational watchdogs. They are read-only/non-destructive and are meant to keep API-surface detection and R2 retention/cold-compaction policy logic close to the always-on archival appliance.",
        "",
        "## Operations",
        "",
        "| operation | ok | returncode |",
        "| --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['operation']} | {row['ok']} | {row['returncode']} |")
    lines.extend(
        [
            "",
            "## Output locations",
            "",
            f"- API surface watch: `{config.root / 'pacifica-api-surface-watch'}`",
            f"- R2 retention policy reports: `{config.root / 'pacifica-r2-retention'}`",
            f"- R2 inventory CSV: `{config.root / 'r2_inventory.csv'}`",
            "",
            "Remote R2 raw deletion is not performed here. Retention reports are planning artifacts only; destructive expiry still requires compacted archive verification and explicit review.",
            "",
        ]
    )
    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines))
    return readme


def run_once(config: WatchdogConfig) -> int:
    config.root.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    python = sys.executable

    api_marker = config.state_dir / "api_surface_last_run.txt"
    if is_due(api_marker, interval_s=config.api_surface_interval_s):
        api_out = config.root / "pacifica-api-surface-watch"
        cmd = [
            python,
            "scripts/watch_pacifica_api_surface.py",
            "--out-dir",
            str(api_out),
            "--timeout-s",
            "20",
            "--fail-on-change",
        ]
        row = run_command("api_surface_watch", cmd, timeout_s=config.command_timeout_s)
        rows.append(row)
        mark_run(api_marker)

    retention_marker = config.state_dir / "r2_retention_last_run.txt"
    if is_due(retention_marker, interval_s=config.r2_retention_interval_s):
        lsjson = config.root / "r2_inventory.lsjson"
        inventory_csv = config.root / "r2_inventory.csv"
        retention_out = config.root / "pacifica-r2-retention"
        remote_path = f"{config.remote_base}/{config.r2_prefix}"
        cmd_inventory = [
            "rclone",
            "lsjson",
            remote_path,
            "--recursive",
            "--files-only",
        ]
        inventory_row = run_command_stdout_to_file(
            "r2_inventory_lsjson",
            cmd_inventory,
            stdout_path=lsjson,
            timeout_s=config.command_timeout_s,
        )
        if inventory_row["ok"]:
            convert_row = run_command(
                "r2_inventory_csv",
                [
                    python,
                    "scripts/pacifica_r2_inventory.py",
                    "--lsjson",
                    str(lsjson),
                    "--out-csv",
                    str(inventory_csv),
                ],
                timeout_s=config.command_timeout_s,
            )
            rows.append(convert_row)
            if convert_row["ok"]:
                rows.append(
                    run_command(
                        "r2_retention_plan",
                        [
                            python,
                            "scripts/plan_pacifica_r2_retention.py",
                            "--inventory-csv",
                            str(inventory_csv),
                            "--out-dir",
                            str(retention_out),
                        ],
                        timeout_s=config.command_timeout_s,
                    )
                )
        rows.insert(0, inventory_row)
        mark_run(retention_marker)

    if not rows:
        rows.append(
            operation_status_row(
                "noop_not_due",
                returncode=0,
                stdout="No watchdog operation due.",
                stderr="",
            )
        )

    summary_dir = config.root / "watchdogs"
    write_watchdog_summary(summary_dir, rows, config=config)

    if config.upload_reports:
        reports_remote = (
            f"{config.remote_base}/{config.reports_prefix}/watchdogs/latest"
        )
        rows.append(
            run_command(
                "upload_watchdog_reports",
                ["rclone", "copy", str(config.root), reports_remote],
                timeout_s=config.command_timeout_s,
            )
        )
        write_watchdog_summary(summary_dir, rows, config=config)

    return 0 if all(row["ok"] for row in rows) else 2


def config_from_env() -> WatchdogConfig:
    root = Path(os.environ.get("PACIFICA_OPS_ROOT", "/data/ops"))
    return WatchdogConfig(
        root=root,
        state_dir=Path(os.environ.get("PACIFICA_OPS_STATE_DIR", str(root / ".state"))),
        remote_base=os.environ.get(
            "PACIFICA_FULL_FIDELITY_REMOTE_BASE", "r2:pacifica-trading-data"
        ),
        r2_prefix=os.environ.get(
            "PACIFICA_FULL_FIDELITY_R2_PREFIX", "raw/pacifica/full_fidelity"
        ),
        reports_prefix=os.environ.get(
            "PACIFICA_OPS_R2_PREFIX", "ops/pacifica/full_fidelity"
        ),
        api_surface_interval_s=int(
            os.environ.get("PACIFICA_API_SURFACE_WATCH_INTERVAL_S", "86400")
        ),
        r2_retention_interval_s=int(
            os.environ.get("PACIFICA_R2_RETENTION_PLAN_INTERVAL_S", "86400")
        ),
        command_timeout_s=int(os.environ.get("PACIFICA_OPS_COMMAND_TIMEOUT_S", "1800")),
        upload_reports=os.environ.get("PACIFICA_OPS_UPLOAD_REPORTS", "1") == "1",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="Run due watchdogs once and exit."
    )
    return parser


def main() -> int:
    build_arg_parser().parse_args()
    return run_once(config_from_env())


if __name__ == "__main__":
    sys.exit(main())

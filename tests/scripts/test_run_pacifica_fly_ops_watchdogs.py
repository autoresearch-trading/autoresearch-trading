import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.run_pacifica_fly_ops_watchdogs import (
    WatchdogConfig,
    is_due,
    operation_status_row,
    run_command_stdout_to_file,
    run_once,
    should_retry_transient_sidecar_lag,
    write_watchdog_summary,
)


def test_is_due_when_marker_missing_or_old(tmp_path):
    marker = tmp_path / "last_run"
    now = datetime(2026, 5, 2, 12, tzinfo=UTC)

    assert is_due(marker, interval_s=3600, now=now) is True

    marker.write_text((now - timedelta(seconds=3599)).isoformat())
    assert is_due(marker, interval_s=3600, now=now) is False

    marker.write_text((now - timedelta(seconds=3600)).isoformat())
    assert is_due(marker, interval_s=3600, now=now) is True


def test_operation_status_row_records_success_and_failure():
    ok = operation_status_row("api_surface_watch", returncode=0, stdout="ok", stderr="")
    bad = operation_status_row(
        "r2_retention_plan", returncode=2, stdout="", stderr="boom"
    )

    assert ok["ok"] is True
    assert ok["operation"] == "api_surface_watch"
    assert bad["ok"] is False
    assert bad["stderr_tail"] == "boom"


def test_write_watchdog_summary_creates_json_and_markdown(tmp_path):
    rows = [
        operation_status_row(
            "api_surface_watch", returncode=0, stdout="unchanged", stderr=""
        ),
        operation_status_row(
            "r2_retention_plan", returncode=2, stdout="", stderr="failed"
        ),
    ]
    config = WatchdogConfig(
        root=tmp_path, remote_base="r2:bucket", r2_prefix="raw/pacifica/full_fidelity"
    )

    readme = write_watchdog_summary(tmp_path, rows, config=config)

    assert readme == tmp_path / "README.md"
    assert "Pacifica Fly Ops Watchdogs" in readme.read_text()
    assert "r2_retention_plan" in readme.read_text()
    status = json.loads((tmp_path / "latest_status.json").read_text())
    assert status["ok"] is False
    assert len(status["operations"]) == 2


def test_run_command_stdout_to_file_preserves_full_stdout(tmp_path):
    out = tmp_path / "large_stdout.txt"
    row = run_command_stdout_to_file(
        "large_stdout",
        ["python", "-c", "print('x' * 12000)"],
        stdout_path=out,
        timeout_s=10,
    )

    assert row["ok"] is True
    assert row["stdout_tail"].startswith("wrote ")
    assert len(out.read_text().strip()) == 12000


def test_run_command_stdout_to_file_handles_timeout_bytes_stderr(tmp_path, monkeypatch):
    out = tmp_path / "partial_stdout.txt"

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["rclone", "lsjson"],
            timeout=1800,
            output=b"partial inventory",
            stderr=b"remote stalled",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    row = run_command_stdout_to_file(
        "r2_inventory_lsjson",
        ["rclone", "lsjson"],
        stdout_path=out,
        timeout_s=1800,
    )

    assert row["ok"] is False
    assert row["returncode"] == 124
    assert "partial stdout retained" in row["stdout_tail"]
    assert "remote stalled" in row["stderr_tail"]
    assert "timeout after 1800s" in row["stderr_tail"]


def test_run_once_uses_line_oriented_r2_inventory_and_converts_it(
    tmp_path, monkeypatch
):
    commands: list[list[str]] = []

    def fake_run(operation, command, *, timeout_s):
        commands.append(command)
        return operation_status_row(
            operation, returncode=0, stdout="ok", stderr="", command=command
        )

    def fake_stdout_to_file(operation, command, *, stdout_path, timeout_s):
        commands.append(command)
        stdout_path.write_text(
            "b.jsonl.gz;2;2026-05-01 00:00:02\na.jsonl.gz;1;2026-05-01 00:00:01\n"
        )
        return operation_status_row(
            operation,
            returncode=0,
            stdout="inventory written",
            stderr="",
            command=command,
        )

    monkeypatch.setattr("scripts.run_pacifica_fly_ops_watchdogs.run_command", fake_run)
    monkeypatch.setattr(
        "scripts.run_pacifica_fly_ops_watchdogs.run_command_stdout_to_file",
        fake_stdout_to_file,
    )

    config = WatchdogConfig(
        root=tmp_path,
        state_dir=tmp_path / ".state",
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        api_surface_interval_s=0,
        r2_retention_interval_s=1,
        r2_freshness_interval_s=0,
        upload_reports=False,
    )

    assert run_once(config) == 0

    inventory_commands = [cmd for cmd in commands if cmd[:2] == ["rclone", "lsf"]]
    assert inventory_commands == [
        [
            "rclone",
            "lsf",
            "r2:bucket/raw/pacifica/full_fidelity",
            "--recursive",
            "--files-only",
            "--format",
            "pst",
            "--separator",
            ";",
        ]
    ]
    assert not any("lsjson" in cmd for command in commands for cmd in command)
    assert (tmp_path / "r2_inventory.csv").exists()


def test_run_once_runs_bounded_r2_freshness_check(tmp_path, monkeypatch):
    commands: list[tuple[str, list[str]]] = []

    def fake_run(operation, command, *, timeout_s):
        commands.append((operation, command))
        if operation == "r2_freshness_check":
            out_path = Path(command[command.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"ok": True, "failures": []}))
        return operation_status_row(
            operation, returncode=0, stdout="ok", stderr="", command=command
        )

    monkeypatch.setattr("scripts.run_pacifica_fly_ops_watchdogs.run_command", fake_run)

    config = WatchdogConfig(
        root=tmp_path,
        state_dir=tmp_path / ".state",
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        api_surface_interval_s=0,
        r2_retention_interval_s=0,
        r2_freshness_interval_s=1,
        upload_reports=False,
    )

    assert run_once(config) == 0

    freshness_commands = [
        cmd for operation, cmd in commands if operation == "r2_freshness_check"
    ]
    assert len(freshness_commands) == 1
    command = freshness_commands[0]
    assert "scripts/check_pacifica_r2_freshness.py" in command
    assert "--remote-base" in command
    assert "r2:bucket" in command
    assert (tmp_path / "pacifica-r2-freshness" / "latest_status.json").exists()


def test_run_once_retries_transient_sidecar_lag_before_failing_watchdog(
    tmp_path, monkeypatch
):
    commands: list[tuple[str, list[str]]] = []
    responses = [
        {
            "ok": False,
            "failures": ["R2_SIDECAR_MISSING"],
            "latest_payload_age_min": 130.0,
            "sidecar_missing_count": 4,
            "listing_errors": [],
        },
        {
            "ok": True,
            "failures": [],
            "latest_payload_age_min": 136.0,
            "sidecar_missing_count": 0,
            "listing_errors": [],
        },
    ]

    def fake_run(operation, command, *, timeout_s):
        commands.append((operation, command))
        if operation == "r2_freshness_check":
            status = responses.pop(0)
            out_path = Path(command[command.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(status))
            return operation_status_row(
                operation,
                returncode=0 if status["ok"] else 2,
                stdout=json.dumps(status),
                stderr="",
                command=command,
            )
        return operation_status_row(
            operation, returncode=0, stdout="ok", stderr="", command=command
        )

    monkeypatch.setattr("scripts.run_pacifica_fly_ops_watchdogs.run_command", fake_run)

    config = WatchdogConfig(
        root=tmp_path,
        state_dir=tmp_path / ".state",
        remote_base="r2:bucket",
        r2_prefix="raw/pacifica/full_fidelity",
        api_surface_interval_s=0,
        r2_retention_interval_s=0,
        r2_freshness_interval_s=1,
        r2_freshness_sidecar_retry_attempts=1,
        r2_freshness_sidecar_retry_delay_s=0,
        upload_reports=False,
    )

    assert run_once(config) == 0

    freshness_commands = [
        cmd for operation, cmd in commands if operation == "r2_freshness_check"
    ]
    assert len(freshness_commands) == 2
    status = json.loads((tmp_path / "watchdogs" / "latest_status.json").read_text())
    assert status["ok"] is True
    assert status["operations"][0]["ok"] is True
    assert "retry" in status["operations"][0]["stderr_tail"]


def test_transient_sidecar_retry_does_not_mask_real_freshness_failures(tmp_path):
    def row_for(status: dict[str, object]) -> dict[str, object]:
        out_path = tmp_path / "freshness.json"
        out_path.write_text(json.dumps(status))
        return operation_status_row(
            "r2_freshness_check",
            returncode=2,
            stdout=json.dumps(status),
            stderr="",
            command=["python", "check", "--out", str(out_path)],
        )

    non_retryable_statuses = [
        {
            "failures": ["R2_REMOTE_FRESHNESS_STALE"],
            "latest_payload_age_min": 181.0,
            "sidecar_missing_count": 0,
            "listing_errors": [],
        },
        {
            "failures": ["R2_SIDECAR_MISSING", "R2_REMOTE_FRESHNESS_STALE"],
            "latest_payload_age_min": 181.0,
            "sidecar_missing_count": 4,
            "listing_errors": [],
        },
        {
            "failures": ["R2_SIDECAR_MISSING"],
            "latest_payload_age_min": 130.0,
            "sidecar_missing_count": 4,
            "listing_errors": [{"sample_prefix": "bad"}],
        },
        {
            "failures": ["R2_SIDECAR_MISSING"],
            "latest_payload_age_min": None,
            "sidecar_missing_count": 4,
            "listing_errors": [],
        },
        {
            "failures": ["R2_SIDECAR_MISSING"],
            "latest_payload_age_min": 130.0,
            "sidecar_missing_count": 0,
            "listing_errors": [],
        },
    ]

    for status in non_retryable_statuses:
        assert (
            should_retry_transient_sidecar_lag(row_for(status), stale_after_min=180.0)
            is False
        )

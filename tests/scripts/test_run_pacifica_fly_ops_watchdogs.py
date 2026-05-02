import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.run_pacifica_fly_ops_watchdogs import (
    WatchdogConfig,
    is_due,
    operation_status_row,
    run_command_stdout_to_file,
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

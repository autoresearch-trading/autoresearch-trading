import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_pacifica_full_fidelity_r2_lifecycle.sh"


def _write_fake_python(bin_dir: Path) -> Path:
    fake = bin_dir / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$PACIFICA_TEST_COMMAND_LOG"\n'
        'case "$*" in\n'
        "  *scan*) echo '{\"scan\":0}' ;;\n"
        '  *upload-batch*) echo \'{"uploaded":0,"failed":0}\' ;;\n'
        '  *repair-sidecars*) echo \'{"sidecars_uploaded":0,"failed":0}\' ;;\n'
        "  *reset-mismatch-errors*) echo '{\"reset\":0}' ;;\n"
        '  *upload-verify*) echo \'{"upload":{"uploaded":0},"verify":{"verified":0}}\' ;;\n'
        '  *prune*) echo \'{"deleted":0,"dry_run":true}\' ;;\n'
        "  *) echo '{}' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _run_lifecycle(tmp_path: Path, *, env_overrides: dict[str, str]) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir)
    log_path = tmp_path / "commands.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PACIFICA_USE_SYSTEM_PYTHON": "1",
            "PACIFICA_TEST_COMMAND_LOG": str(log_path),
            "PACIFICA_FULL_FIDELITY_ROOT": str(tmp_path / "raw"),
            "PACIFICA_FULL_FIDELITY_STATE_DB": str(tmp_path / "state.sqlite"),
            "PACIFICA_FULL_FIDELITY_LIFECYCLE_STATE_DIR": str(tmp_path / ".lifecycle"),
            "PACIFICA_FULL_FIDELITY_RECENT_SCAN_HOURS": "12",
            "PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_LIMIT": "10",
            "PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S": "0",
            "PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT": "5",
            "PACIFICA_FULL_FIDELITY_VERIFY_LIMIT": "5",
            "PACIFICA_R2_PRUNE_EXECUTE": "0",
        }
    )
    env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return log_path.read_text(encoding="utf-8").splitlines()


def test_lifecycle_skips_safety_lane_when_backlog_interval_not_due(tmp_path):
    state_dir = tmp_path / ".lifecycle"
    state_dir.mkdir()
    (state_dir / "backlog_lane_last_run_epoch.txt").write_text("4102444800")

    commands = _run_lifecycle(
        tmp_path,
        env_overrides={"PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S": "3600"},
    )

    assert any("scan --skip-current-hour --recent-hours 12" in cmd for cmd in commands)
    assert any("upload-batch" in cmd for cmd in commands)
    assert any("repair-sidecars" in cmd for cmd in commands)
    assert not any("reset-mismatch-errors" in cmd for cmd in commands)
    assert not any("upload-verify" in cmd for cmd in commands)
    assert not any("prune" in cmd for cmd in commands)


def test_lifecycle_skips_full_scan_when_safety_lane_not_due(tmp_path):
    state_dir = tmp_path / ".lifecycle"
    state_dir.mkdir()
    (state_dir / "backlog_lane_last_run_epoch.txt").write_text("4102444800")

    commands = _run_lifecycle(
        tmp_path,
        env_overrides={
            "PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S": "3600",
            "PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S": "3600",
        },
    )

    broad_scans = [
        cmd for cmd in commands if " scan" in cmd and "--recent-hours" not in cmd
    ]
    assert broad_scans == []
    upload_index = next(i for i, cmd in enumerate(commands) if "upload-batch" in cmd)
    repair_index = next(i for i, cmd in enumerate(commands) if "repair-sidecars" in cmd)
    assert upload_index < repair_index
    assert not any("upload-verify" in cmd for cmd in commands)


def test_lifecycle_runs_and_marks_safety_lane_when_backlog_interval_due(tmp_path):
    commands = _run_lifecycle(
        tmp_path,
        env_overrides={"PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S": "3600"},
    )

    assert any("upload-batch" in cmd for cmd in commands)
    assert any("reset-mismatch-errors" in cmd for cmd in commands)
    assert any("upload-verify" in cmd for cmd in commands)
    assert any("prune" in cmd for cmd in commands)
    marker = tmp_path / ".lifecycle" / "backlog_lane_last_run_epoch.txt"
    assert marker.exists()
    assert marker.read_text().strip().isdigit()


def test_lifecycle_runs_post_safety_fresh_catchup_before_returning(tmp_path):
    commands = _run_lifecycle(
        tmp_path,
        env_overrides={"PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S": "3600"},
    )

    fresh_scans = [
        i
        for i, cmd in enumerate(commands)
        if "scan --skip-current-hour --recent-hours 12" in cmd
    ]
    fresh_uploads = [i for i, cmd in enumerate(commands) if "upload-batch" in cmd]
    sidecar_repairs = [i for i, cmd in enumerate(commands) if "repair-sidecars" in cmd]
    prune_index = next(i for i, cmd in enumerate(commands) if "prune" in cmd)

    assert len(fresh_scans) == 2
    assert len(fresh_uploads) == 2
    assert len(sidecar_repairs) == 2
    assert fresh_scans[0] < fresh_uploads[0] < sidecar_repairs[0] < prune_index
    assert prune_index < fresh_scans[1] < fresh_uploads[1] < sidecar_repairs[1]

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
    assert not any("reset-mismatch-errors" in cmd for cmd in commands)
    assert not any("upload-verify" in cmd for cmd in commands)
    assert not any("prune" in cmd for cmd in commands)


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

from pathlib import Path

import pandas as pd

from scripts.run_pacifica_walk_forward_validation import (
    WalkForwardConfig,
    build_purged_walk_forward_windows,
    evaluate_walk_forward_validation,
    write_walk_forward_report,
)


def _validation_rows(days: int = 32, rows_per_day: int = 2) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for day in range(days):
        for slot in range(rows_per_day):
            ts = start + pd.Timedelta(days=day, minutes=slot)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "BTC" if slot % 2 == 0 else "ETH",
                    "strategy_return_bps": 4.0,
                    "baseline_return_bps": 0.0,
                    "eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_build_purged_walk_forward_windows_are_chronological_and_gap_purged() -> None:
    frame = _validation_rows(days=12, rows_per_day=1)
    config = WalkForwardConfig(train_days=4, test_days=2, purge_days=1, step_days=2)

    windows = build_purged_walk_forward_windows(frame, config=config)

    assert len(windows) == 3
    first = windows.iloc[0]
    assert first["window_id"] == 1
    assert first["train_start"] == "2026-01-01"
    assert first["train_end"] == "2026-01-04"
    assert first["purge_start"] == "2026-01-05"
    assert first["purge_end"] == "2026-01-05"
    assert first["test_start"] == "2026-01-06"
    assert first["test_end"] == "2026-01-07"
    assert first["train_rows"] == 4
    assert first["test_rows"] == 2
    assert first["purged_rows"] == 1


def test_evaluate_walk_forward_validation_fails_young_archive_as_diagnostic() -> None:
    frame = _validation_rows(days=8, rows_per_day=2)
    result = evaluate_walk_forward_validation(
        frame, config=WalkForwardConfig(min_diagnostic_days=10)
    )

    assert result.verdict == "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    assert "insufficient_distinct_days" in result.failure_reasons
    assert result.summary["distinct_days"] == 8
    assert result.summary["validation_windows"] == 0


def test_evaluate_walk_forward_validation_requires_concentration_and_baseline_edges() -> (
    None
):
    concentrated = _validation_rows(days=35, rows_per_day=1)
    concentrated.loc[:, "timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z")

    concentration_result = evaluate_walk_forward_validation(
        concentrated,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            max_day_concentration=0.25,
        ),
    )

    assert concentration_result.verdict == "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    assert "insufficient_distinct_days" in concentration_result.failure_reasons

    frame = _validation_rows(days=35, rows_per_day=2)
    frame["strategy_return_bps"] = -1.0
    frame["baseline_return_bps"] = 0.0
    edge_result = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
        ),
    )

    assert edge_result.verdict == "PROVISIONAL_FAIL"
    assert "nonpositive_post_cost_pnl" in edge_result.failure_reasons
    assert "baseline_not_beaten" in edge_result.failure_reasons


def test_evaluate_walk_forward_validation_adds_random_same_frequency_controls() -> None:
    frame = _validation_rows(days=40, rows_per_day=2)
    result = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            min_validation_grade_days=60,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            random_control_trials=5,
            random_seed=7,
        ),
    )

    assert result.verdict == "PROVISIONAL_PASS"
    assert result.random_controls.shape[0] == 5
    assert set(result.random_controls.columns).issuperset(
        {"control_id", "sampled_rows", "net_pnl_bps", "sortino"}
    )
    assert result.summary["random_controls_beaten_rate"] == 1.0
    assert result.summary["validation_windows"] > 0


def test_sixty_plus_day_good_result_is_validation_grade_pass() -> None:
    frame = _validation_rows(days=75, rows_per_day=2)

    result = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            min_validation_grade_days=60,
            train_days=10,
            test_days=5,
            purge_days=1,
            step_days=5,
            random_control_trials=3,
        ),
    )

    assert result.verdict == "VALIDATION_GRADE_PASS"
    assert result.summary["distinct_days"] == 75
    assert result.summary["distinct_oos_days"] >= 60


def test_validation_cannot_pass_without_random_controls_or_purge_gap() -> None:
    frame = _validation_rows(days=35, rows_per_day=2)

    no_controls = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            random_control_trials=0,
        ),
    )
    no_purge = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=0,
            step_days=2,
            random_control_trials=3,
        ),
    )

    assert no_controls.verdict == "PROVISIONAL_FAIL"
    assert "random_controls_missing" in no_controls.failure_reasons
    assert no_purge.verdict == "PROVISIONAL_FAIL"
    assert "no_purge_gap" in no_purge.failure_reasons


def test_validation_rejects_dirty_inputs_instead_of_silently_dropping_them() -> None:
    frame = _validation_rows(days=35, rows_per_day=2)
    frame["timestamp"] = frame["timestamp"].astype(object)
    frame["baseline_return_bps"] = frame["baseline_return_bps"].astype(object)
    frame.loc[0, "strategy_return_bps"] = float("inf")
    frame.loc[1, "baseline_return_bps"] = "bad"
    frame.loc[2, "timestamp"] = "not-a-time"
    frame.loc[3, "symbol"] = None

    result = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            random_control_trials=3,
        ),
    )

    assert result.verdict == "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    assert "invalid_required_fields" in result.failure_reasons
    assert result.summary["invalid_required_rows"] == 4


def test_string_false_eligible_rows_are_excluded_not_treated_as_true() -> None:
    frame = _validation_rows(days=35, rows_per_day=2)
    frame["eligible"] = "False"

    result = evaluate_walk_forward_validation(
        frame,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            random_control_trials=3,
        ),
    )

    assert result.verdict == "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    assert result.summary["observations"] == 0
    assert result.summary["filtered_ineligible_rows"] == 70


def test_cli_only_allows_insufficient_sample_diagnostic_not_real_failures(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    input_path = tmp_path / "bad_strategy.csv"
    frame = _validation_rows(days=35, rows_per_day=2)
    frame["strategy_return_bps"] = -1.0
    frame.to_csv(input_path, index=False)
    command = [
        sys.executable,
        "scripts/run_pacifica_walk_forward_validation.py",
        "--input",
        str(input_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--allow-fail-diagnostic",
    ]

    failed = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], check=False
    )

    assert failed.returncode == 1


def test_window_day_concentration_blocks_default_validation_grade_pass() -> None:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for day in range(90):
        rows_per_day = 100 if day == 30 else 10
        for slot in range(rows_per_day):
            rows.append(
                {
                    "timestamp": start + pd.Timedelta(days=day, minutes=slot),
                    "symbol": "BTC" if slot % 2 == 0 else "ETH",
                    "strategy_return_bps": 4.0,
                    "baseline_return_bps": 0.0,
                    "eligible": True,
                }
            )
    result = evaluate_walk_forward_validation(
        pd.DataFrame(rows), config=WalkForwardConfig()
    )

    assert result.verdict == "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    assert "window_day_concentration_too_high" in result.failure_reasons
    assert (
        result.summary["max_window_day_concentration"]
        > WalkForwardConfig().max_day_concentration
    )


def test_cli_allow_diagnostic_still_rejects_invalid_required_fields(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    input_path = tmp_path / "dirty.csv"
    frame = _validation_rows(days=8, rows_per_day=2)
    frame.loc[0, "strategy_return_bps"] = float("inf")
    frame.to_csv(input_path, index=False)
    command = [
        sys.executable,
        "scripts/run_pacifica_walk_forward_validation.py",
        "--input",
        str(input_path),
        "--out-dir",
        str(tmp_path / "dirty_out"),
        "--allow-fail-diagnostic",
    ]

    failed = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], check=False
    )

    assert failed.returncode == 1


def test_write_walk_forward_report_creates_readme_and_csvs(tmp_path: Path) -> None:
    input_path = tmp_path / "strategy_returns.csv"
    _validation_rows(days=40, rows_per_day=2).to_csv(input_path, index=False)
    out_dir = tmp_path / "walk_forward"

    result = write_walk_forward_report(
        input_path,
        out_dir,
        config=WalkForwardConfig(
            min_diagnostic_days=10,
            min_provisional_days=30,
            train_days=5,
            test_days=2,
            purge_days=1,
            step_days=2,
            random_control_trials=3,
        ),
    )

    assert result["verdict"] == "PROVISIONAL_PASS"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "windows.csv").exists()
    assert (out_dir / "window_scorecard.csv").exists()
    assert (out_dir / "random_controls.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Pacifica Walk-Forward Validation" in report
    assert "purged chronological windows" in report
    assert "random same-frequency controls" in report
    assert "Do not treat this as an edge claim" in report

import json
from pathlib import Path

from scripts.plan_pacifica_ops_alerts import (
    AlertThresholds,
    evaluate_alerts,
    overall_severity,
    write_alert_plan_report,
)


def _healthy_snapshot() -> dict:
    return {
        "checked_at": "2026-05-08T12:00:00+00:00",
        "fly_state": "started",
        "collector_newest_raw_age_min": 4.0,
        "free_gb": 54.2,
        "rows_with_errors": 0,
        "lifecycle_upload_failed": 0,
        "lifecycle_verify_failed": 0,
        "r2_raw_present": True,
        "r2_latest_remote_age_min": 8.0,
        "r2_sidecar_mismatch_count": 0,
        "watchdog_status_age_min": 20.0,
        "watchdog_ok": True,
        "api_surface_changed": False,
        "archive_inventory_age_hours": 6.0,
        "archive_inventory_timed_out": False,
        "research_refresh_ok": True,
        "delivery_channels": ["origin_chat"],
    }


def test_evaluate_alerts_returns_ok_for_healthy_snapshot() -> None:
    alerts = evaluate_alerts(_healthy_snapshot())

    assert overall_severity(alerts) == "OK"
    assert set(alerts["severity"]) == {"OK"}
    assert "FREE_DISK_LOW" in set(alerts["condition"])
    assert (
        alerts.loc[
            alerts["condition"] == "DELIVERY_CHANNEL_CONFIGURED", "severity"
        ].iloc[0]
        == "OK"
    )


def test_page_alerts_for_stopped_app_stale_raw_low_disk_and_lifecycle_failures() -> (
    None
):
    snapshot = _healthy_snapshot()
    snapshot.update(
        {
            "fly_state": "stopped",
            "collector_newest_raw_age_min": 30.0,
            "free_gb": 49.5,
            "lifecycle_upload_failed": 2,
            "lifecycle_verify_failed": 1,
            "rows_with_errors": 3,
        }
    )

    alerts = evaluate_alerts(snapshot)
    pages = set(alerts.loc[alerts["severity"] == "PAGE", "condition"])

    assert overall_severity(alerts) == "PAGE"
    assert {
        "FLY_APP_STARTED",
        "RAW_FRESHNESS",
        "FREE_DISK_LOW",
        "LIFECYCLE_UPLOAD_FAILURES",
        "LIFECYCLE_VERIFY_FAILURES",
        "LIFECYCLE_DB_ERRORS",
    }.issubset(pages)


def test_r2_watchdog_api_inventory_and_research_conditions_are_classified() -> None:
    snapshot = _healthy_snapshot()
    snapshot.update(
        {
            "r2_raw_present": False,
            "r2_latest_remote_age_min": 75.0,
            "r2_sidecar_mismatch_count": 4,
            "watchdog_status_age_min": 240.0,
            "watchdog_ok": False,
            "api_surface_changed": True,
            "archive_inventory_age_hours": 36.0,
            "research_refresh_ok": False,
            "delivery_channels": [],
        }
    )

    alerts = evaluate_alerts(snapshot)
    by_condition = alerts.set_index("condition")

    assert by_condition.loc["R2_RAW_PREFIX_PRESENT", "severity"] == "PAGE"
    assert by_condition.loc["R2_REMOTE_FRESHNESS", "severity"] == "PAGE"
    assert by_condition.loc["R2_SIDECAR_MISMATCH", "severity"] == "PAGE"
    assert by_condition.loc["WATCHDOG_STATUS_FRESH", "severity"] == "PAGE"
    assert by_condition.loc["API_SURFACE_CHANGED", "severity"] == "PAGE"
    assert by_condition.loc["ARCHIVE_INVENTORY_FRESH", "severity"] == "WARN"
    assert by_condition.loc["RESEARCH_REFRESH_OK", "severity"] == "WARN"
    assert by_condition.loc["DELIVERY_CHANNEL_CONFIGURED", "severity"] == "WARN"


def test_inventory_timeout_is_warn_not_page_when_other_health_is_ok() -> None:
    snapshot = _healthy_snapshot()
    snapshot.update(
        {"archive_inventory_timed_out": True, "archive_inventory_age_hours": 72.0}
    )

    alerts = evaluate_alerts(snapshot)
    inventory = alerts.set_index("condition").loc["ARCHIVE_INVENTORY_FRESH"]

    assert inventory["severity"] == "WARN"
    assert "timeout" in inventory["evidence"]
    assert overall_severity(alerts) == "WARN"


def test_missing_required_signals_fail_closed_to_page() -> None:
    snapshot = _healthy_snapshot()
    del snapshot["free_gb"]
    del snapshot["fly_state"]

    alerts = evaluate_alerts(snapshot)
    missing = alerts[alerts["condition"] == "MISSING_REQUIRED_SIGNALS"].iloc[0]

    assert missing["severity"] == "PAGE"
    assert "free_gb" in missing["evidence"]
    assert "fly_state" in missing["evidence"]
    assert overall_severity(alerts) == "PAGE"


def test_bad_types_fail_closed_to_page_not_silent_ok() -> None:
    snapshot = _healthy_snapshot()
    snapshot["free_gb"] = "fifty-four"
    snapshot["r2_raw_present"] = "yes"

    alerts = evaluate_alerts(snapshot)
    invalid = alerts[alerts["condition"] == "INVALID_REQUIRED_SIGNALS"].iloc[0]

    assert invalid["severity"] == "PAGE"
    assert "free_gb" in invalid["evidence"]
    assert "r2_raw_present" in invalid["evidence"]


def test_write_alert_plan_report_emits_markdown_csv_and_json(tmp_path: Path) -> None:
    snapshot = _healthy_snapshot()
    snapshot["free_gb"] = 49.0
    alerts = evaluate_alerts(
        snapshot, thresholds=AlertThresholds(free_disk_floor_gb=50.0)
    )

    result = write_alert_plan_report(alerts, tmp_path, snapshot=snapshot)

    assert result["overall_severity"] == "PAGE"
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "alert_plan.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert "Pacifica external ops alert plan" in (tmp_path / "README.md").read_text()
    assert "No external delivery credentials" in (tmp_path / "README.md").read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall_severity"] == "PAGE"

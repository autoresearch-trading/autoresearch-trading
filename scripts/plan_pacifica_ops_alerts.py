# scripts/plan_pacifica_ops_alerts.py
"""Plan external operational alerts for the Pacifica full-fidelity collector.

This script separates health-check facts from delivery configuration. It does not
send notifications or require delivery credentials; it classifies a supplied
status snapshot into OK/WARN/PAGE rows and writes diagnostic artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUT_DIR = Path("docs/ops/pacifica-alerting")
SEVERITY_ORDER = {"OK": 0, "WARN": 1, "PAGE": 2}
REQUIRED_SIGNALS = {
    "fly_state": str,
    "collector_newest_raw_age_min": (int, float),
    "free_gb": (int, float),
    "rows_with_errors": (int, float),
    "lifecycle_upload_failed": (int, float),
    "lifecycle_verify_failed": (int, float),
    "r2_raw_present": bool,
    "r2_latest_remote_age_min": (int, float),
    "r2_sidecar_mismatch_count": (int, float),
    "watchdog_status_age_min": (int, float),
    "watchdog_ok": bool,
    "api_surface_changed": bool,
    "archive_inventory_age_hours": (int, float),
    "archive_inventory_timed_out": bool,
    "research_refresh_ok": bool,
    "delivery_channels": list,
}


@dataclass(frozen=True)
class AlertThresholds:
    raw_stale_after_min: float = 15.0
    free_disk_floor_gb: float = 50.0
    r2_remote_stale_after_min: float = 60.0
    watchdog_stale_after_min: float = 180.0
    archive_inventory_stale_after_hours: float = 24.0


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "nan"
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


def _row(condition: str, severity: str, action: str, evidence: str) -> dict[str, str]:
    if severity not in SEVERITY_ORDER:
        raise ValueError(f"unknown severity: {severity}")
    return {
        "condition": condition,
        "severity": severity,
        "action": action,
        "evidence": evidence,
    }


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _missing_and_invalid(snapshot: dict[str, Any]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    for key, expected in REQUIRED_SIGNALS.items():
        if key not in snapshot:
            missing.append(key)
            continue
        value = snapshot[key]
        if expected == (int, float):
            if not _is_finite_number(value):
                invalid.append(key)
        elif expected is bool:
            if not isinstance(value, bool):
                invalid.append(key)
        elif expected is list:
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                invalid.append(key)
        elif expected is str:
            if not isinstance(value, str) or not value.strip():
                invalid.append(key)
        elif not isinstance(value, expected):
            invalid.append(key)
    return sorted(missing), sorted(invalid)


def _get_number(snapshot: dict[str, Any], key: str) -> float | None:
    value = snapshot.get(key)
    if _is_finite_number(value):
        return float(value)
    return None


def _get_bool(snapshot: dict[str, Any], key: str) -> bool | None:
    value = snapshot.get(key)
    return value if isinstance(value, bool) else None


def _numeric_count(snapshot: dict[str, Any], key: str) -> float | None:
    value = _get_number(snapshot, key)
    if value is None or value < 0 or value % 1 != 0:
        return None
    return value


def evaluate_alerts(
    snapshot: dict[str, Any], *, thresholds: AlertThresholds = AlertThresholds()
) -> pd.DataFrame:
    """Classify a health snapshot into OK/WARN/PAGE alert-plan rows."""
    rows: list[dict[str, str]] = []
    missing, invalid = _missing_and_invalid(snapshot)
    if missing:
        rows.append(
            _row(
                "MISSING_REQUIRED_SIGNALS",
                "PAGE",
                "Fix the status snapshot producer before trusting alert state.",
                ";".join(missing),
            )
        )
    # Counts must be nonnegative integers; generic type validation only proves numeric.
    for key in (
        "rows_with_errors",
        "lifecycle_upload_failed",
        "lifecycle_verify_failed",
        "r2_sidecar_mismatch_count",
    ):
        if key in snapshot and _numeric_count(snapshot, key) is None:
            invalid.append(key)
    invalid = sorted(set(invalid))
    if invalid:
        rows.append(
            _row(
                "INVALID_REQUIRED_SIGNALS",
                "PAGE",
                "Fix invalid signal types/ranges before trusting alert state.",
                ";".join(invalid),
            )
        )

    fly_state = snapshot.get("fly_state")
    if isinstance(fly_state, str) and fly_state.strip().lower() == "started":
        rows.append(
            _row("FLY_APP_STARTED", "OK", "No action.", f"fly_state={fly_state}")
        )
    elif fly_state is not None:
        rows.append(
            _row(
                "FLY_APP_STARTED",
                "PAGE",
                "Check Fly app/machine status immediately.",
                f"fly_state={fly_state}",
            )
        )

    raw_age = _get_number(snapshot, "collector_newest_raw_age_min")
    if raw_age is not None:
        severity = "PAGE" if raw_age > thresholds.raw_stale_after_min else "OK"
        rows.append(
            _row(
                "RAW_FRESHNESS",
                severity,
                (
                    "Investigate collector/WebSocket/lifecycle if stale."
                    if severity == "PAGE"
                    else "No action."
                ),
                f"newest_raw_age_min={raw_age:.2f};threshold={thresholds.raw_stale_after_min:.2f}",
            )
        )

    free_gb = _get_number(snapshot, "free_gb")
    if free_gb is not None:
        severity = "PAGE" if free_gb < thresholds.free_disk_floor_gb else "OK"
        rows.append(
            _row(
                "FREE_DISK_LOW",
                severity,
                (
                    "Free disk below Diego's 50 GiB floor; inspect pruning/upload backlog."
                    if severity == "PAGE"
                    else "No action."
                ),
                f"free_gb={free_gb:.2f};floor={thresholds.free_disk_floor_gb:.2f}",
            )
        )

    for key, condition, action in [
        (
            "rows_with_errors",
            "LIFECYCLE_DB_ERRORS",
            "Inspect archive_files.error and repair/verify cycle.",
        ),
        (
            "lifecycle_upload_failed",
            "LIFECYCLE_UPLOAD_FAILURES",
            "Inspect upload logs and R2 credentials/connectivity.",
        ),
        (
            "lifecycle_verify_failed",
            "LIFECYCLE_VERIFY_FAILURES",
            "Inspect verify logs; distinguish active-hour from persistent mismatches.",
        ),
        (
            "r2_sidecar_mismatch_count",
            "R2_SIDECAR_MISMATCH",
            "Inspect affected R2 payload/sidecar pairs before retention work.",
        ),
    ]:
        count = _numeric_count(snapshot, key)
        if count is not None:
            severity = "PAGE" if count > 0 else "OK"
            rows.append(
                _row(
                    condition,
                    severity,
                    action if severity == "PAGE" else "No action.",
                    f"{key}={int(count)}",
                )
            )

    r2_raw_present = _get_bool(snapshot, "r2_raw_present")
    if r2_raw_present is not None:
        rows.append(
            _row(
                "R2_RAW_PREFIX_PRESENT",
                "OK" if r2_raw_present else "PAGE",
                (
                    "Raw prefix missing; stop destructive cleanup and inspect R2 remote."
                    if not r2_raw_present
                    else "No action."
                ),
                f"r2_raw_present={r2_raw_present}",
            )
        )

    remote_age = _get_number(snapshot, "r2_latest_remote_age_min")
    if remote_age is not None:
        severity = "PAGE" if remote_age > thresholds.r2_remote_stale_after_min else "OK"
        rows.append(
            _row(
                "R2_REMOTE_FRESHNESS",
                severity,
                (
                    "Investigate lifecycle upload/verify path if stale."
                    if severity == "PAGE"
                    else "No action."
                ),
                f"r2_latest_remote_age_min={remote_age:.2f};threshold={thresholds.r2_remote_stale_after_min:.2f}",
            )
        )

    watchdog_age = _get_number(snapshot, "watchdog_status_age_min")
    watchdog_ok = _get_bool(snapshot, "watchdog_ok")
    if watchdog_age is not None and watchdog_ok is not None:
        severity = (
            "PAGE"
            if watchdog_age > thresholds.watchdog_stale_after_min or not watchdog_ok
            else "OK"
        )
        rows.append(
            _row(
                "WATCHDOG_STATUS_FRESH",
                severity,
                (
                    "Inspect Fly-side ops watchdog status artifact and logs."
                    if severity == "PAGE"
                    else "No action."
                ),
                f"watchdog_status_age_min={watchdog_age:.2f};watchdog_ok={watchdog_ok}",
            )
        )

    api_changed = _get_bool(snapshot, "api_surface_changed")
    if api_changed is not None:
        rows.append(
            _row(
                "API_SURFACE_CHANGED",
                "PAGE" if api_changed else "OK",
                (
                    "Manually review provider API/docs before collector/baseline changes."
                    if api_changed
                    else "No action."
                ),
                f"api_surface_changed={api_changed}",
            )
        )

    inventory_age = _get_number(snapshot, "archive_inventory_age_hours")
    inventory_timeout = _get_bool(snapshot, "archive_inventory_timed_out")
    if inventory_age is not None and inventory_timeout is not None:
        stale = inventory_age > thresholds.archive_inventory_stale_after_hours
        severity = "WARN" if stale or inventory_timeout else "OK"
        evidence = f"archive_inventory_age_hours={inventory_age:.2f};threshold={thresholds.archive_inventory_stale_after_hours:.2f}"
        if inventory_timeout:
            evidence += ";timeout=true"
        rows.append(
            _row(
                "ARCHIVE_INVENTORY_FRESH",
                severity,
                (
                    "Rerun bounded/partitioned inventory; do not treat timeout alone as data loss."
                    if severity == "WARN"
                    else "No action."
                ),
                evidence,
            )
        )

    research_ok = _get_bool(snapshot, "research_refresh_ok")
    if research_ok is not None:
        rows.append(
            _row(
                "RESEARCH_REFRESH_OK",
                "OK" if research_ok else "WARN",
                (
                    "Research refresh failed; keep separate from collector health and rerun diagnostics."
                    if not research_ok
                    else "No action."
                ),
                f"research_refresh_ok={research_ok}",
            )
        )

    channels = snapshot.get("delivery_channels")
    if isinstance(channels, list):
        configured = bool(channels)
        rows.append(
            _row(
                "DELIVERY_CHANNEL_CONFIGURED",
                "OK" if configured else "WARN",
                (
                    "Configure Hermes cron/chat, Telegram, Discord, or email delivery outside this repo; do not commit credentials."
                    if not configured
                    else "No action."
                ),
                f"delivery_channels={','.join(channels) if channels else 'none'}",
            )
        )

    columns = ["condition", "severity", "action", "evidence"]
    return pd.DataFrame(rows, columns=columns)


def overall_severity(alerts: pd.DataFrame) -> str:
    if alerts.empty:
        return "PAGE"
    severity_rank = alerts["severity"].map(SEVERITY_ORDER)
    if severity_rank.isna().any():
        return "PAGE"
    rank = int(severity_rank.max())
    for name, value in SEVERITY_ORDER.items():
        if value == rank:
            return name
    return "PAGE"


def _summary_counts(alerts: pd.DataFrame) -> dict[str, int]:
    return {
        severity: int((alerts["severity"] == severity).sum())
        for severity in SEVERITY_ORDER
    }


def write_alert_plan_report(
    alerts: pd.DataFrame,
    out_dir: Path,
    *,
    snapshot: dict[str, Any],
    thresholds: AlertThresholds = AlertThresholds(),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = overall_severity(alerts)
    counts = _summary_counts(alerts)
    alerts.to_csv(out_dir / "alert_plan.csv", index=False)
    pd.DataFrame([{**asdict(thresholds)}]).to_csv(
        out_dir / "thresholds.csv", index=False
    )
    (out_dir / "input_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "overall_severity": overall,
        "counts": counts,
        "checked_at": snapshot.get("checked_at"),
        "rows": int(len(alerts)),
        "thresholds": asdict(thresholds),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    page_warn = alerts[alerts["severity"].isin(["PAGE", "WARN"])]
    lines = [
        "# Pacifica external ops alert plan",
        "",
        f"Overall severity: `{overall}`",
        "",
        "This artifact separates health-check classification from notification delivery. It does not send alerts and does not require or store delivery credentials.",
        "",
        "No external delivery credentials should be committed. Wire delivery via Hermes cron/chat, Telegram, Discord, email, or another scheduler outside this repo/configured secrets path.",
        "",
        "## Page/Warn conditions",
        "",
        dataframe_to_markdown_table(page_warn),
        "",
        "## All conditions",
        "",
        dataframe_to_markdown_table(alerts),
        "",
        "## Thresholds",
        "",
        dataframe_to_markdown_table(pd.DataFrame([{**asdict(thresholds)}])),
        "",
        "## Artifacts",
        "",
        "- `alert_plan.csv`",
        "- `summary.json`",
        "- `thresholds.csv`",
        "- `input_snapshot.json`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    return summary


def _load_snapshot(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("snapshot JSON must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan Pacifica external ops alerts from a status snapshot"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--allow-page",
        action="store_true",
        help="Return 0 even if the diagnostic plan has PAGE severity",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args.snapshot)
    alerts = evaluate_alerts(snapshot)
    summary = write_alert_plan_report(alerts, args.out_dir, snapshot=snapshot)
    print(f"overall_severity: {summary['overall_severity']}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    if summary["overall_severity"] == "PAGE" and not args.allow_page:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

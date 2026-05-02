# scripts/plan_pacifica_r2_retention.py
"""Plan non-destructive R2 raw retention and compaction gates.

This tool does not delete remote objects.  It classifies an R2 object inventory so
we can decide when old raw full-fidelity files are safe to expire only after a
verified compacted/cold archive exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUT_DIR = Path("docs/ops/pacifica-r2-retention")


@dataclass(frozen=True)
class R2RetentionPolicy:
    recent_raw_days: int = 60
    compact_after_days: int = 60
    allow_remote_delete_after_days: int = 90
    keep_sidecars: bool = True


def _parse_time(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, utc=True)


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


def _bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def classify_r2_objects(
    inventory: pd.DataFrame,
    *,
    now: datetime | None = None,
    policy: R2RetentionPolicy = R2RetentionPolicy(),
) -> pd.DataFrame:
    """Classify R2 objects under a no-delete-by-default retention policy."""
    required = {"key", "size_bytes", "mod_time"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")
    if now is None:
        now = datetime.now(UTC)
    now_ts = pd.Timestamp(now)

    out = inventory.copy()
    out["size_bytes"] = (
        pd.to_numeric(out["size_bytes"], errors="coerce").fillna(0).astype(int)
    )
    out["mod_time_ts"] = out["mod_time"].apply(_parse_time)
    out["age_days"] = (
        (now_ts - out["mod_time_ts"]).dt.total_seconds() / 86_400
    ).fillna(0.0)
    out["is_sidecar"] = out["key"].astype(str).str.endswith(".sha256")
    out["is_raw_payload"] = (
        out["key"].astype(str).str.endswith((".jsonl.gz", ".jsonl.zst"))
    )
    compacted_verified = _bool_col(out, "compacted_verified")
    manifest_verified = _bool_col(out, "manifest_verified")

    actions: list[str] = []
    delete_allowed: list[bool] = []
    for idx, row in out.iterrows():
        age = float(row["age_days"])
        if bool(row["is_sidecar"]):
            actions.append("retain_checksum_sidecar")
            delete_allowed.append(False)
        elif not bool(row["is_raw_payload"]):
            actions.append("retain_non_raw_object")
            delete_allowed.append(False)
        elif age <= policy.recent_raw_days:
            actions.append("retain_recent_raw")
            delete_allowed.append(False)
        elif age < policy.allow_remote_delete_after_days:
            actions.append("needs_compaction_before_raw_expiry")
            delete_allowed.append(False)
        elif bool(compacted_verified.loc[idx]) and bool(manifest_verified.loc[idx]):
            actions.append("eligible_for_remote_expiry_after_review")
            delete_allowed.append(True)
        else:
            actions.append("blocked_missing_compaction_or_manifest")
            delete_allowed.append(False)
    out["retention_action"] = actions
    out["remote_delete_allowed"] = delete_allowed
    return out.drop(columns=["mod_time_ts"])


def summarize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(columns=["retention_action", "objects", "bytes"])
    return (
        plan.groupby("retention_action", as_index=False)
        .agg(objects=("key", "size"), bytes=("size_bytes", "sum"))
        .sort_values("bytes", ascending=False)
        .reset_index(drop=True)
    )


def write_retention_report(
    inventory_path: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    now: datetime | None = None,
    policy: R2RetentionPolicy = R2RetentionPolicy(),
) -> dict[str, Any]:
    inventory = pd.read_csv(inventory_path)
    plan = classify_r2_objects(inventory, now=now, policy=policy)
    summary = summarize_plan(plan)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "r2_retention_plan.csv"
    summary_path = out_dir / "r2_retention_summary.csv"
    policy_path = out_dir / "r2_retention_policy.csv"
    plan.to_csv(plan_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([asdict(policy)]).to_csv(policy_path, index=False)

    eligible = plan[plan["remote_delete_allowed"]].copy()
    eligible_bytes = int(eligible["size_bytes"].sum()) if not eligible.empty else 0
    lines = [
        "# Pacifica R2 Retention and Cold-Compaction Plan",
        "",
        "No remote deletion was executed. This report is a non-destructive planning artifact.",
        "",
        "## Policy",
        "",
        f"- 0-{policy.recent_raw_days} days: keep raw full-fidelity `.jsonl.gz` in R2.",
        f"- {policy.compact_after_days}+ days: require verified compacted/cold archive before raw expiry can be considered.",
        f"- {policy.allow_remote_delete_after_days}+ days: raw objects may become eligible for remote expiry only if both `compacted_verified` and `manifest_verified` are true.",
        "- `.sha256` sidecars are retained by default.",
        "- Remote deletion requires a separate explicit review/apply step; this script intentionally does not write delete commands.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown_table(summary),
        "",
        f"Objects eligible for remote-expiry review: {len(eligible)}",
        f"Bytes eligible for remote-expiry review: {eligible_bytes}",
        "",
        "## Gates before any R2 raw expiry",
        "",
        "1. A compacted/cold archive exists for the same channel/symbol/date/hour coverage.",
        "2. Manifest row counts, byte totals, and checksum sidecars have been verified.",
        "3. A restore sample has been tested from R2 into the local research pipeline.",
        "4. Current silver/regime diagnostics can be rebuilt from the retained/compacted data.",
        "5. Diego explicitly approves a separate destructive apply step.",
        "",
        "## Output files",
        "",
        "- `r2_retention_plan.csv` — one row per object with classification.",
        "- `r2_retention_summary.csv` — byte/object counts by action.",
        "- `r2_retention_policy.csv` — thresholds used for this report.",
    ]
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    return {
        "readme": str(readme_path),
        "plan": str(plan_path),
        "summary": str(summary_path),
        "objects": len(plan),
        "delete_command_written": False,
        "eligible_for_review": len(eligible),
        "eligible_bytes": eligible_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--recent-raw-days", type=int, default=60)
    parser.add_argument("--allow-delete-after-days", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = R2RetentionPolicy(
        recent_raw_days=args.recent_raw_days,
        compact_after_days=args.recent_raw_days,
        allow_remote_delete_after_days=args.allow_delete_after_days,
    )
    result = write_retention_report(args.inventory_csv, args.out_dir, policy=policy)
    print(f"wrote report: {result['readme']}")
    print(f"objects: {result['objects']}")
    print(f"eligible_for_review: {result['eligible_for_review']}")
    print("delete_command_written: False")


if __name__ == "__main__":
    main()

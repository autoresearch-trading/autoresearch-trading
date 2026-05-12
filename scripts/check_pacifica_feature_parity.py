# scripts/check_pacifica_feature_parity.py
"""Compare offline rebuilt features with online/live-style feature snapshots.

This is a diagnostic parity gate, not a trading system.  It checks whether two
feature tables agree on shared symbol/bucket rows, required metadata, feature
versions, and numeric feature values before any live microbatch path is trusted.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulate_pacifica_execution import dataframe_to_markdown_table

DEFAULT_OFFLINE_PATH = Path(
    "docs/experiments/non-hft-regime-state/regime_state.parquet"
)
DEFAULT_LIVE_PATH = Path("data/pacifica_realtime_research/latest_features.csv")
DEFAULT_OUT_DIR = Path("docs/experiments/feature-parity")
DEFAULT_FEATURE_COLUMNS = [
    "avg_spread_bps",
    "top_depth_notional",
    "toxicity_score",
]
KEY_COLUMNS = ["symbol", "bucket_start_ms"]
REQUIRED_METADATA_COLUMNS = [
    "available_ts",
    "computed_at",
    "watermark_ts",
    "feature_version",
    "provisional_final_flag",
]


@dataclass(frozen=True)
class FeatureParityResult:
    verdict: str
    compared_rows: int
    compared_features: int
    mismatch_count: int
    missing_key_count: int
    version_mismatch_count: int
    metadata_mismatch_count: int
    invalid_metadata_count: int
    invalid_feature_count: int
    invalid_key_count: int
    duplicate_key_count: int
    missing_metadata_columns: list[str]
    failure_reasons: list[str]
    mismatches: pd.DataFrame
    missing_keys: pd.DataFrame
    version_mismatches: pd.DataFrame
    metadata_mismatches: pd.DataFrame
    invalid_metadata: pd.DataFrame
    invalid_features: pd.DataFrame
    invalid_keys: pd.DataFrame
    duplicate_keys: pd.DataFrame


def _read_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported feature table suffix: {path.suffix}")


def _metadata_missing(offline: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for side, frame in [("offline", offline), ("live", live)]:
        for col in REQUIRED_METADATA_COLUMNS:
            if col not in frame.columns:
                missing.append(f"{side}.{col}")
    return missing


def _required_columns(feature_columns: Sequence[str]) -> set[str]:
    return set(KEY_COLUMNS) | set(REQUIRED_METADATA_COLUMNS) | set(feature_columns)


def _validate_columns(
    offline: pd.DataFrame, live: pd.DataFrame, feature_columns: Sequence[str]
) -> list[str]:
    missing = _metadata_missing(offline, live)
    for side, frame in [("offline", offline), ("live", live)]:
        for col in KEY_COLUMNS:
            if col not in frame.columns:
                missing.append(f"{side}.{col}")
        for col in feature_columns:
            if col not in frame.columns:
                missing.append(f"{side}.{col}")
    return missing


def _missing_keys(offline: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    offline_keys = offline[KEY_COLUMNS].drop_duplicates()
    live_keys = live[KEY_COLUMNS].drop_duplicates()
    offline_missing = live_keys.merge(
        offline_keys, on=KEY_COLUMNS, how="left", indicator=True
    )
    offline_missing = offline_missing[offline_missing["_merge"] == "left_only"][
        KEY_COLUMNS
    ]
    offline_missing = offline_missing.assign(side="offline_missing")
    live_missing = offline_keys.merge(
        live_keys, on=KEY_COLUMNS, how="left", indicator=True
    )
    live_missing = live_missing[live_missing["_merge"] == "left_only"][KEY_COLUMNS]
    live_missing = live_missing.assign(side="live_missing")
    return pd.concat([offline_missing, live_missing], ignore_index=True)


def _version_mismatches(joined: pd.DataFrame) -> pd.DataFrame:
    mask = joined["feature_version_offline"].astype(str) != joined[
        "feature_version_live"
    ].astype(str)
    cols = [
        *KEY_COLUMNS,
        "feature_version_offline",
        "feature_version_live",
    ]
    return joined.loc[mask, cols].rename(
        columns={
            "feature_version_offline": "offline_feature_version",
            "feature_version_live": "live_feature_version",
        }
    )


def _metadata_mismatches(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in REQUIRED_METADATA_COLUMNS:
        if col == "feature_version":
            continue
        offline_col = f"{col}_offline"
        live_col = f"{col}_live"
        left = joined[offline_col]
        right = joined[live_col]
        valid = (
            left.notna()
            & right.notna()
            & left.astype(str).str.strip().ne("")
            & right.astype(str).str.strip().ne("")
        )
        mask = valid & (left.astype(str) != right.astype(str))
        for idx in joined.index[mask]:
            rows.append(
                {
                    "symbol": joined.at[idx, "symbol"],
                    "bucket_start_ms": joined.at[idx, "bucket_start_ms"],
                    "metadata_column": col,
                    "offline_value": joined.at[idx, offline_col],
                    "live_value": joined.at[idx, live_col],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "bucket_start_ms",
            "metadata_column",
            "offline_value",
            "live_value",
        ],
    )


def _invalid_metadata(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in REQUIRED_METADATA_COLUMNS:
        for side in ["offline", "live"]:
            value_col = f"{col}_{side}"
            null_mask = joined[value_col].isna() | joined[value_col].astype(
                str
            ).str.strip().eq("")
            for idx in joined.index[null_mask]:
                rows.append(
                    {
                        "symbol": joined.at[idx, "symbol"],
                        "bucket_start_ms": joined.at[idx, "bucket_start_ms"],
                        "side": side,
                        "metadata_column": col,
                        "issue": "missing_or_null",
                    }
                )
    return pd.DataFrame(
        rows, columns=["symbol", "bucket_start_ms", "side", "metadata_column", "issue"]
    )


def _duplicate_keys(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    duplicated = frame[frame.duplicated(KEY_COLUMNS, keep=False)]
    if duplicated.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "side", "rows"])
    return (
        duplicated.groupby(KEY_COLUMNS, as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .assign(side=side)[[*KEY_COLUMNS, "side", "rows"]]
    )


def _invalid_keys(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in KEY_COLUMNS:
        series = frame[col]
        invalid = series.isna() | series.astype(str).str.strip().eq("")
        for idx in frame.index[invalid]:
            rows.append(
                {
                    "symbol": frame.at[idx, "symbol"] if "symbol" in frame else None,
                    "bucket_start_ms": (
                        frame.at[idx, "bucket_start_ms"]
                        if "bucket_start_ms" in frame
                        else None
                    ),
                    "side": side,
                    "key_column": col,
                    "issue": "missing_or_blank_key",
                }
            )
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, "side", "key_column", "issue"])


def _feature_mismatches(
    joined: pd.DataFrame, feature_columns: Sequence[str], tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        offline_col = f"{feature}_offline"
        live_col = f"{feature}_live"
        offline_values = pd.to_numeric(joined[offline_col], errors="coerce")
        live_values = pd.to_numeric(joined[live_col], errors="coerce")
        invalid_mask = (
            offline_values.isna()
            | live_values.isna()
            | ~offline_values.map(math.isfinite)
            | ~live_values.map(math.isfinite)
        )
        for idx in joined.index[invalid_mask]:
            invalid_rows.append(
                {
                    "symbol": joined.at[idx, "symbol"],
                    "bucket_start_ms": joined.at[idx, "bucket_start_ms"],
                    "feature": feature,
                    "offline_value": joined.at[idx, offline_col],
                    "live_value": joined.at[idx, live_col],
                    "issue": "non_numeric_null_or_non_finite",
                }
            )
        diff = (offline_values - live_values).abs()
        mask = (diff > tolerance) & ~invalid_mask
        for idx in joined.index[mask]:
            rows.append(
                {
                    "symbol": joined.at[idx, "symbol"],
                    "bucket_start_ms": joined.at[idx, "bucket_start_ms"],
                    "feature": feature,
                    "offline_value": offline_values.at[idx],
                    "live_value": live_values.at[idx],
                    "abs_diff": diff.at[idx],
                    "tolerance": tolerance,
                }
            )
    return (
        pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "bucket_start_ms",
                "feature",
                "offline_value",
                "live_value",
                "abs_diff",
                "tolerance",
            ],
        ),
        pd.DataFrame(
            invalid_rows,
            columns=[
                "symbol",
                "bucket_start_ms",
                "feature",
                "offline_value",
                "live_value",
                "issue",
            ],
        ),
    )


def compare_feature_frames(
    offline: pd.DataFrame,
    live: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    tolerance: float = 1e-6,
) -> FeatureParityResult:
    """Compare offline/live feature frames with metadata and numeric tolerances."""
    missing_columns = _validate_columns(offline, live, feature_columns)
    empty = pd.DataFrame()
    invalid_config_reasons: list[str] = []
    if not feature_columns:
        invalid_config_reasons.append("no_feature_columns")
    if not math.isfinite(tolerance) or tolerance < 0:
        invalid_config_reasons.append("invalid_tolerance")
    if missing_columns:
        version_mismatches = pd.DataFrame()
        if all(
            col in offline.columns and col in live.columns
            for col in [*KEY_COLUMNS, "feature_version"]
        ):
            joined_versions = offline[[*KEY_COLUMNS, "feature_version"]].merge(
                live[[*KEY_COLUMNS, "feature_version"]],
                on=KEY_COLUMNS,
                how="inner",
                suffixes=("_offline", "_live"),
            )
            version_mismatches = _version_mismatches(joined_versions)
        return FeatureParityResult(
            verdict="PARITY_FAIL_DIAGNOSTIC",
            compared_rows=0,
            compared_features=len(feature_columns),
            mismatch_count=0,
            missing_key_count=0,
            version_mismatch_count=int(len(version_mismatches)),
            metadata_mismatch_count=0,
            invalid_metadata_count=0,
            invalid_feature_count=0,
            invalid_key_count=0,
            duplicate_key_count=0,
            missing_metadata_columns=missing_columns,
            failure_reasons=["missing_required_columns", *invalid_config_reasons],
            mismatches=empty,
            missing_keys=empty,
            version_mismatches=version_mismatches,
            metadata_mismatches=empty,
            invalid_metadata=empty,
            invalid_features=empty,
            invalid_keys=empty,
            duplicate_keys=empty,
        )

    offline_small = offline[
        [*KEY_COLUMNS, *REQUIRED_METADATA_COLUMNS, *feature_columns]
    ].copy()
    live_small = live[
        [*KEY_COLUMNS, *REQUIRED_METADATA_COLUMNS, *feature_columns]
    ].copy()
    duplicate_keys = pd.concat(
        [
            _duplicate_keys(offline_small, "offline"),
            _duplicate_keys(live_small, "live"),
        ],
        ignore_index=True,
    )
    invalid_keys = pd.concat(
        [_invalid_keys(offline_small, "offline"), _invalid_keys(live_small, "live")],
        ignore_index=True,
    )
    missing_keys = _missing_keys(offline_small, live_small)
    joined = offline_small.merge(
        live_small,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_offline", "_live"),
    )
    version_mismatches = _version_mismatches(joined)
    metadata_mismatches = _metadata_mismatches(joined)
    invalid_metadata = _invalid_metadata(joined)
    mismatches, invalid_features = _feature_mismatches(
        joined, feature_columns, tolerance
    )
    failure_reasons: list[str] = list(invalid_config_reasons)
    if len(joined) == 0:
        failure_reasons.append("no_overlapping_rows")
    if len(duplicate_keys):
        failure_reasons.append("duplicate_keys")
    if len(invalid_keys):
        failure_reasons.append("invalid_keys")
    if len(missing_keys):
        failure_reasons.append("missing_keys")
    if len(version_mismatches):
        failure_reasons.append("feature_version_mismatch")
    if len(metadata_mismatches):
        failure_reasons.append("metadata_mismatch")
    if len(invalid_metadata):
        failure_reasons.append("invalid_metadata")
    if len(mismatches):
        failure_reasons.append("feature_mismatch")
    if len(invalid_features):
        failure_reasons.append("invalid_feature_values")
    return FeatureParityResult(
        verdict=(
            "PARITY_FAIL_DIAGNOSTIC" if failure_reasons else "PARITY_PASS_DIAGNOSTIC"
        ),
        compared_rows=int(len(joined)),
        compared_features=int(len(feature_columns)),
        mismatch_count=int(len(mismatches)),
        missing_key_count=int(len(missing_keys)),
        version_mismatch_count=int(len(version_mismatches)),
        metadata_mismatch_count=int(len(metadata_mismatches)),
        invalid_metadata_count=int(len(invalid_metadata)),
        invalid_feature_count=int(len(invalid_features)),
        invalid_key_count=int(len(invalid_keys)),
        duplicate_key_count=int(len(duplicate_keys)),
        missing_metadata_columns=[],
        failure_reasons=failure_reasons,
        mismatches=mismatches,
        missing_keys=missing_keys,
        version_mismatches=version_mismatches,
        metadata_mismatches=metadata_mismatches,
        invalid_metadata=invalid_metadata,
        invalid_features=invalid_features,
        invalid_keys=invalid_keys,
        duplicate_keys=duplicate_keys,
    )


def _summary_frame(result: FeatureParityResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "verdict": result.verdict,
                "compared_rows": result.compared_rows,
                "compared_features": result.compared_features,
                "mismatch_count": result.mismatch_count,
                "missing_key_count": result.missing_key_count,
                "version_mismatch_count": result.version_mismatch_count,
                "metadata_mismatch_count": result.metadata_mismatch_count,
                "invalid_metadata_count": result.invalid_metadata_count,
                "invalid_feature_count": result.invalid_feature_count,
                "invalid_key_count": result.invalid_key_count,
                "duplicate_key_count": result.duplicate_key_count,
                "missing_metadata_columns": ";".join(result.missing_metadata_columns),
                "failure_reasons": ";".join(result.failure_reasons),
            }
        ]
    )


def write_feature_parity_report(
    offline_path: Path = DEFAULT_OFFLINE_PATH,
    live_path: Path = DEFAULT_LIVE_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare two feature artifacts and write Markdown/CSV diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    offline = _read_feature_table(offline_path)
    live = _read_feature_table(live_path)
    result = compare_feature_frames(
        offline,
        live,
        feature_columns=feature_columns,
        tolerance=tolerance,
    )
    summary = _summary_frame(result)
    summary.to_csv(out_dir / "summary.csv", index=False)
    result.mismatches.to_csv(out_dir / "mismatches.csv", index=False)
    result.missing_keys.to_csv(out_dir / "missing_keys.csv", index=False)
    result.version_mismatches.to_csv(out_dir / "version_mismatches.csv", index=False)
    result.metadata_mismatches.to_csv(out_dir / "metadata_mismatches.csv", index=False)
    result.invalid_metadata.to_csv(out_dir / "invalid_metadata.csv", index=False)
    result.invalid_features.to_csv(out_dir / "invalid_features.csv", index=False)
    result.invalid_keys.to_csv(out_dir / "invalid_keys.csv", index=False)
    result.duplicate_keys.to_csv(out_dir / "duplicate_keys.csv", index=False)
    pd.DataFrame({"feature": list(feature_columns)}).to_csv(
        out_dir / "feature_columns.csv", index=False
    )

    report = f"""# Pacifica Feature Parity

Verdict: `{result.verdict}`

This diagnostic compares offline rebuilt features against online/offline
live-style feature snapshots. It does not authorize trading, does not claim edge,
and should gate any future use of live microbatch features in strategy adapters.

## Required metadata

{', '.join(REQUIRED_METADATA_COLUMNS)}

## Summary

{dataframe_to_markdown_table(summary)}

## Compared feature columns

{dataframe_to_markdown_table(pd.DataFrame({'feature': list(feature_columns)}))}

## Mismatches

{dataframe_to_markdown_table(result.mismatches, max_rows=25)}

## Missing keys

{dataframe_to_markdown_table(result.missing_keys, max_rows=25)}

## Version mismatches

{dataframe_to_markdown_table(result.version_mismatches, max_rows=25)}

## Metadata mismatches

{dataframe_to_markdown_table(result.metadata_mismatches, max_rows=25)}

## Invalid metadata

{dataframe_to_markdown_table(result.invalid_metadata, max_rows=25)}

## Invalid feature values

{dataframe_to_markdown_table(result.invalid_features, max_rows=25)}

## Invalid keys

{dataframe_to_markdown_table(result.invalid_keys, max_rows=25)}

## Duplicate keys

{dataframe_to_markdown_table(result.duplicate_keys, max_rows=25)}

## Interpretation discipline

- `PARITY_PASS_DIAGNOSTIC` only means the compared artifacts matched within tolerance.
- It is not a strategy verdict and does not override eligibility or sample-maturity gates.
- `PARITY_FAIL_DIAGNOSTIC` blocks live feature use until mismatches are explained or fixed.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
    return {
        "verdict": result.verdict,
        "out_dir": str(out_dir),
        "mismatch_count": result.mismatch_count,
        "missing_key_count": result.missing_key_count,
        "version_mismatch_count": result.version_mismatch_count,
        "metadata_mismatch_count": result.metadata_mismatch_count,
        "invalid_metadata_count": result.invalid_metadata_count,
        "invalid_feature_count": result.invalid_feature_count,
        "invalid_key_count": result.invalid_key_count,
        "duplicate_key_count": result.duplicate_key_count,
    }


def _parse_feature_columns(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path, default=DEFAULT_OFFLINE_PATH)
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--allow-fail-diagnostic",
        action="store_true",
        help="Write failure diagnostics but exit 0. Default exits 1 on PARITY_FAIL_DIAGNOSTIC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_feature_parity_report(
        args.offline,
        args.live,
        args.out_dir,
        feature_columns=_parse_feature_columns(args.feature_columns),
        tolerance=args.tolerance,
    )
    print(f"verdict: {result['verdict']}")
    print(f"mismatch_count: {result['mismatch_count']}")
    print(f"missing_key_count: {result['missing_key_count']}")
    print(f"version_mismatch_count: {result['version_mismatch_count']}")
    print(f"metadata_mismatch_count: {result['metadata_mismatch_count']}")
    print(f"invalid_metadata_count: {result['invalid_metadata_count']}")
    print(f"invalid_feature_count: {result['invalid_feature_count']}")
    print(f"invalid_key_count: {result['invalid_key_count']}")
    print(f"duplicate_key_count: {result['duplicate_key_count']}")
    print(f"wrote report: {Path(result['out_dir']) / 'README.md'}")
    if result["verdict"] != "PARITY_PASS_DIAGNOSTIC" and not args.allow_fail_diagnostic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

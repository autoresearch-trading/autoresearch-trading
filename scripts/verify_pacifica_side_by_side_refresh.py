# scripts/verify_pacifica_side_by_side_refresh.py
"""Verify candidate Pacifica silver/regime refresh against canonical outputs.

This script is intentionally read-only with respect to canonical directories.  It
writes comparison artifacts under a caller-provided report directory and fails
closed on row-count/coverage regressions, duplicate keys, or null required keys.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_non_hft_regime_state import read_silver_table

DEFAULT_CHANNELS = ("prices", "trades", "bbo", "book", "candle", "mark_price_candle")


def _date_from_ms(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _read_regime_state(regime_dir: Path) -> pd.DataFrame:
    for name in ("regime_state.parquet", "regime_state_delta.parquet"):
        path = regime_dir / name
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def _missing_key_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return []
    return [col for col in columns if col not in frame.columns]


def _key_nulls(frame: pd.DataFrame, columns: list[str]) -> int:
    missing = _missing_key_columns(frame, columns)
    if missing:
        return len(frame)
    return int(frame[columns].isna().any(axis=1).sum())


def _duplicate_keys(frame: pd.DataFrame, columns: list[str]) -> int:
    missing = _missing_key_columns(frame, columns)
    if missing or frame.empty:
        return 0
    return int(frame.duplicated(subset=columns).sum())


def _silver_key_columns(frame: pd.DataFrame) -> list[str]:
    if {"symbol", "event_ts_ms", "recv_ms"}.issubset(frame.columns):
        return ["symbol", "event_ts_ms", "recv_ms"]
    if {"symbol", "event_ts_ms"}.issubset(frame.columns):
        return ["symbol", "event_ts_ms"]
    if {"source_key", "event_ts_ms", "recv_ms"}.issubset(frame.columns):
        return ["source_key", "event_ts_ms", "recv_ms"]
    return ["symbol", "event_ts_ms"]


def _silver_metrics(
    root: Path, channels: list[str] | tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for channel in channels:
        frame = read_silver_table(root, channel)
        count_rows.append({"channel": channel, "rows": len(frame)})
        if frame.empty:
            quality_rows.append(
                {
                    "channel": channel,
                    "key_nulls": 0,
                    "duplicate_keys": 0,
                    "missing_key_columns": "",
                }
            )
            continue
        dates = (
            frame["event_ts_ms"].map(_date_from_ms)
            if "event_ts_ms" in frame
            else pd.Series([None] * len(frame))
        )
        symbols = (
            frame["symbol"] if "symbol" in frame else pd.Series([None] * len(frame))
        )
        for symbol, date in sorted(
            set(zip(symbols.astype(str), dates.astype(str), strict=False))
        ):
            if symbol != "None" and date != "None":
                coverage_rows.append(
                    {"channel": channel, "symbol": symbol, "date": date}
                )
        required_key_cols = ["symbol", "event_ts_ms"]
        key_cols = _silver_key_columns(frame)
        quality_rows.append(
            {
                "channel": channel,
                "key_nulls": _key_nulls(frame, required_key_cols),
                "duplicate_keys": _duplicate_keys(frame, key_cols),
                "missing_key_columns": ";".join(
                    _missing_key_columns(frame, required_key_cols)
                ),
            }
        )
    return (
        pd.DataFrame(count_rows),
        pd.DataFrame(coverage_rows, columns=["channel", "symbol", "date"]),
        pd.DataFrame(quality_rows),
    )


def _regime_metrics(
    regime_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _read_regime_state(regime_dir)
    counts = pd.DataFrame([{"table": "regime_state", "rows": len(frame)}])
    if frame.empty:
        coverage = pd.DataFrame(columns=["symbol", "date"])
        quality = pd.DataFrame(
            [
                {
                    "table": "regime_state",
                    "key_nulls": 0,
                    "duplicate_keys": 0,
                    "missing_key_columns": "",
                }
            ]
        )
        return counts, coverage, quality
    dates = (
        frame["bucket_start_ms"].map(_date_from_ms)
        if "bucket_start_ms" in frame
        else pd.Series([None] * len(frame))
    )
    symbols = frame["symbol"] if "symbol" in frame else pd.Series([None] * len(frame))
    coverage = pd.DataFrame(
        [
            {"symbol": symbol, "date": date}
            for symbol, date in sorted(
                set(zip(symbols.astype(str), dates.astype(str), strict=False))
            )
            if symbol != "None" and date != "None"
        ]
    )
    required_key_cols = ["symbol", "bucket_start_ms"]
    quality = pd.DataFrame(
        [
            {
                "table": "regime_state",
                "key_nulls": _key_nulls(frame, required_key_cols),
                "duplicate_keys": _duplicate_keys(frame, required_key_cols),
                "missing_key_columns": ";".join(
                    _missing_key_columns(frame, required_key_cols)
                ),
            }
        ]
    )
    return counts, coverage, quality


def _coverage_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    if frame.empty or not set(columns).issubset(frame.columns):
        return set()
    return {tuple(str(row[col]) for col in columns) for _, row in frame.iterrows()}


def _write_report_diff(
    canonical_regime: Path, candidate_regime: Path, out_dir: Path
) -> None:
    canonical_report = canonical_regime / "README.md"
    candidate_report = candidate_regime / "README.md"
    if not candidate_report.exists():
        candidate_report = candidate_regime / "incremental_regime_report.md"
    before = (
        canonical_report.read_text(encoding="utf-8").splitlines(keepends=True)
        if canonical_report.exists()
        else []
    )
    after = (
        candidate_report.read_text(encoding="utf-8").splitlines(keepends=True)
        if candidate_report.exists()
        else []
    )
    diff = difflib.unified_diff(
        before,
        after,
        fromfile=str(canonical_report),
        tofile=str(candidate_report),
    )
    (out_dir / "report_diff.patch").write_text("".join(diff), encoding="utf-8")


def compare_side_by_side_refresh(
    canonical_silver_dir: Path,
    candidate_silver_dir: Path,
    canonical_regime_dir: Path,
    candidate_regime_dir: Path,
    out_dir: Path,
    *,
    channels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected = tuple(channels or DEFAULT_CHANNELS)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    can_counts, can_coverage, can_quality = _silver_metrics(
        canonical_silver_dir, selected
    )
    cand_counts, cand_coverage, cand_quality = _silver_metrics(
        candidate_silver_dir, selected
    )
    silver_counts = can_counts.merge(
        cand_counts, on="channel", suffixes=("_canonical", "_candidate")
    )
    silver_counts = silver_counts.rename(
        columns={"rows_canonical": "canonical_rows", "rows_candidate": "candidate_rows"}
    )
    silver_counts["row_delta"] = (
        silver_counts["candidate_rows"] - silver_counts["canonical_rows"]
    )
    silver_counts.to_csv(
        out_dir / "silver_row_counts.csv", index=False, lineterminator="\n"
    )
    silver_coverage = pd.concat(
        [can_coverage.assign(side="canonical"), cand_coverage.assign(side="candidate")],
        ignore_index=True,
    )
    silver_coverage.to_csv(
        out_dir / "silver_coverage.csv", index=False, lineterminator="\n"
    )
    silver_quality = can_quality.merge(
        cand_quality, on="channel", suffixes=("_canonical", "_candidate")
    )
    silver_quality.to_csv(
        out_dir / "silver_duplicates_nulls.csv", index=False, lineterminator="\n"
    )

    if (silver_counts["candidate_rows"] < silver_counts["canonical_rows"]).any():
        failures.append("candidate_silver_row_count_regression")
    if not _coverage_set(can_coverage, ["channel", "symbol", "date"]).issubset(
        _coverage_set(cand_coverage, ["channel", "symbol", "date"])
    ):
        failures.append("candidate_silver_coverage_regression")
    if (silver_quality.get("key_nulls_candidate", pd.Series(dtype=int)) > 0).any():
        failures.append("candidate_silver_key_nulls")
    if (
        silver_quality.get("missing_key_columns_candidate", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .any()
    ):
        failures.append("candidate_silver_missing_key_columns")
    if (
        silver_quality.get("duplicate_keys_candidate", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
        > silver_quality.get("duplicate_keys_canonical", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
    ).any():
        failures.append("candidate_silver_duplicate_keys")

    can_regime_counts, can_regime_cov, can_regime_quality = _regime_metrics(
        canonical_regime_dir
    )
    cand_regime_counts, cand_regime_cov, cand_regime_quality = _regime_metrics(
        candidate_regime_dir
    )
    regime_counts = can_regime_counts.merge(
        cand_regime_counts, on="table", suffixes=("_canonical", "_candidate")
    ).rename(
        columns={"rows_canonical": "canonical_rows", "rows_candidate": "candidate_rows"}
    )
    regime_counts["row_delta"] = (
        regime_counts["candidate_rows"] - regime_counts["canonical_rows"]
    )
    regime_counts.to_csv(
        out_dir / "regime_row_counts.csv", index=False, lineterminator="\n"
    )
    regime_coverage = pd.concat(
        [
            can_regime_cov.assign(side="canonical"),
            cand_regime_cov.assign(side="candidate"),
        ],
        ignore_index=True,
    )
    regime_coverage.to_csv(
        out_dir / "regime_coverage.csv", index=False, lineterminator="\n"
    )
    regime_quality = can_regime_quality.merge(
        cand_regime_quality, on="table", suffixes=("_canonical", "_candidate")
    )
    regime_quality.to_csv(
        out_dir / "regime_duplicates_nulls.csv", index=False, lineterminator="\n"
    )
    _write_report_diff(canonical_regime_dir, candidate_regime_dir, out_dir)

    if (regime_counts["candidate_rows"] < regime_counts["canonical_rows"]).any():
        failures.append("candidate_regime_row_count_regression")
    if not _coverage_set(can_regime_cov, ["symbol", "date"]).issubset(
        _coverage_set(cand_regime_cov, ["symbol", "date"])
    ):
        failures.append("candidate_regime_symbol_coverage_regression")
    if (regime_quality.get("key_nulls_candidate", pd.Series(dtype=int)) > 0).any():
        failures.append("candidate_regime_key_nulls")
    if (
        regime_quality.get("missing_key_columns_candidate", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .any()
    ):
        failures.append("candidate_regime_missing_key_columns")
    if (
        regime_quality.get("duplicate_keys_candidate", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
        > regime_quality.get("duplicate_keys_canonical", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
    ).any():
        failures.append("candidate_regime_duplicate_keys")

    summary = pd.DataFrame(
        [
            {
                "ok": not failures,
                "failures": ";".join(failures),
                "channels": ",".join(selected),
            }
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    (out_dir / "README.md").write_text(
        "# Pacifica Side-by-Side Refresh Verification\n\n"
        f"ok={not failures}\n\n"
        f"failures={failures}\n\n"
        "Promotion to canonical data remains blocked unless this report is green and reviewed.\n",
        encoding="utf-8",
    )
    return {"ok": not failures, "failures": failures, "out_dir": str(out_dir)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-silver-dir", type=Path, required=True)
    parser.add_argument("--candidate-silver-dir", type=Path, required=True)
    parser.add_argument("--canonical-regime-dir", type=Path, required=True)
    parser.add_argument("--candidate-regime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
    result = compare_side_by_side_refresh(
        args.canonical_silver_dir,
        args.candidate_silver_dir,
        args.canonical_regime_dir,
        args.candidate_regime_dir,
        args.out_dir,
        channels=channels,
    )
    print(f"ok={result['ok']}")
    print(f"failures={result['failures']}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

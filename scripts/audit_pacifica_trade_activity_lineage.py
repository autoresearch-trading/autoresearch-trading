# scripts/audit_pacifica_trade_activity_lineage.py
"""Audit Pacifica trade-activity lineage from raw trades to eligibility gates.

This is diagnostic plumbing only.  It traces whether eligibility activity metrics
are supported by raw trades, normalized silver trades, and regime-state
trade_count/trade_notional aggregates before any strategy or paper-trading work
uses those metrics.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pacifica_full_fidelity_silver import (
    iter_raw_records_from_path,
    normalize_trade_record,
)

DEFAULT_RAW_DIR = Path("data/pacifica_full_fidelity")
DEFAULT_SILVER_DIR = Path("data/pacifica_silver_partitioned")
DEFAULT_REGIME_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_ELIGIBILITY_PATH = Path(
    "docs/experiments/paper-trading-eligibility/symbol_eligibility.csv"
)
DEFAULT_OUT_DIR = Path("docs/experiments/trade-activity-lineage")
DEFAULT_MAX_SYMBOLS = 10
REQUIRED_REGIME_COLUMNS = {"symbol", "bucket_start_ms", "trade_count", "trade_notional"}
REQUIRED_SILVER_COLUMNS = {"symbol", "event_ts_ms", "notional"}


@dataclass(frozen=True)
class LineageAuditConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    silver_dir: Path = DEFAULT_SILVER_DIR
    regime_path: Path = DEFAULT_REGIME_PATH
    eligibility_path: Path = DEFAULT_ELIGIBILITY_PATH
    out_dir: Path = DEFAULT_OUT_DIR
    symbols: tuple[str, ...] | None = None
    max_symbols: int = DEFAULT_MAX_SYMBOLS
    bucket: str = "1min"
    notional_abs_tolerance: float = 1e-6
    notional_rel_tolerance: float = 1e-9


@dataclass(frozen=True)
class LineageAuditResult:
    verdict: str
    symbol_summary: pd.DataFrame
    date_summary: pd.DataFrame
    target_symbols: tuple[str, ...]


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


def bucket_ms(bucket: str) -> int:
    bucket = bucket.strip().lower()
    if bucket.endswith("ms"):
        return int(bucket[:-2])
    if bucket.endswith("s"):
        return int(float(bucket[:-1]) * 1_000)
    if bucket.endswith("min"):
        return int(float(bucket[:-3]) * 60_000)
    if bucket.endswith("m"):
        return int(float(bucket[:-1]) * 60_000)
    raise ValueError(f"Unsupported bucket: {bucket}")


def _date_from_ms(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", utc=True, errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )


def _partition_value(path: Path, name: str) -> str | None:
    prefix = f"{name}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def _read_eligibility(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        raise ValueError("eligibility missing required columns: ['symbol']")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    return frame


def _resolve_target_symbols(
    config: LineageAuditConfig, eligibility: pd.DataFrame
) -> tuple[str, ...]:
    if config.symbols:
        return tuple(dict.fromkeys(str(symbol) for symbol in config.symbols))
    if eligibility.empty:
        return ()
    preferred = [
        symbol
        for symbol in ("BTC", "ETH", "SOL")
        if symbol in set(eligibility["symbol"])
    ]
    ordered = [str(symbol) for symbol in eligibility["symbol"].tolist()]
    symbols = list(dict.fromkeys(preferred + ordered))
    return tuple(symbols[: config.max_symbols])


def _empty_date_summary(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "date", *columns])


def _empty_symbol_summary(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", *columns])


def _read_raw_trade_summaries(
    raw_dir: Path, symbols: tuple[str, ...], *, bucket: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "raw_trade_files",
        "raw_trade_rows",
        "raw_trade_notional_sum",
        "raw_trade_bucket_count",
        "raw_invalid_event_ts_rows",
    ]
    if not raw_dir.exists() or not symbols:
        return _empty_date_summary(columns), _empty_symbol_summary(columns)

    wanted = set(symbols)
    trade_root = raw_dir / "channel=trades"
    if not trade_root.exists():
        return _empty_date_summary(columns), _empty_symbol_summary(columns)

    paths: list[Path] = []
    for symbol in symbols:
        symbol_dir = trade_root / f"symbol={symbol}"
        if symbol_dir.exists():
            paths.extend(sorted(symbol_dir.rglob("*.jsonl.gz")))

    ms = bucket_ms(bucket)
    rows: list[dict[str, Any]] = []
    files_seen: set[tuple[str, str, Path]] = set()
    invalid_rows: dict[tuple[str, str], int] = {}
    for path in paths:
        file_partition_symbol = _partition_value(path, "symbol")
        file_partition_date = _partition_value(path, "date")
        for record in iter_raw_records_from_path(path):
            row = normalize_trade_record(record)
            symbol = str(row.get("symbol") or file_partition_symbol or "UNKNOWN")
            if symbol not in wanted:
                continue
            event_ts_ms = pd.to_numeric(
                pd.Series([row.get("event_ts_ms")]), errors="coerce"
            ).iloc[0]
            if pd.isna(event_ts_ms):
                date = file_partition_date or "unknown"
                invalid_rows[(symbol, date)] = invalid_rows.get((symbol, date), 0) + 1
                continue
            event_ts_ms = int(event_ts_ms)
            date = pd.to_datetime(event_ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d")
            files_seen.add((symbol, date, path))
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "bucket_start_ms": (event_ts_ms // ms) * ms,
                    "notional": float(row.get("notional") or 0.0),
                }
            )

    if rows:
        raw = pd.DataFrame(rows)
        date_summary = raw.groupby(["symbol", "date"], as_index=False).agg(
            raw_trade_rows=("notional", "size"),
            raw_trade_notional_sum=("notional", "sum"),
            raw_trade_bucket_count=("bucket_start_ms", "nunique"),
        )
    else:
        date_summary = pd.DataFrame(columns=["symbol", "date"])

    file_counts = pd.DataFrame(
        [
            {"symbol": symbol, "date": date, "raw_trade_files": 1}
            for symbol, date, _ in files_seen
        ]
    )
    if not file_counts.empty:
        file_counts = file_counts.groupby(["symbol", "date"], as_index=False).agg(
            raw_trade_files=("raw_trade_files", "sum")
        )
        date_summary = date_summary.merge(
            file_counts, on=["symbol", "date"], how="outer"
        )
    elif "raw_trade_files" not in date_summary.columns:
        date_summary["raw_trade_files"] = 0

    invalid = pd.DataFrame(
        [
            {"symbol": symbol, "date": date, "raw_invalid_event_ts_rows": count}
            for (symbol, date), count in invalid_rows.items()
        ]
    )
    if not invalid.empty:
        date_summary = date_summary.merge(invalid, on=["symbol", "date"], how="outer")
    elif "raw_invalid_event_ts_rows" not in date_summary.columns:
        date_summary["raw_invalid_event_ts_rows"] = 0

    for column in columns:
        if column not in date_summary.columns:
            date_summary[column] = 0
        date_summary[column] = pd.to_numeric(
            date_summary[column], errors="coerce"
        ).fillna(0)

    symbol_summary = date_summary.groupby("symbol", as_index=False).agg(
        raw_trade_files=("raw_trade_files", "sum"),
        raw_trade_rows=("raw_trade_rows", "sum"),
        raw_trade_notional_sum=("raw_trade_notional_sum", "sum"),
        raw_trade_bucket_count=("raw_trade_bucket_count", "sum"),
        raw_invalid_event_ts_rows=("raw_invalid_event_ts_rows", "sum"),
    )
    return (
        date_summary.sort_values(["symbol", "date"]).reset_index(drop=True),
        symbol_summary,
    )


def _silver_trade_paths(silver_dir: Path, symbols: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    flat_path = silver_dir / "trades.parquet"
    if flat_path.exists():
        paths.append(flat_path)
    channel_dir = silver_dir / "channel=trades"
    if not channel_dir.exists():
        return paths
    if symbols:
        for symbol in symbols:
            symbol_dir = channel_dir / f"symbol={symbol}"
            if symbol_dir.exists():
                paths.extend(sorted(symbol_dir.rglob("*.parquet")))
    else:
        paths.extend(sorted(channel_dir.rglob("*.parquet")))
    return paths


def _read_silver_trade_summaries(
    silver_dir: Path, symbols: tuple[str, ...], *, bucket: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "silver_trade_rows",
        "silver_trade_notional_sum",
        "silver_trade_bucket_count",
        "silver_invalid_event_ts_rows",
    ]
    paths = _silver_trade_paths(silver_dir, symbols)
    if not paths:
        return (
            _empty_date_summary(columns),
            _empty_symbol_summary(columns),
            pd.DataFrame(),
        )

    wanted = set(symbols)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = REQUIRED_SILVER_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"silver trades missing required columns: {sorted(missing)}"
            )
        if frame.empty:
            continue
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str)
        if wanted:
            frame = frame[frame["symbol"].isin(wanted)].copy()
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return (
            _empty_date_summary(columns),
            _empty_symbol_summary(columns),
            pd.DataFrame(),
        )

    trades = pd.concat(frames, ignore_index=True)
    trades["event_ts_ms"] = pd.to_numeric(trades["event_ts_ms"], errors="coerce")
    trades["notional"] = pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0)
    trades["date"] = _date_from_ms(trades["event_ts_ms"])
    valid = trades[trades["event_ts_ms"].notna() & trades["date"].notna()].copy()
    invalid = trades[~(trades["event_ts_ms"].notna() & trades["date"].notna())].copy()
    ms = bucket_ms(bucket)
    valid["bucket_start_ms"] = (valid["event_ts_ms"].astype("int64") // ms) * ms

    if valid.empty:
        date_summary = pd.DataFrame(columns=["symbol", "date"])
    else:
        date_summary = valid.groupby(["symbol", "date"], as_index=False).agg(
            silver_trade_rows=("notional", "size"),
            silver_trade_notional_sum=("notional", "sum"),
            silver_trade_bucket_count=("bucket_start_ms", "nunique"),
        )
    if not invalid.empty:
        invalid_summary = (
            invalid.assign(date="unknown")
            .groupby(["symbol", "date"], as_index=False)
            .size()
        )
        invalid_summary = invalid_summary.rename(
            columns={"size": "silver_invalid_event_ts_rows"}
        )
        date_summary = date_summary.merge(
            invalid_summary, on=["symbol", "date"], how="outer"
        )
    else:
        date_summary["silver_invalid_event_ts_rows"] = 0
    for column in columns:
        if column not in date_summary.columns:
            date_summary[column] = 0
        date_summary[column] = pd.to_numeric(
            date_summary[column], errors="coerce"
        ).fillna(0)

    symbol_summary = date_summary.groupby("symbol", as_index=False).agg(
        silver_trade_rows=("silver_trade_rows", "sum"),
        silver_trade_notional_sum=("silver_trade_notional_sum", "sum"),
        silver_trade_bucket_count=("silver_trade_bucket_count", "sum"),
        silver_invalid_event_ts_rows=("silver_invalid_event_ts_rows", "sum"),
    )
    return (
        date_summary.sort_values(["symbol", "date"]).reset_index(drop=True),
        symbol_summary,
        valid,
    )


def _read_regime_summaries(
    regime_path: Path, symbols: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not regime_path.exists():
        raise ValueError(f"regime_state not found: {regime_path}")
    state = pd.read_parquet(regime_path)
    missing = REQUIRED_REGIME_COLUMNS - set(state.columns)
    if missing:
        raise ValueError(f"regime_state missing required columns: {sorted(missing)}")
    state = state.copy()
    state["symbol"] = state["symbol"].astype(str)
    if symbols:
        state = state[state["symbol"].isin(set(symbols))].copy()
    if state.empty:
        columns = [
            "regime_rows",
            "regime_trade_active_rows",
            "regime_trade_count_sum",
            "regime_trade_notional_sum",
            "regime_trade_notional_median_all_rows",
            "regime_trade_notional_median_active_rows",
            "regime_trade_bucket_count",
        ]
        return _empty_date_summary(columns), _empty_symbol_summary(columns)

    state["bucket_start_ms"] = pd.to_numeric(state["bucket_start_ms"], errors="coerce")
    state["trade_count"] = pd.to_numeric(state["trade_count"], errors="coerce").fillna(
        0.0
    )
    state["trade_notional"] = pd.to_numeric(
        state["trade_notional"], errors="coerce"
    ).fillna(0.0)
    state = state[state["bucket_start_ms"].notna()].copy()
    state["date"] = _date_from_ms(state["bucket_start_ms"])
    state["is_trade_active"] = state["trade_count"] > 0
    active = state[state["is_trade_active"]].copy()

    date_summary = state.groupby(["symbol", "date"], as_index=False).agg(
        regime_rows=("bucket_start_ms", "size"),
        regime_trade_active_rows=("is_trade_active", "sum"),
        regime_trade_count_sum=("trade_count", "sum"),
        regime_trade_notional_sum=("trade_notional", "sum"),
        regime_trade_notional_median_all_rows=("trade_notional", "median"),
        regime_trade_bucket_count=("bucket_start_ms", "nunique"),
    )
    if active.empty:
        date_summary["regime_trade_notional_median_active_rows"] = 0.0
    else:
        active_median = active.groupby(["symbol", "date"], as_index=False).agg(
            regime_trade_notional_median_active_rows=("trade_notional", "median")
        )
        date_summary = date_summary.merge(
            active_median, on=["symbol", "date"], how="left"
        )
        date_summary["regime_trade_notional_median_active_rows"] = date_summary[
            "regime_trade_notional_median_active_rows"
        ].fillna(0.0)

    symbol_summary = state.groupby("symbol", as_index=False).agg(
        regime_rows=("bucket_start_ms", "size"),
        regime_trade_active_rows=("is_trade_active", "sum"),
        regime_trade_count_sum=("trade_count", "sum"),
        regime_trade_notional_sum=("trade_notional", "sum"),
        regime_trade_notional_median_all_rows=("trade_notional", "median"),
        regime_trade_bucket_count=("bucket_start_ms", "nunique"),
    )
    if active.empty:
        symbol_summary["regime_trade_notional_median_active_rows"] = 0.0
    else:
        symbol_active_median = active.groupby("symbol", as_index=False).agg(
            regime_trade_notional_median_active_rows=("trade_notional", "median")
        )
        symbol_summary = symbol_summary.merge(
            symbol_active_median, on="symbol", how="left"
        )
        symbol_summary["regime_trade_notional_median_active_rows"] = symbol_summary[
            "regime_trade_notional_median_active_rows"
        ].fillna(0.0)

    return (
        date_summary.sort_values(["symbol", "date"]).reset_index(drop=True),
        symbol_summary.sort_values("symbol").reset_index(drop=True),
    )


def _merge_summaries(frames: list[pd.DataFrame], on: list[str]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame if out is None else out.merge(frame, on=on, how="outer")
    if out is None:
        return pd.DataFrame(columns=on)
    return out


def _attach_eligibility(
    symbol_summary: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    if eligibility.empty or symbol_summary.empty:
        return symbol_summary
    columns = [
        column
        for column in [
            "symbol",
            "median_trade_notional_per_min",
            "median_bbo_updates_per_min",
            "activity_gate_pass",
            "failure_reasons",
        ]
        if column in eligibility.columns
    ]
    return symbol_summary.merge(eligibility[columns], on="symbol", how="left")


def _fill_numeric_zeros(frame: pd.DataFrame, key_columns: set[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column in key_columns:
            continue
        if column == "failure_reasons":
            out[column] = out[column].fillna("")
            continue
        if column == "activity_gate_pass":
            out[column] = out[column].fillna(False).astype(bool)
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    return out


def _notional_mismatch(
    left: pd.Series, right: pd.Series, config: LineageAuditConfig
) -> pd.Series:
    diff = (left - right).abs()
    allowed = (
        config.notional_abs_tolerance
        + config.notional_rel_tolerance * left.abs().clip(lower=right.abs())
    )
    return diff > allowed


def _add_lineage_deltas(
    frame: pd.DataFrame, config: LineageAuditConfig, *, per_symbol: bool
) -> pd.DataFrame:
    out = _fill_numeric_zeros(frame, {"symbol", "date"})
    for column in [
        "raw_trade_rows",
        "silver_trade_rows",
        "regime_trade_count_sum",
        "raw_trade_notional_sum",
        "silver_trade_notional_sum",
        "regime_trade_notional_sum",
        "regime_rows",
        "regime_trade_active_rows",
        "regime_trade_notional_median_all_rows",
        "silver_trade_bucket_count",
    ]:
        if column not in out.columns:
            out[column] = 0
    out["raw_silver_row_delta"] = out["raw_trade_rows"] - out["silver_trade_rows"]
    out["silver_regime_trade_count_delta"] = (
        out["silver_trade_rows"] - out["regime_trade_count_sum"]
    )
    out["raw_silver_notional_delta_abs"] = (
        out["raw_trade_notional_sum"] - out["silver_trade_notional_sum"]
    ).abs()
    out["silver_regime_notional_delta_abs"] = (
        out["silver_trade_notional_sum"] - out["regime_trade_notional_sum"]
    ).abs()
    out["regime_trade_active_row_share"] = (
        out["regime_trade_active_rows"] / out["regime_rows"].replace(0, pd.NA)
    ).fillna(0.0)
    if per_symbol:
        out["activity_median_zero_reason"] = out.apply(
            _activity_median_zero_reason, axis=1
        )
        out["diagnostic_notes"] = out.apply(
            lambda row: _diagnostic_notes(row, config), axis=1
        )
    return out


def _activity_median_zero_reason(row: pd.Series) -> str:
    median_all = float(row.get("regime_trade_notional_median_all_rows", 0.0) or 0.0)
    silver_rows = float(row.get("silver_trade_rows", 0.0) or 0.0)
    active_share = float(row.get("regime_trade_active_row_share", 0.0) or 0.0)
    count_delta = float(row.get("silver_regime_trade_count_delta", 0.0) or 0.0)
    if median_all != 0.0:
        return "median_nonzero"
    if silver_rows <= 0:
        return "no_silver_trades"
    if count_delta != 0.0:
        return "regime_trade_count_mismatch"
    raw_row_delta = float(row.get("raw_silver_row_delta", 0.0) or 0.0)
    raw_notional_delta = float(row.get("raw_silver_notional_delta_abs", 0.0) or 0.0)
    if raw_row_delta != 0.0 or raw_notional_delta != 0.0:
        return "sparse_trade_minutes_with_raw_silver_gap"
    if active_share <= 0.5:
        return "sparse_trade_minutes_not_missing_trade_lineage"
    return "zero_median_unexplained"


def _diagnostic_notes(row: pd.Series, config: LineageAuditConfig) -> str:
    notes: list[str] = []
    if float(row.get("raw_silver_row_delta", 0.0) or 0.0) != 0.0:
        notes.append("raw_silver_trade_row_mismatch")
    if bool(
        _notional_mismatch(
            pd.Series([float(row.get("raw_trade_notional_sum", 0.0) or 0.0)]),
            pd.Series([float(row.get("silver_trade_notional_sum", 0.0) or 0.0)]),
            config,
        ).iloc[0]
    ):
        notes.append("raw_silver_notional_mismatch")
    if float(row.get("silver_regime_trade_count_delta", 0.0) or 0.0) != 0.0:
        notes.append("silver_regime_trade_count_mismatch")
    if bool(
        _notional_mismatch(
            pd.Series([float(row.get("silver_trade_notional_sum", 0.0) or 0.0)]),
            pd.Series([float(row.get("regime_trade_notional_sum", 0.0) or 0.0)]),
            config,
        ).iloc[0]
    ):
        notes.append("silver_regime_notional_mismatch")
    if (
        float(row.get("raw_trade_rows", 0.0) or 0.0) <= 0
        and float(row.get("silver_trade_rows", 0.0) or 0.0) > 0
    ):
        notes.append("silver_trades_without_local_raw_sample")
    if float(row.get("raw_invalid_event_ts_rows", 0.0) or 0.0) > 0:
        notes.append("raw_invalid_event_ts_rows")
    if float(row.get("silver_invalid_event_ts_rows", 0.0) or 0.0) > 0:
        notes.append("silver_invalid_event_ts_rows")
    if (
        row.get("activity_median_zero_reason")
        == "sparse_trade_minutes_not_missing_trade_lineage"
    ):
        notes.append("median_zero_explained_by_sparse_trade_minutes")
    if row.get("activity_median_zero_reason") == "zero_median_unexplained":
        notes.append("zero_median_unexplained")
    return ";".join(notes)


def _verdict(symbol_summary: pd.DataFrame) -> str:
    if symbol_summary.empty:
        return "LINEAGE_AUDIT_NO_DATA_DIAGNOSTIC"
    failure_notes = [
        "raw_silver_trade_row_mismatch",
        "raw_silver_notional_mismatch",
        "silver_regime_trade_count_mismatch",
        "silver_regime_notional_mismatch",
        "zero_median_unexplained",
        "raw_invalid_event_ts_rows",
        "silver_invalid_event_ts_rows",
    ]
    notes = (
        symbol_summary.get("diagnostic_notes", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
    )
    if any(any(note in cell.split(";") for note in failure_notes) for cell in notes):
        return "LINEAGE_AUDIT_FAIL_DIAGNOSTIC"
    return "LINEAGE_AUDIT_PASS_DIAGNOSTIC"


def audit_trade_activity_lineage(config: LineageAuditConfig) -> LineageAuditResult:
    eligibility = _read_eligibility(config.eligibility_path)
    target_symbols = _resolve_target_symbols(config, eligibility)
    if not target_symbols:
        return LineageAuditResult(
            verdict="LINEAGE_AUDIT_NO_DATA_DIAGNOSTIC",
            symbol_summary=pd.DataFrame(),
            date_summary=pd.DataFrame(),
            target_symbols=(),
        )
    raw_dates, raw_symbols = _read_raw_trade_summaries(
        config.raw_dir, target_symbols, bucket=config.bucket
    )
    silver_dates, silver_symbols, _ = _read_silver_trade_summaries(
        config.silver_dir, target_symbols, bucket=config.bucket
    )
    regime_dates, regime_symbols = _read_regime_summaries(
        config.regime_path, target_symbols
    )

    date_summary = _merge_summaries(
        [raw_dates, silver_dates, regime_dates], ["symbol", "date"]
    )
    if not date_summary.empty:
        date_summary = _add_lineage_deltas(
            config=config, frame=date_summary, per_symbol=False
        )
        date_summary = date_summary.sort_values(["symbol", "date"]).reset_index(
            drop=True
        )

    symbol_summary = _merge_summaries(
        [raw_symbols, silver_symbols, regime_symbols], ["symbol"]
    )
    symbol_summary = _attach_eligibility(symbol_summary, eligibility)
    if not symbol_summary.empty:
        symbol_summary = _add_lineage_deltas(
            config=config, frame=symbol_summary, per_symbol=True
        )
        ordered_cols = [
            "symbol",
            "raw_trade_rows",
            "silver_trade_rows",
            "regime_trade_count_sum",
            "raw_silver_row_delta",
            "silver_regime_trade_count_delta",
            "raw_trade_notional_sum",
            "silver_trade_notional_sum",
            "regime_trade_notional_sum",
            "raw_silver_notional_delta_abs",
            "silver_regime_notional_delta_abs",
            "regime_rows",
            "regime_trade_active_rows",
            "regime_trade_active_row_share",
            "regime_trade_notional_median_all_rows",
            "regime_trade_notional_median_active_rows",
            "median_trade_notional_per_min",
            "activity_gate_pass",
            "activity_median_zero_reason",
            "failure_reasons",
            "diagnostic_notes",
        ]
        for column in ordered_cols:
            if column not in symbol_summary.columns:
                symbol_summary[column] = (
                    "" if column in {"failure_reasons", "diagnostic_notes"} else 0
                )
        symbol_summary = symbol_summary[
            ordered_cols + [c for c in symbol_summary.columns if c not in ordered_cols]
        ]
        symbol_summary = symbol_summary.sort_values("symbol").reset_index(drop=True)

    return LineageAuditResult(
        verdict=_verdict(symbol_summary),
        symbol_summary=symbol_summary,
        date_summary=date_summary,
        target_symbols=target_symbols,
    )


def _markdown_report(result: LineageAuditResult, config: LineageAuditConfig) -> str:
    preview_cols = [
        "symbol",
        "raw_trade_rows",
        "silver_trade_rows",
        "regime_trade_count_sum",
        "raw_silver_row_delta",
        "silver_regime_trade_count_delta",
        "regime_trade_active_row_share",
        "regime_trade_notional_median_all_rows",
        "median_trade_notional_per_min",
        "activity_median_zero_reason",
        "diagnostic_notes",
    ]
    preview = (
        result.symbol_summary[
            [c for c in preview_cols if c in result.symbol_summary.columns]
        ]
        if not result.symbol_summary.empty
        else result.symbol_summary
    )
    explained = 0
    raw_silver_mismatches = 0
    silver_regime_mismatches = 0
    unexplained_zero_medians = 0
    if not result.symbol_summary.empty:
        if "activity_median_zero_reason" in result.symbol_summary:
            explained = int(
                result.symbol_summary["activity_median_zero_reason"]
                .eq("sparse_trade_minutes_not_missing_trade_lineage")
                .sum()
            )
            unexplained_zero_medians = int(
                result.symbol_summary["activity_median_zero_reason"]
                .eq("zero_median_unexplained")
                .sum()
            )
        notes = (
            result.symbol_summary.get("diagnostic_notes", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
        )
        raw_silver_mismatches = int(
            notes.str.contains("raw_silver_", regex=False).sum()
        )
        silver_regime_mismatches = int(
            notes.str.contains("silver_regime_", regex=False).sum()
        )
    failure_counter_rows = pd.DataFrame(
        [
            {"counter": "raw/silver mismatches", "symbols": raw_silver_mismatches},
            {
                "counter": "silver/regime trade-count mismatches",
                "symbols": silver_regime_mismatches,
            },
            {"counter": "sparse-trade zero-median explanations", "symbols": explained},
            {
                "counter": "unexplained zero medians",
                "symbols": unexplained_zero_medians,
            },
        ]
    )
    lines = [
        "# Pacifica Trade Activity Lineage Audit",
        "",
        "This is a diagnostic audit of raw trades -> silver trades -> regime trade_count/trade_notional -> eligibility activity metrics.",
        "It is not a strategy, alpha claim, or paper-trading permission.",
        "",
        f"Verdict: `{result.verdict}`",
        f"Symbols audited: {len(result.target_symbols)}",
        f"Sparse-trade zero-median explanations: {explained}",
        "",
        "## Interpretation discipline",
        "",
        "The paper-trading eligibility activity gate currently uses the median 1-minute `trade_notional` over all regime rows for a symbol. If most otherwise-observed BBO/price/book minutes have no trades, that median can be zero even when raw and silver trades are present and correctly aggregated into regime rows.",
        "",
        "A PASS here only means the inspected trade-activity lineage is internally consistent for the audited symbols. It does not make an edge claim and does not authorize trading.",
        "",
        "## Failure counters",
        "",
        dataframe_to_markdown_table(failure_counter_rows),
        "",
        "If raw/silver mismatches appear while silver/regime mismatches are zero, the first suspect is stale research artifacts: local raw cache contains trades that the current silver/regime/eligibility reports have not ingested yet. Refresh silver and regime before treating eligibility as current.",
        "",
        "## Inputs",
        "",
        f"- Raw dir: `{config.raw_dir}`",
        f"- Silver dir: `{config.silver_dir}`",
        f"- Regime state: `{config.regime_path}`",
        f"- Eligibility: `{config.eligibility_path}`",
        f"- Bucket: `{config.bucket}`",
        f"- Target symbols: `{', '.join(result.target_symbols)}`",
        "",
        "## Symbol summary preview",
        "",
        dataframe_to_markdown_table(preview, max_rows=30),
        "",
        "## Output files",
        "",
        "- `symbol_summary.csv` — one row per audited symbol with raw/silver/regime/eligibility deltas and diagnostic notes.",
        "- `date_summary.csv` — one row per audited symbol/date for locating lineage breaks by day.",
        "- `README.md` — this diagnostic report.",
        "",
    ]
    return "\n".join(lines)


def write_trade_activity_lineage_report(config: LineageAuditConfig) -> dict[str, Any]:
    result = audit_trade_activity_lineage(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    result.symbol_summary.to_csv(config.out_dir / "symbol_summary.csv", index=False)
    result.date_summary.to_csv(config.out_dir / "date_summary.csv", index=False)
    (config.out_dir / "README.md").write_text(
        _markdown_report(result, config), encoding="utf-8"
    )
    return {
        "verdict": result.verdict,
        "symbols_audited": len(result.target_symbols),
        "symbol_summary": str(config.out_dir / "symbol_summary.csv"),
        "date_summary": str(config.out_dir / "date_summary.csv"),
        "readme": str(config.out_dir / "README.md"),
    }


def _parse_symbols(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(symbols) or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER_DIR)
    parser.add_argument("--regime-path", type=Path, default=DEFAULT_REGIME_PATH)
    parser.add_argument(
        "--eligibility-path", type=Path, default=DEFAULT_ELIGIBILITY_PATH
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--symbols", nargs="*", help="Symbols to audit, e.g. BTC ETH SOL or BTC,ETH,SOL"
    )
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--bucket", default="1min")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = write_trade_activity_lineage_report(
        LineageAuditConfig(
            raw_dir=args.raw_dir,
            silver_dir=args.silver_dir,
            regime_path=args.regime_path,
            eligibility_path=args.eligibility_path,
            out_dir=args.out_dir,
            symbols=_parse_symbols(args.symbols),
            max_symbols=args.max_symbols,
            bucket=args.bucket,
        )
    )
    print(f"verdict: {result['verdict']}")
    print(f"symbols_audited: {result['symbols_audited']}")
    print(f"wrote report: {result['readme']}")


if __name__ == "__main__":
    main()

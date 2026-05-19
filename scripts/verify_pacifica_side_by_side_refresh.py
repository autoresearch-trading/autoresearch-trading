# scripts/verify_pacifica_side_by_side_refresh.py
"""Verify candidate Pacifica silver/regime refresh against canonical outputs.

This script is intentionally read-only with respect to canonical directories.  It
writes comparison artifacts under a caller-provided report directory and fails
closed on row-count/coverage regressions, duplicate keys, or null required keys.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


SOURCE_METADATA_COLUMNS = {"source_key", "source_path", "source_sha256"}


def _exact_row_duplicates(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    payload_columns = [
        col for col in frame.columns if col not in SOURCE_METADATA_COLUMNS
    ]
    if not payload_columns:
        return 0
    row_hashes = pd.util.hash_pandas_object(frame[payload_columns], index=False)
    return int(row_hashes.duplicated().sum())


def _row_identity(
    frame: pd.DataFrame, columns: list[str], *, empty_value: str
) -> pd.Series:
    if not columns:
        return pd.Series([empty_value] * len(frame), index=frame.index, dtype="string")
    parts = []
    for col in columns:
        values = (
            frame[col].astype("string")
            if col in frame.columns
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
        parts.append(col + "=" + values.fillna("<NA>"))
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="|")
    return out.fillna(empty_value)


def _base_silver_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        col for col in ["symbol", "event_ts_ms", "recv_ms"] if col in frame.columns
    ]
    if columns:
        return frame[columns].copy()
    if {"source_key", "event_ts_ms", "recv_ms"}.issubset(frame.columns):
        return frame[["source_key", "event_ts_ms", "recv_ms"]].copy()
    return pd.DataFrame(index=frame.index)


def _trade_identity(frame: pd.DataFrame) -> pd.Series:
    history = (
        frame["history_id"].astype("string")
        if "history_id" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    fallback = _row_identity(
        frame,
        [
            col
            for col in ["nonce", "price", "qty", "direction", "trade_class"]
            if col in frame.columns
        ],
        empty_value="fallback:<no_trade_identity>",
    )
    has_history = history.notna() & history.ne("")
    return ("history_id=" + history).where(has_history, "fallback:" + fallback)


def _bbo_identity(frame: pd.DataFrame) -> pd.Series:
    order_cols = [col for col in ["order_id", "last_order_id"] if col in frame.columns]
    quote_cols = [
        col
        for col in ["bid_px", "bid_qty", "ask_px", "ask_qty", "mid", "spread_bps"]
        if col in frame.columns
    ]
    order_identity = _row_identity(
        frame, order_cols, empty_value="order:<no_order_identity>"
    )
    quote_identity = _row_identity(
        frame, quote_cols, empty_value="quote:<no_quote_identity>"
    )
    if not order_cols:
        return quote_identity
    has_order = frame[order_cols].notna().any(axis=1)
    combined_identity = order_identity + "|" + quote_identity
    return combined_identity.where(has_order, "quote:" + quote_identity)


def _silver_key_frame(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    key = _base_silver_key_frame(frame)
    if channel == "trades":
        key = key.assign(trade_identity=_trade_identity(frame))
    elif channel == "bbo":
        key = key.assign(bbo_identity=_bbo_identity(frame))
    elif channel in {"candle", "mark_price_candle"}:
        for col in [
            "interval",
            "start_ts_ms",
            "end_ts_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
        ]:
            if col in frame.columns:
                key[col] = frame[col]
    return key


def _silver_duplicate_keys(frame: pd.DataFrame, channel: str) -> int:
    if frame.empty:
        return 0
    key = _silver_key_frame(frame, channel)
    if key.empty:
        return 0
    return int(key.duplicated().sum())


def _silver_key_schema(frame: pd.DataFrame, channel: str) -> str:
    if frame.empty:
        return ""
    key = _silver_key_frame(frame, channel)
    return ";".join(map(str, key.columns))


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _duckdb_scan_sql() -> str:
    return "read_parquet(?, union_by_name=true, hive_partitioning=false)"


def _has_parquet_file(path: Path) -> bool:
    return path.exists() and next(path.rglob("*.parquet"), None) is not None


def _silver_parquet_inputs(root: Path, channel: str) -> list[str]:
    inputs: list[str] = []
    flat_path = root / f"{channel}.parquet"
    if flat_path.exists():
        inputs.append(str(flat_path))
    partition_dir = root / f"channel={channel}"
    if _has_parquet_file(partition_dir):
        inputs.append(str(partition_dir / "**" / "*.parquet"))
    return inputs


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    spill_dir = ROOT / ".tmp" / "duckdb-verifier-spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = {_sql_literal(str(spill_dir))}")
    memory_limit = os.environ.get("PACIFICA_VERIFIER_DUCKDB_MEMORY_LIMIT", "8GB")
    if memory_limit:
        con.execute(f"SET memory_limit = {_sql_literal(memory_limit)}")
    return con


def _duckdb_columns(con: duckdb.DuckDBPyConnection, inputs: list[str]) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM {_duckdb_scan_sql()}", [inputs]
    ).fetchall()
    return [str(row[0]) for row in rows]


def _duckdb_row_count(con: duckdb.DuckDBPyConnection, inputs: list[str]) -> int:
    value = con.execute(
        f"SELECT COUNT(*) FROM {_duckdb_scan_sql()}", [inputs]
    ).fetchone()[0]
    return int(value or 0)


def _duckdb_key_nulls(
    con: duckdb.DuckDBPyConnection,
    inputs: list[str],
    required_key_cols: list[str],
    *,
    row_count: int,
    columns: list[str],
) -> int:
    if any(col not in columns for col in required_key_cols):
        return row_count
    predicates = " OR ".join(
        f"{_quote_identifier(col)} IS NULL" for col in required_key_cols
    )
    value = con.execute(
        f"SELECT COUNT(*) FROM {_duckdb_scan_sql()} WHERE {predicates}", [inputs]
    ).fetchone()[0]
    return int(value or 0)


def _duckdb_coverage_rows(
    con: duckdb.DuckDBPyConnection, inputs: list[str], channel: str, columns: list[str]
) -> list[dict[str, Any]]:
    if not {"symbol", "event_ts_ms"}.issubset(columns):
        return []
    rows = con.execute(
        f"""
        SELECT DISTINCT
            CAST({_quote_identifier('symbol')} AS VARCHAR) AS symbol,
            strftime(epoch_ms(CAST({_quote_identifier('event_ts_ms')} AS BIGINT)), '%Y-%m-%d') AS date
        FROM {_duckdb_scan_sql()}
        WHERE {_quote_identifier('symbol')} IS NOT NULL
          AND {_quote_identifier('event_ts_ms')} IS NOT NULL
        ORDER BY symbol, date
        """,
        [inputs],
    ).fetchall()
    return [
        {"channel": channel, "symbol": str(symbol), "date": str(date)}
        for symbol, date in rows
        if symbol is not None and date is not None
    ]


def _sql_row_identity(columns: list[str], *, empty_value: str) -> str:
    if not columns:
        return _sql_literal(empty_value)
    parts = [
        f"{_sql_literal(col + '=')} || COALESCE(CAST({_quote_identifier(col)} AS VARCHAR), {_sql_literal('<NA>')})"
        for col in columns
    ]
    return (f" || {_sql_literal('|')} || ").join(parts)


def _trade_identity_sql(columns: list[str]) -> str:
    fallback_cols = [
        col
        for col in ["nonce", "price", "qty", "direction", "trade_class"]
        if col in columns
    ]
    fallback = _sql_row_identity(
        fallback_cols, empty_value="fallback:<no_trade_identity>"
    )
    if "history_id" not in columns:
        return f"{_sql_literal('fallback:')} || {fallback}"
    history = _quote_identifier("history_id")
    return (
        f"CASE WHEN {history} IS NOT NULL AND CAST({history} AS VARCHAR) <> {_sql_literal('')} "
        f"THEN {_sql_literal('history_id=')} || CAST({history} AS VARCHAR) "
        f"ELSE {_sql_literal('fallback:')} || {fallback} END"
    )


def _bbo_identity_sql(columns: list[str]) -> str:
    order_cols = [col for col in ["order_id", "last_order_id"] if col in columns]
    quote_cols = [
        col
        for col in ["bid_px", "bid_qty", "ask_px", "ask_qty", "mid", "spread_bps"]
        if col in columns
    ]
    order_identity = _sql_row_identity(
        order_cols, empty_value="order:<no_order_identity>"
    )
    quote_identity = _sql_row_identity(
        quote_cols, empty_value="quote:<no_quote_identity>"
    )
    if not order_cols:
        return quote_identity
    has_order = " OR ".join(
        f"{_quote_identifier(col)} IS NOT NULL" for col in order_cols
    )
    return (
        f"CASE WHEN {has_order} THEN {order_identity} || {_sql_literal('|')} || {quote_identity} "
        f"ELSE {_sql_literal('quote:')} || {quote_identity} END"
    )


def _base_silver_key_sql(columns: list[str]) -> tuple[list[str], list[str]]:
    names = [col for col in ["symbol", "event_ts_ms", "recv_ms"] if col in columns]
    if names:
        return [_quote_identifier(col) for col in names], names
    fallback_names = ["source_key", "event_ts_ms", "recv_ms"]
    if set(fallback_names).issubset(columns):
        return [_quote_identifier(col) for col in fallback_names], fallback_names
    return [], []


def _silver_key_sql(columns: list[str], channel: str) -> tuple[list[str], list[str]]:
    expressions, names = _base_silver_key_sql(columns)
    if channel == "trades":
        expressions.append(_trade_identity_sql(columns))
        names.append("trade_identity")
    elif channel == "bbo":
        expressions.append(_bbo_identity_sql(columns))
        names.append("bbo_identity")
    elif channel in {"candle", "mark_price_candle"}:
        for col in [
            "interval",
            "start_ts_ms",
            "end_ts_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
        ]:
            if col in columns:
                expressions.append(_quote_identifier(col))
                names.append(col)
    return expressions, names


def _duckdb_duplicate_count(
    con: duckdb.DuckDBPyConnection, inputs: list[str], expressions: list[str]
) -> int:
    if not expressions:
        return 0
    group_by = ", ".join(expressions)
    value = con.execute(
        f"""
        WITH duplicate_groups AS (
            SELECT COUNT(*) AS n
            FROM {_duckdb_scan_sql()}
            GROUP BY {group_by}
            HAVING COUNT(*) > 1
        )
        SELECT COALESCE(SUM(n - 1), 0)::BIGINT FROM duplicate_groups
        """,
        [inputs],
    ).fetchone()[0]
    return int(value or 0)


def _duckdb_exact_payload_duplicates(
    con: duckdb.DuckDBPyConnection,
    inputs: list[str],
    columns: list[str],
    *,
    row_count: int,
) -> int:
    if row_count < 2:
        return 0
    payload_columns = [col for col in columns if col not in SOURCE_METADATA_COLUMNS]
    if not payload_columns:
        return 0
    group_by = ", ".join(_quote_identifier(col) for col in payload_columns)
    value = con.execute(
        f"""
        WITH duplicate_payloads AS (
            SELECT COUNT(*) AS n
            FROM {_duckdb_scan_sql()}
            GROUP BY {group_by}
            HAVING COUNT(*) > 1
        )
        SELECT COALESCE(SUM(n - 1), 0)::BIGINT FROM duplicate_payloads
        """,
        [inputs],
    ).fetchone()[0]
    return int(value or 0)


def _empty_silver_quality_row(channel: str) -> dict[str, Any]:
    return {
        "channel": channel,
        "key_nulls": 0,
        "duplicate_keys": 0,
        "exact_row_duplicates": 0,
        "duplicate_key_schema": "",
        "missing_key_columns": "",
    }


def _silver_channel_metrics(
    root: Path, channel: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    inputs = _silver_parquet_inputs(root, channel)
    if not inputs:
        return {"channel": channel, "rows": 0}, [], _empty_silver_quality_row(channel)

    con = _duckdb_connection()
    try:
        columns = _duckdb_columns(con, inputs)
        row_count = _duckdb_row_count(con, inputs)
        count_row = {"channel": channel, "rows": row_count}
        if row_count == 0:
            return count_row, [], _empty_silver_quality_row(channel)

        coverage_rows = _duckdb_coverage_rows(con, inputs, channel, columns)
        required_key_cols = ["symbol", "event_ts_ms"]
        missing_key_columns = [col for col in required_key_cols if col not in columns]
        key_expressions, key_names = _silver_key_sql(columns, channel)
        quality_row = {
            "channel": channel,
            "key_nulls": _duckdb_key_nulls(
                con,
                inputs,
                required_key_cols,
                row_count=row_count,
                columns=columns,
            ),
            "duplicate_keys": _duckdb_duplicate_count(con, inputs, key_expressions),
            "exact_row_duplicates": _duckdb_exact_payload_duplicates(
                con, inputs, columns, row_count=row_count
            ),
            "duplicate_key_schema": ";".join(key_names),
            "missing_key_columns": ";".join(missing_key_columns),
        }
        return count_row, coverage_rows, quality_row
    finally:
        con.close()


def _silver_metrics(
    root: Path, channels: list[str] | tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for channel in channels:
        count_row, channel_coverage, quality_row = _silver_channel_metrics(
            root, channel
        )
        count_rows.append(count_row)
        coverage_rows.extend(channel_coverage)
        quality_rows.append(quality_row)
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
    if (
        silver_quality.get("exact_row_duplicates_candidate", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
        > silver_quality.get("exact_row_duplicates_canonical", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
    ).any():
        failures.append("candidate_silver_exact_row_duplicates")

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

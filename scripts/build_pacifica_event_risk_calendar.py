# scripts/build_pacifica_event_risk_calendar.py
"""Build a diagnostic event/calendar risk overlay for Pacifica regime rows.

This layer marks known event-risk windows from a local CSV/parquet calendar. It
is context/governor infrastructure only, not a strategy or trade signal.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_EVENTS_PATH = Path("data/pacifica_event_calendar/events.csv")
DEFAULT_OUT_DIR = Path("docs/experiments/event-risk-calendar")
EVENT_RISK_VERSION = "pacifica_event_risk_v1_fixed_diagnostic"
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
REQUIRED_STATE_COLUMNS = ("symbol", "bucket_start_ms")
REQUIRED_EVENT_COLUMNS = (
    "event_timestamp",
    "event_type",
    "pre_window_minutes",
    "post_window_minutes",
    "severity",
    "source_note",
)
SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
RISK_STATE_BY_SEVERITY = {
    "LOW": "EVENT_RISK_LOW",
    "MEDIUM": "EVENT_RISK_MEDIUM",
    "HIGH": "EVENT_RISK_HIGH",
}


@dataclass(frozen=True)
class EventRiskConfig:
    event_risk_version: str = EVENT_RISK_VERSION
    empty_calendar_verdict: str = "NO_EVENTS_CONFIGURED_DIAGNOSTIC"


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


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def _validate_required(
    frame: pd.DataFrame, required: tuple[str, ...], *, source: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _validate_symbols(frame: pd.DataFrame, *, source: str) -> pd.Series:
    raw = frame["symbol"]
    if raw.isna().any():
        raise ValueError(f"{source} contains null or blank symbols")
    as_text = raw.astype(str)
    if (as_text != as_text.str.strip()).any():
        raise ValueError(f"{source} contains noncanonical symbol whitespace")
    if (as_text.str.strip() == "").any():
        raise ValueError(f"{source} contains null or blank symbols")
    reserved = as_text.str.lower().isin({"nan", "none", "null"})
    canonical = as_text.map(lambda value: bool(SYMBOL_RE.fullmatch(value))).astype(bool)
    invalid = (~canonical) | reserved.astype(bool)
    if invalid.any():
        bad = sorted(as_text[invalid].unique())
        raise ValueError(f"{source} contains noncanonical symbol values: {bad}")
    return as_text


def _coerce_bucket_ms(frame: pd.DataFrame, *, source: str) -> pd.Series:
    bucket = pd.to_numeric(frame["bucket_start_ms"], errors="coerce")
    invalid = (
        bucket.isna()
        | ~bucket.map(lambda value: math.isfinite(float(value)))
        | (bucket < 0)
        | (bucket % 1 != 0)
    )
    if invalid.any():
        raise ValueError(f"{source} contains invalid bucket_start_ms")
    return bucket.astype("int64")


def _validate_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    if raw.isna().any():
        raise ValueError(f"event calendar contains blank {column}")
    as_text = raw.astype(str)
    if (as_text.str.strip() == "").any():
        raise ValueError(f"event calendar contains blank {column}")
    if (as_text != as_text.str.strip()).any():
        raise ValueError(f"event calendar contains noncanonical whitespace in {column}")
    unsafe = as_text.str.contains(r"[;\r\n\t]", regex=True)
    if unsafe.any():
        raise ValueError(
            f"event calendar contains unsafe delimiter/control characters in {column}"
        )
    return as_text


def _coerce_window_minutes(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = (
        values.isna()
        | ~values.map(lambda value: math.isfinite(float(value)))
        | (values < 0)
        | (values % 1 != 0)
    )
    if invalid.any():
        raise ValueError(f"event calendar contains invalid {column}")
    return values.astype("int64")


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    _validate_required(events, REQUIRED_EVENT_COLUMNS, source="event calendar")
    out = events.copy()
    if out.empty:
        return out.assign(
            event_ts=pd.to_datetime(pd.Series([], dtype="object"), utc=True),
            pre_window_minutes=pd.Series([], dtype="int64"),
            post_window_minutes=pd.Series([], dtype="int64"),
            event_start_ts=pd.to_datetime(pd.Series([], dtype="object"), utc=True),
            event_end_ts=pd.to_datetime(pd.Series([], dtype="object"), utc=True),
        )
    parsed_ts = pd.to_datetime(out["event_timestamp"], utc=True, errors="coerce")
    if parsed_ts.isna().any():
        raise ValueError("event calendar contains invalid event_timestamp")
    out["event_ts"] = parsed_ts
    out["event_type"] = _validate_text_column(out, "event_type")
    out["source_note"] = _validate_text_column(out, "source_note")
    out["pre_window_minutes"] = _coerce_window_minutes(out, "pre_window_minutes")
    out["post_window_minutes"] = _coerce_window_minutes(out, "post_window_minutes")
    severity_text = out["severity"].astype(str)
    severity = severity_text.str.strip().str.upper()
    unknown = ~severity.isin(SEVERITY_RANK)
    if unknown.any():
        bad = sorted(severity[unknown].unique())
        raise ValueError(f"event calendar contains unknown severity values: {bad}")
    if (severity != severity_text).any():
        raise ValueError(
            "event calendar contains noncanonical whitespace/case in severity"
        )
    out["severity"] = severity
    out["severity_rank"] = out["severity"].map(SEVERITY_RANK).astype(int)
    out["event_start_ts"] = out["event_ts"] - pd.to_timedelta(
        out["pre_window_minutes"], unit="m"
    )
    out["event_end_ts"] = out["event_ts"] + pd.to_timedelta(
        out["post_window_minutes"], unit="m"
    )
    return out


def _normalize_state(state: pd.DataFrame) -> pd.DataFrame:
    _validate_required(state, REQUIRED_STATE_COLUMNS, source="state")
    out = state.copy()
    out["symbol"] = _validate_symbols(out, source="state")
    out["bucket_start_ms"] = _coerce_bucket_ms(out, source="state")
    out["bucket_start_ts"] = pd.to_datetime(out["bucket_start_ms"], unit="ms", utc=True)
    return out


def _phase_for(ts: pd.Timestamp, event: pd.Series) -> str:
    if ts <= event["event_ts"]:
        return "pre_or_at_event"
    return "post_event"


def _event_columns_for_row(ts: pd.Timestamp, active: pd.DataFrame) -> dict[str, Any]:
    if active.empty:
        return {
            "event_risk_state": "NO_KNOWN_EVENT_RISK",
            "max_event_severity": "NONE",
            "active_event_count": 0,
            "active_event_types": "",
            "active_event_source_notes": "",
            "event_window_phase": "none",
        }
    max_rank = int(active["severity_rank"].max())
    max_severity = next(
        name for name, rank in SEVERITY_RANK.items() if rank == max_rank
    )
    event_types = sorted(active["event_type"].unique())
    source_notes = sorted(active["source_note"].unique())
    phases = sorted({_phase_for(ts, row) for _, row in active.iterrows()})
    phase = phases[0] if len(phases) == 1 else "mixed"
    return {
        "event_risk_state": RISK_STATE_BY_SEVERITY[max_severity],
        "max_event_severity": max_severity,
        "active_event_count": int(len(active)),
        "active_event_types": ";".join(event_types),
        "active_event_source_notes": ";".join(source_notes),
        "event_window_phase": phase,
    }


def build_event_risk_calendar(
    state: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: EventRiskConfig = EventRiskConfig(),
) -> pd.DataFrame:
    """Mark state rows with known event-risk windows from a local calendar."""
    state_norm = _normalize_state(state)
    events_norm = _normalize_events(events)
    rows: list[dict[str, Any]] = []
    for _, state_row in state_norm.iterrows():
        ts = state_row["bucket_start_ts"]
        if events_norm.empty:
            active = events_norm
        else:
            active = events_norm[
                (events_norm["event_start_ts"] <= ts)
                & (events_norm["event_end_ts"] >= ts)
            ]
        rows.append(_event_columns_for_row(ts, active))
    event_cols = pd.DataFrame(
        rows,
        columns=[
            "event_risk_state",
            "max_event_severity",
            "active_event_count",
            "active_event_types",
            "active_event_source_notes",
            "event_window_phase",
        ],
    )
    out = pd.concat([state_norm.reset_index(drop=True), event_cols], axis=1)
    out["events_configured"] = not events_norm.empty
    out["event_risk_version"] = config.event_risk_version
    first = [
        "symbol",
        "bucket_start_ms",
        "bucket_start_ts",
        "event_risk_state",
        "max_event_severity",
        "active_event_count",
        "active_event_types",
        "event_window_phase",
        "active_event_source_notes",
        "events_configured",
        "event_risk_version",
    ]
    rest = [column for column in out.columns if column not in first]
    return (
        out[first + rest]
        .sort_values(["symbol", "bucket_start_ms"])
        .reset_index(drop=True)
    )


def summarize_event_risk(marked: pd.DataFrame) -> pd.DataFrame:
    if marked.empty:
        return pd.DataFrame(columns=["event_risk_state", "rows", "row_share"])
    total = len(marked)
    summary = marked.groupby("event_risk_state", as_index=False).agg(
        rows=("symbol", "size")
    )
    summary["row_share"] = summary["rows"] / total
    return summary.sort_values(
        ["rows", "event_risk_state"], ascending=[False, True]
    ).reset_index(drop=True)


def summarize_symbol_event_risk(marked: pd.DataFrame) -> pd.DataFrame:
    if marked.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "rows",
                "event_risk_rows",
                "event_risk_rate",
                "max_event_severity",
            ]
        )
    summary = (
        marked.groupby("symbol", as_index=False)
        .agg(
            rows=("bucket_start_ms", "size"),
            event_risk_rows=("active_event_count", lambda s: int((s > 0).sum())),
            max_event_severity=("max_event_severity", lambda s: _max_severity(s)),
        )
        .sort_values(
            ["event_risk_rows", "rows", "symbol"], ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )
    summary["event_risk_rate"] = summary["event_risk_rows"] / summary["rows"].replace(
        0, pd.NA
    )
    summary["event_risk_rate"] = summary["event_risk_rate"].fillna(0.0)
    return summary


def _max_severity(values: pd.Series) -> str:
    ranks = [SEVERITY_RANK.get(str(value), 0) for value in values]
    max_rank = max(ranks) if ranks else 0
    for severity, rank in SEVERITY_RANK.items():
        if rank == max_rank:
            return severity
    return "NONE"


def _verdict(
    marked: pd.DataFrame, *, config: EventRiskConfig = EventRiskConfig()
) -> str:
    if marked.empty:
        return "NO_STATE_ROWS_DIAGNOSTIC"
    if "events_configured" in marked.columns and not bool(
        marked["events_configured"].any()
    ):
        return config.empty_calendar_verdict
    if int((marked["active_event_count"] > 0).sum()) == 0:
        return "NO_ACTIVE_EVENT_WINDOWS_DIAGNOSTIC"
    return "EVENT_RISK_CONTEXT_BUILT_DIAGNOSTIC"


def write_event_risk_report(
    marked: pd.DataFrame,
    out_dir: Path,
    *,
    config: EventRiskConfig = EventRiskConfig(),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_summary = summarize_event_risk(marked)
    symbol_summary = summarize_symbol_event_risk(marked)
    config_frame = pd.DataFrame([asdict(config)])
    verdict = _verdict(marked, config=config)

    marked.to_csv(out_dir / "event_risk_rows.csv", index=False)
    risk_summary.to_csv(out_dir / "event_risk_summary.csv", index=False)
    symbol_summary.to_csv(out_dir / "symbol_event_risk_summary.csv", index=False)
    config_frame.to_csv(out_dir / "config.csv", index=False)

    lines = [
        "# Event/calendar risk layer",
        "",
        f"Verdict: `{verdict}`",
        "",
        "This is a diagnostic context/governor layer, not a trade signal and not permission to paper/live trade.",
        "It marks known event-risk windows from local CSV/parquet calendar input so future strategies can skip, reduce, or annotate event windows without post-hoc tuning.",
        "",
        "## Interpretation discipline",
        "",
        "- Event windows come from an explicit local calendar; no hidden external API is queried here.",
        "- `NO_KNOWN_EVENT_RISK` only means no configured event window matched the row, not that the market is safe.",
        "- Event severity labels are fixed diagnostic annotations, not optimized alpha parameters.",
        "",
        "## Event-risk summary",
        "",
        dataframe_to_markdown_table(risk_summary),
        "",
        "## Symbol event-risk summary",
        "",
        dataframe_to_markdown_table(symbol_summary, max_rows=25),
        "",
        "## Config",
        "",
        dataframe_to_markdown_table(config_frame),
        "",
        "## Artifacts",
        "",
        "- `event_risk_rows.csv` (generated locally for full row-level inspection; intentionally ignored in git for full-run outputs because it can exceed GitHub blob limits)",
        "- `event_risk_summary.csv`",
        "- `symbol_event_risk_summary.csv`",
        "- `config.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    return {
        "verdict": verdict,
        "rows": int(len(marked)),
        "event_risk_rows": (
            int((marked["active_event_count"] > 0).sum()) if not marked.empty else 0
        ),
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Pacifica diagnostic event/calendar risk context"
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    state = _read_frame(args.state)
    events = _read_frame(args.events)
    marked = build_event_risk_calendar(state, events)
    result = write_event_risk_report(marked, args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"event_risk_rows: {result['event_risk_rows']}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

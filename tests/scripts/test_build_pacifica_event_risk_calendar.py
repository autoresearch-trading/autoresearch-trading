from pathlib import Path

import pandas as pd

from scripts.build_pacifica_event_risk_calendar import (
    EventRiskConfig,
    build_event_risk_calendar,
    write_event_risk_report,
)


def _state_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_767_272_340_000,
                "toxicity_score": 0.10,
            },  # 12:59
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_767_272_400_000,
                "toxicity_score": 0.20,
            },  # 13:00
            {
                "symbol": "ETH",
                "bucket_start_ms": 1_767_272_460_000,
                "toxicity_score": 0.30,
            },  # 13:01
            {
                "symbol": "SOL",
                "bucket_start_ms": 1_767_272_820_000,
                "toxicity_score": 0.40,
            },  # 13:07
        ]
    )


def _event_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_timestamp": "2026-01-01T13:00:00Z",
                "event_type": "FOMC_MINUTES",
                "pre_window_minutes": 1,
                "post_window_minutes": 2,
                "severity": "HIGH",
                "source_note": "manual calendar fixture",
            },
            {
                "event_timestamp": "2026-01-01T13:05:00Z",
                "event_type": "ETF_FLOW_PRINT",
                "pre_window_minutes": 1,
                "post_window_minutes": 5,
                "severity": "LOW",
                "source_note": "manual calendar fixture",
            },
        ]
    )


def test_build_event_risk_calendar_marks_pre_and_post_event_windows() -> None:
    marked = build_event_risk_calendar(_state_rows(), _event_rows())
    by_key = marked.set_index(["symbol", "bucket_start_ms"])

    btc_pre = by_key.loc[("BTC", 1_767_272_340_000)]
    assert btc_pre["event_risk_state"] == "EVENT_RISK_HIGH"
    assert btc_pre["active_event_count"] == 1
    assert btc_pre["active_event_types"] == "FOMC_MINUTES"
    assert btc_pre["event_window_phase"] == "pre_or_at_event"

    eth_post = by_key.loc[("ETH", 1_767_272_460_000)]
    assert eth_post["event_risk_state"] == "EVENT_RISK_HIGH"
    assert eth_post["event_window_phase"] == "post_event"

    sol = by_key.loc[("SOL", 1_767_272_820_000)]
    assert sol["event_risk_state"] == "EVENT_RISK_LOW"
    assert sol["active_event_types"] == "ETF_FLOW_PRINT"


def test_overlapping_events_choose_highest_severity_and_keep_all_event_types() -> None:
    events = pd.concat(
        [
            _event_rows().head(1),
            pd.DataFrame(
                [
                    {
                        "event_timestamp": "2026-01-01T13:00:30Z",
                        "event_type": "EXCHANGE_OUTAGE_RISK",
                        "pre_window_minutes": 1,
                        "post_window_minutes": 2,
                        "severity": "MEDIUM",
                        "source_note": "manual incident note",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    marked = build_event_risk_calendar(_state_rows().head(2), events)
    row = marked[marked["bucket_start_ms"] == 1_767_272_400_000].iloc[0]

    assert row["event_risk_state"] == "EVENT_RISK_HIGH"
    assert row["active_event_count"] == 2
    assert row["active_event_types"] == "EXCHANGE_OUTAGE_RISK;FOMC_MINUTES"
    assert row["max_event_severity"] == "HIGH"


def test_empty_calendar_marks_no_known_event_risk_without_failing() -> None:
    empty_events = pd.DataFrame(
        columns=[
            "event_timestamp",
            "event_type",
            "pre_window_minutes",
            "post_window_minutes",
            "severity",
            "source_note",
        ]
    )

    marked = build_event_risk_calendar(_state_rows(), empty_events)

    assert set(marked["event_risk_state"]) == {"NO_KNOWN_EVENT_RISK"}
    assert int(marked["active_event_count"].sum()) == 0


def test_event_calendar_fails_closed_on_missing_required_columns() -> None:
    missing_state = _state_rows().drop(columns=["bucket_start_ms"])
    try:
        build_event_risk_calendar(missing_state, _event_rows())
    except ValueError as exc:
        assert "state missing required columns" in str(exc)
        assert "bucket_start_ms" in str(exc)
    else:
        raise AssertionError("missing state column did not fail closed")

    missing_events = _event_rows().drop(columns=["source_note"])
    try:
        build_event_risk_calendar(_state_rows(), missing_events)
    except ValueError as exc:
        assert "event calendar missing required columns" in str(exc)
        assert "source_note" in str(exc)
    else:
        raise AssertionError("missing event column did not fail closed")


def test_event_calendar_rejects_invalid_timestamps_windows_and_severity() -> None:
    invalid_ts = _event_rows().head(1).copy()
    invalid_ts.loc[0, "event_timestamp"] = "not-a-date"
    try:
        build_event_risk_calendar(_state_rows(), invalid_ts)
    except ValueError as exc:
        assert "invalid event_timestamp" in str(exc)
    else:
        raise AssertionError("invalid timestamp did not fail closed")

    invalid_window = _event_rows().head(1).copy()
    invalid_window.loc[0, "pre_window_minutes"] = -1
    try:
        build_event_risk_calendar(_state_rows(), invalid_window)
    except ValueError as exc:
        assert "invalid pre_window_minutes" in str(exc)
    else:
        raise AssertionError("negative window did not fail closed")

    invalid_severity = _event_rows().head(1).copy()
    invalid_severity.loc[0, "severity"] = "critical"
    try:
        build_event_risk_calendar(_state_rows(), invalid_severity)
    except ValueError as exc:
        assert "unknown severity" in str(exc)
    else:
        raise AssertionError("unknown severity did not fail closed")


def test_state_keys_and_event_text_are_strict() -> None:
    dirty_symbol = _state_rows().head(1).copy()
    dirty_symbol.loc[0, "symbol"] = " BTC"
    try:
        build_event_risk_calendar(dirty_symbol, _event_rows())
    except ValueError as exc:
        assert "noncanonical symbol" in str(exc)
    else:
        raise AssertionError("dirty symbol did not fail closed")

    blank_type = _event_rows().head(1).copy()
    blank_type.loc[0, "event_type"] = "   "
    try:
        build_event_risk_calendar(_state_rows(), blank_type)
    except ValueError as exc:
        assert "blank event_type" in str(exc)
    else:
        raise AssertionError("blank event type did not fail closed")

    delimiter_type = _event_rows().head(1).copy()
    delimiter_type.loc[0, "event_type"] = "TYPE;INJECT"
    try:
        build_event_risk_calendar(_state_rows(), delimiter_type)
    except ValueError as exc:
        assert "unsafe delimiter/control characters" in str(exc)
    else:
        raise AssertionError("delimiter in event type did not fail closed")

    newline_note = _event_rows().head(1).copy()
    newline_note.loc[0, "source_note"] = "note\nother"
    try:
        build_event_risk_calendar(_state_rows(), newline_note)
    except ValueError as exc:
        assert "unsafe delimiter/control characters" in str(exc)
    else:
        raise AssertionError("newline in source note did not fail closed")


def test_write_event_risk_report_distinguishes_empty_calendar_from_inactive_events(
    tmp_path: Path,
) -> None:
    empty_events = pd.DataFrame(
        columns=[
            "event_timestamp",
            "event_type",
            "pre_window_minutes",
            "post_window_minutes",
            "severity",
            "source_note",
        ]
    )
    marked = build_event_risk_calendar(_state_rows(), empty_events)
    result = write_event_risk_report(marked, tmp_path, config=EventRiskConfig())

    assert result["verdict"] == "NO_EVENTS_CONFIGURED_DIAGNOSTIC"
    assert "no configured event window matched" in (tmp_path / "README.md").read_text()


def test_empty_state_rows_emit_no_state_diagnostic(tmp_path: Path) -> None:
    empty_state = pd.DataFrame(columns=["symbol", "bucket_start_ms"])

    marked = build_event_risk_calendar(empty_state, _event_rows())
    result = write_event_risk_report(marked, tmp_path, config=EventRiskConfig())

    assert marked.empty
    assert result["verdict"] == "NO_STATE_ROWS_DIAGNOSTIC"


def test_write_event_risk_report_emits_markdown_and_csvs(tmp_path: Path) -> None:
    marked = build_event_risk_calendar(_state_rows(), _event_rows())
    result = write_event_risk_report(marked, tmp_path, config=EventRiskConfig())

    assert result["verdict"] == "EVENT_RISK_CONTEXT_BUILT_DIAGNOSTIC"
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "event_risk_rows.csv").exists()
    assert (tmp_path / "event_risk_summary.csv").exists()
    assert (tmp_path / "symbol_event_risk_summary.csv").exists()
    assert (tmp_path / "config.csv").exists()
    text = (tmp_path / "README.md").read_text()
    assert "Event/calendar risk layer" in text
    assert "not a trade signal" in text

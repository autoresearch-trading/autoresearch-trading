import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.check_pacifica_r2_freshness import (
    analyze_lsf_samples,
    parse_lsf_listing,
    parse_rclone_time,
)


def _set_tz(monkeypatch: pytest.MonkeyPatch, tz: str | None) -> None:
    if not hasattr(time, "tzset"):
        pytest.skip("process timezone changes require time.tzset")
    if tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", tz)
    time.tzset()


def test_parse_rclone_time_treats_lsf_timestamp_as_local_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_tz = os.environ.get("TZ")
    try:
        _set_tz(monkeypatch, "Etc/GMT+5")

        parsed = parse_rclone_time("2026-05-12 08:06:41")

        assert parsed == datetime(2026, 5, 12, 13, 6, 41, tzinfo=UTC)
    finally:
        _set_tz(monkeypatch, old_tz)


def test_parse_lsf_listing_keeps_payloads_and_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tz(monkeypatch, "UTC")
    rows = parse_lsf_listing(
        "hour=05/run-a.jsonl.gz;123;2026-05-12 05:01:02\n"
        "hour=05/run-a.jsonl.gz.sha256;96;2026-05-12 05:02:03\n"
        "hour=05/tmp.partial;7;2026-05-12 05:03:04\n",
        sample_prefix="channel=bbo/symbol=BTC/date=2026-05-12",
    )

    assert [row["relative_path"] for row in rows] == [
        "hour=05/run-a.jsonl.gz",
        "hour=05/run-a.jsonl.gz.sha256",
    ]
    assert rows[0]["size_bytes"] == 123
    assert rows[0]["modified_at"] == "2026-05-12T05:01:02+00:00"


def test_analyze_lsf_samples_pages_when_latest_payload_is_stale_or_sidecar_missing() -> (
    None
):
    now = datetime(2026, 5, 12, 10, 0, tzinfo=UTC)
    stale = now - timedelta(hours=5)
    rows = [
        {
            "sample_prefix": "channel=bbo/symbol=BTC/date=2026-05-12",
            "relative_path": "hour=05/run-a.jsonl.gz",
            "size_bytes": 123,
            "modified_at": stale.isoformat(),
        }
    ]

    status = analyze_lsf_samples(rows, now=now, stale_after_min=180)

    assert status["ok"] is False
    assert status["latest_payload_age_min"] == 300.0
    assert status["sidecar_missing_count"] == 1
    assert "R2_REMOTE_FRESHNESS_STALE" in status["failures"]
    assert "R2_SIDECAR_MISSING" in status["failures"]


def test_analyze_lsf_samples_ok_for_fresh_payload_with_sidecar() -> None:
    now = datetime(2026, 5, 12, 10, 0, tzinfo=UTC)
    fresh = now - timedelta(minutes=80)
    rows = [
        {
            "sample_prefix": "channel=trades/symbol=BTC/date=2026-05-12",
            "relative_path": "hour=08/run-a.jsonl.gz",
            "size_bytes": 123,
            "modified_at": fresh.isoformat(),
        },
        {
            "sample_prefix": "channel=trades/symbol=BTC/date=2026-05-12",
            "relative_path": "hour=08/run-a.jsonl.gz.sha256",
            "size_bytes": 96,
            "modified_at": fresh.isoformat(),
        },
    ]

    status = analyze_lsf_samples(rows, now=now, stale_after_min=180)

    assert status["ok"] is True
    assert status["failures"] == []
    assert status["latest_payload_age_min"] == 80.0
    assert status["sidecar_missing_count"] == 0

import gzip
import json
from pathlib import Path

import pytest

from scripts.collect_pacifica_full_fidelity import (
    DEFAULT_INTERVALS,
    build_subscriptions,
    channel_symbol_date,
    ensure_min_free_disk,
    event_rows_from_message,
    parse_symbol_filter,
    write_jsonl_records,
)


def test_parse_symbol_filter_deduplicates_and_strips_values():
    assert parse_symbol_filter(" BTC, ETH ,BTC,,SOL ") == ["BTC", "ETH", "SOL"]


def test_build_subscriptions_covers_full_public_market_stream_grid():
    subscriptions = build_subscriptions(
        ["BTC", "ETH"], intervals=["1m"], agg_levels=[1]
    )

    assert {tuple(sorted(sub["params"].items())) for sub in subscriptions} >= {
        tuple(sorted({"source": "prices"}.items())),
        tuple(sorted({"source": "trades", "symbol": "BTC"}.items())),
        tuple(sorted({"source": "book", "symbol": "BTC", "agg_level": 1}.items())),
        tuple(sorted({"source": "bbo", "symbol": "BTC"}.items())),
        tuple(sorted({"source": "candle", "symbol": "BTC", "interval": "1m"}.items())),
        tuple(
            sorted(
                {
                    "source": "mark_price_candle",
                    "symbol": "BTC",
                    "interval": "1m",
                }.items()
            )
        ),
    }
    assert all(sub["method"] == "subscribe" for sub in subscriptions)
    assert len(subscriptions) == 1 + 2 * 5


def test_channel_symbol_date_for_batched_prices_uses_symbol_and_event_time():
    record = {
        "recv_ms": 1777582375000,
        "channel": "prices",
        "data": {"symbol": "BTC", "timestamp": 1777582374310},
    }

    assert channel_symbol_date(record) == ("prices", "BTC", "2026-04-30")


def test_event_rows_from_message_compacts_raw_payload_and_splits_batched_data():
    message = {
        "channel": "trades",
        "data": [
            {
                "h": 1,
                "s": "BTC",
                "a": "0.1",
                "p": "100",
                "d": "open_long",
                "tc": "normal",
                "t": 1777582374310,
                "li": 9,
            },
            {
                "h": 2,
                "s": "ETH",
                "a": "1",
                "p": "10",
                "d": "close_short",
                "tc": "market_liquidation",
                "t": 1777582375310,
                "li": 10,
            },
        ],
    }

    rows = event_rows_from_message(
        message, recv_ms=1777582376000, raw_text=json.dumps(message)
    )

    assert [row["symbol"] for row in rows] == ["BTC", "ETH"]
    assert rows[0]["channel"] == "trades"
    assert rows[0]["data"]["h"] == 1
    assert rows[0]["raw_message"] == {"channel": "trades", "data": message["data"][0]}
    assert rows[1]["raw_message"] == {"channel": "trades", "data": message["data"][1]}
    assert "raw_text" not in rows[0]
    assert rows[0]["raw_text_sha256"] == rows[1]["raw_text_sha256"]
    assert rows[0]["raw_payload_mode"] == "compact"


def test_event_rows_from_message_can_preserve_full_raw_payload_when_requested():
    message = {
        "channel": "trades",
        "data": [
            {"h": 1, "s": "BTC", "t": 1777582374310},
            {"h": 2, "s": "ETH", "t": 1777582375310},
        ],
    }
    raw_text = json.dumps(message)

    rows = event_rows_from_message(
        message,
        recv_ms=1777582376000,
        raw_text=raw_text,
        raw_payload_mode="full",
    )

    assert rows[0]["raw_message"] == message
    assert rows[0]["raw_text"] == raw_text


def test_ensure_min_free_disk_raises_when_floor_exceeds_available_space(tmp_path):
    with pytest.raises(RuntimeError, match="free disk below safety floor"):
        ensure_min_free_disk(tmp_path, min_free_gb=10**9)


def test_write_jsonl_records_partitions_by_channel_symbol_date_and_keeps_raw_json(
    tmp_path,
):
    rows = event_rows_from_message(
        {
            "channel": "book",
            "data": {
                "s": "BTC",
                "t": 1777582374310,
                "li": 123,
                "l": [
                    [{"a": "1", "n": 2, "p": "100"}],
                    [{"a": "2", "n": 1, "p": "101"}],
                ],
            },
        },
        recv_ms=1777582376000,
        raw_text='{"channel":"book"}',
    )

    written = write_jsonl_records(tmp_path, rows, run_id="testrun")

    assert len(written) == 1
    path = written[0]
    assert path.match("**/channel=book/symbol=BTC/date=2026-04-30/hour=20/*.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        saved = json.loads(fh.readline())
    assert saved["data"]["li"] == 123
    assert saved["data"]["l"][0][0]["n"] == 2


def test_default_intervals_include_all_documented_candle_intervals():
    assert DEFAULT_INTERVALS == (
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "8h",
        "12h",
        "1d",
    )

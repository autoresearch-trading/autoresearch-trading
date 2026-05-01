import gzip
import json
from pathlib import Path

import pandas as pd

from scripts.watch_pacifica_realtime_research import (
    build_realtime_report,
    inventory_raw_archive,
    inventory_silver_archive,
    write_realtime_outputs,
)


def _write_raw(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _record(channel: str, symbol: str, data: dict, recv_ms: int) -> dict:
    return {
        "recv_ms": recv_ms,
        "event_ts_ms": data.get("timestamp") or data.get("t") or recv_ms,
        "channel": channel,
        "symbol": symbol,
        "data": data,
        "raw_message": {"channel": channel, "data": data},
        "raw_text": json.dumps({"channel": channel, "data": data}),
    }


def test_inventory_raw_archive_reports_freshness_by_channel_and_symbol(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_raw(
        raw / "channel=trades" / "symbol=BTC" / "date=2026-05-01" / "run.jsonl.gz",
        [
            _record(
                "trades",
                "BTC",
                {"s": "BTC", "a": "1", "p": "100", "d": "open_long", "t": 1_000},
                recv_ms=2_000,
            ),
            _record(
                "trades",
                "BTC",
                {"s": "BTC", "a": "2", "p": "101", "d": "open_short", "t": 4_000},
                recv_ms=5_000,
            ),
        ],
    )
    _write_raw(
        raw / "channel=bbo" / "symbol=ETH" / "date=2026-05-01" / "run.jsonl.gz",
        [
            _record(
                "bbo",
                "ETH",
                {"s": "ETH", "b": "99", "B": "3", "a": "101", "A": "4", "t": 3_000},
                recv_ms=3_500,
            )
        ],
    )

    inventory = inventory_raw_archive(raw, now_ms=8_000)

    assert inventory.file_count == 2
    assert inventory.symbol_count == 2
    assert inventory.latest_recv_ms == 5_000
    assert inventory.latest_age_s == 3.0
    by_key = {(row.channel, row.symbol): row for row in inventory.rows}
    assert by_key[("trades", "BTC")].row_count == 2
    assert by_key[("trades", "BTC")].latest_event_ts_ms == 4_000
    assert by_key[("bbo", "ETH")].latest_age_s == 4.5


def test_build_realtime_report_computes_latest_1m_market_quality_features(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    minute = 60_000
    _write_raw(
        raw / "channel=trades" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [
            _record(
                "trades",
                "BTC",
                {
                    "s": "BTC",
                    "a": "2",
                    "p": "100",
                    "d": "open_long",
                    "t": minute + 1_000,
                },
                recv_ms=minute + 1_100,
            ),
            _record(
                "trades",
                "BTC",
                {
                    "s": "BTC",
                    "a": "3",
                    "p": "102",
                    "d": "open_short",
                    "t": minute + 20_000,
                },
                recv_ms=minute + 20_100,
            ),
        ],
    )
    _write_raw(
        raw / "channel=bbo" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [
            _record(
                "bbo",
                "BTC",
                {
                    "s": "BTC",
                    "b": "100",
                    "B": "5",
                    "a": "102",
                    "A": "7",
                    "t": minute + 30_000,
                },
                recv_ms=minute + 30_100,
            )
        ],
    )
    _write_raw(
        raw / "channel=prices" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [
            _record(
                "prices",
                "BTC",
                {
                    "symbol": "BTC",
                    "timestamp": minute + 40_000,
                    "mid": "101",
                    "mark": "102",
                    "oracle": "100",
                    "funding": "0.001",
                    "open_interest": "42",
                },
                recv_ms=minute + 40_100,
            )
        ],
    )

    report = build_realtime_report(raw, now_ms=minute + 70_000, stale_after_s=120)

    assert report.inventory.file_count == 3
    assert report.generated_at_ms == minute + 70_000
    assert len(report.features) == 1
    feature = report.features[0]
    assert feature.symbol == "BTC"
    assert feature.window_start_ms == minute
    assert feature.trade_count_1m == 2
    assert feature.trade_volume_1m == 5.0
    assert feature.trade_notional_1m == 506.0
    assert feature.last_price == 102.0
    assert feature.return_bps_1m == 200.0
    assert feature.spread_bps == 198.01980198019803
    assert feature.top_depth_notional == 1214.0
    assert feature.mark_oracle_basis_bps == 200.0
    assert feature.stress_score > 0
    assert report.warnings == []


def _write_silver(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_inventory_silver_archive_reports_freshness_without_reading_raw_gzip(
    tmp_path: Path,
) -> None:
    silver = tmp_path / "silver"
    _write_silver(
        silver
        / "channel=trades"
        / "symbol=BTC"
        / "date=1970-01-01"
        / "part-000000.parquet",
        [
            {
                "event_ts_ms": 1_000,
                "recv_ms": 1_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 100.0,
                "qty": 1.0,
                "signed_qty": 1.0,
                "notional": 100.0,
            },
            {
                "event_ts_ms": 4_000,
                "recv_ms": 4_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 101.0,
                "qty": 2.0,
                "signed_qty": -2.0,
                "notional": 202.0,
            },
        ],
    )
    _write_silver(
        silver
        / "channel=bbo"
        / "symbol=ETH"
        / "date=1970-01-01"
        / "part-000000.parquet",
        [
            {
                "event_ts_ms": 3_000,
                "recv_ms": 3_100,
                "symbol": "ETH",
                "channel": "bbo",
                "spread_bps": 10.0,
            }
        ],
    )

    inventory = inventory_silver_archive(silver, now_ms=8_000)

    assert inventory.file_count == 2
    assert inventory.row_count == 3
    assert inventory.symbol_count == 2
    assert inventory.latest_recv_ms == 4_100
    by_key = {(row.channel, row.symbol): row for row in inventory.rows}
    assert by_key[("trades", "BTC")].row_count == 2
    assert by_key[("trades", "BTC")].latest_event_ts_ms == 4_000
    assert by_key[("bbo", "ETH")].latest_age_s == 4.9


def test_build_realtime_report_can_use_silver_partitions_for_features(
    tmp_path: Path,
) -> None:
    silver = tmp_path / "silver"
    minute = 60_000
    _write_silver(
        silver
        / "channel=trades"
        / "symbol=BTC"
        / "date=1970-01-01"
        / "part-000000.parquet",
        [
            {
                "event_ts_ms": minute + 1_000,
                "recv_ms": minute + 1_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 100.0,
                "qty": 2.0,
                "signed_qty": 2.0,
                "notional": 200.0,
            },
            {
                "event_ts_ms": minute + 20_000,
                "recv_ms": minute + 20_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 102.0,
                "qty": 3.0,
                "signed_qty": -3.0,
                "notional": 306.0,
            },
        ],
    )
    _write_silver(
        silver
        / "channel=bbo"
        / "symbol=BTC"
        / "date=1970-01-01"
        / "part-000000.parquet",
        [
            {
                "event_ts_ms": minute + 30_000,
                "recv_ms": minute + 30_100,
                "symbol": "BTC",
                "channel": "bbo",
                "bid_px": 100.0,
                "ask_px": 102.0,
                "spread_bps": 198.01980198019803,
                "top_bid_notional": 500.0,
                "top_ask_notional": 714.0,
            }
        ],
    )
    _write_silver(
        silver
        / "channel=prices"
        / "symbol=BTC"
        / "date=1970-01-01"
        / "part-000000.parquet",
        [
            {
                "event_ts_ms": minute + 40_000,
                "recv_ms": minute + 40_100,
                "symbol": "BTC",
                "channel": "prices",
                "mid": 101.0,
                "mark": 102.0,
                "oracle": 100.0,
                "funding": 0.001,
                "open_interest": 42.0,
                "mark_oracle_basis_bps": 200.0,
                "mid_oracle_basis_bps": 100.0,
            }
        ],
    )

    report = build_realtime_report(
        silver,
        source="silver",
        now_ms=minute + 70_000,
        stale_after_s=120,
    )

    assert report.raw_dir == str(silver)
    assert report.inventory.file_count == 3
    assert report.warnings == []
    feature = report.features[0]
    assert feature.symbol == "BTC"
    assert feature.trade_count_1m == 2
    assert feature.trade_volume_1m == 5.0
    assert feature.trade_notional_1m == 506.0
    assert feature.return_bps_1m == 200.0
    assert feature.spread_bps == 198.01980198019803
    assert feature.top_depth_notional == 1214.0
    assert feature.mark_oracle_basis_bps == 200.0


def test_write_realtime_outputs_writes_markdown_and_csv_artifacts(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw(
        raw / "channel=trades" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [
            _record(
                "trades",
                "BTC",
                {"s": "BTC", "a": "1", "p": "100", "d": "open_long", "t": 1_000},
                recv_ms=1_100,
            )
        ],
    )

    report = build_realtime_report(raw, now_ms=61_000, stale_after_s=120)
    paths = write_realtime_outputs(report, out)

    assert paths["markdown"].exists()
    assert paths["features_csv"].exists()
    assert paths["inventory_csv"].exists()
    markdown = paths["markdown"].read_text()
    assert "Pacifica realtime research monitor" in markdown
    assert "This is read-only diagnostics" in markdown
    features = pd.read_csv(paths["features_csv"])
    assert list(features["symbol"]) == ["BTC"]
    inventory = pd.read_csv(paths["inventory_csv"])
    assert set(inventory["channel"]) == {"trades"}

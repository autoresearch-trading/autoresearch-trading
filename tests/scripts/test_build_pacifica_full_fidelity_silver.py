import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_pacifica_full_fidelity_silver import (
    DEFAULT_OUT_DIR,
    iter_raw_records,
    normalize_bbo_record,
    normalize_book_record,
    normalize_price_record,
    normalize_trade_record,
    write_incremental_silver_tables,
    write_partitioned_silver_tables,
    write_silver_tables,
)
from scripts.build_pacifica_source_manifest import build_source_manifest


def _record(channel: str, symbol: str, data: dict, ts: int = 1_700_000_000_000) -> dict:
    return {
        "recv_ms": ts + 25,
        "event_ts_ms": data.get("timestamp") or data.get("t") or ts,
        "channel": channel,
        "symbol": symbol,
        "data": data,
        "raw_message": {"channel": channel, "data": data},
        "raw_text": json.dumps({"channel": channel, "data": data}),
    }


def _write_raw(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_sealed_raw(path: Path, rows: list[dict]) -> None:
    _write_raw(path, rows)
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def test_incremental_silver_cli_help_imports_when_run_as_script() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "scripts/build_pacifica_full_fidelity_silver.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--layout" in result.stdout


def test_iter_raw_records_reads_partitioned_jsonl_gz(tmp_path: Path) -> None:
    row = _record(
        "trades",
        "BTC",
        {
            "s": "BTC",
            "a": "2",
            "p": "100",
            "d": "open_long",
            "tc": "normal",
            "t": 1,
            "h": 7,
            "li": 8,
        },
    )
    _write_raw(
        tmp_path / "channel=trades" / "symbol=BTC" / "date=2026-04-30" / "run.jsonl.gz",
        [row],
    )

    records = list(iter_raw_records(tmp_path, channels=["trades"]))

    assert records == [row]


def test_iter_raw_records_reads_hour_partitioned_jsonl_gz(tmp_path: Path) -> None:
    row = _record(
        "bbo",
        "ETH",
        {
            "s": "ETH",
            "b": "99",
            "B": "3",
            "a": "101",
            "A": "4",
            "i": 10,
            "li": 12,
            "t": 2,
        },
    )
    _write_raw(
        tmp_path
        / "channel=bbo"
        / "symbol=ETH"
        / "date=2026-04-30"
        / "hour=22"
        / "run.jsonl.gz",
        [row],
    )

    records = list(iter_raw_records(tmp_path, channels=["bbo"]))

    assert records == [row]


def test_iter_raw_records_skips_incomplete_active_gzip_file(tmp_path: Path) -> None:
    row = _record(
        "trades",
        "BTC",
        {
            "s": "BTC",
            "a": "2",
            "p": "100",
            "d": "open_long",
            "tc": "normal",
            "t": 1,
            "h": 7,
            "li": 8,
        },
    )
    _write_raw(
        tmp_path
        / "channel=trades"
        / "symbol=BTC"
        / "date=2026-04-30"
        / "complete.jsonl.gz",
        [row],
    )
    active_path = (
        tmp_path
        / "channel=trades"
        / "symbol=ETH"
        / "date=2026-04-30"
        / "active.jsonl.gz"
    )
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(b"not a complete gzip stream")

    records = list(iter_raw_records(tmp_path, channels=["trades"]))

    assert records == [row]


def test_normalize_trade_preserves_full_fidelity_ids_and_classifies_side() -> None:
    row = _record(
        "trades",
        "BTC",
        {
            "s": "BTC",
            "a": "2.5",
            "p": "100.5",
            "d": "close_short",
            "tc": "liquidation",
            "t": 123,
            "h": 44,
            "li": 55,
            "it": 0,
        },
    )

    flat = normalize_trade_record(row)

    assert flat["symbol"] == "BTC"
    assert flat["qty"] == 2.5
    assert flat["price"] == 100.5
    assert flat["signed_qty"] == 2.5
    assert flat["notional"] == 251.25
    assert flat["trade_class"] == "liquidation"
    assert flat["history_id"] == 44
    assert flat["nonce"] == 55


def test_normalize_bbo_computes_mid_spread_depth_and_keeps_order_ids() -> None:
    row = _record(
        "bbo",
        "ETH",
        {
            "s": "ETH",
            "b": "99",
            "B": "3",
            "a": "101",
            "A": "4",
            "i": 10,
            "li": 12,
            "t": 123,
        },
    )

    flat = normalize_bbo_record(row)

    assert flat["bid_px"] == 99.0
    assert flat["ask_px"] == 101.0
    assert flat["mid"] == 100.0
    assert flat["spread_bps"] == 200.0
    assert flat["top_bid_notional"] == 297.0
    assert flat["top_ask_notional"] == 404.0
    assert flat["order_id"] == 10
    assert flat["last_order_id"] == 12


def test_normalize_book_computes_l1_l5_depth_order_counts_and_nonce() -> None:
    levels = [
        [{"p": "99", "a": "1", "n": 2}, {"p": "98", "a": "2", "n": 3}],
        [{"p": "101", "a": "4", "n": 5}, {"p": "102", "a": "6", "n": 7}],
    ]
    row = _record("book", "SOL", {"s": "SOL", "l": levels, "li": 99, "t": 123})

    flat = normalize_book_record(row)

    assert flat["bid_px_l1"] == 99.0
    assert flat["ask_px_l1"] == 101.0
    assert flat["mid_l1"] == 100.0
    assert flat["spread_bps_l1"] == 200.0
    assert flat["bid_depth_l5"] == 3.0
    assert flat["ask_depth_l5"] == 10.0
    assert flat["bid_orders_l5"] == 5
    assert flat["ask_orders_l5"] == 12
    assert flat["nonce"] == 99


def test_normalize_price_computes_basis_and_keeps_funding_oi() -> None:
    row = _record(
        "prices",
        "BTC",
        {
            "symbol": "BTC",
            "timestamp": 123,
            "mid": "100",
            "mark": "101",
            "oracle": "99",
            "funding": "0.001",
            "next_funding": "0.002",
            "open_interest": "42",
        },
    )

    flat = normalize_price_record(row)

    assert flat["symbol"] == "BTC"
    assert flat["mid"] == 100.0
    assert flat["mark_oracle_basis_bps"] == pytest.approx(202.02020202020202)
    assert flat["mid_oracle_basis_bps"] == pytest.approx(101.01010101010101)
    assert flat["funding"] == 0.001
    assert flat["next_funding"] == 0.002
    assert flat["open_interest"] == 42.0


def test_write_silver_tables_outputs_channel_parquet_and_quality_csv(
    tmp_path: Path,
) -> None:
    rows = [
        _record(
            "trades",
            "BTC",
            {
                "s": "BTC",
                "a": "1",
                "p": "100",
                "d": "open_long",
                "tc": "normal",
                "t": 1,
                "h": 7,
                "li": 8,
            },
        ),
        _record(
            "bbo",
            "BTC",
            {
                "s": "BTC",
                "b": "99",
                "B": "3",
                "a": "101",
                "A": "4",
                "i": 10,
                "li": 12,
                "t": 2,
            },
        ),
    ]
    raw = tmp_path / "raw"
    _write_raw(
        raw / "channel=trades" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [rows[0]],
    )
    _write_raw(
        raw / "channel=bbo" / "symbol=BTC" / "date=1970-01-01" / "run.jsonl.gz",
        [rows[1]],
    )

    written = write_silver_tables(raw, tmp_path / "silver", channels=["trades", "bbo"])

    assert (tmp_path / "silver" / "trades.parquet").exists()
    assert (tmp_path / "silver" / "bbo.parquet").exists()
    assert (tmp_path / "silver" / "quality_summary.csv").exists()
    assert written["trades"] == 1
    assert written["bbo"] == 1
    trades = pd.read_parquet(tmp_path / "silver" / "trades.parquet")
    assert trades.loc[0, "notional"] == 100.0
    quality = pd.read_csv(tmp_path / "silver" / "quality_summary.csv")
    assert set(quality["channel"]) == {"trades", "bbo"}


def test_write_partitioned_silver_tables_flushes_by_channel_symbol_date_without_flat_table(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    rows = [
        _record(
            "trades",
            "BTC",
            {
                "s": "BTC",
                "a": "1",
                "p": "100",
                "d": "open_long",
                "tc": "normal",
                "t": 1_700_000_000_000,
                "h": 1,
                "li": 10,
            },
        ),
        _record(
            "trades",
            "ETH",
            {
                "s": "ETH",
                "a": "2",
                "p": "50",
                "d": "open_short",
                "tc": "normal",
                "t": 1_700_000_060_000,
                "h": 2,
                "li": 11,
            },
        ),
        _record(
            "bbo",
            "BTC",
            {
                "s": "BTC",
                "b": "99",
                "B": "1",
                "a": "101",
                "A": "1",
                "i": 20,
                "li": 21,
                "t": 1_700_000_000_000,
            },
        ),
    ]
    _write_raw(
        raw / "channel=trades" / "symbol=BTC" / "date=2023-11-14" / "run.jsonl.gz",
        [rows[0]],
    )
    _write_raw(
        raw / "channel=trades" / "symbol=ETH" / "date=2023-11-14" / "run.jsonl.gz",
        [rows[1]],
    )
    _write_raw(
        raw / "channel=bbo" / "symbol=BTC" / "date=2023-11-14" / "run.jsonl.gz",
        [rows[2]],
    )

    written = write_partitioned_silver_tables(
        raw, tmp_path / "silver", channels=["trades", "bbo"], chunk_size=1
    )

    assert written["trades"] == 2
    assert written["bbo"] == 1
    assert not (tmp_path / "silver" / "trades.parquet").exists()
    btc_trade_parts = list(
        (tmp_path / "silver" / "channel=trades" / "symbol=BTC").glob("date=*/*.parquet")
    )
    eth_trade_parts = list(
        (tmp_path / "silver" / "channel=trades" / "symbol=ETH").glob("date=*/*.parquet")
    )
    assert len(btc_trade_parts) == 1
    assert len(eth_trade_parts) == 1
    assert pd.read_parquet(btc_trade_parts[0]).loc[0, "notional"] == 100.0
    quality = pd.read_csv(tmp_path / "silver" / "quality_summary.csv")
    assert quality.set_index("channel").loc["trades", "rows"] == 2


def test_write_incremental_silver_tables_processes_only_new_changed_sealed_chunks(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "silver_candidate"
    first = _record(
        "trades",
        "BTC",
        {"s": "BTC", "a": "1", "p": "100", "d": "open_long", "t": 1_700_000_000_000},
    )
    second = _record(
        "trades",
        "ETH",
        {"s": "ETH", "a": "2", "p": "50", "d": "open_short", "t": 1_700_000_060_000},
    )
    unsealed = _record(
        "bbo",
        "BTC",
        {"s": "BTC", "b": "99", "B": "1", "a": "101", "A": "1", "t": 1_700_000_000_000},
    )
    first_path = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "hour=22"
        / "run-first.jsonl.gz"
    )
    second_path = (
        raw
        / "channel=trades"
        / "symbol=ETH"
        / "date=2023-11-14"
        / "hour=22"
        / "run-second.jsonl.gz"
    )
    unsealed_path = (
        raw
        / "channel=bbo"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "hour=22"
        / "run-active.jsonl.gz"
    )
    _write_sealed_raw(first_path, [first])
    _write_raw(unsealed_path, [unsealed])
    previous = build_source_manifest(
        raw, channels=["trades", "bbo"], verify_sha=True, count_rows=True
    )

    _write_sealed_raw(second_path, [second])
    current = build_source_manifest(
        raw, channels=["trades", "bbo"], verify_sha=True, count_rows=True
    )
    result = write_incremental_silver_tables(
        raw,
        out,
        current_manifest=current,
        previous_manifest=previous,
        channels=["trades", "bbo"],
    )

    assert result["processed_source_objects"] == 1
    assert result["written_rows_by_channel"] == {"trades": 1, "bbo": 0}
    assert (out / "source_manifest.csv").exists()
    assert (out / "incremental_plan.csv").exists()
    assert not list((out / "channel=bbo").glob("**/*.parquet"))
    trade_parts = list((out / "channel=trades" / "symbol=ETH").glob("**/*.parquet"))
    assert len(trade_parts) == 1
    trades = pd.read_parquet(trade_parts[0])
    assert trades.loc[0, "symbol"] == "ETH"
    assert trades.loc[0, "source_key"].endswith("run=run-second")


def test_write_incremental_silver_tables_skips_unchanged_manifest_rows(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "silver_candidate"
    path = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "hour=22"
        / "run-first.jsonl.gz"
    )
    _write_sealed_raw(
        path,
        [
            _record(
                "trades",
                "BTC",
                {
                    "s": "BTC",
                    "a": "1",
                    "p": "100",
                    "d": "open_long",
                    "t": 1_700_000_000_000,
                },
            )
        ],
    )
    manifest = build_source_manifest(
        raw, channels=["trades"], verify_sha=True, count_rows=True
    )

    first = write_incremental_silver_tables(
        raw,
        out,
        current_manifest=manifest,
        previous_manifest=None,
        channels=["trades"],
    )
    second = write_incremental_silver_tables(
        raw,
        out,
        current_manifest=manifest,
        previous_manifest=manifest,
        channels=["trades"],
    )

    assert first["processed_source_objects"] == 1
    assert second["processed_source_objects"] == 0
    assert second["written_rows_by_channel"] == {"trades": 0}


def test_incremental_silver_refuses_canonical_default_out_dir(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    with pytest.raises(ValueError, match="side-by-side candidate"):
        write_incremental_silver_tables(
            raw,
            DEFAULT_OUT_DIR,
            current_manifest=pd.DataFrame(),
            previous_manifest=None,
            channels=["trades"],
        )


def test_incremental_silver_rejects_manifest_source_path_traversal(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    manifest = pd.DataFrame(
        [
            {
                "source_key": "channel=trades/symbol=BTC/date=2023-11-14/hour=22/run=escape",
                "channel": "trades",
                "symbol": "BTC",
                "date": "2023-11-14",
                "hour": "22",
                "run": "escape",
                "source_path": "../escape.jsonl.gz",
                "size_bytes": 1,
                "mtime_ns": 1,
                "sha256": "0" * 64,
                "status": "sealed",
                "sha_verified": True,
                "gzip_readable": True,
                "row_count": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="outside raw_dir"):
        write_incremental_silver_tables(
            raw,
            out,
            current_manifest=manifest,
            previous_manifest=None,
            channels=["trades"],
        )


def test_incremental_silver_requires_verified_manifest_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    path = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "hour=22"
        / "run-unverified.jsonl.gz"
    )
    _write_sealed_raw(
        path,
        [
            _record(
                "trades",
                "BTC",
                {
                    "s": "BTC",
                    "a": "1",
                    "p": "100",
                    "d": "open_long",
                    "t": 1_700_000_000_000,
                },
            )
        ],
    )
    manifest = build_source_manifest(
        raw, channels=["trades"], verify_sha=False, count_rows=False
    )

    with pytest.raises(ValueError, match="verified sealed"):
        write_incremental_silver_tables(
            raw,
            out,
            current_manifest=manifest,
            previous_manifest=None,
            channels=["trades"],
        )


def test_incremental_silver_rejects_flat_base_silver_seed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    canonical = tmp_path / "canonical_silver"
    canonical.mkdir()
    pd.DataFrame([{"event_ts_ms": 1, "symbol": "BTC", "qty": 1.0}]).to_parquet(
        canonical / "trades.parquet", index=False
    )

    with pytest.raises(ValueError, match="flat canonical seed"):
        write_incremental_silver_tables(
            raw,
            tmp_path / "candidate_silver",
            current_manifest=pd.DataFrame(),
            previous_manifest=None,
            channels=["trades"],
            base_silver_dir=canonical,
        )


def test_incremental_silver_rebuilds_affected_seeded_partition_to_avoid_duplicates(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical_silver"
    out = tmp_path / "candidate_silver"
    source = (
        raw
        / "channel=trades"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "hour=22"
        / "run-rewrite.jsonl.gz"
    )
    old_row = _record(
        "trades",
        "BTC",
        {"s": "BTC", "a": "1", "p": "100", "d": "open_long", "t": 1_700_000_000_000},
    )
    new_row = _record(
        "trades",
        "BTC",
        {"s": "BTC", "a": "3", "p": "100", "d": "open_long", "t": 1_700_000_000_000},
    )
    _write_sealed_raw(source, [old_row])
    previous = build_source_manifest(
        raw, channels=["trades"], verify_sha=True, count_rows=True
    )
    old_part = (
        canonical
        / "channel=trades"
        / "symbol=BTC"
        / "date=2023-11-14"
        / "part-000000.parquet"
    )
    old_part.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"event_ts_ms": old_row["event_ts_ms"], "symbol": "BTC", "qty": 1.0}]
    ).to_parquet(old_part, index=False)

    _write_sealed_raw(source, [new_row])
    current = build_source_manifest(
        raw, channels=["trades"], verify_sha=True, count_rows=True
    )
    result = write_incremental_silver_tables(
        raw,
        out,
        current_manifest=current,
        previous_manifest=previous,
        channels=["trades"],
        base_silver_dir=canonical,
    )

    assert result["planned_source_objects"] == 1
    parts = list(
        (out / "channel=trades" / "symbol=BTC" / "date=2023-11-14").glob("**/*.parquet")
    )
    assert len(parts) == 1
    table = pd.read_parquet(parts[0])
    assert len(table) == 1
    assert table.loc[0, "qty"] == 3.0

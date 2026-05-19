import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.verify_pacifica_side_by_side_refresh as verifier
from scripts.verify_pacifica_side_by_side_refresh import compare_side_by_side_refresh


def test_side_by_side_verifier_cli_help_imports_when_run_as_script() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "scripts/verify_pacifica_side_by_side_refresh.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--canonical-silver-dir" in result.stdout


def _write_silver_part(
    root: Path, channel: str, symbol: str, date: str, rows: list[dict]
) -> None:
    path = (
        root
        / f"channel={channel}"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_compare_side_by_side_refresh_reports_counts_coverage_nulls_duplicates_and_diff(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    _write_silver_part(
        canonical_silver,
        "bbo",
        "BTC",
        "2023-11-14",
        [{"event_ts_ms": 1_700_000_000_000, "symbol": "BTC", "channel": "bbo"}],
    )
    _write_silver_part(
        candidate_silver,
        "bbo",
        "BTC",
        "2023-11-14",
        [{"event_ts_ms": 1_700_000_000_000, "symbol": "BTC", "channel": "bbo"}],
    )
    _write_silver_part(
        candidate_silver,
        "bbo",
        "ETH",
        "2023-11-15",
        [{"event_ts_ms": 1_700_086_400_000, "symbol": "ETH", "channel": "bbo"}],
    )
    canonical_regime.mkdir()
    candidate_regime.mkdir()
    pd.DataFrame([{"symbol": "BTC", "bucket_start_ms": 1_700_000_000_000}]).to_parquet(
        canonical_regime / "regime_state.parquet", index=False
    )
    pd.DataFrame(
        [
            {"symbol": "BTC", "bucket_start_ms": 1_700_000_000_000},
            {"symbol": "ETH", "bucket_start_ms": 1_700_086_400_000},
        ]
    ).to_parquet(candidate_regime / "regime_state.parquet", index=False)
    (canonical_regime / "README.md").write_text("old report\n", encoding="utf-8")
    (candidate_regime / "README.md").write_text("new report\n", encoding="utf-8")

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["bbo"],
    )

    assert result["ok"] is True
    assert result["failures"] == []
    assert (out / "silver_row_counts.csv").exists()
    assert (out / "silver_coverage.csv").exists()
    assert (out / "silver_duplicates_nulls.csv").exists()
    assert (out / "regime_row_counts.csv").exists()
    assert (out / "report_diff.patch").exists()
    silver_counts = pd.read_csv(out / "silver_row_counts.csv")
    assert silver_counts.set_index("channel").loc["bbo", "candidate_rows"] == 2
    assert "-old report" in (out / "report_diff.patch").read_text(encoding="utf-8")


def test_compare_side_by_side_refresh_fails_on_missing_candidate_key_columns(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    _write_silver_part(
        candidate_silver,
        "bbo",
        "BTC",
        "2023-11-14",
        [{"symbol": "BTC", "channel": "bbo"}],
    )
    canonical_regime.mkdir()
    candidate_regime.mkdir()
    pd.DataFrame(columns=["symbol", "bucket_start_ms"]).to_parquet(
        canonical_regime / "regime_state.parquet", index=False
    )
    pd.DataFrame([{"symbol": "BTC"}]).to_parquet(
        candidate_regime / "regime_state.parquet", index=False
    )

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["bbo"],
    )

    assert result["ok"] is False
    assert "candidate_silver_missing_key_columns" in result["failures"]
    assert "candidate_regime_missing_key_columns" in result["failures"]
    silver_quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    assert "missing_key_columns_candidate" in silver_quality.columns
    assert (
        silver_quality.set_index("channel").loc["bbo", "missing_key_columns_candidate"]
        == "event_ts_ms"
    )


def _write_matching_regime_pair(canonical_regime: Path, candidate_regime: Path) -> None:
    canonical_regime.mkdir()
    candidate_regime.mkdir()
    row = {"symbol": "BTC", "bucket_start_ms": 1_700_000_000_000}
    pd.DataFrame([row]).to_parquet(
        canonical_regime / "regime_state.parquet", index=False
    )
    pd.DataFrame([row]).to_parquet(
        candidate_regime / "regime_state.parquet", index=False
    )


def test_silver_metrics_do_not_use_full_pandas_read_silver_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    row = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "BTC",
        "channel": "prices",
        "mid": 100.0,
    }
    _write_silver_part(canonical_silver, "prices", "BTC", "2023-11-14", [row])
    _write_silver_part(candidate_silver, "prices", "BTC", "2023-11-14", [row])
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    def fail_if_called(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError(
            "production silver metrics must not materialize channels via pandas"
        )

    monkeypatch.setattr(verifier, "read_silver_table", fail_if_called, raising=False)

    result = verifier.compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["prices"],
    )

    assert result["ok"] is True
    counts = pd.read_csv(out / "silver_row_counts.csv")
    assert counts.set_index("channel").loc["prices", "candidate_rows"] == 1


def test_compare_side_by_side_refresh_allows_semantically_distinct_trade_ids_sharing_base_timestamps(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    _write_silver_part(
        canonical_silver,
        "trades",
        "BTC",
        "2023-11-14",
        [
            {
                "event_ts_ms": 1_700_000_000_000,
                "recv_ms": 1_700_000_000_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 100.0,
                "qty": 1.0,
                "direction": "open_long",
                "history_id": 10,
                "nonce": 1,
            }
        ],
    )
    _write_silver_part(
        candidate_silver,
        "trades",
        "BTC",
        "2023-11-14",
        [
            {
                "event_ts_ms": 1_700_000_000_000,
                "recv_ms": 1_700_000_000_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 100.0,
                "qty": 1.0,
                "direction": "open_long",
                "history_id": 10,
                "nonce": 1,
            },
            {
                "event_ts_ms": 1_700_000_000_000,
                "recv_ms": 1_700_000_000_100,
                "symbol": "BTC",
                "channel": "trades",
                "price": 100.5,
                "qty": 2.0,
                "direction": "open_short",
                "history_id": 11,
                "nonce": 2,
            },
        ],
    )
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["trades"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    row = quality.set_index("channel").loc["trades"]
    assert row["duplicate_keys_candidate"] == 0
    assert row["exact_row_duplicates_candidate"] == 0


def test_compare_side_by_side_refresh_allows_distinct_bbo_order_ids_sharing_base_timestamps(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    base = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "BTC",
        "channel": "bbo",
        "bid_px": 99.0,
        "ask_px": 101.0,
    }
    _write_silver_part(
        canonical_silver,
        "bbo",
        "BTC",
        "2023-11-14",
        [base | {"order_id": 10, "last_order_id": 9, "ask_qty": 1.0}],
    )
    _write_silver_part(
        candidate_silver,
        "bbo",
        "BTC",
        "2023-11-14",
        [
            base | {"order_id": 10, "last_order_id": 9, "ask_qty": 1.0},
            base | {"order_id": 11, "last_order_id": 10, "ask_qty": 2.0},
        ],
    )
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["bbo"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    assert quality.set_index("channel").loc["bbo", "duplicate_keys_candidate"] == 0


def test_compare_side_by_side_refresh_allows_bbo_quote_updates_sharing_order_ids(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    base = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "ETH",
        "channel": "bbo",
        "bid_px": 2122.1,
        "ask_px": 2122.2,
        "ask_qty": 27.5968,
        "mid": 2122.15,
        "spread_bps": 0.4712,
        "order_id": 8689985892,
        "last_order_id": 8689985892,
    }
    _write_silver_part(
        canonical_silver,
        "bbo",
        "ETH",
        "2023-11-14",
        [base | {"bid_qty": 17.4502}],
    )
    _write_silver_part(
        candidate_silver,
        "bbo",
        "ETH",
        "2023-11-14",
        [base | {"bid_qty": 17.4502}, base | {"bid_qty": 8.0253}],
    )
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["bbo"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    assert quality.set_index("channel").loc["bbo", "duplicate_keys_candidate"] == 0


def test_compare_side_by_side_refresh_allows_distinct_candle_intervals_sharing_base_timestamps(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    base = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "BTC",
        "channel": "candle",
        "start_ts_ms": 1_700_000_000_000,
    }
    _write_silver_part(
        canonical_silver,
        "candle",
        "BTC",
        "2023-11-14",
        [base | {"interval": "1m", "end_ts_ms": 1_700_000_060_000, "close": 100.0}],
    )
    _write_silver_part(
        candidate_silver,
        "candle",
        "BTC",
        "2023-11-14",
        [
            base | {"interval": "1m", "end_ts_ms": 1_700_000_060_000, "close": 100.0},
            base | {"interval": "5m", "end_ts_ms": 1_700_000_300_000, "close": 101.0},
        ],
    )
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["candle"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    assert quality.set_index("channel").loc["candle", "duplicate_keys_candidate"] == 0


def test_compare_side_by_side_refresh_allows_candle_revisions_sharing_candle_key(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    base = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "BNB",
        "channel": "candle",
        "interval": "15m",
        "start_ts_ms": 1_700_000_000_000,
        "end_ts_ms": 1_700_000_900_000,
        "open": 640.21,
        "high": 640.21,
        "low": 639.42,
        "close": 639.53,
    }
    _write_silver_part(
        canonical_silver,
        "candle",
        "BNB",
        "2023-11-14",
        [base | {"volume": 0.779, "trade_count": 67}],
    )
    _write_silver_part(
        candidate_silver,
        "candle",
        "BNB",
        "2023-11-14",
        [
            base | {"volume": 0.779, "trade_count": 67},
            base | {"volume": 0.780, "trade_count": 68},
        ],
    )
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["candle"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    assert quality.set_index("channel").loc["candle", "duplicate_keys_candidate"] == 0


def test_compare_side_by_side_refresh_fails_on_increased_exact_row_duplicates(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    row = {
        "event_ts_ms": 1_700_000_000_000,
        "recv_ms": 1_700_000_000_100,
        "symbol": "BTC",
        "channel": "prices",
        "mid": 100.0,
        "mark": 100.0,
        "oracle": 100.0,
    }
    canonical_row = row | {
        "source_key": "source-a",
        "source_path": "channel=prices/symbol=BTC/date=2023-11-14/run-a.jsonl.gz",
        "source_sha256": "a",
    }
    candidate_rows = [
        canonical_row,
        row
        | {
            "source_key": "source-b",
            "source_path": "channel=prices/symbol=BTC/date=2023-11-14/run-b.jsonl.gz",
            "source_sha256": "b",
        },
    ]
    _write_silver_part(canonical_silver, "prices", "BTC", "2023-11-14", [canonical_row])
    _write_silver_part(candidate_silver, "prices", "BTC", "2023-11-14", candidate_rows)
    _write_matching_regime_pair(canonical_regime, candidate_regime)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["prices"],
    )

    assert result["ok"] is False
    assert "candidate_silver_exact_row_duplicates" in result["failures"]
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    row = quality.set_index("channel").loc["prices"]
    assert row["exact_row_duplicates_canonical"] == 0
    assert row["exact_row_duplicates_candidate"] == 1


def test_compare_side_by_side_refresh_allows_preexisting_duplicate_key_baseline_without_regression(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    duplicate_rows = [
        {"event_ts_ms": 1_700_000_000_000, "symbol": "BTC", "channel": "bbo"},
        {"event_ts_ms": 1_700_000_000_000, "symbol": "BTC", "channel": "bbo"},
    ]
    _write_silver_part(canonical_silver, "bbo", "BTC", "2023-11-14", duplicate_rows)
    _write_silver_part(candidate_silver, "bbo", "BTC", "2023-11-14", duplicate_rows)
    canonical_regime.mkdir()
    candidate_regime.mkdir()
    pd.DataFrame([{"symbol": "BTC", "bucket_start_ms": 1_700_000_000_000}]).to_parquet(
        canonical_regime / "regime_state.parquet", index=False
    )
    pd.DataFrame([{"symbol": "BTC", "bucket_start_ms": 1_700_000_000_000}]).to_parquet(
        candidate_regime / "regime_state.parquet", index=False
    )

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["bbo"],
    )

    assert result["ok"] is True
    quality = pd.read_csv(out / "silver_duplicates_nulls.csv")
    row = quality.set_index("channel").loc["bbo"]
    assert row["duplicate_keys_canonical"] == 1
    assert row["duplicate_keys_candidate"] == 1


def test_compare_side_by_side_refresh_fails_on_candidate_row_regression_or_bad_keys(
    tmp_path: Path,
) -> None:
    canonical_silver = tmp_path / "canonical_silver"
    candidate_silver = tmp_path / "candidate_silver"
    canonical_regime = tmp_path / "canonical_regime"
    candidate_regime = tmp_path / "candidate_regime"
    out = tmp_path / "verify"
    _write_silver_part(
        canonical_silver,
        "trades",
        "BTC",
        "2023-11-14",
        [
            {"event_ts_ms": 1, "symbol": "BTC", "channel": "trades"},
            {"event_ts_ms": 2, "symbol": "BTC", "channel": "trades"},
        ],
    )
    _write_silver_part(
        candidate_silver,
        "trades",
        "BTC",
        "2023-11-14",
        [
            {"event_ts_ms": None, "symbol": "BTC", "channel": "trades"},
            {"event_ts_ms": None, "symbol": "BTC", "channel": "trades"},
        ],
    )
    canonical_regime.mkdir()
    candidate_regime.mkdir()
    pd.DataFrame(
        [
            {"symbol": "BTC", "bucket_start_ms": 1},
            {"symbol": "ETH", "bucket_start_ms": 1},
        ]
    ).to_parquet(canonical_regime / "regime_state.parquet", index=False)
    pd.DataFrame(
        [
            {"symbol": "BTC", "bucket_start_ms": 1},
            {"symbol": "BTC", "bucket_start_ms": 1},
        ]
    ).to_parquet(candidate_regime / "regime_state.parquet", index=False)

    result = compare_side_by_side_refresh(
        canonical_silver,
        candidate_silver,
        canonical_regime,
        candidate_regime,
        out,
        channels=["trades"],
    )

    assert result["ok"] is False
    assert "candidate_silver_key_nulls" in result["failures"]
    assert "candidate_regime_duplicate_keys" in result["failures"]
    assert "candidate_regime_symbol_coverage_regression" in result["failures"]

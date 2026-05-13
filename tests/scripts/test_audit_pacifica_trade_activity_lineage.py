import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.audit_pacifica_trade_activity_lineage import (
    LineageAuditConfig,
    audit_trade_activity_lineage,
    write_trade_activity_lineage_report,
)


def _write_raw_trade(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _raw_trade(symbol: str, ts_ms: int, qty: float, price: float) -> dict:
    return {
        "channel": "trades",
        "symbol": symbol,
        "event_ts_ms": ts_ms,
        "recv_ms": ts_ms + 100,
        "data": {
            "symbol": symbol,
            "t": ts_ms,
            "a": str(qty),
            "p": str(price),
            "d": "open_long",
            "tc": "normal",
            "h": ts_ms // 1000,
            "li": ts_ms // 10,
            "it": 0,
        },
    }


def _write_silver_trades(
    silver_dir: Path, symbol: str, date: str, rows: list[dict]
) -> None:
    path = (
        silver_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _silver_trade(symbol: str, ts_ms: int, qty: float, price: float) -> dict:
    return {
        "event_ts_ms": ts_ms,
        "recv_ms": ts_ms + 100,
        "symbol": symbol,
        "channel": "trades",
        "price": price,
        "qty": qty,
        "signed_qty": qty,
        "notional": qty * price,
        "direction": "open_long",
        "trade_class": "normal",
        "history_id": ts_ms // 1000,
        "nonce": ts_ms // 10,
        "is_taker_internal": 0,
    }


def _write_regime(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_eligibility(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_lineage_audit_explains_zero_median_when_trade_minutes_are_sparse(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    symbol = "BTC"
    date = "2026-05-01"
    base_ms = 1_777_593_600_000
    trade_times = [base_ms, base_ms + 60_000]

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [
            _raw_trade(symbol, trade_times[0], 0.5, 100.0),
            _raw_trade(symbol, trade_times[1], 1.5, 101.0),
        ],
    )
    _write_silver_trades(
        silver_dir,
        symbol,
        date,
        [
            _silver_trade(symbol, trade_times[0], 0.5, 100.0),
            _silver_trade(symbol, trade_times[1], 1.5, 101.0),
        ],
    )
    _write_regime(
        regime_path,
        [
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms,
                "trade_count": 1,
                "trade_notional": 50.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 60_000,
                "trade_count": 1,
                "trade_notional": 151.5,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 120_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 180_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 240_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
        ],
    )
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 0.0,
                "activity_gate_pass": False,
                "failure_reasons": "sample;activity",
            }
        ],
    )

    result = audit_trade_activity_lineage(
        LineageAuditConfig(
            raw_dir=raw_dir,
            silver_dir=silver_dir,
            regime_path=regime_path,
            eligibility_path=eligibility_path,
            symbols=(symbol,),
        )
    )

    assert result.verdict == "LINEAGE_AUDIT_PASS_DIAGNOSTIC"
    summary = result.symbol_summary.set_index("symbol").loc[symbol]
    assert summary["raw_trade_rows"] == 2
    assert summary["silver_trade_rows"] == 2
    assert summary["regime_trade_count_sum"] == 2
    assert summary["raw_silver_row_delta"] == 0
    assert summary["silver_regime_trade_count_delta"] == 0
    assert summary["regime_trade_notional_median_all_rows"] == 0.0
    assert (
        summary["activity_median_zero_reason"]
        == "sparse_trade_minutes_not_missing_trade_lineage"
    )
    assert (
        "median_zero_explained_by_sparse_trade_minutes" in summary["diagnostic_notes"]
    )


def test_zero_median_reason_is_not_overclaimed_when_raw_silver_mismatches(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    symbol = "BTC"
    date = "2026-05-01"
    base_ms = 1_777_593_600_000

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [
            _raw_trade(symbol, base_ms, 1.0, 100.0),
            _raw_trade(symbol, base_ms + 60_000, 1.0, 100.0),
        ],
    )
    _write_silver_trades(
        silver_dir, symbol, date, [_silver_trade(symbol, base_ms, 1.0, 100.0)]
    )
    _write_regime(
        regime_path,
        [
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms,
                "trade_count": 1,
                "trade_notional": 100.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 60_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 120_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
        ],
    )
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 0.0,
                "activity_gate_pass": False,
                "failure_reasons": "activity",
            }
        ],
    )

    result = audit_trade_activity_lineage(
        LineageAuditConfig(
            raw_dir=raw_dir,
            silver_dir=silver_dir,
            regime_path=regime_path,
            eligibility_path=eligibility_path,
            symbols=(symbol,),
        )
    )

    summary = result.symbol_summary.set_index("symbol").loc[symbol]
    assert result.verdict == "LINEAGE_AUDIT_FAIL_DIAGNOSTIC"
    assert (
        summary["activity_median_zero_reason"]
        == "sparse_trade_minutes_with_raw_silver_gap"
    )
    assert (
        "median_zero_explained_by_sparse_trade_minutes"
        not in summary["diagnostic_notes"]
    )


def test_empty_target_symbols_returns_no_data_diagnostic(tmp_path: Path) -> None:
    regime_path = tmp_path / "regime_state.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_777_593_600_000,
                "trade_count": 1,
                "trade_notional": 100.0,
            }
        ]
    ).to_parquet(regime_path, index=False)

    result = audit_trade_activity_lineage(
        LineageAuditConfig(
            raw_dir=tmp_path / "raw",
            silver_dir=tmp_path / "silver",
            regime_path=regime_path,
            eligibility_path=tmp_path / "missing_eligibility.csv",
            max_symbols=0,
        )
    )

    assert result.target_symbols == ()
    assert result.verdict == "LINEAGE_AUDIT_NO_DATA_DIAGNOSTIC"
    assert result.symbol_summary.empty


def test_invalid_raw_trade_timestamps_are_counted_and_fail_closed(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    symbol = "BTC"
    date = "2026-05-01"

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [
            {
                "channel": "trades",
                "symbol": symbol,
                "recv_ms": 1_777_593_600_100,
                "data": {"symbol": symbol, "a": "1", "p": "100"},
            }
        ],
    )
    _write_regime(regime_path, [])
    # Preserve required empty-regime schema.
    pd.DataFrame(
        columns=["symbol", "bucket_start_ms", "trade_count", "trade_notional"]
    ).to_parquet(regime_path, index=False)
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 0.0,
                "activity_gate_pass": False,
                "failure_reasons": "activity",
            }
        ],
    )

    result = audit_trade_activity_lineage(
        LineageAuditConfig(
            raw_dir=raw_dir,
            silver_dir=silver_dir,
            regime_path=regime_path,
            eligibility_path=eligibility_path,
            symbols=(symbol,),
        )
    )

    summary = result.symbol_summary.set_index("symbol").loc[symbol]
    assert result.verdict == "LINEAGE_AUDIT_FAIL_DIAGNOSTIC"
    assert summary["raw_invalid_event_ts_rows"] == 1
    assert "raw_invalid_event_ts_rows" in summary["diagnostic_notes"]


def test_lineage_audit_fails_when_silver_and_regime_trade_counts_disagree(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    symbol = "ETH"
    date = "2026-05-01"
    base_ms = 1_777_593_600_000

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [
            _raw_trade(symbol, base_ms, 1.0, 200.0),
            _raw_trade(symbol, base_ms + 60_000, 1.0, 201.0),
        ],
    )
    _write_silver_trades(
        silver_dir,
        symbol,
        date,
        [
            _silver_trade(symbol, base_ms, 1.0, 200.0),
            _silver_trade(symbol, base_ms + 60_000, 1.0, 201.0),
        ],
    )
    _write_regime(
        regime_path,
        [
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms,
                "trade_count": 1,
                "trade_notional": 200.0,
                "bbo_updates": 10,
            },
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms + 60_000,
                "trade_count": 0,
                "trade_notional": 0.0,
                "bbo_updates": 10,
            },
        ],
    )
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 0.0,
                "activity_gate_pass": False,
                "failure_reasons": "activity",
            }
        ],
    )

    result = audit_trade_activity_lineage(
        LineageAuditConfig(
            raw_dir=raw_dir,
            silver_dir=silver_dir,
            regime_path=regime_path,
            eligibility_path=eligibility_path,
            symbols=(symbol,),
        )
    )

    assert result.verdict == "LINEAGE_AUDIT_FAIL_DIAGNOSTIC"
    summary = result.symbol_summary.set_index("symbol").loc[symbol]
    assert summary["silver_regime_trade_count_delta"] == 1
    assert "silver_regime_trade_count_mismatch" in summary["diagnostic_notes"]


def test_trade_activity_lineage_report_writes_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    out_dir = tmp_path / "report"
    symbol = "SOL"
    date = "2026-05-01"
    base_ms = 1_777_593_600_000

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [_raw_trade(symbol, base_ms, 2.0, 50.0)],
    )
    _write_silver_trades(
        silver_dir, symbol, date, [_silver_trade(symbol, base_ms, 2.0, 50.0)]
    )
    _write_regime(
        regime_path,
        [
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms,
                "trade_count": 1,
                "trade_notional": 100.0,
                "bbo_updates": 10,
            }
        ],
    )
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 100.0,
                "activity_gate_pass": True,
                "failure_reasons": "sample",
            }
        ],
    )

    result = write_trade_activity_lineage_report(
        LineageAuditConfig(
            raw_dir=raw_dir,
            silver_dir=silver_dir,
            regime_path=regime_path,
            eligibility_path=eligibility_path,
            out_dir=out_dir,
            symbols=(symbol,),
        )
    )

    assert result["verdict"] == "LINEAGE_AUDIT_PASS_DIAGNOSTIC"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "symbol_summary.csv").exists()
    assert (out_dir / "date_summary.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Trade Activity Lineage Audit" in report
    assert "not a strategy, alpha claim, or paper-trading permission" in report
    assert (
        "raw trades -> silver trades -> regime trade_count/trade_notional -> eligibility activity metrics"
        in report
    )
    assert "Failure counters" in report
    assert "silver/regime trade-count mismatches" in report


def test_trade_activity_lineage_cli_runs_as_direct_script(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver"
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    out_dir = tmp_path / "report"
    symbol = "BTC"
    date = "2026-05-01"
    base_ms = 1_777_593_600_000

    _write_raw_trade(
        raw_dir
        / "channel=trades"
        / f"symbol={symbol}"
        / f"date={date}"
        / "part.jsonl.gz",
        [_raw_trade(symbol, base_ms, 1.0, 100.0)],
    )
    _write_silver_trades(
        silver_dir, symbol, date, [_silver_trade(symbol, base_ms, 1.0, 100.0)]
    )
    _write_regime(
        regime_path,
        [
            {
                "symbol": symbol,
                "bucket_start_ms": base_ms,
                "trade_count": 1,
                "trade_notional": 100.0,
                "bbo_updates": 10,
            }
        ],
    )
    _write_eligibility(
        eligibility_path,
        [
            {
                "symbol": symbol,
                "median_trade_notional_per_min": 100.0,
                "activity_gate_pass": True,
                "failure_reasons": "",
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_pacifica_trade_activity_lineage.py",
            "--raw-dir",
            str(raw_dir),
            "--silver-dir",
            str(silver_dir),
            "--regime-path",
            str(regime_path),
            "--eligibility-path",
            str(eligibility_path),
            "--out-dir",
            str(out_dir),
            "--symbols",
            symbol,
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "verdict: LINEAGE_AUDIT_PASS_DIAGNOSTIC" in completed.stdout
    assert (out_dir / "README.md").exists()


def test_lineage_audit_fails_closed_on_missing_regime_columns(tmp_path: Path) -> None:
    regime_path = tmp_path / "regime_state.parquet"
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    pd.DataFrame([{"symbol": "BTC", "bucket_start_ms": 1_777_593_600_000}]).to_parquet(
        regime_path, index=False
    )
    _write_eligibility(eligibility_path, [{"symbol": "BTC"}])

    try:
        audit_trade_activity_lineage(
            LineageAuditConfig(
                raw_dir=tmp_path / "raw",
                silver_dir=tmp_path / "silver",
                regime_path=regime_path,
                eligibility_path=eligibility_path,
                symbols=("BTC",),
            )
        )
    except ValueError as exc:
        assert "regime_state missing required columns" in str(exc)
    else:
        raise AssertionError("expected missing required regime columns to fail closed")

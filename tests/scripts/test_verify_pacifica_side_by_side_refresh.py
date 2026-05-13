import subprocess
import sys
from pathlib import Path

import pandas as pd

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

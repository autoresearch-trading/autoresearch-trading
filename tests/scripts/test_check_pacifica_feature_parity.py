import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.check_pacifica_feature_parity import (
    REQUIRED_METADATA_COLUMNS,
    compare_feature_frames,
    write_feature_parity_report,
)


def _features(
    *, spread_delta: float = 0.0, feature_version: str = "regime_v1"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_777_507_200_000,
                "available_ts": 1_777_507_260_000,
                "computed_at": 1_777_507_261_000,
                "watermark_ts": 1_777_507_260_000,
                "feature_version": feature_version,
                "provisional_final_flag": "final",
                "avg_spread_bps": 5.0 + spread_delta,
                "top_depth_notional": 25_000.0,
                "toxicity_score": 0.25,
            },
            {
                "symbol": "ETH",
                "bucket_start_ms": 1_777_507_200_000,
                "available_ts": 1_777_507_260_000,
                "computed_at": 1_777_507_262_000,
                "watermark_ts": 1_777_507_260_000,
                "feature_version": feature_version,
                "provisional_final_flag": "final",
                "avg_spread_bps": 8.0,
                "top_depth_notional": 12_000.0,
                "toxicity_score": 0.35,
            },
        ]
    )


def test_compare_feature_frames_passes_equal_features_with_metadata() -> None:
    offline = _features()
    live = _features()

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps", "top_depth_notional", "toxicity_score"],
        tolerance=1e-9,
    )

    assert result.verdict == "PARITY_PASS_DIAGNOSTIC"
    assert result.mismatch_count == 0
    assert result.missing_metadata_columns == []
    assert result.mismatches.empty


def test_compare_feature_frames_fails_changed_feature_with_clear_mismatch_row() -> None:
    offline = _features()
    live = _features(spread_delta=0.25)

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps", "top_depth_notional", "toxicity_score"],
        tolerance=0.01,
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.mismatch_count == 1
    mismatch = result.mismatches.iloc[0]
    assert mismatch["symbol"] == "BTC"
    assert mismatch["feature"] == "avg_spread_bps"
    assert mismatch["offline_value"] == 5.0
    assert mismatch["live_value"] == 5.25
    assert mismatch["abs_diff"] == 0.25


def test_compare_feature_frames_fails_missing_metadata_and_feature_version_mismatch() -> (
    None
):
    offline = _features()
    live = _features(feature_version="regime_v2").drop(columns=["watermark_ts"])

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.missing_metadata_columns == ["live.watermark_ts"]
    assert result.version_mismatch_count == 2
    assert set(REQUIRED_METADATA_COLUMNS).issuperset(
        {
            "available_ts",
            "computed_at",
            "watermark_ts",
            "feature_version",
            "provisional_final_flag",
        }
    )


def test_compare_feature_frames_flags_missing_keys_on_either_side() -> None:
    offline = _features()
    live = _features().query("symbol == 'BTC'")

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.missing_key_count == 1
    assert result.missing_keys.iloc[0]["symbol"] == "ETH"
    assert result.missing_keys.iloc[0]["side"] == "live_missing"


def test_compare_feature_frames_fails_empty_inputs() -> None:
    empty = _features().head(0)

    result = compare_feature_frames(
        empty,
        empty,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert "no_overlapping_rows" in result.failure_reasons


def test_compare_feature_frames_fails_duplicate_keys() -> None:
    offline = pd.concat([_features().head(1), _features().head(1)], ignore_index=True)
    live = _features().head(1)

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.duplicate_key_count == 1
    assert result.duplicate_keys.iloc[0]["side"] == "offline"


def test_compare_feature_frames_fails_metadata_value_mismatch_and_nulls() -> None:
    offline = _features()
    live = _features()
    live.loc[0, "watermark_ts"] = live.loc[0, "watermark_ts"] - 60_000
    live.loc[1, "provisional_final_flag"] = None

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.metadata_mismatch_count == 1
    assert result.invalid_metadata_count == 1
    assert set(result.metadata_mismatches["metadata_column"]) == {"watermark_ts"}
    assert set(result.invalid_metadata["metadata_column"]) == {"provisional_final_flag"}


def test_compare_feature_frames_fails_non_numeric_features_even_when_both_bad() -> None:
    offline = _features()
    live = _features()
    offline["avg_spread_bps"] = offline["avg_spread_bps"].astype(object)
    live["avg_spread_bps"] = live["avg_spread_bps"].astype(object)
    offline.loc[0, "avg_spread_bps"] = "bad"
    live.loc[0, "avg_spread_bps"] = "bad"

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.invalid_feature_count == 1
    assert result.invalid_features.iloc[0]["feature"] == "avg_spread_bps"


def test_cli_exits_nonzero_on_parity_failure_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    offline_path = tmp_path / "offline.parquet"
    live_path = tmp_path / "live.csv"
    _features().to_parquet(offline_path, index=False)
    _features(spread_delta=1.0).to_csv(live_path, index=False)

    command = [
        sys.executable,
        "scripts/check_pacifica_feature_parity.py",
        "--offline",
        str(offline_path),
        "--live",
        str(live_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--feature-columns",
        "avg_spread_bps",
    ]
    failed = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], check=False
    )
    allowed = subprocess.run(
        [*command, "--allow-fail-diagnostic"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert failed.returncode == 1
    assert allowed.returncode == 0


def test_compare_feature_frames_rejects_empty_feature_column_list() -> None:
    result = compare_feature_frames(_features(), _features(), feature_columns=[])

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert "no_feature_columns" in result.failure_reasons


def test_compare_feature_frames_rejects_invalid_tolerance() -> None:
    for bad_tolerance in [-1.0, float("nan"), float("inf")]:
        result = compare_feature_frames(
            _features(),
            _features(spread_delta=999.0),
            feature_columns=["avg_spread_bps"],
            tolerance=bad_tolerance,
        )

        assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
        assert "invalid_tolerance" in result.failure_reasons


def test_compare_feature_frames_fails_non_finite_features_even_when_both_match() -> (
    None
):
    offline = _features()
    live = _features()
    offline.loc[0, "avg_spread_bps"] = float("inf")
    live.loc[0, "avg_spread_bps"] = float("inf")

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.invalid_feature_count == 1
    assert result.invalid_features.iloc[0]["issue"] == "non_numeric_null_or_non_finite"


def test_compare_feature_frames_fails_null_or_blank_keys_even_when_both_match() -> None:
    offline = _features()
    live = _features()
    offline.loc[0, "symbol"] = ""
    live.loc[0, "symbol"] = ""
    offline.loc[1, "bucket_start_ms"] = None
    live.loc[1, "bucket_start_ms"] = None

    result = compare_feature_frames(
        offline,
        live,
        feature_columns=["avg_spread_bps"],
    )

    assert result.verdict == "PARITY_FAIL_DIAGNOSTIC"
    assert result.invalid_key_count == 4
    assert "invalid_keys" in result.failure_reasons


def test_write_feature_parity_report_creates_markdown_and_csvs(tmp_path: Path) -> None:
    offline_path = tmp_path / "offline.parquet"
    live_path = tmp_path / "live.csv"
    out_dir = tmp_path / "feature-parity"
    _features().to_parquet(offline_path, index=False)
    _features(spread_delta=0.25).to_csv(live_path, index=False)

    result = write_feature_parity_report(
        offline_path,
        live_path,
        out_dir,
        feature_columns=["avg_spread_bps", "top_depth_notional", "toxicity_score"],
        tolerance=0.01,
    )

    assert result["verdict"] == "PARITY_FAIL_DIAGNOSTIC"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "mismatches.csv").exists()
    assert (out_dir / "missing_keys.csv").exists()
    assert (out_dir / "summary.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Pacifica Feature Parity" in report
    assert "online/offline" in report
    assert "does not authorize trading" in report

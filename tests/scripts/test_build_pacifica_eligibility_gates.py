from pathlib import Path

import pandas as pd

from scripts.build_pacifica_eligibility_gates import (
    DEFAULT_THRESHOLDS,
    EligibilityThresholds,
    evaluate_symbol_eligibility,
    write_eligibility_report,
)


def _state_rows() -> pd.DataFrame:
    rows = []
    # Liquid enough on two distinct days, but still fails default 30-day sample gate.
    for i in range(6):
        rows.append(
            {
                "symbol": "LIQ",
                "bucket_start_ms": 1_777_507_200_000
                + i * 60_000
                + (86_400_000 if i >= 3 else 0),
                "avg_spread_bps": 4.0,
                "top_depth_notional": 25_000.0,
                "trade_notional": 1_200.0,
                "bbo_updates": 50.0,
                "book_updates": 70.0,
                "toxicity_score": 0.25,
            }
        )
    # Bad spread/depth/activity symbol.
    for i in range(4):
        rows.append(
            {
                "symbol": "ILLQ",
                "bucket_start_ms": 1_777_507_200_000 + i * 60_000,
                "avg_spread_bps": 80.0,
                "top_depth_notional": 100.0,
                "trade_notional": 0.0,
                "bbo_updates": 3.0,
                "book_updates": 2.0,
                "toxicity_score": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_evaluate_symbol_eligibility_reports_gate_columns_and_failure_reasons() -> None:
    thresholds = EligibilityThresholds(
        min_days=2,
        min_observations=5,
        min_median_top_depth_notional=5_000.0,
        max_median_spread_bps=10.0,
        min_median_trade_notional_per_min=100.0,
        min_median_bbo_updates_per_min=10.0,
        max_day_observation_concentration=0.8,
    )

    result = evaluate_symbol_eligibility(_state_rows(), thresholds=thresholds)

    liq = result.set_index("symbol").loc["LIQ"]
    assert bool(liq["eligible"]) is True
    assert bool(liq["sample_gate_pass"]) is True
    assert bool(liq["liquidity_gate_pass"]) is True
    assert bool(liq["spread_cost_gate_pass"]) is True
    assert bool(liq["activity_gate_pass"]) is True
    assert liq["failure_reasons"] == ""

    illq = result.set_index("symbol").loc["ILLQ"]
    assert bool(illq["eligible"]) is False
    assert "sample" in illq["failure_reasons"]
    assert "liquidity" in illq["failure_reasons"]
    assert "spread_cost" in illq["failure_reasons"]
    assert "activity" in illq["failure_reasons"]


def test_default_thresholds_keep_young_diagnostic_archive_ineligible() -> None:
    result = evaluate_symbol_eligibility(_state_rows(), thresholds=DEFAULT_THRESHOLDS)

    assert result["eligible"].sum() == 0
    assert result["sample_gate_pass"].sum() == 0
    assert set(result["verdict"]) == {"INELIGIBLE_DIAGNOSTIC"}


def test_write_eligibility_report_creates_readme_and_csvs(tmp_path: Path) -> None:
    state_path = tmp_path / "regime_state.parquet"
    _state_rows().to_parquet(state_path, index=False)
    out_dir = tmp_path / "eligibility"

    result = write_eligibility_report(
        state_path,
        out_dir,
        thresholds=EligibilityThresholds(
            min_days=2,
            min_observations=5,
            min_median_top_depth_notional=5_000.0,
            max_median_spread_bps=10.0,
            min_median_trade_notional_per_min=100.0,
            min_median_bbo_updates_per_min=10.0,
            max_day_observation_concentration=0.8,
        ),
    )

    assert result["verdict"] == "HAS_ELIGIBLE_SYMBOLS_DIAGNOSTIC"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "symbol_eligibility.csv").exists()
    assert (out_dir / "eligible_symbols.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Pacifica Paper-Trading Eligibility Gates" in report
    assert "non-HFT" in report
    assert "Do not trade every collected symbol" in report

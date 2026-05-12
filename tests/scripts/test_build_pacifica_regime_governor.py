from pathlib import Path

import pandas as pd

from scripts.build_pacifica_regime_governor import (
    GovernorThresholds,
    classify_regime_rows,
    summarize_governor_decisions,
    write_regime_governor_report,
)


def _fixture_state() -> pd.DataFrame:
    base_ts = 1_777_507_200_000
    rows = [
        {
            "symbol": "BTC",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 4.0,
            "top_depth_notional": 50_000.0,
            "toxicity_score": 0.20,
            "mark_oracle_basis_abs_bps": 2.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 50.0,
            "trade_notional": 1_000.0,
        },
        {
            "symbol": "BTC",
            "bucket_start_ms": base_ts + 60_000,
            "avg_spread_bps": 12.0,
            "top_depth_notional": 20_000.0,
            "toxicity_score": 0.62,
            "mark_oracle_basis_abs_bps": 3.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 45.0,
            "trade_notional": 800.0,
        },
        {
            "symbol": "ETH",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 80.0,
            "top_depth_notional": 30_000.0,
            "toxicity_score": 0.25,
            "mark_oracle_basis_abs_bps": 2.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 30.0,
            "trade_notional": 500.0,
        },
        {
            "symbol": "SOL",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 5.0,
            "top_depth_notional": 500.0,
            "toxicity_score": 0.20,
            "mark_oracle_basis_abs_bps": 1.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 25.0,
            "trade_notional": 200.0,
        },
        {
            "symbol": "DOGE",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 4.0,
            "top_depth_notional": 50_000.0,
            "toxicity_score": 0.95,
            "mark_oracle_basis_abs_bps": 1.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 40.0,
            "trade_notional": 900.0,
        },
        {
            "symbol": "XRP",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 4.0,
            "top_depth_notional": 50_000.0,
            "toxicity_score": 0.20,
            "mark_oracle_basis_abs_bps": 120.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 40.0,
            "trade_notional": 900.0,
        },
        {
            "symbol": "AVAX",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 4.0,
            "top_depth_notional": 50_000.0,
            "toxicity_score": 0.20,
            "mark_oracle_basis_abs_bps": 1.0,
            "liquidation_notional": 250_000.0,
            "bbo_updates": 40.0,
            "trade_notional": 900.0,
        },
        {
            "symbol": "STALE",
            "bucket_start_ms": base_ts,
            "avg_spread_bps": 4.0,
            "top_depth_notional": 50_000.0,
            "toxicity_score": 0.20,
            "mark_oracle_basis_abs_bps": 1.0,
            "liquidation_notional": 0.0,
            "bbo_updates": 0.0,
            "trade_notional": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_classify_regime_rows_assigns_fixed_no_trade_decisions() -> None:
    thresholds = GovernorThresholds(
        reduce_toxicity_score=0.60,
        skip_toxicity_score=0.90,
        max_spread_bps=40.0,
        min_top_depth_notional=1_000.0,
        max_mark_oracle_basis_abs_bps=100.0,
        forced_flow_liquidation_notional=100_000.0,
        min_bbo_updates=1.0,
        min_trade_notional=1.0,
    )

    result = classify_regime_rows(_fixture_state(), thresholds=thresholds)
    first_btc = result.loc[
        (result["symbol"] == "BTC") & (result["toxicity_score"] < 0.6),
        "governor_decision",
    ].iloc[0]
    reduced_btc = result.loc[
        (result["symbol"] == "BTC") & (result["toxicity_score"] > 0.6),
        "governor_decision",
    ].iloc[0]
    decisions = (
        result.drop_duplicates("symbol")
        .set_index("symbol")["governor_decision"]
        .to_dict()
    )

    assert first_btc == "TRADABLE_DIAGNOSTIC"
    assert reduced_btc == "REDUCE_SIZE_DIAGNOSTIC"
    assert decisions["ETH"] == "SKIP_WIDE_SPREAD"
    assert decisions["SOL"] == "SKIP_THIN_DEPTH"
    assert decisions["DOGE"] == "SKIP_TOXIC_REGIME"
    assert decisions["XRP"] == "SKIP_MARK_DISLOCATION"
    assert decisions["AVAX"] == "SKIP_FORCED_FLOW_AFTERSHOCK"
    assert decisions["STALE"] == "SKIP_STALE_DATA"


def test_classify_regime_rows_records_reasons_without_tuning_thresholds() -> None:
    result = classify_regime_rows(_fixture_state())

    wide = result.set_index("symbol").loc["ETH"]
    assert wide["governor_action"] == "skip"
    assert "spread" in wide["governor_reasons"]
    assert "threshold_version" in result.columns
    assert set(result["threshold_version"]) == {"pacifica_governor_v1_fixed_diagnostic"}


def test_summarize_governor_decisions_counts_decisions_and_skip_rate() -> None:
    decisions = classify_regime_rows(_fixture_state())

    summary = summarize_governor_decisions(decisions)

    counts = summary.set_index("governor_decision")["rows"].to_dict()
    assert counts["TRADABLE_DIAGNOSTIC"] == 1
    assert counts["SKIP_TOXIC_REGIME"] == 1
    assert counts["SKIP_FORCED_FLOW_AFTERSHOCK"] == 1
    assert set(counts) == {
        "TRADABLE_DIAGNOSTIC",
        "REDUCE_SIZE_DIAGNOSTIC",
        "SKIP_TOXIC_REGIME",
        "SKIP_WIDE_SPREAD",
        "SKIP_THIN_DEPTH",
        "SKIP_STALE_DATA",
        "SKIP_MARK_DISLOCATION",
        "SKIP_FORCED_FLOW_AFTERSHOCK",
    }
    assert summary["skip_rate"].max() <= 1.0


def test_summary_includes_zero_count_fixed_states() -> None:
    tradable_only = classify_regime_rows(_fixture_state().head(1))

    summary = summarize_governor_decisions(tradable_only)

    forced_flow = summary.set_index("governor_decision").loc[
        "SKIP_FORCED_FLOW_AFTERSHOCK"
    ]
    assert forced_flow["rows"] == 0
    assert forced_flow["skip_rate"] == 0.0


def test_missing_safety_columns_raise_instead_of_failing_open() -> None:
    state = _fixture_state().drop(columns=["toxicity_score"])

    try:
        classify_regime_rows(state)
    except ValueError as exc:
        assert "toxicity_score" in str(exc)
    else:  # pragma: no cover - test should fail before this branch
        raise AssertionError("expected ValueError")


def test_nan_safety_metrics_fail_closed_to_skip_state() -> None:
    state = _fixture_state().head(1).copy()
    state.loc[0, "avg_spread_bps"] = None

    result = classify_regime_rows(state)

    assert result.loc[0, "governor_decision"] == "SKIP_WIDE_SPREAD"
    assert "spread" in result.loc[0, "governor_reasons"]


def test_missing_one_market_quality_feed_fails_closed_to_stale() -> None:
    state = _fixture_state().head(1).copy()
    state.loc[0, "bbo_updates"] = 0.0
    state.loc[0, "trade_notional"] = 1_000.0

    result = classify_regime_rows(state)

    assert result.loc[0, "governor_decision"] == "SKIP_STALE_DATA"
    assert "stale_data" in result.loc[0, "governor_reasons"]


def test_write_regime_governor_report_creates_markdown_and_csvs(tmp_path: Path) -> None:
    state_path = tmp_path / "regime_state.parquet"
    _fixture_state().to_parquet(state_path, index=False)
    out_dir = tmp_path / "regime-governor"

    result = write_regime_governor_report(state_path, out_dir)

    assert result["verdict"] == "DIAGNOSTIC_GOVERNOR_RULES_ONLY"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "governor_decisions.csv").exists()
    assert (out_dir / "decision_summary.csv").exists()

    report = (out_dir / "README.md").read_text()
    assert "Pacifica No-Trade Regime Governor" in report
    assert "does not authorize paper trading" in report
    assert "fixed diagnostic rules" in report

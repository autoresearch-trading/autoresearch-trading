from pathlib import Path

import pandas as pd
import pytest

from scripts.build_pacifica_regime_governor import (
    GovernorThresholds,
    classify_regime_rows,
    summarize_governor_decisions,
    write_regime_governor_report,
)

EXPECTED_DIAGNOSTIC_STATES = {
    "SKIP_STALE_DATA",
    "SKIP_WIDE_SPREAD",
    "SKIP_LOW_DEPTH",
    "SKIP_TOXIC_REGIME",
    "SKIP_MARK_ORACLE_DISLOCATION",
    "TRADABLE_DIAGNOSTIC",
}


def _base_row(symbol: str = "BTC") -> dict[str, float | int | str]:
    return {
        "symbol": symbol,
        "bucket_start_ms": 1_777_507_200_000,
        "bbo_updates": 50.0,
        "avg_spread_bps": 4.0,
        "top_depth_notional": 50_000.0,
        "trade_count": 5.0,
        "trade_notional": 1_000.0,
        "price_updates": 3.0,
        "toxicity_score": 0.20,
        "mark_oracle_basis_abs_bps": 2.0,
    }


def _fixture_state() -> pd.DataFrame:
    rows = []

    rows.append(_base_row("BTC"))

    wide = _base_row("ETH")
    wide["avg_spread_bps"] = 80.0
    rows.append(wide)

    shallow = _base_row("SOL")
    shallow["top_depth_notional"] = 500.0
    rows.append(shallow)

    toxic = _base_row("DOGE")
    toxic["toxicity_score"] = 0.95
    rows.append(toxic)

    dislocated = _base_row("XRP")
    dislocated["mark_oracle_basis_abs_bps"] = 120.0
    rows.append(dislocated)

    stale_bbo = _base_row("STALE_BBO")
    stale_bbo["bbo_updates"] = 0.0
    rows.append(stale_bbo)

    stale_trades = _base_row("STALE_TRADES")
    stale_trades["trade_count"] = 0.0
    stale_trades["trade_notional"] = 0.0
    rows.append(stale_trades)

    stale_prices = _base_row("STALE_PRICES")
    stale_prices["price_updates"] = 0.0
    rows.append(stale_prices)

    conflict = _base_row("CONFLICT")
    conflict["bbo_updates"] = 0.0
    conflict["avg_spread_bps"] = 999.0
    conflict["toxicity_score"] = 1.0
    conflict["mark_oracle_basis_abs_bps"] = 500.0
    rows.append(conflict)

    return pd.DataFrame(rows)


def test_classify_regime_rows_assigns_fixed_diagnostic_only_states() -> None:
    thresholds = GovernorThresholds(
        skip_toxicity_score=0.90,
        max_spread_bps=40.0,
        min_top_depth_notional=1_000.0,
        max_mark_oracle_basis_abs_bps=100.0,
        min_bbo_updates=1.0,
        min_trade_count=1.0,
        min_trade_notional=1.0,
        min_price_updates=1.0,
    )

    result = classify_regime_rows(_fixture_state(), thresholds=thresholds)
    decisions = result.set_index("symbol")["governor_decision"].to_dict()

    assert decisions["BTC"] == "TRADABLE_DIAGNOSTIC"
    assert decisions["ETH"] == "SKIP_WIDE_SPREAD"
    assert decisions["SOL"] == "SKIP_LOW_DEPTH"
    assert decisions["DOGE"] == "SKIP_TOXIC_REGIME"
    assert decisions["XRP"] == "SKIP_MARK_ORACLE_DISLOCATION"
    assert decisions["STALE_BBO"] == "SKIP_STALE_DATA"
    assert decisions["STALE_TRADES"] == "SKIP_STALE_DATA"
    assert decisions["STALE_PRICES"] == "SKIP_STALE_DATA"
    assert decisions["CONFLICT"] == "SKIP_STALE_DATA"
    assert set(result["governor_decision"]) == EXPECTED_DIAGNOSTIC_STATES
    assert set(result["threshold_version"]) == {"pacifica_governor_v2_fixed_diagnostic"}
    assert (
        result.loc[result["symbol"] == "BTC", "governor_action"].iloc[0]
        == "diagnostic_only"
    )


def test_summarize_governor_decisions_includes_every_fixed_state_and_no_retired_states() -> (
    None
):
    decisions = classify_regime_rows(_fixture_state())

    summary = summarize_governor_decisions(decisions)
    counts = summary.set_index("governor_decision")["rows"].to_dict()

    assert set(counts) == EXPECTED_DIAGNOSTIC_STATES
    assert counts["TRADABLE_DIAGNOSTIC"] == 1
    assert counts["SKIP_STALE_DATA"] == 4
    assert counts["SKIP_LOW_DEPTH"] == 1
    assert counts["SKIP_MARK_ORACLE_DISLOCATION"] == 1
    assert "REDUCE_SIZE_DIAGNOSTIC" not in counts
    assert "SKIP_FORCED_FLOW_AFTERSHOCK" not in counts
    assert "SKIP_THIN_DEPTH" not in counts
    assert "SKIP_MARK_DISLOCATION" not in counts
    assert summary["skip_rate"].max() <= 1.0


def test_missing_latest_regime_schema_columns_raise_instead_of_failing_open() -> None:
    for column in ["toxicity_score", "price_updates", "trade_count", "avg_spread_bps"]:
        state = _fixture_state().drop(columns=[column])
        with pytest.raises(ValueError, match=column):
            classify_regime_rows(state)


@pytest.mark.parametrize(
    ("column", "expected_decision", "expected_reason"),
    [
        ("bbo_updates", "SKIP_STALE_DATA", "stale_data"),
        ("trade_count", "SKIP_STALE_DATA", "stale_data"),
        ("trade_notional", "SKIP_STALE_DATA", "stale_data"),
        ("price_updates", "SKIP_STALE_DATA", "stale_data"),
        ("avg_spread_bps", "SKIP_WIDE_SPREAD", "spread"),
        ("top_depth_notional", "SKIP_LOW_DEPTH", "depth"),
        ("toxicity_score", "SKIP_TOXIC_REGIME", "toxicity"),
        (
            "mark_oracle_basis_abs_bps",
            "SKIP_MARK_ORACLE_DISLOCATION",
            "mark_oracle_dislocation",
        ),
    ],
)
def test_nan_safety_metrics_fail_closed_to_skip_state(
    column: str, expected_decision: str, expected_reason: str
) -> None:
    state = pd.DataFrame([_base_row()])
    state.loc[0, column] = None

    result = classify_regime_rows(state)

    assert result.loc[0, "governor_decision"] == expected_decision
    assert expected_reason in result.loc[0, "governor_reasons"]


def test_boundary_thresholds_are_deterministic_and_fail_closed() -> None:
    thresholds = GovernorThresholds(
        skip_toxicity_score=0.90,
        max_spread_bps=40.0,
        min_top_depth_notional=1_000.0,
        max_mark_oracle_basis_abs_bps=100.0,
        min_bbo_updates=1.0,
        min_trade_count=1.0,
        min_trade_notional=1.0,
        min_price_updates=1.0,
    )

    spread = pd.DataFrame([{**_base_row(), "avg_spread_bps": 40.0}])
    toxicity = pd.DataFrame([{**_base_row(), "toxicity_score": 0.90}])
    dislocation = pd.DataFrame([{**_base_row(), "mark_oracle_basis_abs_bps": 100.0}])
    minimum_freshness = pd.DataFrame(
        [
            {
                **_base_row(),
                "bbo_updates": 1.0,
                "trade_count": 1.0,
                "trade_notional": 1.0,
                "price_updates": 1.0,
            }
        ]
    )

    assert (
        classify_regime_rows(spread, thresholds=thresholds).loc[0, "governor_decision"]
        == "SKIP_WIDE_SPREAD"
    )
    assert (
        classify_regime_rows(toxicity, thresholds=thresholds).loc[
            0, "governor_decision"
        ]
        == "SKIP_TOXIC_REGIME"
    )
    assert (
        classify_regime_rows(dislocation, thresholds=thresholds).loc[
            0, "governor_decision"
        ]
        == "SKIP_MARK_ORACLE_DISLOCATION"
    )
    assert (
        classify_regime_rows(minimum_freshness, thresholds=thresholds).loc[
            0, "governor_decision"
        ]
        == "TRADABLE_DIAGNOSTIC"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip_toxicity_score": -0.01},
        {"skip_toxicity_score": 1.01},
        {"skip_toxicity_score": float("nan")},
        {"max_spread_bps": 0.0},
        {"min_top_depth_notional": -1.0},
        {"max_mark_oracle_basis_abs_bps": float("inf")},
        {"min_bbo_updates": 0.0},
        {"min_trade_count": 0.0},
        {"min_trade_notional": 0.0},
        {"min_price_updates": 0.0},
    ],
)
def test_invalid_thresholds_raise_before_classification(
    kwargs: dict[str, float],
) -> None:
    thresholds = GovernorThresholds(**kwargs)

    with pytest.raises(ValueError, match="invalid threshold"):
        classify_regime_rows(_fixture_state(), thresholds=thresholds)


def test_write_regime_governor_report_creates_diagnostic_only_markdown_and_csvs(
    tmp_path: Path,
) -> None:
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
    assert "diagnostic-only" in report
    assert "not a trade signal" in report
    assert "TRADABLE_DIAGNOSTIC` means only" in report
    assert "SKIP_LOW_DEPTH" in report
    assert "SKIP_MARK_ORACLE_DISLOCATION" in report
    assert "REDUCE_SIZE_DIAGNOSTIC" not in report
    assert "SKIP_FORCED_FLOW_AFTERSHOCK" not in report

from pathlib import Path

import pandas as pd

from scripts.build_pacifica_symbol_lifecycle import (
    LifecycleConfig,
    assign_symbol_lifecycle,
    write_symbol_lifecycle_report,
)


def _eligibility_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "eligible": True,
                "sample_gate_pass": True,
                "liquidity_gate_pass": True,
                "spread_cost_gate_pass": True,
                "activity_gate_pass": True,
                "stability_gate_pass": True,
                "concentration_gate_pass": True,
                "failure_reasons": "",
                "n_days": 45,
                "n_observations": 20_000,
            },
            {
                "symbol": "ETH",
                "eligible": False,
                "sample_gate_pass": True,
                "liquidity_gate_pass": True,
                "spread_cost_gate_pass": False,
                "activity_gate_pass": True,
                "stability_gate_pass": True,
                "concentration_gate_pass": True,
                "failure_reasons": "spread_cost",
                "n_days": 45,
                "n_observations": 18_000,
            },
            {
                "symbol": "NEW",
                "eligible": False,
                "sample_gate_pass": False,
                "liquidity_gate_pass": True,
                "spread_cost_gate_pass": True,
                "activity_gate_pass": True,
                "stability_gate_pass": True,
                "concentration_gate_pass": True,
                "failure_reasons": "sample",
                "n_days": 5,
                "n_observations": 2_000,
            },
            {
                "symbol": "THIN",
                "eligible": False,
                "sample_gate_pass": True,
                "liquidity_gate_pass": False,
                "spread_cost_gate_pass": True,
                "activity_gate_pass": False,
                "stability_gate_pass": True,
                "concentration_gate_pass": True,
                "failure_reasons": "liquidity;activity",
                "n_days": 45,
                "n_observations": 15_000,
            },
        ]
    )


def test_assign_symbol_lifecycle_maps_eligibility_and_reasons() -> None:
    lifecycle = assign_symbol_lifecycle(_eligibility_rows())
    by_symbol = lifecycle.set_index("symbol")

    assert by_symbol.loc["BTC", "lifecycle_state"] == "ELIGIBLE"
    assert by_symbol.loc["BTC", "reason_codes"] == "eligible"
    assert bool(by_symbol.loc["BTC", "paper_trading_allowed_diagnostic"]) is True

    assert by_symbol.loc["ETH", "lifecycle_state"] == "RESEARCHABLE"
    assert "spread_too_wide" in by_symbol.loc["ETH", "reason_codes"]
    assert bool(by_symbol.loc["ETH", "paper_trading_allowed_diagnostic"]) is False

    assert by_symbol.loc["NEW", "lifecycle_state"] == "COLLECTED"
    assert "insufficient_days" in by_symbol.loc["NEW", "reason_codes"]

    assert by_symbol.loc["THIN", "lifecycle_state"] == "DISABLED"
    assert "depth_too_thin" in by_symbol.loc["THIN", "reason_codes"]
    assert "insufficient_activity" in by_symbol.loc["THIN", "reason_codes"]


def test_previous_lifecycle_turns_demotion_into_probation_not_silent_eligible() -> None:
    previous = pd.DataFrame(
        [
            {"symbol": "ETH", "lifecycle_state": "ELIGIBLE"},
            {"symbol": "OLD", "lifecycle_state": "ELIGIBLE"},
            {"symbol": "DEAD", "lifecycle_state": "RETIRED"},
        ]
    )
    current = pd.concat(
        [
            _eligibility_rows(),
            pd.DataFrame(
                [
                    {
                        "symbol": "DEAD",
                        "eligible": True,
                        "sample_gate_pass": True,
                        "liquidity_gate_pass": True,
                        "spread_cost_gate_pass": True,
                        "activity_gate_pass": True,
                        "stability_gate_pass": True,
                        "concentration_gate_pass": True,
                        "failure_reasons": "",
                        "n_days": 45,
                        "n_observations": 20_000,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    lifecycle = assign_symbol_lifecycle(current, previous_lifecycle=previous)
    by_symbol = lifecycle.set_index("symbol")

    assert by_symbol.loc["ETH", "lifecycle_state"] == "PROBATION"
    assert "demoted_from_eligible" in by_symbol.loc["ETH", "reason_codes"]
    assert by_symbol.loc["OLD", "lifecycle_state"] == "DISABLED"
    assert "missing_current_eligibility" in by_symbol.loc["OLD", "reason_codes"]
    assert by_symbol.loc["DEAD", "lifecycle_state"] == "RETIRED"
    assert "retired_override" in by_symbol.loc["DEAD", "reason_codes"]


def test_bad_post_cost_baseline_disables_otherwise_eligible_symbol() -> None:
    baselines = pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "post_cost_baseline_pass": False,
                "baseline_net_pnl_bps": -12.5,
            },
            {
                "symbol": "ETH",
                "post_cost_baseline_pass": True,
                "baseline_net_pnl_bps": 20.0,
            },
        ]
    )

    lifecycle = assign_symbol_lifecycle(
        _eligibility_rows(), baseline_scorecard=baselines
    )
    btc = lifecycle.set_index("symbol").loc["BTC"]

    assert btc["lifecycle_state"] == "DISABLED"
    assert "bad_post_cost_baseline" in btc["reason_codes"]
    assert bool(btc["paper_trading_allowed_diagnostic"]) is False


def test_inconsistent_eligible_snapshot_is_disabled_not_promoted() -> None:
    dirty = _eligibility_rows().head(1).copy()
    dirty.loc[0, "eligible"] = True
    dirty.loc[0, "activity_gate_pass"] = False

    lifecycle = assign_symbol_lifecycle(dirty)
    btc = lifecycle.set_index("symbol").loc["BTC"]

    assert btc["lifecycle_state"] == "DISABLED"
    assert "inconsistent_eligibility_snapshot" in btc["reason_codes"]
    assert bool(btc["paper_trading_allowed_diagnostic"]) is False


def test_retired_previous_state_is_sticky_and_previous_input_is_strict() -> None:
    previous = pd.DataFrame([{"symbol": "BTC", "lifecycle_state": "retired"}])

    try:
        assign_symbol_lifecycle(
            _eligibility_rows().head(1), previous_lifecycle=previous
        )
    except ValueError as exc:
        assert "unknown lifecycle_state" in str(exc)
    else:
        raise AssertionError(
            "noncanonical previous lifecycle state did not fail closed"
        )


def test_dirty_symbols_and_unknown_boolean_values_fail_closed() -> None:
    dirty_symbol = _eligibility_rows().head(1).copy()
    dirty_symbol.loc[0, "symbol"] = " BTC"
    try:
        assign_symbol_lifecycle(dirty_symbol)
    except ValueError as exc:
        assert "noncanonical symbol" in str(exc)
    else:
        raise AssertionError("dirty symbol did not fail closed")

    dirty_bool = _eligibility_rows().head(1).copy()
    dirty_bool["eligible"] = dirty_bool["eligible"].astype(object)
    dirty_bool.loc[0, "eligible"] = "maybe"
    try:
        assign_symbol_lifecycle(dirty_bool)
    except ValueError as exc:
        assert "invalid boolean" in str(exc)
    else:
        raise AssertionError("unknown boolean did not fail closed")


def test_missing_baseline_for_promotable_symbol_disables_when_scorecard_supplied() -> (
    None
):
    baselines = pd.DataFrame(
        [
            {
                "symbol": "ETH",
                "post_cost_baseline_pass": True,
                "baseline_net_pnl_bps": 20.0,
            }
        ]
    )

    lifecycle = assign_symbol_lifecycle(
        _eligibility_rows().head(1), baseline_scorecard=baselines
    )
    btc = lifecycle.set_index("symbol").loc["BTC"]

    assert btc["lifecycle_state"] == "DISABLED"
    assert "missing_post_cost_baseline" in btc["reason_codes"]
    assert bool(btc["paper_trading_allowed_diagnostic"]) is False


def test_invalid_counts_fail_closed() -> None:
    invalid = _eligibility_rows().head(1).copy()
    invalid.loc[0, "n_observations"] = -1

    try:
        assign_symbol_lifecycle(invalid)
    except ValueError as exc:
        assert "invalid numeric count" in str(exc)
    else:
        raise AssertionError("negative count did not fail closed")


def test_empty_supplied_baseline_scorecard_disables_promotable_symbols() -> None:
    empty_baseline = pd.DataFrame(columns=["symbol", "post_cost_baseline_pass"])

    lifecycle = assign_symbol_lifecycle(
        _eligibility_rows().head(1), baseline_scorecard=empty_baseline
    )
    btc = lifecycle.set_index("symbol").loc["BTC"]

    assert btc["lifecycle_state"] == "DISABLED"
    assert "missing_post_cost_baseline" in btc["reason_codes"]
    assert bool(btc["paper_trading_allowed_diagnostic"]) is False


def test_assign_symbol_lifecycle_fails_closed_on_missing_required_columns() -> None:
    missing = _eligibility_rows().drop(columns=["activity_gate_pass"])

    try:
        assign_symbol_lifecycle(missing)
    except ValueError as exc:
        assert "activity_gate_pass" in str(exc)
    else:
        raise AssertionError("missing required lifecycle input column did not fail")


def test_write_symbol_lifecycle_report_creates_readme_and_csvs(tmp_path: Path) -> None:
    eligibility_path = tmp_path / "symbol_eligibility.csv"
    _eligibility_rows().to_csv(eligibility_path, index=False)
    out_dir = tmp_path / "symbol_lifecycle"

    result = write_symbol_lifecycle_report(
        eligibility_path, out_dir, config=LifecycleConfig()
    )

    assert result["verdict"] == "HAS_ELIGIBLE_SYMBOLS_DIAGNOSTIC"
    assert result["eligible_symbols"] == 1
    assert (out_dir / "README.md").exists()
    assert (out_dir / "symbol_lifecycle.csv").exists()
    assert (out_dir / "state_counts.csv").exists()
    assert (out_dir / "transitions.csv").exists()
    report = (out_dir / "README.md").read_text()
    assert "Pacifica Symbol Lifecycle" in report
    assert "diagnostic lifecycle" in report
    assert "does not authorize live trading" in report

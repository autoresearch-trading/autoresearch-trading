import math
from pathlib import Path

import pandas as pd
import pytest

from scripts.non_hft_toxic_overlay_probe import (
    add_forward_path_metrics,
    assign_toxicity_buckets,
    dataframe_to_markdown_table,
    evaluate_overlay,
    summarize_toxicity_buckets,
    verdict_from_scorecard,
    write_toxic_overlay_report,
)


def _sample_state() -> pd.DataFrame:
    rows = []
    mids = [100.0, 101.0, 99.0, 98.0, 102.0, 103.0]
    toxicity = [0.1, 0.95, 0.2, 0.85, 0.3, 0.4]
    for i, (mid, tox) in enumerate(zip(mids, toxicity, strict=True)):
        rows.append(
            {
                "symbol": "BTC",
                "bucket_start_ms": i * 60_000,
                "last_mid": mid,
                "toxicity_score": tox,
            }
        )
    return pd.DataFrame(rows)


def test_add_forward_path_metrics_computes_returns_and_adverse_excursions() -> None:
    state = _sample_state()

    out = add_forward_path_metrics(state, horizons_minutes=(2,), bucket_minutes=1)

    first = out.iloc[0]
    assert first["forward_return_2m_bps"] == pytest.approx(-100.0)
    assert first["long_adverse_excursion_2m_bps"] == pytest.approx(-100.0)
    assert first["short_adverse_excursion_2m_bps"] == pytest.approx(-100.0)
    assert math.isnan(out.iloc[-1]["forward_return_2m_bps"])


def test_assign_toxicity_buckets_uses_fixed_quantile_bins() -> None:
    state = _sample_state()

    out = assign_toxicity_buckets(state, n_buckets=5)

    assert out["toxicity_decile"].min() == 1
    assert out["toxicity_decile"].max() == 5
    assert out.loc[out["toxicity_score"].idxmax(), "toxicity_decile"] == 5


def test_summarize_toxicity_buckets_reports_removed_high_toxicity_risk() -> None:
    state = assign_toxicity_buckets(
        add_forward_path_metrics(
            _sample_state(), horizons_minutes=(2,), bucket_minutes=1
        ),
        n_buckets=5,
    )

    summary = summarize_toxicity_buckets(state, horizons_minutes=(2,))

    top = summary[summary["toxicity_decile"] == 5].iloc[0]
    assert int(top["n_obs"]) >= 1
    assert "p05_long_adverse_excursion_bps" in summary.columns
    assert "p05_short_adverse_excursion_bps" in summary.columns


def test_evaluate_overlay_compares_baseline_accepted_and_removed() -> None:
    state = assign_toxicity_buckets(
        add_forward_path_metrics(
            _sample_state(), horizons_minutes=(2,), bucket_minutes=1
        ),
        n_buckets=10,
    )

    scorecard = evaluate_overlay(
        state,
        horizons_minutes=(2,),
        toxicity_cutoffs=(0.8,),
        min_days=1,
        min_removed=1,
    )

    row = scorecard.iloc[0]
    assert row["cutoff"] == 0.8
    assert int(row["n_baseline"]) == 4
    assert int(row["n_removed"]) >= 1
    assert 0.0 < row["retention_rate"] < 1.0
    assert "sortino_proxy_baseline" in scorecard.columns
    assert "sortino_proxy_accepted" in scorecard.columns
    assert "delta_downside_deviation" in scorecard.columns


def test_verdict_policy_distinguishes_insufficient_sample_from_fail_or_pass() -> None:
    insufficient = pd.DataFrame(
        [
            {
                "cutoff": 0.9,
                "horizon_minutes": 15,
                "n_days": 2,
                "n_removed": 10,
                "retention_rate": 0.9,
                "delta_p05_long_adverse_excursion_bps": 1.0,
                "delta_downside_deviation": -1.0,
                "delta_sortino_proxy": 0.1,
            }
        ]
    )
    assert verdict_from_scorecard(
        insufficient, min_days=30, min_removed=100
    ).startswith("INSUFFICIENT_SAMPLE")

    passing = pd.DataFrame(
        [
            {
                "cutoff": 0.9,
                "horizon_minutes": 15,
                "n_days": 35,
                "n_removed": 120,
                "retention_rate": 0.9,
                "delta_p05_long_adverse_excursion_bps": 6.0,
                "delta_p05_short_adverse_excursion_bps": 6.0,
                "delta_downside_deviation_pct": -4.0,
                "delta_sortino_proxy": 0.1,
            },
            {
                "cutoff": 0.9,
                "horizon_minutes": 30,
                "n_days": 35,
                "n_removed": 120,
                "retention_rate": 0.9,
                "delta_p05_long_adverse_excursion_bps": 5.5,
                "delta_p05_short_adverse_excursion_bps": 5.5,
                "delta_downside_deviation_pct": -3.5,
                "delta_sortino_proxy": 0.1,
            },
        ]
    )
    assert (
        verdict_from_scorecard(passing, min_days=30, min_removed=100)
        == "PASS_DIAGNOSTIC"
    )


def test_markdown_table_does_not_require_tabulate() -> None:
    df = pd.DataFrame({"a": [1], "b": [2.5]})

    out = dataframe_to_markdown_table(df)

    assert "| a | b |" in out
    assert "| 1 | 2.5000 |" in out


def test_write_toxic_overlay_report_creates_expected_outputs(tmp_path: Path) -> None:
    state_path = tmp_path / "regime_state.parquet"
    _sample_state().to_parquet(state_path, index=False)
    out_dir = tmp_path / "report"

    result = write_toxic_overlay_report(
        state_path,
        out_dir,
        horizons_minutes=(2,),
        bucket_minutes=1,
        min_days=1,
        min_removed=1,
    )

    assert (out_dir / "README.md").exists()
    assert (out_dir / "toxic_bucket_summary.csv").exists()
    assert (out_dir / "overlay_scorecard.csv").exists()
    assert (out_dir / "symbol_summary.csv").exists()
    assert result["state_with_forward_metrics"].endswith(
        "state_with_forward_metrics.parquet"
    )
    assert "Toxic Regime Overlay" in (out_dir / "README.md").read_text()

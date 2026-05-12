# scripts/build_pacifica_regime_governor.py
"""Build fixed diagnostic no-trade/size-reduction decisions from regime state.

This script is a risk/governor layer, not a strategy.  It converts existing
non-HFT regime-state rows into explicit diagnostic decisions so future strategy
adapters can obey fixed no-trade rules before they reach the paper ledger.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulate_pacifica_execution import dataframe_to_markdown_table

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_OUT_DIR = Path("docs/experiments/regime-governor")
THRESHOLD_VERSION = "pacifica_governor_v1_fixed_diagnostic"


@dataclass(frozen=True)
class GovernorThresholds:
    """Fixed diagnostic thresholds for the v1 no-trade regime governor."""

    reduce_toxicity_score: float = 0.60
    skip_toxicity_score: float = 0.90
    max_spread_bps: float = 40.0
    min_top_depth_notional: float = 1_000.0
    max_mark_oracle_basis_abs_bps: float = 100.0
    forced_flow_liquidation_notional: float = 100_000.0
    min_bbo_updates: float = 1.0
    min_trade_notional: float = 1.0


DEFAULT_THRESHOLDS = GovernorThresholds()
FIXED_DECISION_ORDER = [
    "TRADABLE_DIAGNOSTIC",
    "REDUCE_SIZE_DIAGNOSTIC",
    "SKIP_TOXIC_REGIME",
    "SKIP_WIDE_SPREAD",
    "SKIP_THIN_DEPTH",
    "SKIP_STALE_DATA",
    "SKIP_MARK_DISLOCATION",
    "SKIP_FORCED_FLOW_AFTERSHOCK",
]
SAFETY_COLUMNS = {
    "avg_spread_bps",
    "top_depth_notional",
    "toxicity_score",
    "mark_oracle_basis_abs_bps",
    "liquidation_notional",
    "bbo_updates",
    "trade_notional",
}


def _numeric(
    frame: pd.DataFrame, column: str, *, fail_closed_value: float
) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(fail_closed_value)


def _fail_closed_value(column: str, thresholds: GovernorThresholds) -> float:
    if column == "avg_spread_bps":
        return thresholds.max_spread_bps
    if column == "top_depth_notional":
        return 0.0
    if column == "toxicity_score":
        return thresholds.skip_toxicity_score
    if column == "mark_oracle_basis_abs_bps":
        return thresholds.max_mark_oracle_basis_abs_bps
    if column == "liquidation_notional":
        return thresholds.forced_flow_liquidation_notional
    if column in {"bbo_updates", "trade_notional"}:
        return 0.0
    raise ValueError(f"unsupported safety column: {column}")


def _decision_for_row(
    row: pd.Series, thresholds: GovernorThresholds
) -> tuple[str, str, str]:
    reasons: list[str] = []
    # Stale/missing either market-quality feed fails closed. A governor should not
    # allow rows just because one feed is active while the other safety feed is
    # absent or invalid.
    if (
        row["bbo_updates"] < thresholds.min_bbo_updates
        or row["trade_notional"] < thresholds.min_trade_notional
    ):
        reasons.append("stale_data")
        return "SKIP_STALE_DATA", "skip", ";".join(reasons)
    if row["liquidation_notional"] >= thresholds.forced_flow_liquidation_notional:
        reasons.append("forced_flow_aftershock")
        return "SKIP_FORCED_FLOW_AFTERSHOCK", "skip", ";".join(reasons)
    if row["mark_oracle_basis_abs_bps"] >= thresholds.max_mark_oracle_basis_abs_bps:
        reasons.append("mark_oracle_dislocation")
        return "SKIP_MARK_DISLOCATION", "skip", ";".join(reasons)
    if row["toxicity_score"] >= thresholds.skip_toxicity_score:
        reasons.append("toxicity")
        return "SKIP_TOXIC_REGIME", "skip", ";".join(reasons)
    if row["avg_spread_bps"] >= thresholds.max_spread_bps:
        reasons.append("spread")
        return "SKIP_WIDE_SPREAD", "skip", ";".join(reasons)
    if row["top_depth_notional"] < thresholds.min_top_depth_notional:
        reasons.append("depth")
        return "SKIP_THIN_DEPTH", "skip", ";".join(reasons)
    if row["toxicity_score"] >= thresholds.reduce_toxicity_score:
        reasons.append("elevated_toxicity")
        return "REDUCE_SIZE_DIAGNOSTIC", "reduce_size", ";".join(reasons)
    return "TRADABLE_DIAGNOSTIC", "allow_diagnostic", ""


def classify_regime_rows(
    state: pd.DataFrame, *, thresholds: GovernorThresholds = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Classify each regime-state row with fixed diagnostic governor rules."""
    required = {"symbol", "bucket_start_ms", *SAFETY_COLUMNS}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"state missing required columns: {sorted(missing)}")
    if state.empty:
        return pd.DataFrame()

    out = state.copy()
    for col in [
        "avg_spread_bps",
        "top_depth_notional",
        "toxicity_score",
        "mark_oracle_basis_abs_bps",
        "liquidation_notional",
        "bbo_updates",
        "trade_notional",
    ]:
        out[col] = _numeric(
            out, col, fail_closed_value=_fail_closed_value(col, thresholds)
        )

    classified = out.apply(
        _decision_for_row, axis=1, result_type="expand", args=(thresholds,)
    )
    out["governor_decision"] = classified[0]
    out["governor_action"] = classified[1]
    out["governor_reasons"] = classified[2]
    out["threshold_version"] = THRESHOLD_VERSION
    cols_first = [
        "symbol",
        "bucket_start_ms",
        "governor_decision",
        "governor_action",
        "governor_reasons",
        "threshold_version",
    ]
    remaining = [col for col in out.columns if col not in cols_first]
    return (
        out[cols_first + remaining]
        .sort_values(["symbol", "bucket_start_ms"])
        .reset_index(drop=True)
    )


def summarize_governor_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize governor decisions with row counts and skip rates."""
    if decisions.empty:
        return pd.DataFrame(
            columns=["governor_decision", "rows", "skip_rows", "skip_rate"]
        )
    total_rows = len(decisions)
    raw_summary = decisions.groupby("governor_decision", as_index=False).agg(
        rows=("symbol", "size"),
        skip_rows=("governor_action", lambda s: int((s == "skip").sum())),
    )
    fixed = pd.DataFrame({"governor_decision": FIXED_DECISION_ORDER})
    summary = fixed.merge(raw_summary, on="governor_decision", how="left").fillna(
        {"rows": 0, "skip_rows": 0}
    )
    summary["rows"] = summary["rows"].astype(int)
    summary["skip_rows"] = summary["skip_rows"].astype(int)
    summary["row_share"] = summary["rows"] / total_rows
    summary["skip_rate"] = summary["skip_rows"] / summary["rows"].replace(0, pd.NA)
    summary["skip_rate"] = summary["skip_rate"].fillna(0.0)
    return summary


def _thresholds_frame(thresholds: GovernorThresholds) -> pd.DataFrame:
    return pd.DataFrame(
        [{"threshold_version": THRESHOLD_VERSION, **asdict(thresholds)}]
    )


def write_regime_governor_report(
    state_path: Path = DEFAULT_STATE_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    thresholds: GovernorThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Write diagnostic no-trade governor decisions and report artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = pd.read_parquet(state_path)
    decisions = classify_regime_rows(state, thresholds=thresholds)
    summary = summarize_governor_decisions(decisions)
    thresholds_df = _thresholds_frame(thresholds)

    decisions.to_csv(out_dir / "governor_decisions.csv", index=False)
    summary.to_csv(out_dir / "decision_summary.csv", index=False)
    thresholds_df.to_csv(out_dir / "thresholds.csv", index=False)

    report = f"""# Pacifica No-Trade Regime Governor

Verdict: `DIAGNOSTIC_GOVERNOR_RULES_ONLY`

This report converts non-HFT Pacifica regime-state rows into fixed diagnostic
rules for skipping, reducing size, or allowing diagnostic-only trading states.
It does not authorize paper trading, does not create a strategy, and does not
change the current archive maturity verdict. The fixed diagnostic rules below
must remain fixed while enough fresh days accrue for validation.

## Fixed diagnostic rules

Threshold version: `{THRESHOLD_VERSION}`

{dataframe_to_markdown_table(thresholds_df)}

Decision precedence:

1. `SKIP_STALE_DATA`
2. `SKIP_FORCED_FLOW_AFTERSHOCK`
3. `SKIP_MARK_DISLOCATION`
4. `SKIP_TOXIC_REGIME`
5. `SKIP_WIDE_SPREAD`
6. `SKIP_THIN_DEPTH`
7. `REDUCE_SIZE_DIAGNOSTIC`
8. `TRADABLE_DIAGNOSTIC`

## Decision summary

{dataframe_to_markdown_table(summary)}

## Interpretation discipline

- `TRADABLE_DIAGNOSTIC` means the row passed this diagnostic governor only; it is not a trade signal.
- `REDUCE_SIZE_DIAGNOSTIC` is a future sizing constraint, not permission to trade.
- All `SKIP_*` rows should be excluded from future strategy adapters unless a later pre-registered experiment explicitly tests otherwise.
- These thresholds are intentionally fixed and should not be tuned on the current young archive.

## Next system-upgrade phase

Build online/offline feature parity checks before using live microbatch features
for decisions.

## Large local artifact

- `governor_decisions.csv` is generated by this script for local inspection, but full-run row dumps are intentionally ignored in git because they can exceed GitHub blob limits. Commit summary tables and README diagnostics instead.
"""
    (out_dir / "README.md").write_text(report)
    return {
        "verdict": "DIAGNOSTIC_GOVERNOR_RULES_ONLY",
        "out_dir": str(out_dir),
        "rows": int(len(decisions)),
        "decisions": (
            int(decisions["governor_decision"].nunique()) if not decisions.empty else 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_regime_governor_report(args.state, args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"wrote report: {Path(result['out_dir']) / 'README.md'}")


if __name__ == "__main__":
    main()

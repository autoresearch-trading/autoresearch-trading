# scripts/build_pacifica_regime_governor.py
"""Build fixed diagnostic no-trade decisions from regime state.

This script is a risk/governor layer, not a strategy. It converts existing
non-HFT regime-state rows into explicit diagnostic-only decisions so future
strategy adapters can obey fixed fail-closed no-trade rules before they reach a
paper ledger.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulate_pacifica_execution import dataframe_to_markdown_table

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_OUT_DIR = Path("docs/experiments/regime-governor")
THRESHOLD_VERSION = "pacifica_governor_v2_fixed_diagnostic"


@dataclass(frozen=True)
class GovernorThresholds:
    """Fixed diagnostic thresholds for the v2 no-trade regime governor.

    These are deliberately not optimized alpha parameters. Invalid/non-finite
    values raise before any row can be marked ``TRADABLE_DIAGNOSTIC``.
    """

    skip_toxicity_score: float = 0.90
    max_spread_bps: float = 40.0
    min_top_depth_notional: float = 1_000.0
    max_mark_oracle_basis_abs_bps: float = 100.0
    min_bbo_updates: float = 1.0
    min_trade_count: float = 1.0
    min_trade_notional: float = 1.0
    min_price_updates: float = 1.0


DEFAULT_THRESHOLDS = GovernorThresholds()
FIXED_DECISION_ORDER = [
    "SKIP_STALE_DATA",
    "SKIP_WIDE_SPREAD",
    "SKIP_LOW_DEPTH",
    "SKIP_TOXIC_REGIME",
    "SKIP_MARK_ORACLE_DISLOCATION",
    "TRADABLE_DIAGNOSTIC",
]
SAFETY_COLUMNS = {
    "avg_spread_bps",
    "top_depth_notional",
    "toxicity_score",
    "mark_oracle_basis_abs_bps",
    "bbo_updates",
    "trade_count",
    "trade_notional",
    "price_updates",
}
KEY_COLUMNS = {"symbol", "bucket_start_ms"}
REQUIRED_COLUMNS = KEY_COLUMNS | SAFETY_COLUMNS
STALE_FEED_COLUMNS = {
    "bbo_updates",
    "trade_count",
    "trade_notional",
    "price_updates",
}


def _finite_float(value: float) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def validate_thresholds(thresholds: GovernorThresholds) -> None:
    """Fail closed on invalid governor thresholds before classification."""
    values = {
        field.name: _finite_float(getattr(thresholds, field.name))
        for field in fields(thresholds)
    }
    invalid = [name for name, value in values.items() if value is None]
    if values["skip_toxicity_score"] is not None and not (
        0.0 < values["skip_toxicity_score"] <= 1.0
    ):
        invalid.append("skip_toxicity_score")
    for positive_name in [
        "max_spread_bps",
        "min_top_depth_notional",
        "max_mark_oracle_basis_abs_bps",
        "min_bbo_updates",
        "min_trade_count",
        "min_trade_notional",
        "min_price_updates",
    ]:
        value = values[positive_name]
        if value is not None and value <= 0.0:
            invalid.append(positive_name)
    if invalid:
        raise ValueError(f"invalid threshold values: {sorted(set(invalid))}")


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
    if column in STALE_FEED_COLUMNS:
        return 0.0
    raise ValueError(f"unsupported safety column: {column}")


def _validate_state_schema(state: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(state.columns)
    if missing:
        raise ValueError(f"state missing required columns: {sorted(missing)}")
    if state.empty:
        return
    null_key_columns = [column for column in KEY_COLUMNS if state[column].isna().any()]
    if null_key_columns:
        raise ValueError(
            f"state has null required key columns: {sorted(null_key_columns)}"
        )


def _decision_for_row(
    row: pd.Series, thresholds: GovernorThresholds
) -> tuple[str, str, str]:
    reasons: list[str] = []
    if (
        row["bbo_updates"] < thresholds.min_bbo_updates
        or row["trade_count"] < thresholds.min_trade_count
        or row["trade_notional"] < thresholds.min_trade_notional
        or row["price_updates"] < thresholds.min_price_updates
    ):
        reasons.append("stale_data")
        return "SKIP_STALE_DATA", "skip", ";".join(reasons)
    if row["avg_spread_bps"] >= thresholds.max_spread_bps:
        reasons.append("spread")
        return "SKIP_WIDE_SPREAD", "skip", ";".join(reasons)
    if row["top_depth_notional"] < thresholds.min_top_depth_notional:
        reasons.append("depth")
        return "SKIP_LOW_DEPTH", "skip", ";".join(reasons)
    if row["toxicity_score"] >= thresholds.skip_toxicity_score:
        reasons.append("toxicity")
        return "SKIP_TOXIC_REGIME", "skip", ";".join(reasons)
    if row["mark_oracle_basis_abs_bps"] >= thresholds.max_mark_oracle_basis_abs_bps:
        reasons.append("mark_oracle_dislocation")
        return "SKIP_MARK_ORACLE_DISLOCATION", "skip", ";".join(reasons)
    return "TRADABLE_DIAGNOSTIC", "diagnostic_only", ""


def classify_regime_rows(
    state: pd.DataFrame, *, thresholds: GovernorThresholds = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Classify each regime-state row with fixed diagnostic governor rules."""
    validate_thresholds(thresholds)
    _validate_state_schema(state)
    if state.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "bucket_start_ms",
                "governor_decision",
                "governor_action",
                "governor_reasons",
                "threshold_version",
                *[column for column in state.columns if column not in KEY_COLUMNS],
            ]
        )

    out = state.copy()
    for col in sorted(SAFETY_COLUMNS):
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
    fixed = pd.DataFrame({"governor_decision": FIXED_DECISION_ORDER})
    if decisions.empty:
        summary = fixed.copy()
        summary["rows"] = 0
        summary["skip_rows"] = 0
        summary["row_share"] = 0.0
        summary["skip_rate"] = 0.0
        return summary

    total_rows = len(decisions)
    raw_summary = decisions.groupby("governor_decision", as_index=False).agg(
        rows=("symbol", "size"),
        skip_rows=("governor_action", lambda s: int((s == "skip").sum())),
    )
    unexpected = set(raw_summary["governor_decision"]) - set(FIXED_DECISION_ORDER)
    if unexpected:
        raise ValueError(f"unexpected governor decisions: {sorted(unexpected)}")
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
    validate_thresholds(thresholds)
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

This report converts non-HFT Pacifica regime-state rows into fixed diagnostic-only
skip states plus `TRADABLE_DIAGNOSTIC`. It does not authorize paper trading, does
not create a strategy, and does not change the current archive maturity verdict.
The fixed diagnostic rules below must remain fixed while enough fresh days accrue
for validation.

`TRADABLE_DIAGNOSTIC` means only that this diagnostic governor did not block the
row. It is not a trade signal, not alpha evidence, and not permission to paper or
live trade.

## Latest regime-state schema requirements

The governor fails closed unless every input row has these required columns:

{dataframe_to_markdown_table(pd.DataFrame({"required_column": sorted(REQUIRED_COLUMNS)}))}

Missing columns raise. Null key columns raise. NaN/non-numeric safety metrics are
coerced to conservative values that trigger `SKIP_*` states rather than
`TRADABLE_DIAGNOSTIC`. Invalid/non-finite thresholds raise before classification.

## Fixed diagnostic rules

Threshold version: `{THRESHOLD_VERSION}`

{dataframe_to_markdown_table(thresholds_df)}

Decision precedence:

1. `SKIP_STALE_DATA` — stale/missing BBO, trade, or price feed activity.
2. `SKIP_WIDE_SPREAD` — average spread is at or above the fixed maximum.
3. `SKIP_LOW_DEPTH` — top-of-book depth is below the fixed minimum.
4. `SKIP_TOXIC_REGIME` — toxicity score is at or above the fixed skip threshold.
5. `SKIP_MARK_ORACLE_DISLOCATION` — mark/oracle basis is at or above the fixed maximum.
6. `TRADABLE_DIAGNOSTIC` — diagnostic-only state; not a trade signal.

## Decision summary

{dataframe_to_markdown_table(summary)}

## Interpretation discipline

- `TRADABLE_DIAGNOSTIC` means only "not blocked by this diagnostic governor"; it is not a trade signal.
- All `SKIP_*` rows should be excluded from future strategy adapters unless a later pre-registered experiment explicitly tests otherwise.
- These thresholds are intentionally fixed and should not be tuned on the current young archive.
- A future paper-trading adapter still needs explicit symbol eligibility, cost/slippage, sample-size, stability, concentration, and post-cost validation gates.

## Large local artifact

- `governor_decisions.csv` is generated by this script for local inspection, but full-run row dumps are intentionally ignored in git because they can exceed GitHub blob limits. Commit summary tables and README diagnostics instead.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
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

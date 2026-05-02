# scripts/build_pacifica_eligibility_gates.py
"""Build pre-registered Pacifica paper-trading eligibility gates.

This script separates the broad raw/research universe from the subset that is
allowed to enter any future non-HFT paper-trading strategy.  It is intentionally
not a strategy or backtest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_OUT_DIR = Path("docs/experiments/paper-trading-eligibility")


@dataclass(frozen=True)
class EligibilityThresholds:
    """Pre-registered minimum gates for future paper-trading eligibility."""

    min_days: int = 30
    min_observations: int = 10_000
    min_median_top_depth_notional: float = 5_000.0
    max_median_spread_bps: float = 25.0
    min_median_trade_notional_per_min: float = 25.0
    min_median_bbo_updates_per_min: float = 10.0
    max_day_observation_concentration: float = 0.25
    max_p95_spread_bps: float = 75.0
    max_mean_toxicity_score: float = 0.75


DEFAULT_THRESHOLDS = EligibilityThresholds()


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown_table(
    df: pd.DataFrame, *, max_rows: int | None = None
) -> str:
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows) if max_rows is not None else df
    headers = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def _date_from_bucket_ms(bucket_start_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(bucket_start_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")


def _required_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _concentration_by_date(group: pd.DataFrame) -> float:
    if group.empty:
        return 1.0
    counts = group.groupby("date").size()
    if counts.sum() == 0:
        return 1.0
    return float(counts.max() / counts.sum())


def _failure_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    for name, col in [
        ("sample", "sample_gate_pass"),
        ("liquidity", "liquidity_gate_pass"),
        ("spread_cost", "spread_cost_gate_pass"),
        ("activity", "activity_gate_pass"),
        ("stability", "stability_gate_pass"),
        ("concentration", "concentration_gate_pass"),
    ]:
        if not bool(row[col]):
            reasons.append(name)
    return ";".join(reasons)


def evaluate_symbol_eligibility(
    state: pd.DataFrame, *, thresholds: EligibilityThresholds = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Return one row per symbol with explicit non-HFT eligibility gates."""
    required = {"symbol", "bucket_start_ms"}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"state missing required columns: {sorted(missing)}")
    if state.empty:
        return pd.DataFrame()

    work = state.copy()
    work["date"] = _date_from_bucket_ms(work["bucket_start_ms"])
    work["avg_spread_bps"] = _required_numeric(work, "avg_spread_bps")
    work["top_depth_notional"] = _required_numeric(work, "top_depth_notional")
    work["trade_notional"] = _required_numeric(work, "trade_notional")
    work["bbo_updates"] = _required_numeric(work, "bbo_updates")
    work["toxicity_score"] = _required_numeric(work, "toxicity_score")

    rows: list[dict[str, Any]] = []
    for symbol, group in work.groupby("symbol", sort=True):
        n_obs = int(len(group))
        n_days = int(group["date"].nunique())
        median_depth = float(group["top_depth_notional"].median())
        median_spread = float(group["avg_spread_bps"].median())
        p95_spread = float(group["avg_spread_bps"].quantile(0.95))
        median_trade_notional = float(group["trade_notional"].median())
        median_bbo_updates = float(group["bbo_updates"].median())
        mean_toxicity = float(group["toxicity_score"].mean())
        max_day_concentration = _concentration_by_date(group)

        sample_gate = (
            n_days >= thresholds.min_days and n_obs >= thresholds.min_observations
        )
        liquidity_gate = median_depth >= thresholds.min_median_top_depth_notional
        spread_gate = median_spread <= thresholds.max_median_spread_bps
        activity_gate = (
            median_trade_notional >= thresholds.min_median_trade_notional_per_min
            and median_bbo_updates >= thresholds.min_median_bbo_updates_per_min
        )
        stability_gate = (
            p95_spread <= thresholds.max_p95_spread_bps
            and mean_toxicity <= thresholds.max_mean_toxicity_score
        )
        concentration_gate = (
            max_day_concentration <= thresholds.max_day_observation_concentration
        )
        eligible = all(
            [
                sample_gate,
                liquidity_gate,
                spread_gate,
                activity_gate,
                stability_gate,
                concentration_gate,
            ]
        )
        rows.append(
            {
                "symbol": symbol,
                "eligible": bool(eligible),
                "verdict": (
                    "ELIGIBLE_DIAGNOSTIC" if eligible else "INELIGIBLE_DIAGNOSTIC"
                ),
                "n_observations": n_obs,
                "n_days": n_days,
                "median_top_depth_notional": median_depth,
                "median_spread_bps": median_spread,
                "p95_spread_bps": p95_spread,
                "median_trade_notional_per_min": median_trade_notional,
                "median_bbo_updates_per_min": median_bbo_updates,
                "mean_toxicity_score": mean_toxicity,
                "max_day_observation_concentration": max_day_concentration,
                "sample_gate_pass": bool(sample_gate),
                "liquidity_gate_pass": bool(liquidity_gate),
                "spread_cost_gate_pass": bool(spread_gate),
                "activity_gate_pass": bool(activity_gate),
                "stability_gate_pass": bool(stability_gate),
                "concentration_gate_pass": bool(concentration_gate),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["failure_reasons"] = out.apply(_failure_reasons, axis=1)
        out = out.sort_values(
            [
                "eligible",
                "sample_gate_pass",
                "liquidity_gate_pass",
                "spread_cost_gate_pass",
                "median_top_depth_notional",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
    return out


def overall_verdict(symbol_eligibility: pd.DataFrame) -> str:
    if symbol_eligibility.empty:
        return "NO_DATA_DIAGNOSTIC"
    if bool(symbol_eligibility["eligible"].any()):
        return "HAS_ELIGIBLE_SYMBOLS_DIAGNOSTIC"
    if not bool(symbol_eligibility["sample_gate_pass"].any()):
        return "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    return "NO_ELIGIBLE_SYMBOLS_DIAGNOSTIC"


def _gate_counts(symbol_eligibility: pd.DataFrame) -> pd.DataFrame:
    gate_cols = [
        "sample_gate_pass",
        "liquidity_gate_pass",
        "spread_cost_gate_pass",
        "activity_gate_pass",
        "stability_gate_pass",
        "concentration_gate_pass",
        "eligible",
    ]
    rows = []
    total = len(symbol_eligibility)
    for col in gate_cols:
        count = int(symbol_eligibility[col].sum()) if col in symbol_eligibility else 0
        rows.append({"gate": col, "passing_symbols": count, "total_symbols": total})
    return pd.DataFrame(rows)


def write_eligibility_report(
    state_path: Path,
    out_dir: Path,
    *,
    thresholds: EligibilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    state = pd.read_parquet(state_path)
    symbol_eligibility = evaluate_symbol_eligibility(state, thresholds=thresholds)
    verdict = overall_verdict(symbol_eligibility)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_counts = _gate_counts(symbol_eligibility)
    eligible_symbols = symbol_eligibility[symbol_eligibility["eligible"]].copy()
    symbol_eligibility.to_csv(out_dir / "symbol_eligibility.csv", index=False)
    gate_counts.to_csv(out_dir / "gate_counts.csv", index=False)
    eligible_symbols.to_csv(out_dir / "eligible_symbols.csv", index=False)

    thresholds_df = pd.DataFrame(
        [
            {"threshold": key, "value": value}
            for key, value in asdict(thresholds).items()
        ]
    )
    thresholds_df.to_csv(out_dir / "thresholds.csv", index=False)

    preview_cols = [
        "symbol",
        "eligible",
        "n_days",
        "n_observations",
        "median_top_depth_notional",
        "median_spread_bps",
        "median_trade_notional_per_min",
        "max_day_observation_concentration",
        "failure_reasons",
    ]
    preview = (
        symbol_eligibility[preview_cols]
        if not symbol_eligibility.empty
        else symbol_eligibility
    )

    lines = [
        "# Pacifica Paper-Trading Eligibility Gates",
        "",
        "This report defines the pre-trade eligible universe for the non-HFT Pacifica paper-trading program.",
        "It is not a strategy, alpha claim, or backtest.",
        "",
        f"Verdict: `{verdict}`",
        f"Symbols evaluated: {len(symbol_eligibility)}",
        f"Eligible symbols: {len(eligible_symbols)}",
        "",
        "## Interpretation discipline",
        "",
        "Do not trade every collected symbol. The raw collector intentionally captures the broad live public universe; paper trading may only use symbols that pass explicit sample, liquidity, spread/cost, activity, stability, and concentration gates.",
        "The current archive is still young, so this report is diagnostic until enough full distinct days accrue.",
        "",
        "## Thresholds",
        "",
        dataframe_to_markdown_table(thresholds_df),
        "",
        "## Gate counts",
        "",
        dataframe_to_markdown_table(gate_counts),
        "",
        "## Symbol eligibility preview",
        "",
        dataframe_to_markdown_table(preview, max_rows=30),
        "",
        "## Output files",
        "",
        "- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.",
        "- `eligible_symbols.csv` — subset that passed all gates.",
        "- `gate_counts.csv` — count of symbols passing each gate.",
        "- `thresholds.csv` — fixed thresholds used for this run.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    return {
        "verdict": verdict,
        "symbols_evaluated": len(symbol_eligibility),
        "eligible_symbols": len(eligible_symbols),
        "symbol_eligibility": str(out_dir / "symbol_eligibility.csv"),
        "eligible_symbols_path": str(out_dir / "eligible_symbols.csv"),
        "readme": str(out_dir / "README.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_eligibility_report(args.state_path, args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"symbols_evaluated: {result['symbols_evaluated']}")
    print(f"eligible_symbols: {result['eligible_symbols']}")
    print(f"wrote report: {result['readme']}")


if __name__ == "__main__":
    main()

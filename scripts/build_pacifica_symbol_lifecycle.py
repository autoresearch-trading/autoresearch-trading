# scripts/build_pacifica_symbol_lifecycle.py
"""Build a diagnostic symbol lifecycle from Pacifica eligibility gates.

This layer turns a static eligibility snapshot into explicit promotion/demotion
states. It is still diagnostic infrastructure, not permission to live trade.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ELIGIBILITY_PATH = Path(
    "docs/experiments/paper-trading-eligibility/symbol_eligibility.csv"
)
DEFAULT_OUT_DIR = Path("docs/experiments/symbol-lifecycle")
LIFECYCLE_STATES = (
    "COLLECTED",
    "RESEARCHABLE",
    "ELIGIBLE",
    "PROBATION",
    "DISABLED",
    "RETIRED",
)
REQUIRED_ELIGIBILITY_COLUMNS = (
    "symbol",
    "eligible",
    "sample_gate_pass",
    "liquidity_gate_pass",
    "spread_cost_gate_pass",
    "activity_gate_pass",
    "stability_gate_pass",
    "concentration_gate_pass",
    "n_days",
    "n_observations",
)


SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
BOOL_COLUMNS = (
    "eligible",
    "sample_gate_pass",
    "liquidity_gate_pass",
    "spread_cost_gate_pass",
    "activity_gate_pass",
    "stability_gate_pass",
    "concentration_gate_pass",
)


@dataclass(frozen=True)
class LifecycleConfig:
    min_researchable_days: int = 10
    min_researchable_observations: int = 1_000
    probation_previous_states: tuple[str, ...] = ("ELIGIBLE", "PROBATION")


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("invalid boolean value: null")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _validate_symbols(frame: pd.DataFrame, *, source: str) -> pd.Series:
    if "symbol" not in frame.columns:
        raise ValueError(f"{source} missing symbol column")
    raw = frame["symbol"]
    if raw.isna().any():
        raise ValueError(f"{source} contains null or blank symbols")
    as_text = raw.astype(str)
    if (as_text != as_text.str.strip()).any():
        raise ValueError(f"{source} contains noncanonical symbol whitespace")
    if (as_text.str.strip() == "").any():
        raise ValueError(f"{source} contains null or blank symbols")
    reserved = as_text.str.lower().isin({"nan", "none", "null"})
    invalid = ~as_text.map(lambda value: bool(SYMBOL_RE.fullmatch(value))) | reserved
    if invalid.any():
        bad = sorted(as_text[invalid].unique())
        raise ValueError(f"{source} contains noncanonical symbol values: {bad}")
    duplicates = as_text.duplicated()
    if duplicates.any():
        duplicate_values = sorted(as_text[duplicates].unique())
        raise ValueError(f"{source} contains duplicate symbols: {duplicate_values}")
    return as_text


def _validate_numeric_counts(frame: pd.DataFrame) -> None:
    for column in ("n_days", "n_observations"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = (
            numeric.isna()
            | ~numeric.map(lambda value: math.isfinite(float(value)))
            | (numeric < 0)
            | (numeric % 1 != 0)
        )
        if invalid.any():
            raise ValueError(f"invalid numeric count in {column}")


def _validate_eligibility(eligibility: pd.DataFrame) -> None:
    missing = [
        col for col in REQUIRED_ELIGIBILITY_COLUMNS if col not in eligibility.columns
    ]
    if missing:
        raise ValueError(f"symbol lifecycle input missing required columns: {missing}")
    _validate_symbols(eligibility, source="symbol lifecycle input")
    _validate_numeric_counts(eligibility)


def _reason_codes(row: pd.Series, baseline_bad: bool) -> list[str]:
    reasons: list[str] = []
    if bool(row["eligible"]):
        reasons.append("eligible")
    if not bool(row["sample_gate_pass"]):
        reasons.append("insufficient_days")
    if not bool(row["liquidity_gate_pass"]):
        reasons.append("depth_too_thin")
    if not bool(row["spread_cost_gate_pass"]):
        reasons.append("spread_too_wide")
    if not bool(row["activity_gate_pass"]):
        reasons.append("insufficient_activity")
    if not bool(row["stability_gate_pass"]):
        reasons.append("unstable_feed")
    if not bool(row["concentration_gate_pass"]):
        reasons.append("too_concentrated")
    if baseline_bad:
        reasons.append("bad_post_cost_baseline")
    return reasons or ["collected_only"]


def _read_optional_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported optional input extension: {path.suffix}")


def assign_symbol_lifecycle(
    eligibility: pd.DataFrame,
    *,
    previous_lifecycle: pd.DataFrame | None = None,
    baseline_scorecard: pd.DataFrame | None = None,
    config: LifecycleConfig = LifecycleConfig(),
) -> pd.DataFrame:
    """Return one row per symbol with diagnostic lifecycle state and transition."""
    _validate_eligibility(eligibility)
    current = eligibility.copy()
    current["symbol"] = _validate_symbols(current, source="symbol lifecycle input")
    for col in BOOL_COLUMNS:
        try:
            current[col] = current[col].map(_parse_bool)
        except ValueError as exc:
            raise ValueError(f"invalid boolean in {col}: {exc}") from exc
    current["n_days"] = pd.to_numeric(current["n_days"], errors="raise").astype(int)
    current["n_observations"] = pd.to_numeric(
        current["n_observations"], errors="raise"
    ).astype(int)

    previous_by_symbol: dict[str, str] = {}
    retired_symbols: set[str] = set()
    if previous_lifecycle is not None and not previous_lifecycle.empty:
        if not {"symbol", "lifecycle_state"}.issubset(previous_lifecycle.columns):
            raise ValueError(
                "previous lifecycle input must include symbol and lifecycle_state"
            )
        prev = previous_lifecycle.copy()
        prev["symbol"] = _validate_symbols(prev, source="previous lifecycle input")
        prev["lifecycle_state"] = prev["lifecycle_state"].astype(str)
        unknown_states = ~prev["lifecycle_state"].isin(LIFECYCLE_STATES)
        if unknown_states.any():
            bad = sorted(prev.loc[unknown_states, "lifecycle_state"].unique())
            raise ValueError(
                f"previous lifecycle input has unknown lifecycle_state values: {bad}"
            )
        previous_by_symbol = dict(
            zip(prev["symbol"], prev["lifecycle_state"], strict=False)
        )
        retired_symbols = set(prev.loc[prev["lifecycle_state"] == "RETIRED", "symbol"])

    baseline_bad_by_symbol: dict[str, bool] = {}
    if baseline_scorecard is not None:
        required = {"symbol", "post_cost_baseline_pass"}
        if not required.issubset(baseline_scorecard.columns):
            raise ValueError(
                "baseline scorecard must include symbol and post_cost_baseline_pass"
            )
        if not baseline_scorecard.empty:
            score = baseline_scorecard.copy()
            score["symbol"] = _validate_symbols(score, source="baseline scorecard")
            try:
                score["post_cost_baseline_pass"] = score["post_cost_baseline_pass"].map(
                    _parse_bool
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid boolean in post_cost_baseline_pass: {exc}"
                ) from exc
            baseline_bad_by_symbol = dict(
                zip(score["symbol"], ~score["post_cost_baseline_pass"], strict=False)
            )

    rows: list[dict[str, Any]] = []
    current_symbols = set(current["symbol"])
    all_symbols = sorted(current_symbols | set(previous_by_symbol))
    current_by_symbol = current.set_index("symbol")

    for symbol in all_symbols:
        previous_state = previous_by_symbol.get(symbol, "NEW")
        if symbol in retired_symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "previous_state": previous_state,
                    "lifecycle_state": "RETIRED",
                    "transition": f"{previous_state}->RETIRED",
                    "reason_codes": "retired_override",
                    "paper_trading_allowed_diagnostic": False,
                    "n_days": 0,
                    "n_observations": 0,
                }
            )
            continue
        if symbol not in current_symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "previous_state": previous_state,
                    "lifecycle_state": "DISABLED",
                    "transition": f"{previous_state}->DISABLED",
                    "reason_codes": "missing_current_eligibility",
                    "paper_trading_allowed_diagnostic": False,
                    "n_days": 0,
                    "n_observations": 0,
                }
            )
            continue

        row = current_by_symbol.loc[symbol]
        baseline_scorecard_supplied = baseline_scorecard is not None
        missing_baseline = (
            baseline_scorecard_supplied and symbol not in baseline_bad_by_symbol
        )
        baseline_bad = bool(baseline_bad_by_symbol.get(symbol, False))
        all_gates_pass = all(
            bool(row[col])
            for col in (
                "sample_gate_pass",
                "liquidity_gate_pass",
                "spread_cost_gate_pass",
                "activity_gate_pass",
                "stability_gate_pass",
                "concentration_gate_pass",
            )
        )
        inconsistent_eligible = bool(row["eligible"]) and not all_gates_pass
        reasons = _reason_codes(row, baseline_bad)
        if missing_baseline:
            reasons.append("missing_post_cost_baseline")
        if inconsistent_eligible:
            reasons.append("inconsistent_eligibility_snapshot")
        hard_disabled = (
            baseline_bad
            or missing_baseline
            or inconsistent_eligible
            or not bool(row["liquidity_gate_pass"])
            or not bool(row["activity_gate_pass"])
            or not bool(row["stability_gate_pass"])
            or not bool(row["concentration_gate_pass"])
        )
        if (
            bool(row["eligible"])
            and all_gates_pass
            and not baseline_bad
            and not missing_baseline
        ):
            state = "ELIGIBLE"
        elif hard_disabled:
            state = "DISABLED"
        elif (
            row["n_days"] >= config.min_researchable_days
            and row["n_observations"] >= config.min_researchable_observations
        ):
            state = "RESEARCHABLE"
        else:
            state = "COLLECTED"
        if previous_state in config.probation_previous_states and state in {
            "RESEARCHABLE",
            "COLLECTED",
        }:
            state = "PROBATION"
            reasons.append("demoted_from_eligible")

        rows.append(
            {
                "symbol": symbol,
                "previous_state": previous_state,
                "lifecycle_state": state,
                "transition": f"{previous_state}->{state}",
                "reason_codes": ";".join(dict.fromkeys(reasons)),
                "paper_trading_allowed_diagnostic": state == "ELIGIBLE",
                "n_days": int(row["n_days"]),
                "n_observations": int(row["n_observations"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["lifecycle_state", "symbol"])
        .reset_index(drop=True)
    )


def _state_counts(lifecycle: pd.DataFrame) -> pd.DataFrame:
    counts = (
        lifecycle["lifecycle_state"].value_counts().to_dict()
        if not lifecycle.empty
        else {}
    )
    return pd.DataFrame(
        [
            {"lifecycle_state": state, "symbols": int(counts.get(state, 0))}
            for state in LIFECYCLE_STATES
        ]
    )


def _overall_verdict(lifecycle: pd.DataFrame) -> str:
    if lifecycle.empty:
        return "NO_SYMBOLS_DIAGNOSTIC"
    if (lifecycle["lifecycle_state"] == "ELIGIBLE").any():
        return "HAS_ELIGIBLE_SYMBOLS_DIAGNOSTIC"
    if (lifecycle["lifecycle_state"] == "RESEARCHABLE").any():
        return "RESEARCHABLE_ONLY_DIAGNOSTIC"
    return "NO_ELIGIBLE_SYMBOLS_DIAGNOSTIC"


def write_symbol_lifecycle_report(
    eligibility_path: Path,
    out_dir: Path,
    *,
    previous_lifecycle_path: Path | None = None,
    baseline_scorecard_path: Path | None = None,
    config: LifecycleConfig = LifecycleConfig(),
) -> dict[str, Any]:
    eligibility = pd.read_csv(eligibility_path)
    previous = _read_optional_frame(previous_lifecycle_path)
    baseline = _read_optional_frame(baseline_scorecard_path)
    lifecycle = assign_symbol_lifecycle(
        eligibility,
        previous_lifecycle=previous,
        baseline_scorecard=baseline,
        config=config,
    )
    state_counts = _state_counts(lifecycle)
    transitions = (
        lifecycle.groupby("transition", dropna=False)
        .size()
        .reset_index(name="symbols")
        .sort_values(["symbols", "transition"], ascending=[False, True])
        if not lifecycle.empty
        else pd.DataFrame(columns=["transition", "symbols"])
    )
    verdict = _overall_verdict(lifecycle)
    out_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.to_csv(out_dir / "symbol_lifecycle.csv", index=False)
    state_counts.to_csv(out_dir / "state_counts.csv", index=False)
    transitions.to_csv(out_dir / "transitions.csv", index=False)
    pd.DataFrame([asdict(config)]).to_csv(out_dir / "config.csv", index=False)

    readme = f"""# Pacifica Symbol Lifecycle

Verdict: `{verdict}`

This is a diagnostic lifecycle layer. It turns static eligibility gates into promotion/demotion states for the collect-broad/trade-selectively universe policy. It does not authorize live trading. `paper_trading_allowed_diagnostic=True` only means the symbol passed this diagnostic lifecycle snapshot and still requires the broader paper ledger, governor, parity, walk-forward, portfolio, and maturity gates.

## State counts

{dataframe_to_markdown_table(state_counts)}

## Transition counts

{dataframe_to_markdown_table(transitions)}

## Lifecycle preview

{dataframe_to_markdown_table(lifecycle, max_rows=30)}

## Artifacts

- `symbol_lifecycle.csv`
- `state_counts.csv`
- `transitions.csv`
- `config.csv`
"""
    (out_dir / "README.md").write_text(readme)
    return {
        "verdict": verdict,
        "symbols": int(len(lifecycle)),
        "eligible_symbols": (
            int((lifecycle["lifecycle_state"] == "ELIGIBLE").sum())
            if not lifecycle.empty
            else 0
        ),
        "researchable_symbols": (
            int((lifecycle["lifecycle_state"] == "RESEARCHABLE").sum())
            if not lifecycle.empty
            else 0
        ),
        "disabled_symbols": (
            int((lifecycle["lifecycle_state"] == "DISABLED").sum())
            if not lifecycle.empty
            else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY_PATH)
    parser.add_argument("--previous-lifecycle", type=Path)
    parser.add_argument("--baseline-scorecard", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    result = write_symbol_lifecycle_report(
        args.eligibility,
        args.out_dir,
        previous_lifecycle_path=args.previous_lifecycle,
        baseline_scorecard_path=args.baseline_scorecard,
    )
    print(f"verdict: {result['verdict']}")
    print(f"eligible_symbols: {result['eligible_symbols']}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

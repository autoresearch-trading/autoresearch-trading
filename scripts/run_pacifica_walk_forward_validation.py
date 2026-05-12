# scripts/run_pacifica_walk_forward_validation.py
"""Run purged chronological walk-forward validation for Pacifica strategy outputs.

This harness is strategy-neutral. It validates an already materialized return stream
and refuses to frame young or concentrated samples as edge evidence.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_PATH = Path(
    "docs/experiments/walk-forward-validation/bootstrap_strategy_returns.csv"
)
DEFAULT_OUT_DIR = Path("docs/experiments/walk-forward-validation")
REQUIRED_COLUMNS = ("timestamp", "symbol", "strategy_return_bps", "baseline_return_bps")


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 14
    test_days: int = 7
    purge_days: int = 1
    step_days: int = 7
    min_diagnostic_days: int = 10
    min_provisional_days: int = 30
    min_validation_grade_days: int = 60
    min_test_rows: int = 20
    max_day_concentration: float = 0.25
    max_symbol_concentration: float = 0.50
    random_control_trials: int = 100
    random_seed: int = 17
    min_random_control_beaten_rate: float = 0.50


DEFAULT_CONFIG = WalkForwardConfig()


@dataclass(frozen=True)
class WalkForwardResult:
    verdict: str
    failure_reasons: list[str]
    summary: dict[str, Any]
    windows: pd.DataFrame
    window_scorecard: pd.DataFrame
    random_controls: pd.DataFrame


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
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


def _date_string(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def _parse_eligible(value: Any) -> bool | None:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", ""}:
            return False
    return None


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"validation input missing required columns: {missing}")
    out = frame.copy()

    timestamp = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    strategy = pd.to_numeric(out["strategy_return_bps"], errors="coerce")
    baseline = pd.to_numeric(out["baseline_return_bps"], errors="coerce")
    symbol_raw = out["symbol"]
    symbol_missing = symbol_raw.isna() | (symbol_raw.astype(str).str.strip() == "")
    timestamp_invalid = timestamp.isna()
    strategy_invalid = strategy.isna() | ~strategy.map(math.isfinite)
    baseline_invalid = baseline.isna() | ~baseline.map(math.isfinite)

    eligible_invalid = pd.Series(False, index=out.index)
    filtered_ineligible = 0
    if "eligible" in out.columns:
        parsed_eligible = out["eligible"].map(_parse_eligible)
        eligible_invalid = parsed_eligible.isna()
        eligible_true = parsed_eligible.fillna(False).astype(bool)
        filtered_ineligible = int((~eligible_true & ~eligible_invalid).sum())
    else:
        eligible_true = pd.Series(True, index=out.index)

    invalid_mask = (
        timestamp_invalid
        | strategy_invalid
        | baseline_invalid
        | symbol_missing
        | eligible_invalid
    )
    out["timestamp"] = timestamp
    out["strategy_return_bps"] = strategy
    out["baseline_return_bps"] = baseline
    out["symbol"] = symbol_raw.astype("string").str.strip()
    out = out[~invalid_mask & eligible_true].copy()
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    out.attrs["invalid_timestamp_rows"] = int(timestamp_invalid.sum())
    out.attrs["invalid_symbol_rows"] = int(symbol_missing.sum())
    out.attrs["invalid_strategy_return_rows"] = int(strategy_invalid.sum())
    out.attrs["invalid_baseline_return_rows"] = int(baseline_invalid.sum())
    out.attrs["invalid_eligible_rows"] = int(eligible_invalid.sum())
    out.attrs["invalid_required_rows"] = int(invalid_mask.sum())
    out.attrs["filtered_ineligible_rows"] = filtered_ineligible
    return out


def _validate_config(config: WalkForwardConfig) -> None:
    positive_ints = {
        "train_days": config.train_days,
        "test_days": config.test_days,
        "step_days": config.step_days,
        "min_diagnostic_days": config.min_diagnostic_days,
        "min_provisional_days": config.min_provisional_days,
        "min_validation_grade_days": config.min_validation_grade_days,
        "min_test_rows": config.min_test_rows,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.purge_days < 0:
        raise ValueError("purge_days must be non-negative")
    if config.step_days < config.test_days:
        raise ValueError(
            "step_days must be greater than or equal to test_days to avoid overlapping OOS windows"
        )
    if config.random_control_trials < 0:
        raise ValueError("random_control_trials must be non-negative")
    for name, value in {
        "max_day_concentration": config.max_day_concentration,
        "max_symbol_concentration": config.max_symbol_concentration,
        "min_random_control_beaten_rate": config.min_random_control_beaten_rate,
    }.items():
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")


def build_purged_walk_forward_windows(
    frame: pd.DataFrame, *, config: WalkForwardConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Build train/purge/test windows over distinct UTC dates."""
    _validate_config(config)
    data = _prepare_frame(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "train_start",
                "train_end",
                "purge_start",
                "purge_end",
                "test_start",
                "test_end",
                "train_rows",
                "purged_rows",
                "test_rows",
            ]
        )
    dates = sorted(pd.to_datetime(data["date"].unique(), utc=True))
    rows: list[dict[str, Any]] = []
    span = config.train_days + config.purge_days + config.test_days
    window_id = 1
    for start_idx in range(0, max(0, len(dates) - span + 1), config.step_days):
        train_start = dates[start_idx]
        train_end = dates[start_idx + config.train_days - 1]
        purge_start_idx = start_idx + config.train_days
        purge_end_idx = purge_start_idx + config.purge_days - 1
        test_start_idx = start_idx + config.train_days + config.purge_days
        test_end_idx = test_start_idx + config.test_days - 1
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        if config.purge_days:
            purge_start = dates[purge_start_idx]
            purge_end = dates[purge_end_idx]
            purge_mask = (pd.to_datetime(data["date"], utc=True) >= purge_start) & (
                pd.to_datetime(data["date"], utc=True) <= purge_end
            )
            purge_start_label = _date_string(purge_start)
            purge_end_label = _date_string(purge_end)
        else:
            purge_mask = pd.Series(False, index=data.index)
            purge_start_label = ""
            purge_end_label = ""
        date_series = pd.to_datetime(data["date"], utc=True)
        train_mask = (date_series >= train_start) & (date_series <= train_end)
        test_mask = (date_series >= test_start) & (date_series <= test_end)
        rows.append(
            {
                "window_id": window_id,
                "train_start": _date_string(train_start),
                "train_end": _date_string(train_end),
                "purge_start": purge_start_label,
                "purge_end": purge_end_label,
                "test_start": _date_string(test_start),
                "test_end": _date_string(test_end),
                "train_rows": int(train_mask.sum()),
                "purged_rows": int(purge_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
        )
        window_id += 1
    return pd.DataFrame(rows)


def _downside_deviation(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    downside = vals.clip(upper=0.0)
    if (downside < 0).sum() == 0:
        return 0.0
    return float((downside.pow(2).mean()) ** 0.5)


def _sortino(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    mean = float(vals.mean())
    dd = _downside_deviation(vals)
    if dd == 0:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return math.nan
    return float(mean / dd)


def _max_drawdown_bps(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.empty:
        return math.nan
    equity = vals.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min())


def _max_concentration(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    counts = frame[column].value_counts(dropna=True)
    if counts.empty or counts.sum() == 0:
        return math.nan
    return float(counts.max() / counts.sum())


def _score_windows(data: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if windows.empty:
        return pd.DataFrame()
    date_series = pd.to_datetime(data["date"], utc=True)
    for _, window in windows.iterrows():
        start = pd.Timestamp(window["test_start"], tz="UTC")
        end = pd.Timestamp(window["test_end"], tz="UTC")
        test = data[(date_series >= start) & (date_series <= end)].copy()
        returns = (
            test["strategy_return_bps"] if not test.empty else pd.Series(dtype=float)
        )
        baseline = (
            test["baseline_return_bps"] if not test.empty else pd.Series(dtype=float)
        )
        rows.append(
            {
                "window_id": int(window["window_id"]),
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "test_rows": int(len(test)),
                "distinct_test_days": (
                    int(test["date"].nunique()) if not test.empty else 0
                ),
                "distinct_symbols": (
                    int(test["symbol"].nunique()) if not test.empty else 0
                ),
                "net_pnl_bps": float(returns.sum()),
                "baseline_pnl_bps": float(baseline.sum()),
                "excess_vs_baseline_bps": float(returns.sum() - baseline.sum()),
                "mean_return_bps": (
                    float(returns.mean()) if not returns.empty else math.nan
                ),
                "sortino": _sortino(returns),
                "max_drawdown_bps": _max_drawdown_bps(returns),
                "max_day_concentration": _max_concentration(test, "date"),
                "max_symbol_concentration": _max_concentration(test, "symbol"),
            }
        )
    return pd.DataFrame(rows)


def _oos_test_rows(data: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if data.empty or windows.empty:
        return data.head(0).copy()
    date_series = pd.to_datetime(data["date"], utc=True)
    masks = []
    for _, window in windows.iterrows():
        start = pd.Timestamp(window["test_start"], tz="UTC")
        end = pd.Timestamp(window["test_end"], tz="UTC")
        masks.append((date_series >= start) & (date_series <= end))
    if not masks:
        return data.head(0).copy()
    mask = masks[0].copy()
    for other in masks[1:]:
        mask = mask | other
    return data[mask].copy()


def _random_controls(
    data: pd.DataFrame, *, rows_to_sample: int, config: WalkForwardConfig
) -> pd.DataFrame:
    if config.random_control_trials == 0 or rows_to_sample <= 0 or data.empty:
        return pd.DataFrame(
            columns=["control_id", "sampled_rows", "net_pnl_bps", "sortino"]
        )
    baseline_returns = data["baseline_return_bps"].reset_index(drop=True)
    replace = rows_to_sample > len(baseline_returns)
    rows: list[dict[str, Any]] = []
    for control_id in range(1, config.random_control_trials + 1):
        sample = baseline_returns.sample(
            n=rows_to_sample,
            replace=replace,
            random_state=config.random_seed + control_id,
        ).reset_index(drop=True)
        rows.append(
            {
                "control_id": control_id,
                "sampled_rows": int(len(sample)),
                "net_pnl_bps": float(sample.sum()),
                "sortino": _sortino(sample),
            }
        )
    return pd.DataFrame(rows)


def _summary_from_scores(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    scores: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, Any]:
    total_net = float(scores["net_pnl_bps"].sum()) if not scores.empty else 0.0
    total_baseline = (
        float(scores["baseline_pnl_bps"].sum()) if not scores.empty else 0.0
    )
    beaten_rate = math.nan
    if not controls.empty:
        beaten_rate = float((total_net > controls["net_pnl_bps"]).mean())
    oos = _oos_test_rows(data, windows)
    summary = {
        "observations": int(len(data)),
        "distinct_days": int(data["date"].nunique()) if not data.empty else 0,
        "distinct_symbols": int(data["symbol"].nunique()) if not data.empty else 0,
        "max_day_concentration": _max_concentration(data, "date"),
        "max_symbol_concentration": _max_concentration(data, "symbol"),
        "distinct_oos_days": int(oos["date"].nunique()) if not oos.empty else 0,
        "distinct_oos_symbols": int(oos["symbol"].nunique()) if not oos.empty else 0,
        "max_oos_day_concentration": _max_concentration(oos, "date"),
        "max_oos_symbol_concentration": _max_concentration(oos, "symbol"),
        "max_window_day_concentration": (
            float(scores["max_day_concentration"].max())
            if not scores.empty
            else math.nan
        ),
        "max_window_symbol_concentration": (
            float(scores["max_symbol_concentration"].max())
            if not scores.empty
            else math.nan
        ),
        "validation_windows": int(len(windows)),
        "scored_windows": int(len(scores)),
        "total_test_rows": int(scores["test_rows"].sum()) if not scores.empty else 0,
        "unique_oos_rows": int(len(oos)),
        "net_pnl_bps": total_net,
        "baseline_pnl_bps": total_baseline,
        "excess_vs_baseline_bps": total_net - total_baseline,
        "sortino": _sortino(scores["net_pnl_bps"]) if not scores.empty else math.nan,
        "max_drawdown_bps": (
            _max_drawdown_bps(scores["net_pnl_bps"]) if not scores.empty else math.nan
        ),
        "random_control_trials": int(len(controls)),
        "random_controls_beaten_rate": beaten_rate,
    }
    for key in (
        "invalid_timestamp_rows",
        "invalid_symbol_rows",
        "invalid_strategy_return_rows",
        "invalid_baseline_return_rows",
        "invalid_eligible_rows",
        "invalid_required_rows",
        "filtered_ineligible_rows",
    ):
        summary[key] = int(data.attrs.get(key, 0))
    return summary


def _verdict(
    summary: dict[str, Any], config: WalkForwardConfig
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if summary["invalid_required_rows"] > 0:
        failures.append("invalid_required_fields")
    if summary["distinct_days"] < config.min_diagnostic_days:
        failures.append("insufficient_distinct_days")
    if summary["observations"] == 0:
        failures.append("no_validation_rows")
    if (
        math.isfinite(summary["max_day_concentration"])
        and summary["max_day_concentration"] > config.max_day_concentration
    ):
        failures.append("day_concentration_too_high")
    if (
        math.isfinite(summary["max_symbol_concentration"])
        and summary["max_symbol_concentration"] > config.max_symbol_concentration
    ):
        failures.append("symbol_concentration_too_high")
    if (
        math.isfinite(summary["max_oos_day_concentration"])
        and summary["max_oos_day_concentration"] > config.max_day_concentration
    ):
        failures.append("oos_day_concentration_too_high")
    if (
        math.isfinite(summary["max_oos_symbol_concentration"])
        and summary["max_oos_symbol_concentration"] > config.max_symbol_concentration
    ):
        failures.append("oos_symbol_concentration_too_high")
    if (
        math.isfinite(summary["max_window_symbol_concentration"])
        and summary["max_window_symbol_concentration"] > config.max_symbol_concentration
    ):
        failures.append("window_symbol_concentration_too_high")
    minimum_days_for_window_day_gate = math.ceil(1.0 / config.max_day_concentration)
    if (
        config.test_days >= minimum_days_for_window_day_gate
        and math.isfinite(summary["max_window_day_concentration"])
        and summary["max_window_day_concentration"] > config.max_day_concentration
    ):
        failures.append("window_day_concentration_too_high")
    if summary["validation_windows"] == 0:
        failures.append("no_purged_validation_windows")
    if (
        summary["total_test_rows"] < config.min_test_rows
        and summary["validation_windows"] > 0
    ):
        failures.append("insufficient_test_rows")

    sample_failures = {
        "invalid_required_fields",
        "insufficient_distinct_days",
        "no_validation_rows",
        "day_concentration_too_high",
        "symbol_concentration_too_high",
        "oos_day_concentration_too_high",
        "oos_symbol_concentration_too_high",
        "window_symbol_concentration_too_high",
        "window_day_concentration_too_high",
        "no_purged_validation_windows",
        "insufficient_test_rows",
    }
    if any(reason in sample_failures for reason in failures):
        return "INSUFFICIENT_SAMPLE_DIAGNOSTIC", failures

    if config.purge_days == 0:
        failures.append("no_purge_gap")
    if config.random_control_trials == 0 or summary["random_control_trials"] == 0:
        failures.append("random_controls_missing")
    if not math.isfinite(summary["net_pnl_bps"]) or summary["net_pnl_bps"] <= 0:
        failures.append("nonpositive_post_cost_pnl")
    if (
        not math.isfinite(summary["excess_vs_baseline_bps"])
        or summary["excess_vs_baseline_bps"] <= 0
    ):
        failures.append("baseline_not_beaten")
    if summary["random_control_trials"] > 0 and (
        not math.isfinite(summary["random_controls_beaten_rate"])
        or summary["random_controls_beaten_rate"]
        < config.min_random_control_beaten_rate
    ):
        failures.append("random_controls_not_beaten")

    validation_grade = (
        summary["distinct_days"] >= config.min_validation_grade_days
        and summary["distinct_oos_days"] >= config.min_validation_grade_days
    )
    provisional = (
        summary["distinct_days"] >= config.min_provisional_days
        and summary["distinct_oos_days"] >= config.min_provisional_days
    )

    if failures:
        return (
            "VALIDATION_GRADE_FAIL" if validation_grade else "PROVISIONAL_FAIL"
        ), failures
    if validation_grade:
        return "VALIDATION_GRADE_PASS", []
    if provisional:
        return "PROVISIONAL_PASS", []
    return "EARLY_SANITY_ONLY", []


def evaluate_walk_forward_validation(
    frame: pd.DataFrame, *, config: WalkForwardConfig = DEFAULT_CONFIG
) -> WalkForwardResult:
    """Evaluate a strategy-neutral return stream with purged walk-forward gates."""
    _validate_config(config)
    data = _prepare_frame(frame)
    windows = build_purged_walk_forward_windows(data, config=config)
    scores = _score_windows(data, windows)
    oos = _oos_test_rows(data, windows)
    rows_to_sample = int(scores["test_rows"].sum()) if not scores.empty else 0
    controls = _random_controls(oos, rows_to_sample=rows_to_sample, config=config)
    summary = _summary_from_scores(data, windows, scores, controls)
    verdict, failures = _verdict(summary, config)
    return WalkForwardResult(
        verdict=verdict,
        failure_reasons=failures,
        summary=summary,
        windows=windows,
        window_scorecard=scores,
        random_controls=controls,
    )


def _summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )


def write_walk_forward_report(
    input_path: Path,
    out_dir: Path,
    *,
    config: WalkForwardConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    frame = _read_frame(input_path)
    result = evaluate_walk_forward_validation(frame, config=config)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.windows.to_csv(out_dir / "windows.csv", index=False)
    result.window_scorecard.to_csv(out_dir / "window_scorecard.csv", index=False)
    result.random_controls.to_csv(out_dir / "random_controls.csv", index=False)
    _summary_table(result.summary).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame([asdict(config)]).to_csv(out_dir / "config.csv", index=False)

    readme = f"""# Pacifica Walk-Forward Validation

Verdict: `{result.verdict}`

Failure reasons: `{';'.join(result.failure_reasons) if result.failure_reasons else 'none'}`

This is a strategy-neutral, non-HFT validation harness. It uses purged chronological windows and random same-frequency controls before any future strategy result can be discussed as evidence. Do not treat this as an edge claim unless sample maturity, concentration, economics, baseline, and control gates all pass.

## Interpretation discipline

- `INSUFFICIENT_SAMPLE_DIAGNOSTIC`: plumbing/sample diagnostic only.
- `EARLY_SANITY_ONLY`: at least {config.min_diagnostic_days} distinct days, but below provisional maturity.
- `PROVISIONAL_PASS` / `PROVISIONAL_FAIL`: at least {config.min_provisional_days} distinct days.
- `VALIDATION_GRADE_PASS` / `VALIDATION_GRADE_FAIL`: at least {config.min_validation_grade_days} distinct days.

## Summary

{dataframe_to_markdown_table(_summary_table(result.summary))}

## Config

{dataframe_to_markdown_table(pd.DataFrame([asdict(config)]))}

## Window scorecard preview

{dataframe_to_markdown_table(result.window_scorecard, max_rows=20)}

## Random same-frequency controls

{dataframe_to_markdown_table(result.random_controls, max_rows=20)}

## Artifacts

- `summary.csv`
- `config.csv`
- `windows.csv`
- `window_scorecard.csv`
- `random_controls.csv`
"""
    (out_dir / "README.md").write_text(readme)
    return {
        "verdict": result.verdict,
        "failure_reasons": result.failure_reasons,
        **result.summary,
    }


def _write_bootstrap_input(path: Path) -> None:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for day in range(8):
        for symbol in ("BTC", "ETH"):
            ts = start + pd.Timedelta(days=day)
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "strategy_return_bps": 0.0,
                    "baseline_return_bps": 0.0,
                    "eligible": True,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    parser.add_argument("--allow-fail-diagnostic", action="store_true")
    args = parser.parse_args()

    if args.bootstrap_if_missing and not args.input.exists():
        _write_bootstrap_input(args.input)
    result = write_walk_forward_report(args.input, args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    if result["verdict"] == "INSUFFICIENT_SAMPLE_DIAGNOSTIC":
        allow_clean_insufficient_sample = (
            args.allow_fail_diagnostic
            and "invalid_required_fields" not in result.get("failure_reasons", [])
        )
        return 0 if allow_clean_insufficient_sample else 1
    if result["verdict"] not in {"PROVISIONAL_PASS", "VALIDATION_GRADE_PASS"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

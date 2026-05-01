# scripts/non_hft_toxic_overlay_probe.py
"""Evaluate a non-HFT toxic-regime no-trade overlay.

This probe asks whether high 1-minute toxicity regimes have worse future adverse
excursion/downside risk, and whether excluding top-toxicity windows improves a
slow Sortino/downside proxy. It is intentionally not an HFT alpha model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_OUT_DIR = Path("docs/experiments/toxic-regime-overlay")
DEFAULT_HORIZONS = (5, 15, 30, 60)
DEFAULT_CUTOFFS = (0.90, 0.80, 0.70)


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
    """Render a small Markdown table without pandas' optional tabulate dependency."""
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


def add_forward_path_metrics(
    state: pd.DataFrame,
    *,
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS,
    bucket_minutes: int = 1,
) -> pd.DataFrame:
    """Add future return and adverse-excursion metrics for each symbol bucket."""
    if state.empty:
        return state.copy()
    required = {"symbol", "bucket_start_ms", "last_mid"}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"state missing required columns: {sorted(missing)}")

    out = state.copy().sort_values(["symbol", "bucket_start_ms"]).reset_index(drop=True)
    out["date"] = _date_from_bucket_ms(out["bucket_start_ms"])
    for horizon in horizons_minutes:
        steps = max(1, int(horizon / bucket_minutes))
        fwd_col = f"forward_return_{horizon}m_bps"
        long_col = f"long_adverse_excursion_{horizon}m_bps"
        short_col = f"short_adverse_excursion_{horizon}m_bps"
        out[fwd_col] = math.nan
        out[long_col] = math.nan
        out[short_col] = math.nan

        for _, idx in out.groupby("symbol", sort=False).groups.items():
            group = out.loc[list(idx)].sort_values("bucket_start_ms")
            mids = pd.to_numeric(group["last_mid"], errors="coerce").to_numpy(
                dtype=float
            )
            positions = group.index.to_list()
            for i, pos in enumerate(positions):
                start = mids[i]
                end_i = i + steps
                if not math.isfinite(start) or start <= 0 or end_i >= len(mids):
                    continue
                path = mids[i + 1 : end_i + 1]
                path = path[pd.notna(path)]
                if len(path) < steps:
                    continue
                path_returns = (path / start - 1.0) * 10_000
                out.at[pos, fwd_col] = float(path_returns[-1])
                out.at[pos, long_col] = float(path_returns.min())
                out.at[pos, short_col] = float((-path_returns).min())
    return out


def assign_toxicity_buckets(
    state: pd.DataFrame, *, n_buckets: int = 10
) -> pd.DataFrame:
    """Assign low=1 to high=n_buckets toxicity buckets."""
    if "toxicity_score" not in state.columns:
        raise ValueError("state missing required column: toxicity_score")
    out = state.copy()
    score = pd.to_numeric(out["toxicity_score"], errors="coerce")
    valid = score.notna()
    out["toxicity_decile"] = pd.NA
    if valid.any():
        ranked = score[valid].rank(method="first", pct=True)
        buckets = (ranked * n_buckets).apply(math.ceil).clip(1, n_buckets).astype(int)
        out.loc[valid, "toxicity_decile"] = buckets
    return out


def _downside_deviation(values: pd.Series, mar: float = 0.0) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    downside = (vals - mar).clip(upper=0.0)
    return float((downside.pow(2).mean()) ** 0.5)


def _sortino_proxy(values: pd.Series, mar: float = 0.0) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    dd = _downside_deviation(vals, mar=mar)
    if not math.isfinite(dd) or dd == 0:
        return math.nan
    return float((vals.mean() - mar) / dd)


def _tail_loss_probability(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    threshold = vals.quantile(0.10)
    return float((vals <= threshold).mean())


def _risk_stats(rows: pd.DataFrame, horizon: int) -> dict[str, float]:
    fwd = f"forward_return_{horizon}m_bps"
    long_adv = f"long_adverse_excursion_{horizon}m_bps"
    short_adv = f"short_adverse_excursion_{horizon}m_bps"
    return {
        "mean_forward_return": float(pd.to_numeric(rows[fwd], errors="coerce").mean()),
        "median_forward_return": float(
            pd.to_numeric(rows[fwd], errors="coerce").median()
        ),
        "p05_long_adverse_excursion_bps": float(
            pd.to_numeric(rows[long_adv], errors="coerce").quantile(0.05)
        ),
        "p05_short_adverse_excursion_bps": float(
            pd.to_numeric(rows[short_adv], errors="coerce").quantile(0.05)
        ),
        "downside_deviation": _downside_deviation(rows[fwd]),
        "sortino_proxy": _sortino_proxy(rows[fwd]),
        "tail_loss_probability": _tail_loss_probability(rows[fwd]),
    }


def summarize_toxicity_buckets(
    state: pd.DataFrame, *, horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    rows = []
    if state.empty or "toxicity_decile" not in state.columns:
        return pd.DataFrame()
    for horizon in horizons_minutes:
        fwd = f"forward_return_{horizon}m_bps"
        if fwd not in state.columns:
            continue
        eligible = state[state[fwd].notna()].copy()
        for decile, group in eligible.groupby("toxicity_decile", dropna=True):
            stats = _risk_stats(group, horizon)
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "toxicity_decile": int(decile),
                    "n_obs": len(group),
                    "n_days": group["date"].nunique() if "date" in group else 0,
                    "n_symbols": group["symbol"].nunique() if "symbol" in group else 0,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _max_day_concentration(rows: pd.DataFrame) -> float:
    if rows.empty or "date" not in rows.columns:
        return math.nan
    counts = rows["date"].value_counts()
    if counts.empty:
        return math.nan
    return float(counts.max() / counts.sum())


def evaluate_overlay(
    state: pd.DataFrame,
    *,
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS,
    toxicity_cutoffs: tuple[float, ...] = DEFAULT_CUTOFFS,
    min_days: int = 30,
    min_removed: int = 100,
) -> pd.DataFrame:
    rows = []
    if state.empty:
        return pd.DataFrame()
    for horizon in horizons_minutes:
        fwd = f"forward_return_{horizon}m_bps"
        if fwd not in state.columns:
            continue
        eligible = state[state[fwd].notna()].copy()
        if eligible.empty:
            continue
        for cutoff in toxicity_cutoffs:
            threshold = eligible["toxicity_score"].quantile(cutoff)
            removed = eligible[eligible["toxicity_score"] >= threshold]
            accepted = eligible[eligible["toxicity_score"] < threshold]
            base = _risk_stats(eligible, horizon)
            acc = _risk_stats(accepted, horizon) if not accepted.empty else {}
            rem = _risk_stats(removed, horizon) if not removed.empty else {}
            base_dd = base.get("downside_deviation", math.nan)
            acc_dd = acc.get("downside_deviation", math.nan)
            delta_dd = (
                acc_dd - base_dd
                if math.isfinite(acc_dd) and math.isfinite(base_dd)
                else math.nan
            )
            delta_dd_pct = (
                (delta_dd / base_dd * 100.0)
                if base_dd and math.isfinite(delta_dd)
                else math.nan
            )
            rows.append(
                {
                    "cutoff": cutoff,
                    "horizon_minutes": horizon,
                    "threshold": threshold,
                    "n_baseline": len(eligible),
                    "n_accepted": len(accepted),
                    "n_removed": len(removed),
                    "retention_rate": len(accepted) / len(eligible),
                    "n_days": eligible["date"].nunique() if "date" in eligible else 0,
                    "removed_days": (
                        removed["date"].nunique() if "date" in removed else 0
                    ),
                    "max_day_concentration": _max_day_concentration(removed),
                    "sample_gate_pass": bool(
                        (eligible["date"].nunique() if "date" in eligible else 0)
                        >= min_days
                        and len(removed) >= min_removed
                    ),
                    "retention_gate_pass": bool(len(accepted) / len(eligible) >= 0.70),
                    "mean_forward_return_baseline": base.get(
                        "mean_forward_return", math.nan
                    ),
                    "mean_forward_return_accepted": acc.get(
                        "mean_forward_return", math.nan
                    ),
                    "mean_forward_return_removed": rem.get(
                        "mean_forward_return", math.nan
                    ),
                    "p05_long_adverse_excursion_baseline": base.get(
                        "p05_long_adverse_excursion_bps", math.nan
                    ),
                    "p05_long_adverse_excursion_accepted": acc.get(
                        "p05_long_adverse_excursion_bps", math.nan
                    ),
                    "p05_long_adverse_excursion_removed": rem.get(
                        "p05_long_adverse_excursion_bps", math.nan
                    ),
                    "p05_short_adverse_excursion_baseline": base.get(
                        "p05_short_adverse_excursion_bps", math.nan
                    ),
                    "p05_short_adverse_excursion_accepted": acc.get(
                        "p05_short_adverse_excursion_bps", math.nan
                    ),
                    "p05_short_adverse_excursion_removed": rem.get(
                        "p05_short_adverse_excursion_bps", math.nan
                    ),
                    "downside_deviation_baseline": base_dd,
                    "downside_deviation_accepted": acc_dd,
                    "delta_downside_deviation": delta_dd,
                    "delta_downside_deviation_pct": delta_dd_pct,
                    "sortino_proxy_baseline": base.get("sortino_proxy", math.nan),
                    "sortino_proxy_accepted": acc.get("sortino_proxy", math.nan),
                    "delta_sortino_proxy": acc.get("sortino_proxy", math.nan)
                    - base.get("sortino_proxy", math.nan),
                    "tail_loss_probability_baseline": base.get(
                        "tail_loss_probability", math.nan
                    ),
                    "tail_loss_probability_accepted": acc.get(
                        "tail_loss_probability", math.nan
                    ),
                    "delta_p05_long_adverse_excursion_bps": acc.get(
                        "p05_long_adverse_excursion_bps", math.nan
                    )
                    - base.get("p05_long_adverse_excursion_bps", math.nan),
                    "delta_p05_short_adverse_excursion_bps": acc.get(
                        "p05_short_adverse_excursion_bps", math.nan
                    )
                    - base.get("p05_short_adverse_excursion_bps", math.nan),
                }
            )
    return pd.DataFrame(rows)


def verdict_from_scorecard(
    scorecard: pd.DataFrame,
    *,
    primary_cutoff: float = 0.90,
    min_days: int = 30,
    min_removed: int = 100,
) -> str:
    if scorecard.empty:
        return "INSUFFICIENT_SAMPLE_EMPTY"
    primary = scorecard[scorecard["cutoff"].round(6) == round(primary_cutoff, 6)].copy()
    if primary.empty:
        primary = scorecard.copy()
    if (primary["n_days"].max() < min_days) or (
        primary["n_removed"].max() < min_removed
    ):
        return "INSUFFICIENT_SAMPLE_DIAGNOSTIC"
    candidates = primary[primary["horizon_minutes"].isin([15, 30, 60])]
    pass_rows = candidates[
        (candidates["retention_rate"] >= 0.70)
        & (candidates["delta_downside_deviation_pct"] <= -3.0)
        & (candidates["delta_p05_long_adverse_excursion_bps"] >= 5.0)
        & (candidates["delta_p05_short_adverse_excursion_bps"] >= 5.0)
        & (candidates["delta_sortino_proxy"] >= 0.0)
    ]
    if len(pass_rows) >= 2 and any(pass_rows["horizon_minutes"].isin([15, 30])):
        return "PASS_DIAGNOSTIC"
    mixed = primary[
        (primary["delta_downside_deviation_pct"] < 0)
        | (primary["delta_p05_long_adverse_excursion_bps"] > 0)
        | (primary["delta_p05_short_adverse_excursion_bps"] > 0)
    ]
    if not mixed.empty:
        return "INCONCLUSIVE_MIXED"
    return "FAIL"


def _symbol_summary(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame()
    return (
        state.groupby("symbol", as_index=False)
        .agg(
            n_obs=("bucket_start_ms", "size"),
            mean_toxicity=("toxicity_score", "mean"),
            max_toxicity=("toxicity_score", "max"),
        )
        .sort_values("mean_toxicity", ascending=False)
    )


def _hour_summary(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame()
    out = state.copy()
    out["hour_utc"] = pd.to_datetime(
        out["bucket_start_ms"], unit="ms", utc=True
    ).dt.hour
    return (
        out.groupby("hour_utc", as_index=False)
        .agg(
            n_obs=("bucket_start_ms", "size"),
            mean_toxicity=("toxicity_score", "mean"),
            max_toxicity=("toxicity_score", "max"),
        )
        .sort_values("hour_utc")
    )


def _markdown_report(
    *,
    state: pd.DataFrame,
    scorecard: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    verdict: str,
    horizons_minutes: tuple[int, ...],
    toxicity_cutoffs: tuple[float, ...],
    min_days: int,
    min_removed: int,
) -> str:
    n_days = state["date"].nunique() if "date" in state else 0
    lines = [
        "# Toxic Regime Overlay Probe",
        "",
        "This is a non-HFT risk-filter probe. It tests whether high 1-minute toxicity buckets precede worse future adverse excursion/downside risk, and whether excluding high-toxicity windows improves a slow Sortino proxy.",
        "",
        f"Verdict: `{verdict}`",
        f"Rows: {len(state)}",
        f"Symbols: {state['symbol'].nunique() if not state.empty else 0}",
        f"Distinct dates: {n_days}",
        f"Horizons minutes: `{list(horizons_minutes)}`",
        f"Toxicity cutoffs: `{list(toxicity_cutoffs)}`",
        "",
        "## Interpretation discipline",
        "",
        "The first live full-fidelity archive is still young, so early runs should be treated as diagnostics unless the sample/day gates pass.",
        f"Minimum serious-validation gates used here: `{min_days}` distinct days and `{min_removed}` removed high-toxicity observations.",
        "",
        "## Current handoff",
        "",
        "Active goal: build a highly profitable non-HFT Pacifica paper-trading system. Sortino > 2 is a quality bar, but success also requires positive net PnL after fees/slippage/funding, bounded drawdown, enough trades/days, and no single symbol/day dominating results.",
        "",
        "This is a diagnostic no-trade/risk-overlay probe over the collected live public universe. Do not claim an edge from this diagnostic run, do not tune cutoffs on it, and do not assume all collected symbols should be traded.",
        "",
        "Paper trading requires explicit liquidity, spread/cost, sample-size, stability, concentration, and post-cost eligibility gates before any symbol is tradable.",
        "",
        "## Overlay scorecard",
        "",
        dataframe_to_markdown_table(scorecard.head(30)),
        "",
        "## Toxicity bucket summary",
        "",
        dataframe_to_markdown_table(bucket_summary.head(30)),
        "",
        "## Highest mean-toxicity symbols",
        "",
        dataframe_to_markdown_table(symbol_summary.head(20)),
        "",
        "## Hour summary",
        "",
        dataframe_to_markdown_table(hour_summary.head(24)),
        "",
        "## Next step",
        "",
        "Keep this probe fixed while the collector accrues fresh days. Do not tune cutoffs based on the diagnostic run. Once 30+ full days exist, rerun for provisional validation; once 60+ days exist, use chronological/purged validation.",
    ]
    return "\n".join(lines)


def write_toxic_overlay_report(
    state_path: Path,
    out_dir: Path,
    *,
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS,
    toxicity_cutoffs: tuple[float, ...] = DEFAULT_CUTOFFS,
    bucket_minutes: int = 1,
    min_days: int = 30,
    min_removed: int = 100,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = pd.read_parquet(state_path)
    state = add_forward_path_metrics(
        state, horizons_minutes=horizons_minutes, bucket_minutes=bucket_minutes
    )
    state = assign_toxicity_buckets(state, n_buckets=10)
    bucket_summary = summarize_toxicity_buckets(
        state, horizons_minutes=horizons_minutes
    )
    scorecard = evaluate_overlay(
        state,
        horizons_minutes=horizons_minutes,
        toxicity_cutoffs=toxicity_cutoffs,
        min_days=min_days,
        min_removed=min_removed,
    )
    symbol_summary = _symbol_summary(state)
    hour_summary = _hour_summary(state)
    verdict = verdict_from_scorecard(
        scorecard, min_days=min_days, min_removed=min_removed
    )

    state_with_forward = out_dir / "state_with_forward_metrics.parquet"
    state.to_parquet(state_with_forward, index=False)
    bucket_summary.to_csv(out_dir / "toxic_bucket_summary.csv", index=False)
    scorecard.to_csv(out_dir / "overlay_scorecard.csv", index=False)
    symbol_summary.to_csv(out_dir / "symbol_summary.csv", index=False)
    hour_summary.to_csv(out_dir / "hour_summary.csv", index=False)
    (out_dir / "README.md").write_text(
        _markdown_report(
            state=state,
            scorecard=scorecard,
            bucket_summary=bucket_summary,
            symbol_summary=symbol_summary,
            hour_summary=hour_summary,
            verdict=verdict,
            horizons_minutes=horizons_minutes,
            toxicity_cutoffs=toxicity_cutoffs,
            min_days=min_days,
            min_removed=min_removed,
        ),
        encoding="utf-8",
    )
    return {
        "verdict": verdict,
        "state_with_forward_metrics": str(state_with_forward),
        "scorecard": str(out_dir / "overlay_scorecard.csv"),
        "report": str(out_dir / "README.md"),
    }


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    parser.add_argument("--bucket-minutes", type=int, default=1)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--min-removed", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = write_toxic_overlay_report(
        args.state_path,
        args.out_dir,
        horizons_minutes=_parse_int_tuple(args.horizons),
        toxicity_cutoffs=_parse_float_tuple(args.cutoffs),
        bucket_minutes=args.bucket_minutes,
        min_days=args.min_days,
        min_removed=args.min_removed,
    )
    print(f"verdict: {result['verdict']}")
    print(f"wrote report: {result['report']}")


if __name__ == "__main__":
    main()

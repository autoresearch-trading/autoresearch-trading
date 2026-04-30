# scripts/post_cascade_regime_probe.py
"""Post-cascade regime discovery probe.

This script intentionally studies what happens *after* observed liquidation
bursts.  It does not train an encoder, predict pre-cascade direction, or use
maker fills as alpha.  Current-data output is discovery-only because the April
holdout has already been consumed by prior Goal-A work.

Usage:
    uv run python scripts/post_cascade_regime_probe.py \
        --cache-dir data/cache \
        --out-dir docs/experiments/post-cascade-regime-sprint
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from tape.constants import FEATURE_NAMES, SYMBOLS

LIQ_CAUSES: tuple[str, ...] = ("market_liquidation", "backstop_liquidation")
LOG_RETURN_IDX: int = FEATURE_NAMES.index("log_return")
DEFAULT_MAX_GAP_MS: int = 60_000
DEFAULT_TAKER_FEE_BPS_PER_SIDE: float = 6.0
DEFAULT_SLIPPAGE_BPS_PER_SIDE: float = 1.0


@dataclass(frozen=True)
class ProbeConfig:
    cache_dir: Path
    out_dir: Path
    max_gap_ms: int = DEFAULT_MAX_GAP_MS
    taker_fee_bps_per_side: float = DEFAULT_TAKER_FEE_BPS_PER_SIDE
    slippage_bps_per_side: float = DEFAULT_SLIPPAGE_BPS_PER_SIDE
    include_consumed_holdout: bool = True

    @property
    def round_trip_cost_bps(self) -> float:
        return 2.0 * self.taker_fee_bps_per_side + 2.0 * self.slippage_bps_per_side


def fixed_parameter_grid() -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    """Tiny pre-registered discovery grid.

    Keep this intentionally small.  Expanding it turns the sprint into a broad
    consumed-holdout search.
    """
    return {
        "delays": (0, 10, 50),
        "horizons": (50, 100, 500),
        "min_abs_move_bps": (10.0,),
    }


def _empty_bursts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start_ts",
            "end_ts",
            "start_price",
            "end_price",
            "n_trades",
            "qty_sum",
            "notional_sum",
            "return_bps",
            "abs_return_bps",
            "sign",
        ]
    )


def group_liquidation_bursts(
    trades: pd.DataFrame,
    *,
    max_gap_ms: int = DEFAULT_MAX_GAP_MS,
    min_abs_move_bps: float = 10.0,
) -> pd.DataFrame:
    """Group liquidation trades into bursts by timestamp gap.

    A burst is retained only when it contains at least two liquidation trades and
    the signed start-to-end price move clears `min_abs_move_bps`.
    """
    required = {"ts_ms", "price", "qty", "cause"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    liq = trades[trades["cause"].isin(LIQ_CAUSES)].copy()
    if liq.empty:
        return _empty_bursts()
    liq = liq.sort_values("ts_ms").reset_index(drop=True)

    rows: list[dict[str, float | int]] = []
    start = 0
    ts = liq["ts_ms"].to_numpy(dtype=np.int64)
    for i in range(1, len(liq) + 1):
        boundary = i == len(liq) or int(ts[i] - ts[i - 1]) > max_gap_ms
        if not boundary:
            continue
        burst = liq.iloc[start:i]
        start = i
        if len(burst) < 2:
            continue
        start_price = float(burst["price"].iloc[0])
        end_price = float(burst["price"].iloc[-1])
        if start_price <= 0 or end_price <= 0:
            continue
        ret_bps = math.log(end_price / start_price) * 1e4
        abs_ret_bps = abs(ret_bps)
        if abs_ret_bps < min_abs_move_bps:
            continue
        qty = burst["qty"].astype(float).to_numpy()
        px = burst["price"].astype(float).to_numpy()
        rows.append(
            {
                "start_ts": int(burst["ts_ms"].iloc[0]),
                "end_ts": int(burst["ts_ms"].iloc[-1]),
                "start_price": start_price,
                "end_price": end_price,
                "n_trades": int(len(burst)),
                "qty_sum": float(qty.sum()),
                "notional_sum": float((qty * px).sum()),
                "return_bps": float(ret_bps),
                "abs_return_bps": float(abs_ret_bps),
                "sign": int(1 if ret_bps > 0 else -1),
            }
        )

    if not rows:
        return _empty_bursts()
    return pd.DataFrame(rows)


def forward_log_return_after_delay(
    log_returns: np.ndarray,
    *,
    anchor_idx: int,
    delay_events: int,
    horizon: int,
) -> float:
    """Forward log return after entry delay.

    Entry index is `anchor_idx + delay_events`.  The return sums events
    `(entry_idx, entry_idx + horizon]`, matching the project convention that
    the anchor event itself is already known at decision time.
    """
    entry_idx = int(anchor_idx) + int(delay_events)
    end_idx = entry_idx + int(horizon)
    if entry_idx < 0 or end_idx >= len(log_returns):
        return float("nan")
    return float(
        np.asarray(log_returns, dtype=float)[entry_idx + 1 : end_idx + 1].sum()
    )


def compute_regime_metrics(
    *,
    burst_return_bps: float,
    forward_log_return: float,
    round_trip_cost_bps: float,
) -> dict[str, float | bool]:
    """Compute continuation/reversion bps after an observed burst."""
    if not np.isfinite(forward_log_return) or not np.isfinite(burst_return_bps):
        return {
            "gross_reversion_bps": float("nan"),
            "net_reversion_bps": float("nan"),
            "gross_continuation_bps": float("nan"),
            "net_reversion_positive": False,
        }
    if burst_return_bps == 0:
        sign = 0.0
    else:
        sign = 1.0 if burst_return_bps > 0 else -1.0
    fwd_bps = float(forward_log_return) * 1e4
    gross_continuation = sign * fwd_bps
    gross_reversion = -sign * fwd_bps
    net_reversion = gross_reversion - float(round_trip_cost_bps)
    return {
        "gross_reversion_bps": float(gross_reversion),
        "net_reversion_bps": float(net_reversion),
        "gross_continuation_bps": float(gross_continuation),
        "net_reversion_positive": bool(net_reversion > 0.0),
    }


def summarize_regime_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-cascade regime rows by horizon and delay."""
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "delay_events",
                "n_events",
                "n_days",
                "n_symbols",
                "median_net_bps",
                "mean_net_bps",
                "frac_positive",
                "median_gross_reversion_bps",
            ]
        )

    grouped = rows.groupby(["horizon", "delay_events"], dropna=False)
    out = grouped.agg(
        n_events=("net_reversion_bps", "size"),
        n_days=("date", "nunique"),
        n_symbols=("symbol", "nunique"),
        median_net_bps=("net_reversion_bps", "median"),
        mean_net_bps=("net_reversion_bps", "mean"),
        frac_positive=("net_reversion_bps", lambda x: float((x > 0).mean())),
        median_gross_reversion_bps=("gross_reversion_bps", "median"),
    ).reset_index()
    return out.sort_values(["horizon", "delay_events"]).reset_index(drop=True)


def _load_shard(
    cache_dir: Path, symbol: str, date_str: str
) -> tuple[np.ndarray, np.ndarray] | None:
    path = cache_dir / f"{symbol}__{date_str}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return z["features"].astype(np.float64), z["event_ts"].astype(np.int64)


def _load_trade_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    path = Path(f"data/trades/symbol={symbol}/date={date_str}")
    if not path.exists():
        return None
    q = f"SELECT ts_ms, price, qty, cause FROM read_parquet('{path}/*.parquet') ORDER BY ts_ms"
    try:
        return duckdb.query(q).to_df()
    except Exception:
        return None


def _iter_symbol_dates() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    root = Path("data/trades")
    for symbol_dir in sorted(root.glob("symbol=*")):
        symbol = symbol_dir.name.removeprefix("symbol=")
        if symbol not in SYMBOLS:
            continue
        for date_dir in sorted(symbol_dir.glob("date=2026-04-*")):
            date_str = date_dir.name.removeprefix("date=")
            pairs.append((symbol, date_str))
    return pairs


def build_post_cascade_rows(config: ProbeConfig) -> pd.DataFrame:
    """Build event-study rows for observed liquidation bursts."""
    grid = fixed_parameter_grid()
    rows: list[dict[str, object]] = []
    round_trip_cost = config.round_trip_cost_bps

    for symbol, date_str in _iter_symbol_dates():
        if not config.include_consumed_holdout and date_str >= "2026-04-14":
            continue
        shard = _load_shard(config.cache_dir, symbol, date_str)
        if shard is None:
            continue
        features, event_ts = shard
        log_returns = features[:, LOG_RETURN_IDX]
        trades = _load_trade_day(symbol, date_str)
        if trades is None or trades.empty:
            continue
        bursts = group_liquidation_bursts(
            trades,
            max_gap_ms=config.max_gap_ms,
            min_abs_move_bps=float(grid["min_abs_move_bps"][0]),
        )
        if bursts.empty:
            continue

        for _, burst in bursts.iterrows():
            # Use last cached event at or before burst end as decision anchor.
            anchor_idx = int(
                np.searchsorted(event_ts, int(burst["end_ts"]), side="right") - 1
            )
            if anchor_idx < 0 or anchor_idx >= len(event_ts):
                continue
            for delay in grid["delays"]:
                for horizon in grid["horizons"]:
                    fwd = forward_log_return_after_delay(
                        log_returns,
                        anchor_idx=anchor_idx,
                        delay_events=int(delay),
                        horizon=int(horizon),
                    )
                    if not np.isfinite(fwd):
                        continue
                    metrics = compute_regime_metrics(
                        burst_return_bps=float(burst["return_bps"]),
                        forward_log_return=fwd,
                        round_trip_cost_bps=round_trip_cost,
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": date_str,
                            "burst_start_ts": int(burst["start_ts"]),
                            "burst_end_ts": int(burst["end_ts"]),
                            "anchor_idx": anchor_idx,
                            "delay_events": int(delay),
                            "horizon": int(horizon),
                            "n_trades": int(burst["n_trades"]),
                            "notional_sum": float(burst["notional_sum"]),
                            "burst_return_bps": float(burst["return_bps"]),
                            "burst_abs_return_bps": float(burst["abs_return_bps"]),
                            "burst_sign": int(burst["sign"]),
                            "forward_log_return": float(fwd),
                            "round_trip_cost_bps": round_trip_cost,
                            **metrics,
                        }
                    )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional tabulate dependency."""
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]

    def fmt(value: object) -> str:
        if isinstance(value, float) or isinstance(value, np.floating):
            if math.isnan(float(value)):
                return "nan"
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.4f}"
        if isinstance(value, int) or isinstance(value, np.integer):
            return str(int(value))
        return str(value)

    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_report(
    out_dir: Path, rows: pd.DataFrame, summary: pd.DataFrame, config: ProbeConfig
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "README.md"
    if rows.empty:
        body = "# Post-Cascade Regime Sprint\n\nNo eligible liquidation bursts were found.\n"
        report_path.write_text(body)
        return report_path

    best = summary.sort_values(
        ["median_net_bps", "frac_positive"], ascending=[False, False]
    ).iloc[0]
    pass_gate = (
        float(best["median_net_bps"]) >= 3.0
        and float(best["frac_positive"]) >= 0.57
        and int(best["n_events"]) >= 400
        and int(best["n_symbols"]) >= 5
    )
    verdict = "PASS_DISCOVERY" if pass_gate else "KILL_OR_UNDERPOWERED"

    lines: list[str] = []
    lines.append("# Post-Cascade Regime Sprint")
    lines.append("")
    lines.append("**Status:** discovery-only; prior April holdout is already consumed.")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Observed liquidation trades are grouped into bursts by timestamp gap. The burst sign is observed from the start-to-end burst price move. The probe then tests contrarian post-burst reversion after fixed delays and horizons, net of taker round-trip fees and a conservative slippage placeholder."
    )
    lines.append("")
    lines.append("No encoder, RL, maker fills, or pre-cascade direction model is used.")
    lines.append("")
    lines.append("## Cost assumptions")
    lines.append("")
    lines.append(f"- taker fee per side: {config.taker_fee_bps_per_side:.2f} bp")
    lines.append(
        f"- slippage per side placeholder: {config.slippage_bps_per_side:.2f} bp"
    )
    lines.append(f"- round-trip cost: {config.round_trip_cost_bps:.2f} bp")
    lines.append("")
    lines.append("## Universe summary")
    lines.append("")
    lines.append(f"- rows: {len(rows):,}")
    lines.append(
        f"- unique bursts: {rows[['symbol','date','burst_start_ts','burst_end_ts']].drop_duplicates().shape[0]:,}"
    )
    lines.append(f"- symbols: {rows['symbol'].nunique()}")
    lines.append(f"- days: {rows['date'].nunique()}")
    lines.append("")
    lines.append("## Horizon/delay summary")
    lines.append("")
    lines.append(dataframe_to_markdown_table(summary))
    lines.append("")
    lines.append("## Best cell")
    lines.append("")
    lines.append(dataframe_to_markdown_table(best.to_frame().T))
    lines.append("")
    lines.append("## Discovery gate")
    lines.append("")
    lines.append(
        "Pass requires median net >= +3 bp, frac-positive >= 0.57, n_events >= 400, and n_symbols >= 5. If this gate fails, do not continue Pacifica-only modeling without a new hypothesis or fresh data."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if pass_gate:
        lines.append(
            "A post-cascade reversion cell clears the mechanical discovery gate. Next step is to freeze this exact rule and validate on fresh untouched post-April data before any modeling."
        )
    else:
        lines.append(
            "No post-cascade reversion cell clears the mechanical discovery gate. Treat this as a kill or underpowered result; do not optimize thresholds on the consumed data."
        )
    lines.append("")
    report_path.write_text("\n".join(lines))
    return report_path


def run_probe(config: ProbeConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    rows = build_post_cascade_rows(config)
    summary = summarize_regime_table(rows)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = config.out_dir / "post_cascade_regime_rows.csv"
    summary_path = config.out_dir / "post_cascade_regime_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = _write_report(config.out_dir, rows, summary, config)
    return rows, summary, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/post-cascade-regime-sprint"),
    )
    p.add_argument("--max-gap-ms", type=int, default=DEFAULT_MAX_GAP_MS)
    p.add_argument(
        "--taker-fee-bps-per-side", type=float, default=DEFAULT_TAKER_FEE_BPS_PER_SIDE
    )
    p.add_argument(
        "--slippage-bps-per-side", type=float, default=DEFAULT_SLIPPAGE_BPS_PER_SIDE
    )
    p.add_argument("--exclude-consumed-holdout", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ProbeConfig(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        max_gap_ms=args.max_gap_ms,
        taker_fee_bps_per_side=args.taker_fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        include_consumed_holdout=not args.exclude_consumed_holdout,
    )
    rows, summary, report = run_probe(config)
    print(f"wrote {len(rows):,} rows")
    print(f"wrote summary rows: {len(summary):,}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()

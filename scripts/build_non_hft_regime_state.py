# scripts/build_non_hft_regime_state.py
"""Build slow non-HFT regime-state features from Pacifica silver tables.

The output is a risk/state layer for 30s/1m/5m decision cadences.  It is not an
HFT trigger and intentionally aggregates away sub-second event timing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SILVER_DIR = Path("data/pacifica_silver_partitioned")
DEFAULT_OUT_DIR = Path("docs/experiments/non-hft-regime-state")


def bucket_ms(bucket: str) -> int:
    bucket = bucket.strip().lower()
    if bucket.endswith("ms"):
        return int(bucket[:-2])
    if bucket.endswith("s"):
        return int(float(bucket[:-1]) * 1_000)
    if bucket.endswith("min"):
        return int(float(bucket[:-3]) * 60_000)
    if bucket.endswith("m"):
        return int(float(bucket[:-1]) * 60_000)
    raise ValueError(f"Unsupported bucket: {bucket}")


def _partition_value(path: Path, name: str) -> str | None:
    prefix = f"{name}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def _event_date_from_ms(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def affected_symbol_dates_from_source_plan(
    source_plan: pd.DataFrame | None,
) -> set[tuple[str, str]] | None:
    if source_plan is None:
        return None
    if source_plan.empty:
        return set()
    if not {"symbol", "date"}.issubset(source_plan.columns):
        return set()
    plan = source_plan.copy()
    if "status" in plan.columns:
        plan = plan[plan["status"].astype(str).eq("sealed")]
    if "change_status" in plan.columns:
        plan = plan[plan["change_status"].astype(str).isin(["new", "changed"])]
    return {
        (str(row["symbol"]), str(row["date"]))
        for _, row in plan.iterrows()
        if pd.notna(row.get("symbol")) and pd.notna(row.get("date"))
    }


def _filter_frame_by_symbol_dates(
    frame: pd.DataFrame, symbol_dates: set[tuple[str, str]] | None
) -> pd.DataFrame:
    if symbol_dates is None or frame.empty or "symbol" not in frame.columns:
        return frame
    if not symbol_dates:
        return frame.iloc[0:0].copy()
    dates = (
        frame["event_ts_ms"].map(_event_date_from_ms)
        if "event_ts_ms" in frame
        else None
    )
    if dates is None:
        return frame.iloc[0:0].copy()
    mask = [
        (str(symbol), str(date)) in symbol_dates
        for symbol, date in zip(frame["symbol"], dates, strict=False)
    ]
    return frame.loc[mask].copy()


def _partitioned_silver_parts(
    silver_dir: Path,
    channel: str,
    symbol_dates: set[tuple[str, str]] | None,
) -> list[Path]:
    channel_dir = silver_dir / f"channel={channel}"
    if not channel_dir.exists():
        return []
    parts = sorted(
        set(channel_dir.glob("symbol=*/date=*/*.parquet"))
        | set(channel_dir.glob("symbol=*/date=*/**/*.parquet"))
    )
    if symbol_dates is None:
        return parts
    filtered: list[Path] = []
    for part in parts:
        symbol = _partition_value(part, "symbol")
        date = _partition_value(part, "date")
        if symbol is not None and date is not None and (symbol, date) in symbol_dates:
            filtered.append(part)
    return filtered


def read_silver_table(
    silver_dir: Path,
    channel: str,
    *,
    symbol_dates: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Read v1 flat silver parquet and/or partitioned source-object layouts."""
    frames: list[pd.DataFrame] = []
    flat_path = silver_dir / f"{channel}.parquet"
    if flat_path.exists():
        frames.append(
            _filter_frame_by_symbol_dates(pd.read_parquet(flat_path), symbol_dates)
        )
    parts = _partitioned_silver_parts(silver_dir, channel, symbol_dates)
    frames.extend(pd.read_parquet(part) for part in parts)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _add_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if df.empty:
        return df
    ms = bucket_ms(bucket)
    out = df.copy()
    out = out[out["event_ts_ms"].notna()].copy()
    out["bucket_start_ms"] = (out["event_ts_ms"].astype("int64") // ms) * ms
    return out


def _bbo_features(
    silver_dir: Path, bucket: str, symbol_dates: set[tuple[str, str]] | None = None
) -> pd.DataFrame:
    bbo = _add_bucket(
        read_silver_table(silver_dir, "bbo", symbol_dates=symbol_dates), bucket
    )
    if bbo.empty:
        return pd.DataFrame(columns=["symbol", "bucket_start_ms"])
    grouped = bbo.groupby(["symbol", "bucket_start_ms"], as_index=False).agg(
        bbo_updates=("event_ts_ms", "size"),
        avg_spread_bps=("spread_bps", "mean"),
        max_spread_bps=("spread_bps", "max"),
        avg_top_bid_notional=("top_bid_notional", "mean"),
        avg_top_ask_notional=("top_ask_notional", "mean"),
        first_mid=("mid", "first"),
        last_mid=("mid", "last"),
        first_order_id=("last_order_id", "first"),
        last_order_id=("last_order_id", "last"),
    )
    grouped["mid_return_bps"] = (
        grouped["last_mid"] / grouped["first_mid"] - 1.0
    ) * 10_000
    minutes = bucket_ms(bucket) / 60_000
    grouped["bbo_churn_per_min"] = grouped["bbo_updates"] / minutes
    grouped["order_id_delta"] = grouped["last_order_id"] - grouped["first_order_id"]
    grouped["top_depth_notional"] = (
        grouped["avg_top_bid_notional"] + grouped["avg_top_ask_notional"]
    )
    return grouped


def _trade_features(
    silver_dir: Path, bucket: str, symbol_dates: set[tuple[str, str]] | None = None
) -> pd.DataFrame:
    trades = _add_bucket(
        read_silver_table(silver_dir, "trades", symbol_dates=symbol_dates), bucket
    )
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "bucket_start_ms"])
    trade_class = (
        trades.get("trade_class", pd.Series(index=trades.index, dtype=object))
        .fillna("")
        .str.lower()
    )
    cause = (
        trades.get("cause", pd.Series(index=trades.index, dtype=object))
        .fillna("")
        .str.lower()
    )
    trades["is_liquidation"] = (
        trade_class.eq("liquidation")
        | trade_class.str.endswith("_liquidation")
        | cause.isin(["market_liquidation", "backstop_liquidation"])
    )
    trades["liq_notional"] = trades["notional"].where(trades["is_liquidation"], 0.0)
    grouped = trades.groupby(["symbol", "bucket_start_ms"], as_index=False).agg(
        trade_count=("event_ts_ms", "size"),
        trade_qty=("qty", "sum"),
        signed_trade_qty=("signed_qty", "sum"),
        trade_notional=("notional", "sum"),
        liquidation_count=("is_liquidation", "sum"),
        liquidation_notional=("liq_notional", "sum"),
    )
    grouped["abs_trade_imbalance"] = (
        grouped["signed_trade_qty"].abs() / grouped["trade_qty"].replace(0, pd.NA)
    ).fillna(0.0)
    return grouped


def _price_features(
    silver_dir: Path, bucket: str, symbol_dates: set[tuple[str, str]] | None = None
) -> pd.DataFrame:
    prices = _add_bucket(
        read_silver_table(silver_dir, "prices", symbol_dates=symbol_dates), bucket
    )
    if prices.empty:
        return pd.DataFrame(columns=["symbol", "bucket_start_ms"])
    grouped = prices.groupby(["symbol", "bucket_start_ms"], as_index=False).agg(
        mark_oracle_basis_bps=("mark_oracle_basis_bps", "last"),
        mid_oracle_basis_bps=("mid_oracle_basis_bps", "last"),
        funding=("funding", "last"),
        next_funding=("next_funding", "last"),
        open_interest=("open_interest", "last"),
        price_updates=("event_ts_ms", "size"),
    )
    grouped["mark_oracle_basis_abs_bps"] = grouped["mark_oracle_basis_bps"].abs()
    return grouped


def _book_features(
    silver_dir: Path, bucket: str, symbol_dates: set[tuple[str, str]] | None = None
) -> pd.DataFrame:
    book = _add_bucket(
        read_silver_table(silver_dir, "book", symbol_dates=symbol_dates), bucket
    )
    if book.empty:
        return pd.DataFrame(columns=["symbol", "bucket_start_ms"])
    return book.groupby(["symbol", "bucket_start_ms"], as_index=False).agg(
        book_updates=("event_ts_ms", "size"),
        avg_book_spread_bps=("spread_bps_l1", "mean"),
        avg_bid_depth_l5=("bid_depth_l5", "mean"),
        avg_ask_depth_l5=("ask_depth_l5", "mean"),
        avg_bid_orders_l5=("bid_orders_l5", "mean"),
        avg_ask_orders_l5=("ask_orders_l5", "mean"),
        first_book_nonce=("nonce", "first"),
        last_book_nonce=("nonce", "last"),
    )


def _merge_features(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    state = frames[0]
    for frame in frames[1:]:
        state = state.merge(frame, on=["symbol", "bucket_start_ms"], how="outer")
    return state.sort_values(["symbol", "bucket_start_ms"]).reset_index(drop=True)


def compute_toxicity_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metrics = [
        "avg_spread_bps",
        "bbo_churn_per_min",
        "abs_trade_imbalance",
        "realized_vol_bps",
        "liquidation_notional",
        "mark_oracle_basis_abs_bps",
    ]
    scores = []
    for metric in metrics:
        if metric not in out.columns:
            continue
        series = pd.to_numeric(out[metric], errors="coerce").fillna(0.0)
        if len(series) <= 1 or series.max() == series.min():
            scores.append(pd.Series(0.0, index=out.index))
        else:
            scores.append(series.rank(pct=True))
    if scores:
        out["toxicity_score"] = sum(scores) / len(scores)
    else:
        out["toxicity_score"] = 0.0
    out["toxicity_score"] = out["toxicity_score"].clip(0.0, 1.0)
    return out


def build_regime_state(
    silver_dir: Path,
    *,
    bucket: str = "1min",
    source_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    symbol_dates = affected_symbol_dates_from_source_plan(source_plan)
    state = _merge_features(
        [
            _bbo_features(silver_dir, bucket, symbol_dates),
            _trade_features(silver_dir, bucket, symbol_dates),
            _price_features(silver_dir, bucket, symbol_dates),
            _book_features(silver_dir, bucket, symbol_dates),
        ]
    )
    if state.empty:
        return state
    if "mid_return_bps" in state.columns:
        state["realized_vol_bps"] = (
            state.groupby("symbol")["mid_return_bps"]
            .transform(lambda s: s.rolling(5, min_periods=1).std())
            .fillna(0.0)
        )
    else:
        state["realized_vol_bps"] = 0.0
    for col in [
        "trade_count",
        "trade_qty",
        "signed_trade_qty",
        "trade_notional",
        "liquidation_count",
        "liquidation_notional",
        "abs_trade_imbalance",
        "bbo_updates",
        "bbo_churn_per_min",
        "mark_oracle_basis_abs_bps",
    ]:
        if col not in state.columns:
            state[col] = 0.0
    state = compute_toxicity_score(state)
    return state


def _markdown_summary(
    state: pd.DataFrame, bucket: str, quality: pd.DataFrame | None = None
) -> str:
    lines = [
        "# Non-HFT Pacifica Regime State",
        "",
        "This report is a slow regime-state layer built from full-fidelity Pacifica data.",
        "It is intentionally non-HFT: features are aggregated to decision buckets and should be used for risk/no-trade overlays before alpha tests.",
        "The upstream silver layer is now partitioned by channel/symbol/date so raw JSONL.GZ can be normalized in bounded chunks instead of one in-memory table.",
        "",
        f"Bucket: `{bucket}`",
        f"Rows: {len(state)}",
        f"Symbols: {state['symbol'].nunique() if not state.empty and 'symbol' in state else 0}",
        "",
    ]
    if quality is not None and not quality.empty:
        lines.extend(["## Silver input quality", "", quality.to_csv(index=False), ""])
    lines.extend(
        [
            "## Current handoff",
            "",
            "Active goal: build a highly profitable non-HFT Pacifica paper-trading system. Sortino > 2 is a quality bar, but success also requires positive net PnL after fees/slippage/funding, bounded drawdown, enough trades/days, and no single symbol/day dominating results.",
            "",
            "This regime-state layer is a research/risk substrate across the full live public Pacifica universe. Do not assume all collected symbols should be traded; paper trading must use explicit liquidity, spread/cost, sample-size, stability, concentration, and post-cost eligibility gates.",
            "",
            "Current local runs are diagnostic until enough distinct full-fidelity days accrue. Keep downstream toxicity thresholds fixed while new data arrives; do not tune cutoffs on the initial diagnostic sample.",
            "",
            "## Intended next probe",
            "",
            "Use this table to test whether high-toxicity buckets predict worse adverse excursion/slippage and whether avoiding them improves a non-HFT Sortino proxy.",
            "",
        ]
    )
    if not state.empty:
        top = state.sort_values("toxicity_score", ascending=False).head(10)
        lines.extend(["## Highest toxicity preview", "", top.to_csv(index=False), ""])
    return "\n".join(lines)


def write_regime_state(
    silver_dir: Path, out_dir: Path, *, bucket: str = "1min"
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_regime_state(silver_dir, bucket=bucket)
    quality_path = silver_dir / "quality_summary.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    state.to_parquet(out_dir / "regime_state.parquet", index=False)
    state.head(200).to_csv(out_dir / "regime_state_preview.csv", index=False)
    if not quality.empty:
        quality.to_csv(out_dir / "silver_quality_summary.csv", index=False)
    (out_dir / "README.md").write_text(
        _markdown_summary(state, bucket, quality), encoding="utf-8"
    )
    return state


def _incremental_regime_summary(
    state: pd.DataFrame, bucket: str, source_plan: pd.DataFrame
) -> str:
    affected = affected_symbol_dates_from_source_plan(source_plan) or set()
    lines = [
        "# Incremental Non-HFT Pacifica Regime State Delta",
        "",
        "This is a side-by-side delta built only from symbol/date partitions touched by new or changed sealed source objects.",
        "It intentionally writes `regime_state_delta.parquet` instead of overwriting canonical `regime_state.parquet`.",
        "",
        f"Bucket: `{bucket}`",
        f"Delta rows: {len(state)}",
        f"Affected symbol/date partitions: {len(affected)}",
        "",
        "Canonical promotion remains blocked until side-by-side verification checks row counts, symbol/date/channel coverage, duplicate/null keys, and report diffs.",
        "",
    ]
    if not source_plan.empty:
        lines.extend(["## Source-object plan", "", source_plan.to_csv(index=False), ""])
    return "\n".join(lines)


def write_incremental_regime_state(
    silver_dir: Path,
    out_dir: Path,
    *,
    source_plan: pd.DataFrame,
    bucket: str = "1min",
) -> pd.DataFrame:
    """Write a regime-state delta for changed source-object symbol/date partitions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_regime_state(silver_dir, bucket=bucket, source_plan=source_plan)
    state.to_parquet(out_dir / "regime_state_delta.parquet", index=False)
    state.head(200).to_csv(out_dir / "regime_state_delta_preview.csv", index=False)
    source_plan.to_csv(out_dir / "incremental_source_plan.csv", index=False)
    (out_dir / "incremental_regime_report.md").write_text(
        _incremental_regime_summary(state, bucket, source_plan), encoding="utf-8"
    )
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--bucket", default="1min", help="Slow decision bucket, e.g. 30s, 1min, 5min"
    )
    parser.add_argument(
        "--source-plan",
        type=Path,
        help="Optional incremental_plan.csv from silver refresh. Writes delta artifacts instead of canonical regime_state.parquet.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.source_plan:
        source_plan = pd.read_csv(args.source_plan, dtype={"hour": "string"})
        state = write_incremental_regime_state(
            args.silver_dir, args.out_dir, source_plan=source_plan, bucket=args.bucket
        )
        print(
            f"wrote {len(state)} incremental regime-state delta rows to {args.out_dir}"
        )
    else:
        state = write_regime_state(args.silver_dir, args.out_dir, bucket=args.bucket)
        print(f"wrote {len(state)} regime-state rows to {args.out_dir}")


if __name__ == "__main__":
    main()

"""Goal-A feasibility: per-symbol slippage + edge headroom table.

For each (symbol, notional_size in {$1k, $10k, $100k}, horizon in {H10, H50,
H100, H500}):

  * Walk the OB at each window's last event for that target notional, both
    legs (buy entry → sell exit).
  * Compute realized log-return between the window's last event and h events
    later (from the cached per-event log_return feature, summed).
  * gross_edge_bps = |realized_return| * 1e4   (one-sided absolute, what a
    perfect direction signal could capture)
  * cost_bps      = 2 * fees_bps + 2 * |slip_bps|   (round-trip)
  * headroom_bps  = gross_edge_bps - cost_bps

Reads:
  * Cached shards under data/cache/*.npz   (event-aligned 17-feat tensors)
  * Raw OB parquet under data/orderbook/...   (10-level snapshots, ~24s)

Hard constraints
  * Skips any shard with date >= 2026-04-14 (April hold-out, gotcha #17).
  * Uses tape.io_parquet.load_ob_day for raw OB load — no re-implementation.

Methodological notes
  * Pacifica taker fee assumed at 6 bp per side (per task spec). The v1 spec
    docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md does
    not document maker rebates — flat 6bp/side used both legs.
  * The OB cache cadence is ~24 s; we use the latest snapshot at-or-before the
    window's last event timestamp. Slippage at $100k on illiquid alts is a
    snapshot-of-the-book estimate, not an order-flow-aware market-impact model.
  * Sampling: stride=200 (eval stride) windows per shard, capped at 200 windows
    per shard for runtime. With 4178 shards × 200 windows × 3 sizes × 4
    horizons that is ~10M cell evaluations — vectorized.
  * edge_bps is unsigned (|return|): we are asking "could a perfect direction
    signal have made money", not measuring directional skill.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tape.constants import (
    APRIL_HELDOUT_START,
    DIRECTION_HORIZONS,
    FEATURE_NAMES,
    STRIDE_EVAL,
    SYMBOLS,
    WINDOW_LEN,
)
from tape.io_parquet import load_ob_day

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT = Path("data/cache")
OUT_DIR_DEFAULT = Path("docs/experiments/goal-a-feasibility")
NOTIONAL_SIZES_USD: tuple[float, ...] = (1_000.0, 10_000.0, 100_000.0)
TAKER_FEE_BPS_PER_SIDE: float = 6.0  # Pacifica taker, per task spec
N_LEVELS: int = 10
WINDOWS_PER_SHARD_CAP: int = 200  # subsample cap to keep runtime bounded

# Index of log_return in the cached features tensor (col 0 per FEATURE_NAMES)
_LOG_RETURN_IDX = FEATURE_NAMES.index("log_return")


# ---------------------------------------------------------------------------
# Book-walk primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillResult:
    fillable: bool
    fill_price: float
    filled_qty: float
    slippage_bps: float  # signed: + for buy above mid, - for sell below mid


def simulate_taker_fill(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
    *,
    target_notional: float,
    side: str,
) -> FillResult:
    """Walk the book on `side` ('buy' → asks, 'sell' → bids) until cumulative
    notional >= target_notional. Levels with qty == 0 are skipped (missing
    raw level → zero liquidity). If the full book cannot fill the target,
    fillable=False, but we still report the realized fill_price/slippage from
    what was available (so the caller can spot near-misses).
    """
    if side == "buy":
        prices, qtys = ask_prices, ask_qtys
    elif side == "sell":
        prices, qtys = bid_prices, bid_qtys
    else:  # pragma: no cover
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    # mid: best bid + best ask / 2 (use first non-empty level on each side)
    bid_valid = (bid_prices > 0) & (bid_qtys > 0)
    ask_valid = (ask_prices > 0) & (ask_qtys > 0)
    if not bid_valid.any() or not ask_valid.any():
        return FillResult(False, np.nan, 0.0, np.nan)
    best_bid = bid_prices[bid_valid][0]
    best_ask = ask_prices[ask_valid][0]
    mid = 0.5 * (best_bid + best_ask)

    remaining = float(target_notional)
    spent_notional = 0.0
    filled_qty = 0.0

    for p, q in zip(prices, qtys, strict=True):
        if q <= 0 or p <= 0:
            continue  # missing level
        level_notional = float(p) * float(q)
        if level_notional >= remaining:
            # Partial fill at this level, exits the loop
            take_qty = remaining / float(p)
            filled_qty += take_qty
            spent_notional += remaining
            remaining = 0.0
            break
        # Consume the whole level
        filled_qty += float(q)
        spent_notional += level_notional
        remaining -= level_notional

    fillable = remaining <= 1e-9
    if filled_qty <= 0:
        return FillResult(False, np.nan, 0.0, np.nan)

    fill_price = spent_notional / filled_qty
    # Signed slippage: buy above mid is +, sell below mid is -
    slip_bps = (fill_price - mid) / mid * 1e4
    return FillResult(fillable, fill_price, filled_qty, float(slip_bps))


# ---------------------------------------------------------------------------
# Forward log-return from cached log_return column
# ---------------------------------------------------------------------------


def forward_log_return(
    log_returns: np.ndarray,
    anchor_idx: int,
    horizon: int,
) -> float:
    """Sum log_returns[anchor_idx+1 .. anchor_idx+horizon]. Returns NaN if
    horizon extends past the array.
    """
    end = anchor_idx + horizon
    if end >= len(log_returns):
        return float("nan")
    return float(log_returns[anchor_idx + 1 : end + 1].sum())


# ---------------------------------------------------------------------------
# Headroom math
# ---------------------------------------------------------------------------


def headroom_bps(*, edge_bps: float, slip_bps: float, fees_bps: float) -> float:
    """Round-trip headroom: |edge| - (2 * fees + 2 * |slip|)."""
    return abs(edge_bps) - (2.0 * fees_bps + 2.0 * abs(slip_bps))


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------


def _expand_snap_levels(
    snap_row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull the 10 bid/ask prices+qtys out of a flat snap row (post-expand)."""
    bp = np.array(
        [snap_row[f"bid{i}_price"] for i in range(1, N_LEVELS + 1)], dtype=float
    )
    bq = np.array(
        [snap_row[f"bid{i}_qty"] for i in range(1, N_LEVELS + 1)], dtype=float
    )
    ap = np.array(
        [snap_row[f"ask{i}_price"] for i in range(1, N_LEVELS + 1)], dtype=float
    )
    aq = np.array(
        [snap_row[f"ask{i}_qty"] for i in range(1, N_LEVELS + 1)], dtype=float
    )
    return bp, bq, ap, aq


def _list_window_starts(n_events: int, stride: int = STRIDE_EVAL) -> np.ndarray:
    """Window starts s.t. [s, s+WINDOW_LEN) is fully in-bounds."""
    if n_events < WINDOW_LEN:
        return np.array([], dtype=np.int64)
    last_start = n_events - WINDOW_LEN
    return np.arange(0, last_start + 1, stride, dtype=np.int64)


def _subsample_starts(
    starts: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Random subsample without replacement to keep at most `cap` starts."""
    if len(starts) <= cap:
        return starts
    idx = rng.choice(len(starts), size=cap, replace=False)
    idx.sort()
    return starts[idx]


def process_shard(
    shard_path: Path,
    *,
    rng: np.random.Generator,
    horizons: Iterable[int] = DIRECTION_HORIZONS,
    sizes: Iterable[float] = NOTIONAL_SIZES_USD,
    cap: int = WINDOWS_PER_SHARD_CAP,
) -> list[dict]:
    """Process a single (symbol, date) shard → list of per-window per-cell rows.

    Returns one row per (window, size, horizon).  Empty list if the shard is
    in the April hold-out, has fewer than WINDOW_LEN events, or its raw OB
    parquet is missing.
    """
    sym, date = shard_path.stem.split("__")
    if date >= APRIL_HELDOUT_START:
        return []

    with np.load(shard_path, allow_pickle=False) as z:
        features = z["features"]
        event_ts = z["event_ts"]
    n_events = features.shape[0]
    starts = _list_window_starts(n_events, stride=STRIDE_EVAL)
    if len(starts) == 0:
        return []
    starts = _subsample_starts(starts, cap, rng)

    # Anchor: last event of each window → starts + WINDOW_LEN - 1
    anchors = starts + WINDOW_LEN - 1
    anchor_ts = event_ts[anchors]

    # Per-event log_return column (already in cache — col 0)
    log_returns = features[:, _LOG_RETURN_IDX].astype(np.float64)

    # Forward returns per horizon, vectorised cumulative trick:
    #   cum[i] = sum log_returns[0..i-1] (so cum[i+h]-cum[i] = sum at i+1..i+h
    #   when we shift). We want sum from anchor+1..anchor+h inclusive.
    cum = np.concatenate([[0.0], np.cumsum(log_returns)])
    # cum[k] = sum(log_returns[:k]) for k in [0, n].

    fwd_by_h: dict[int, np.ndarray] = {}
    for h in horizons:
        end = anchors + h
        valid = end < n_events
        out = np.full(len(anchors), np.nan, dtype=np.float64)
        # cum[end+1] - cum[anchor+1] = sum log_returns[anchor+1 .. end]
        out[valid] = cum[end[valid] + 1] - cum[anchors[valid] + 1]
        fwd_by_h[h] = out

    # Load raw OB and align via searchsorted (most recent prior snapshot)
    ob = load_ob_day(sym, date)
    if ob is None or len(ob) == 0:
        return []
    snap_ts = ob["ts_ms"].to_numpy(dtype=np.int64)
    # right-1 → most recent snapshot at or before anchor_ts (causal alignment)
    snap_idx = np.searchsorted(snap_ts, anchor_ts, side="right") - 1
    valid_snap = snap_idx >= 0
    if not valid_snap.any():
        return []

    # Pre-extract level matrices for fast row access
    bid_prices_all = np.stack(
        [ob[f"bid{i}_price"].to_numpy(dtype=float) for i in range(1, N_LEVELS + 1)],
        axis=1,
    )
    bid_qtys_all = np.stack(
        [ob[f"bid{i}_qty"].to_numpy(dtype=float) for i in range(1, N_LEVELS + 1)],
        axis=1,
    )
    ask_prices_all = np.stack(
        [ob[f"ask{i}_price"].to_numpy(dtype=float) for i in range(1, N_LEVELS + 1)],
        axis=1,
    )
    ask_qtys_all = np.stack(
        [ob[f"ask{i}_qty"].to_numpy(dtype=float) for i in range(1, N_LEVELS + 1)],
        axis=1,
    )

    rows: list[dict] = []
    for w_idx in range(len(anchors)):
        if not valid_snap[w_idx]:
            continue
        s = int(snap_idx[w_idx])
        bp, bq = bid_prices_all[s], bid_qtys_all[s]
        ap, aq = ask_prices_all[s], ask_qtys_all[s]
        for size in sizes:
            buy = simulate_taker_fill(bp, bq, ap, aq, target_notional=size, side="buy")
            sell = simulate_taker_fill(
                bp, bq, ap, aq, target_notional=size, side="sell"
            )
            # Round-trip slippage cost: |buy slip| + |sell slip|
            # Both legs must be fillable for the round-trip to count
            fillable = buy.fillable and sell.fillable
            if buy.filled_qty <= 0 or sell.filled_qty <= 0:
                slip_avg = float("nan")
            else:
                slip_avg = 0.5 * (abs(buy.slippage_bps) + abs(sell.slippage_bps))
            for h in horizons:
                fr = fwd_by_h[h][w_idx]
                if not np.isfinite(fr):
                    edge_bps = float("nan")
                    head = float("nan")
                else:
                    edge_bps = abs(fr) * 1e4
                    if np.isfinite(slip_avg):
                        head = headroom_bps(
                            edge_bps=edge_bps,
                            slip_bps=slip_avg,
                            fees_bps=TAKER_FEE_BPS_PER_SIDE,
                        )
                    else:
                        head = float("nan")
                rows.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "window_start": int(starts[w_idx]),
                        "anchor_ts": int(anchor_ts[w_idx]),
                        "size_usd": float(size),
                        "horizon": int(h),
                        "fillable": bool(fillable),
                        "slip_buy_bps": float(buy.slippage_bps),
                        "slip_sell_bps": float(sell.slippage_bps),
                        "slip_avg_bps": (
                            float(slip_avg) if np.isfinite(slip_avg) else float("nan")
                        ),
                        "edge_bps": (
                            float(edge_bps) if np.isfinite(edge_bps) else float("nan")
                        ),
                        "headroom_bps": (
                            float(head) if np.isfinite(head) else float("nan")
                        ),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cells(per_window_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(symbol, size, horizon) summary."""
    rows: list[dict] = []
    for (sym, size, h), g in per_window_df.groupby(
        ["symbol", "size_usd", "horizon"], sort=True
    ):
        n_total = len(g)
        if n_total == 0:
            continue
        fillable_frac = g["fillable"].mean()
        # All slip / edge / headroom stats: condition on rows with finite values
        slips = g["slip_avg_bps"].dropna().to_numpy()
        edges = g["edge_bps"].dropna().to_numpy()
        # headroom requires both fillable AND finite edge
        head_g = g[g["fillable"] & np.isfinite(g["headroom_bps"])]
        heads = head_g["headroom_bps"].to_numpy()
        n_head = len(heads)
        rows.append(
            {
                "symbol": sym,
                "size_usd": float(size),
                "horizon": int(h),
                "n_windows": int(n_total),
                "n_fillable_with_edge": int(n_head),
                "fillable_frac": float(fillable_frac),
                "slip_median_bps": (
                    float(np.median(slips)) if len(slips) else float("nan")
                ),
                "slip_p90_bps": (
                    float(np.quantile(slips, 0.90)) if len(slips) else float("nan")
                ),
                "edge_median_bps": (
                    float(np.median(edges)) if len(edges) else float("nan")
                ),
                "edge_p75_bps": (
                    float(np.quantile(edges, 0.75)) if len(edges) else float("nan")
                ),
                "edge_p90_bps": (
                    float(np.quantile(edges, 0.90)) if len(edges) else float("nan")
                ),
                "headroom_median_bps": (
                    float(np.median(heads)) if n_head else float("nan")
                ),
                "headroom_p75_bps": (
                    float(np.quantile(heads, 0.75)) if n_head else float("nan")
                ),
                "headroom_p90_bps": (
                    float(np.quantile(heads, 0.90)) if n_head else float("nan")
                ),
                "frac_positive_headroom": (
                    float((heads > 0).mean()) if n_head else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["symbol", "size_usd", "horizon"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these symbols (default: all in cache).",
    )
    parser.add_argument(
        "--max-shards-per-symbol",
        type=int,
        default=None,
        help="If set, sample up to N shards per symbol (uniform).",
    )
    parser.add_argument("--seed", type=int, default=0xCAFE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cache_dir = args.cache_dir
    all_shards = sorted(cache_dir.glob("*.npz"))
    # Hard-skip April hold-out shards (defence-in-depth; no such files exist
    # in cache today but cheap insurance for future re-runs)
    all_shards = [p for p in all_shards if p.stem.split("__")[1] < APRIL_HELDOUT_START]

    if args.symbols:
        wanted = set(args.symbols)
        all_shards = [p for p in all_shards if p.stem.split("__")[0] in wanted]
    else:
        wanted = set(SYMBOLS)
        all_shards = [p for p in all_shards if p.stem.split("__")[0] in wanted]

    # Optional per-symbol shard cap (for fast smoke runs)
    if args.max_shards_per_symbol is not None:
        from collections import defaultdict

        by_sym: dict[str, list[Path]] = defaultdict(list)
        for p in all_shards:
            by_sym[p.stem.split("__")[0]].append(p)
        capped: list[Path] = []
        for sym, paths in by_sym.items():
            if len(paths) <= args.max_shards_per_symbol:
                capped.extend(paths)
            else:
                idx = rng.choice(
                    len(paths), size=args.max_shards_per_symbol, replace=False
                )
                idx.sort()
                capped.extend([paths[i] for i in idx])
        all_shards = sorted(capped)

    print(f"Processing {len(all_shards)} shards...", flush=True)
    all_rows: list[dict] = []
    for i, p in enumerate(all_shards):
        if i % 50 == 0:
            print(f"  [{i}/{len(all_shards)}] {p.name}", flush=True)
        rows = process_shard(p, rng=rng)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No rows produced — check cache_dir and symbols filter.")

    per_window_df = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_window_path = args.out_dir / "per_window.parquet"
    per_window_df.to_parquet(per_window_path, index=False)
    print(f"Wrote {len(per_window_df)} per-window rows → {per_window_path}", flush=True)

    summary_df = aggregate_cells(per_window_df)
    csv_path = args.out_dir / "headroom_table.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"Wrote {len(summary_df)} (sym,size,horizon) cells → {csv_path}", flush=True)


if __name__ == "__main__":
    main()

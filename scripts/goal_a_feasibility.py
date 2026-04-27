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
from typing import Iterable, cast

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

# Accuracy stress regimes — directional skill levels at which to evaluate
# expected per-round-trip PnL. 0.55 / 0.575 / 0.60 bracket the upper end of
# what realistic models on this data have produced (v1 hit ~51.4% at H500).
ACCURACY_REGIMES: tuple[float, ...] = (0.55, 0.575, 0.60)
SURVIVOR_FRAC_POS_THRESHOLD: float = 0.55
SURVIVOR_HEADROOM_THRESHOLD_BPS: float = 0.0

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


def headroom_at_accuracy_bps(
    *, edge_bps: float, slip_bps: float, fees_bps: float, accuracy: float
) -> float:
    """Expected per-round-trip PnL in bps under iid directional accuracy `p`.

    Math: when the model is right with probability p and wrong with (1-p),
    expected gross PnL per trip = p · |edge| + (1-p) · (-|edge|)
                                = (2p - 1) · |edge|.
    Subtract the round-trip cost band (2 fees + 2 |slip|).

    p=1.0 → recovers headroom_bps (perfect oracle).
    p=0.5 → -(cost) (pure cost; no signal).
    p=0.55, 0.575, 0.60 → 0.10 / 0.15 / 0.20 × edge − cost.

    Caveat (documented in README): this is a first-order calc that treats
    realized |edge| as the magnitude and accuracy `p` as iid. It does NOT
    model the joint distribution of (signal-correctness, edge-size). A more
    honest sim would condition on a model output ↔ realized-return joint,
    which we do not have. Slippage scaling-with-thinness is also not modelled.
    """
    edge_signal = (2.0 * accuracy - 1.0) * abs(edge_bps)
    cost = 2.0 * fees_bps + 2.0 * abs(slip_bps)
    return edge_signal - cost


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------


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


def _accuracy_suffix(accuracy: float) -> str:
    """'_55', '_575', '_60' for 0.55, 0.575, 0.60 — used in column names.

    Encodes accuracy as percent with no decimal point: 0.55→55, 0.575→575,
    0.60→60. Stable for the three regimes we use; not a general-purpose helper.
    """
    return "_" + ("{:g}".format(accuracy * 100)).replace(".", "")


def add_accuracy_stress_columns(
    per_window_df: pd.DataFrame,
    *,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
    fees_bps: float = TAKER_FEE_BPS_PER_SIDE,
) -> pd.DataFrame:
    """Append `headroom_acc_<X>_bps` columns for each accuracy level.

    Vectorised over the parquet. NaN propagates from edge_bps / slip_avg_bps.
    Cost band uses the same fee + |slip| as the oracle column.
    """
    df = per_window_df.copy()
    edge = df["edge_bps"].to_numpy(dtype=np.float64)
    slip = np.abs(df["slip_avg_bps"].to_numpy(dtype=np.float64))
    cost = 2.0 * fees_bps + 2.0 * slip
    for acc in accuracies:
        suffix = _accuracy_suffix(acc)
        col = f"headroom_acc{suffix}_bps"
        df[col] = (2.0 * acc - 1.0) * np.abs(edge) - cost
    return df


def aggregate_cells(
    per_window_df: pd.DataFrame,
    *,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
) -> pd.DataFrame:
    """Per-(symbol, size, horizon) summary.

    Preserves the original oracle (perfect-direction) columns and appends
    per-accuracy regime stats `headroom_<X>_{median,p75,p90}_bps` and
    `frac_pos_acc_<X>` for each `X` in `accuracies` if the corresponding
    `headroom_acc<X>_bps` column is present in the input.
    """
    accuracy_list = list(accuracies)
    rows: list[dict] = []
    for key, g in per_window_df.groupby(["symbol", "size_usd", "horizon"], sort=True):
        sym, size, h = cast(tuple[str, float, int], key)
        n_total = len(g)
        if n_total == 0:
            continue
        fillable_frac = float(g["fillable"].to_numpy().astype(float).mean())
        # All slip / edge / headroom stats: condition on rows with finite values
        slips = g["slip_avg_bps"].dropna().to_numpy()
        edges = g["edge_bps"].dropna().to_numpy()
        # headroom requires both fillable AND finite edge
        fill_mask = g["fillable"].to_numpy()
        head_mask = fill_mask & np.isfinite(g["headroom_bps"].to_numpy())
        head_g = g.loc[head_mask]
        heads = head_g["headroom_bps"].to_numpy()
        n_head = len(heads)
        row: dict[str, float | int | str] = {
            "symbol": sym,
            "size_usd": float(size),
            "horizon": int(h),
            "n_windows": int(n_total),
            "n_fillable_with_edge": int(n_head),
            "fillable_frac": fillable_frac,
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
        # Per-accuracy regime stats — same fillable+finite filter
        for acc in accuracy_list:
            suffix = _accuracy_suffix(acc)
            src_col = f"headroom_acc{suffix}_bps"
            if src_col not in g.columns:
                continue
            acc_vals_full = g[src_col].to_numpy(dtype=np.float64)
            mask_acc = fill_mask & np.isfinite(acc_vals_full)
            acc_vals = acc_vals_full[mask_acc]
            n_acc = len(acc_vals)
            row[f"headroom{suffix}_median_bps"] = (
                float(np.median(acc_vals)) if n_acc else float("nan")
            )
            row[f"headroom{suffix}_p75_bps"] = (
                float(np.quantile(acc_vals, 0.75)) if n_acc else float("nan")
            )
            row[f"headroom{suffix}_p90_bps"] = (
                float(np.quantile(acc_vals, 0.90)) if n_acc else float("nan")
            )
            row[f"frac_pos_acc{suffix}"] = (
                float((acc_vals > 0).mean()) if n_acc else float("nan")
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["symbol", "size_usd", "horizon"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Survivors emission
# ---------------------------------------------------------------------------


def survivors_for_accuracy(
    cell_df: pd.DataFrame,
    *,
    accuracy: float,
    frac_pos_threshold: float = SURVIVOR_FRAC_POS_THRESHOLD,
    headroom_threshold_bps: float = SURVIVOR_HEADROOM_THRESHOLD_BPS,
) -> pd.DataFrame:
    """Filter aggregated cells to those that survive at `accuracy`.

    Survivor = `frac_pos_acc_<X> > frac_pos_threshold` AND
               `headroom_<X>_median_bps > headroom_threshold_bps`.
    Returns a copy with normalised column names: `headroom_median_bps`,
    `headroom_p75_bps`, `headroom_p90_bps`, `frac_pos`.
    """
    suffix = _accuracy_suffix(accuracy)
    med_col = f"headroom{suffix}_median_bps"
    p75_col = f"headroom{suffix}_p75_bps"
    p90_col = f"headroom{suffix}_p90_bps"
    frac_col = f"frac_pos_acc{suffix}"
    if frac_col not in cell_df.columns or med_col not in cell_df.columns:
        return pd.DataFrame()
    mask = (cell_df[frac_col] > frac_pos_threshold) & (
        cell_df[med_col] > headroom_threshold_bps
    )
    out = cell_df.loc[mask, :].copy()
    out["headroom_median_bps_at_acc"] = out[med_col]
    out["headroom_p75_bps_at_acc"] = out[p75_col]
    out["headroom_p90_bps_at_acc"] = out[p90_col]
    out["frac_pos_at_acc"] = out[frac_col]
    out["accuracy"] = accuracy
    cols = [
        "symbol",
        "size_usd",
        "horizon",
        "n_windows",
        "n_fillable_with_edge",
        "fillable_frac",
        "edge_median_bps",
        "slip_median_bps",
        "headroom_median_bps_at_acc",
        "headroom_p75_bps_at_acc",
        "headroom_p90_bps_at_acc",
        "frac_pos_at_acc",
        "accuracy",
    ]
    return (
        out[cols].sort_values(["horizon", "size_usd", "symbol"]).reset_index(drop=True)
    )


def _format_survivors_table(df: pd.DataFrame) -> str:
    """Render a survivors sub-table as a markdown grouped block by horizon."""
    if df.empty:
        return "_No cells survive at this accuracy._\n"
    lines: list[str] = []
    horizons = sorted({int(v) for v in df["horizon"].tolist()})
    for h_int in horizons:
        g = df.loc[df["horizon"] == h_int]
        lines.append(f"\n#### H{h_int}\n")
        lines.append(
            "| symbol | size | n_windows | fillable | edge_med | slip_med "
            "| headroom_med | headroom_p75 | frac_pos |"
        )
        lines.append(
            "|--------|------|-----------|----------|----------|----------"
            "|--------------|--------------|----------|"
        )
        # Iterate via tolist() for properly typed scalars (not Series)
        symbols = g["symbol"].tolist()
        sizes = g["size_usd"].tolist()
        nwin = g["n_windows"].tolist()
        fillable_fracs = g["fillable_frac"].tolist()
        edge_meds = g["edge_median_bps"].tolist()
        slip_meds = g["slip_median_bps"].tolist()
        head_meds = g["headroom_median_bps_at_acc"].tolist()
        head_p75s = g["headroom_p75_bps_at_acc"].tolist()
        frac_poses = g["frac_pos_at_acc"].tolist()
        for sym, sz, nw, ff, em, sm, hm, hp75, fp in zip(
            symbols,
            sizes,
            nwin,
            fillable_fracs,
            edge_meds,
            slip_meds,
            head_meds,
            head_p75s,
            frac_poses,
            strict=True,
        ):
            size_label = f"${int(float(sz)) // 1000}k"
            lines.append(
                f"| {sym} | {size_label} | "
                f"{int(float(nw))} | "
                f"{float(ff):.1%} | "
                f"{float(em):.2f} bp | "
                f"{float(sm):.2f} bp | "
                f"{float(hm):.2f} bp | "
                f"{float(hp75):.2f} bp | "
                f"{float(fp):.1%} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_survivors_md(
    cell_df: pd.DataFrame,
    out_path: Path,
    *,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
) -> dict[float, pd.DataFrame]:
    """Emit `survivors.md` with per-accuracy sub-tables. Returns the
    dict {accuracy: DataFrame} for downstream programmatic use.
    """
    survivors_by_acc: dict[float, pd.DataFrame] = {}
    body: list[str] = []
    body.append("# Goal-A Survivors at Realistic Directional Accuracy\n")
    body.append(
        "Cells that satisfy `frac_pos_acc_X > "
        f"{SURVIVOR_FRAC_POS_THRESHOLD:.2f}` AND "
        f"`headroom_X_median_bps > {SURVIVOR_HEADROOM_THRESHOLD_BPS:.0f}`, "
        "where X is the directional accuracy regime.\n"
    )
    body.append(
        "**Cost band**: 2 × 6 bp fee + 2 × |slip| (round-trip). "
        "**Edge math**: expected gross PnL per round trip = (2p − 1) × |edge|. "
        "Headroom = expected gross − cost. See README §"
        "Survivors at realistic directional accuracy for the modelling caveats.\n"
    )
    for acc in accuracies:
        df = survivors_for_accuracy(cell_df, accuracy=acc)
        survivors_by_acc[acc] = df
        body.append(f"\n## Accuracy = {acc:.3f} ({acc*100:g}%)\n")
        body.append(f"_Survivor count: **{len(df)}** of 300 cells._\n")
        body.append(_format_survivors_table(df))
        # Near-misses: top-5 by median headroom even if they fail the gate.
        # Useful for diagnosing how far off the universe is.
        suffix = _accuracy_suffix(acc)
        med_col = f"headroom{suffix}_median_bps"
        frac_col = f"frac_pos_acc{suffix}"
        if med_col in cell_df.columns:
            top = (
                cell_df.sort_values(med_col, ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            body.append(f"\n### Top 5 cells by median headroom@{acc:g} (gate-aware)\n")
            body.append(
                "| symbol | size | horizon | edge_med | slip_med "
                "| headroom_med | frac_pos | gate? |"
            )
            body.append(
                "|--------|------|---------|----------|----------"
                "|--------------|----------|-------|"
            )
            symbols = top["symbol"].tolist()
            sizes = top["size_usd"].tolist()
            horizons_l = top["horizon"].tolist()
            edges = top["edge_median_bps"].tolist()
            slips = top["slip_median_bps"].tolist()
            meds = top[med_col].tolist()
            fracs = top[frac_col].tolist()
            for sym, sz, ho, em, sm, hm, fp in zip(
                symbols, sizes, horizons_l, edges, slips, meds, fracs, strict=True
            ):
                gate = (
                    "PASS"
                    if (
                        float(fp) > SURVIVOR_FRAC_POS_THRESHOLD
                        and float(hm) > SURVIVOR_HEADROOM_THRESHOLD_BPS
                    )
                    else "fail"
                )
                size_label = f"${int(float(sz)) // 1000}k"
                body.append(
                    f"| {sym} | {size_label} | H{int(float(ho))} | "
                    f"{float(em):.2f} bp | {float(sm):.2f} bp | "
                    f"{float(hm):.2f} bp | {float(fp):.1%} | {gate} |"
                )
            body.append("")
    out_path.write_text("\n".join(body))
    return survivors_by_acc


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
    parser.add_argument(
        "--accuracy-stress-only",
        action="store_true",
        help=(
            "Skip the book-walk pipeline and operate on an existing "
            "per_window.parquet under --out-dir. Re-emits the parquet with "
            "headroom_acc_<X>_bps columns appended, regenerates the summary "
            "CSV with per-accuracy stats, and writes survivors.md."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.accuracy_stress_only:
        per_window_path = args.out_dir / "per_window.parquet"
        if not per_window_path.exists():
            raise FileNotFoundError(
                f"--accuracy-stress-only requires {per_window_path} to exist. "
                "Run the full pipeline first."
            )
        print(f"Loading {per_window_path}...", flush=True)
        per_window_df = pd.read_parquet(per_window_path)
        print(
            f"  Loaded {len(per_window_df)} rows. Adding accuracy-stress "
            f"columns for {ACCURACY_REGIMES}...",
            flush=True,
        )
        per_window_df = add_accuracy_stress_columns(per_window_df)
        per_window_df.to_parquet(per_window_path, index=False)
        print(f"  Re-emitted {per_window_path}", flush=True)

        summary_df = aggregate_cells(per_window_df)
        csv_path = args.out_dir / "headroom_table.csv"
        summary_df.to_csv(csv_path, index=False, float_format="%.6g")
        print(
            f"  Wrote {len(summary_df)} cells (now with per-acc stats) → "
            f"{csv_path}",
            flush=True,
        )

        survivors_path = args.out_dir / "survivors.md"
        survivors_by_acc = write_survivors_md(summary_df, survivors_path)
        for acc, df in survivors_by_acc.items():
            print(
                f"  acc={acc:.3f}: {len(df)} survivor cells",
                flush=True,
            )
        print(f"  Wrote {survivors_path}", flush=True)
        return

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
        for paths in by_sym.values():
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
    # Always compute accuracy-stress columns alongside the oracle headroom.
    per_window_df = add_accuracy_stress_columns(per_window_df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_window_path = args.out_dir / "per_window.parquet"
    per_window_df.to_parquet(per_window_path, index=False)
    print(f"Wrote {len(per_window_df)} per-window rows → {per_window_path}", flush=True)

    summary_df = aggregate_cells(per_window_df)
    csv_path = args.out_dir / "headroom_table.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"Wrote {len(summary_df)} (sym,size,horizon) cells → {csv_path}", flush=True)

    survivors_path = args.out_dir / "survivors.md"
    survivors_by_acc = write_survivors_md(summary_df, survivors_path)
    for acc, df in survivors_by_acc.items():
        print(f"  acc={acc:.3f}: {len(df)} survivor cells", flush=True)
    print(f"Wrote {survivors_path}", flush=True)


if __name__ == "__main__":
    main()

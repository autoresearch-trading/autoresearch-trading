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

# Maker-mode sensitivity sweep range (bps per side). Negative = rebate;
# positive = fee. Pacifica taker is +6 bp/side; we sweep below that to map
# how much rebate would be needed to flip the universe alive.
MAKER_FEES_BPS_SWEEP: tuple[float, ...] = (
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
)

# Fill-proxy threshold: |edge_bps| >= 1 bp counts as "mid traversed >=1 tick"
# This is a defensible cross-symbol simplification — true tick sizes are
# symbol-dependent. See `add_maker_headroom_columns` docstring for caveats.
MAKER_FILL_PROXY_BPS: float = 1.0

# Adverse-selection simulator — empirical fill + realized PnL on cached OB
# Offsets in bps from anchor mid at which we post symmetric limits.
ADVERSE_SELECTION_OFFSETS_BPS: tuple[float, ...] = (1.0, 2.0, 5.0)
# Pacifica actual maker fee: +1.5 bp/side (paid, not rebated)
PACIFICA_MAKER_FEE_BPS: float = 1.5
# Subsample cap per shard for the adverse-selection sim (smaller than the
# main book-walk pipeline because we now do per-window OB-snapshot range scans
# over the horizon, not just an at-anchor snapshot lookup).
ADVERSE_WINDOWS_PER_SHARD_CAP: int = 200

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


# ---------------------------------------------------------------------------
# Maker-mode cost band (parameterised by maker fee/rebate, NO slippage)
# ---------------------------------------------------------------------------


def maker_headroom_at_accuracy_bps(
    *,
    edge_bps: float,
    maker_fee_bps: float,
    accuracy: float,
) -> float:
    """Maker-mode headroom (bps) under iid directional accuracy `p`.

    Math:
        gross_pnl_bps      = (2p − 1) × |edge_bps|        # accuracy-stressed
        maker_cost_bps     = 2 × maker_fee_bps            # both legs, signed
        maker_headroom_bps = gross_pnl_bps − maker_cost_bps  # NO slippage

    Sign convention: `maker_fee_bps > 0` is a fee paid both legs, `< 0` is a
    rebate earned both legs. cost = 2 × maker_fee_bps preserves that sign.

    Load-bearing simplification: maker execution does NOT cross the spread, so
    slippage contribution is zero. In reality this is replaced by:
      (a) fill-rate risk (we may not get filled),
      (b) adverse-selection risk (we get filled when we shouldn't have).
    Both are out of scope for this first-cut analysis. The fill-proxy column
    in `add_maker_headroom_columns` is a coarse partial mitigant for (a).
    """
    edge_signal = (2.0 * accuracy - 1.0) * abs(edge_bps)
    cost = 2.0 * maker_fee_bps  # signed: positive = fee, negative = rebate
    return edge_signal - cost


def _fmt_fee_for_col(fee: float) -> str:
    """Render a maker-fee number for a column suffix.  -1.0→'-1', 0.0→'0',
    3.5→'3p5'.  Stable for the sweep we run (integer bps); gracefully handles
    halves if we ever extend the grid.
    """
    if float(fee).is_integer():
        return str(int(fee))
    return ("{:g}".format(fee)).replace(".", "p")


def add_maker_headroom_columns(
    per_window_df: pd.DataFrame,
    *,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
    maker_fees_bps: Iterable[float] = MAKER_FEES_BPS_SWEEP,
    fill_proxy_bps: float = MAKER_FILL_PROXY_BPS,
) -> pd.DataFrame:
    """Append maker-mode headroom columns + a fill-proxy boolean per window.

    Adds:
      * `maker_fill_proxy_bool`: True iff `|edge_bps| >= fill_proxy_bps`.
        Approximates "mid traversed >=1 tick from anchor mid within the
        horizon" by reusing the realised log-return magnitude. NaN edge → False.
        Caveats: edge is computed from VWAP cumsum, not anchor-mid → forward-mid;
        we don't have anchor-mid-aligned forward mid in the parquet. VWAP-based
        proxy is mid-anchored enough for a coarse cross-symbol filter.
      * `maker_headroom_acc_<X>_fee_<F>_bps` for each (accuracy, maker_fee) pair:
        (2p − 1) × |edge_bps| − 2 × maker_fee_bps. NO slippage by design.

    Limit explicitly NOT modelled:
      - queue position / time priority
      - partial fills (we treat fill as binary)
      - adverse selection (we'd get filled when the model is wrong, magnifying
        the (1-p) leg of the gross PnL — a bigger effect than fees in practice)
    A real fill-rate study needs raw limit-order event data which we don't have.
    """
    df = per_window_df.copy()
    edge = df["edge_bps"].to_numpy(dtype=np.float64)
    abs_edge = np.abs(edge)

    # Fill proxy: |edge_bps| >= threshold (1 bp = ~1 tick proxy). NaN → False.
    fill_proxy = np.isfinite(edge) & (abs_edge >= fill_proxy_bps)
    df["maker_fill_proxy_bool"] = fill_proxy

    for acc in accuracies:
        acc_suffix = _accuracy_suffix(acc).lstrip("_")  # '55', '575', '60'
        # Compute the gross signal once per accuracy
        gross = (2.0 * acc - 1.0) * abs_edge  # NaN propagates from edge
        for fee in maker_fees_bps:
            fee_suffix = _fmt_fee_for_col(fee)
            cost = 2.0 * fee  # signed scalar
            col = f"maker_headroom_acc_{acc_suffix}_fee_{fee_suffix}_bps"
            df[col] = gross - cost
    return df


def compute_maker_sensitivity_table(
    per_window_df: pd.DataFrame,
    *,
    maker_fees_bps: Iterable[float] = MAKER_FEES_BPS_SWEEP,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
    frac_pos_threshold: float = SURVIVOR_FRAC_POS_THRESHOLD,
    headroom_threshold_bps: float = SURVIVOR_HEADROOM_THRESHOLD_BPS,
    fill_proxy_bps: float = MAKER_FILL_PROXY_BPS,
) -> pd.DataFrame:
    """Per (maker_fee, accuracy) → number of (symbol,size,horizon) cells alive.

    A cell is "alive" iff:
        median maker_headroom > headroom_threshold_bps  AND
        frac_positive maker_headroom > frac_pos_threshold

    Two flavours reported:
      * `n_cells_alive`: cells that pass on raw headroom.
      * `n_cells_alive_with_fill_proxy`: cells that ALSO have median
         `maker_fill_proxy_bool` > frac_pos_threshold (i.e. a majority of the
         windows in the cell have edge >= 1 bp — proxy for "would have filled").

    Also reports `top_5_cells_by_median_headroom` as a comma-joined string of
    `SYM:$Sk:Hh` tags, sorted by median headroom desc.
    """
    fee_list = list(maker_fees_bps)
    acc_list = list(accuracies)

    df = add_maker_headroom_columns(
        per_window_df,
        accuracies=acc_list,
        maker_fees_bps=fee_list,
        fill_proxy_bps=fill_proxy_bps,
    )
    fillable_mask_full = df["fillable"].to_numpy(dtype=bool)
    fill_proxy_full = df["maker_fill_proxy_bool"].to_numpy(dtype=bool)

    rows: list[dict] = []
    # Pre-group once for speed — same groupby reused per (acc, fee) cell.
    grouped = df.groupby(["symbol", "size_usd", "horizon"], sort=True)
    # Cache group indices to avoid repeated label scans
    group_indices: list[tuple[tuple[str, float, int], np.ndarray]] = []
    for key, idx in grouped.indices.items():
        group_indices.append((cast(tuple[str, float, int], key), np.asarray(idx)))

    for acc in acc_list:
        acc_suffix = _accuracy_suffix(acc).lstrip("_")
        for fee in fee_list:
            fee_suffix = _fmt_fee_for_col(fee)
            col = f"maker_headroom_acc_{acc_suffix}_fee_{fee_suffix}_bps"
            head_full = df[col].to_numpy(dtype=np.float64)
            valid_full = fillable_mask_full & np.isfinite(head_full)

            n_alive = 0
            n_alive_proxy = 0
            cell_records: list[tuple[float, str]] = []  # (median, label)
            for key, idx in group_indices:
                sym, size, h = key
                local_valid = valid_full[idx]
                if not local_valid.any():
                    continue
                local_head = head_full[idx][local_valid]
                if local_head.size == 0:
                    continue
                med = float(np.median(local_head))
                frac_pos = float((local_head > 0).mean())
                cell_alive = (
                    med > headroom_threshold_bps and frac_pos > frac_pos_threshold
                )
                if cell_alive:
                    n_alive += 1
                    label = f"{sym}:${int(size)//1000}k:H{int(h)}"
                    cell_records.append((med, label))
                    # Apply fill-proxy filter on the same valid windows
                    local_fp = fill_proxy_full[idx][local_valid]
                    if local_fp.size > 0:
                        frac_fp = float(local_fp.mean())
                        if frac_fp > frac_pos_threshold:
                            n_alive_proxy += 1

            top5 = sorted(cell_records, key=lambda t: -t[0])[:5]
            top5_str = ", ".join(f"{label}({med:+.2f}bp)" for med, label in top5)

            rows.append(
                {
                    "maker_fee_bps": float(fee),
                    "accuracy": float(acc),
                    "n_cells_alive": int(n_alive),
                    "n_cells_alive_with_fill_proxy": int(n_alive_proxy),
                    "top_5_cells_by_median_headroom": top5_str,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["accuracy", "maker_fee_bps"])
        .reset_index(drop=True)
    )


def write_maker_sensitivity_md(
    sens_df: pd.DataFrame,
    out_path: Path,
    *,
    accuracies: Iterable[float] = ACCURACY_REGIMES,
    fill_proxy_bps: float = MAKER_FILL_PROXY_BPS,
) -> dict[float, float | None]:
    """Emit `maker_sensitivity.md`. Returns the per-accuracy "breakeven" maker
    fee — the highest maker_fee_bps at which `n_cells_alive >= 1`.

    Returns None for an accuracy regime if no cells are alive at any swept fee.
    """
    breakeven: dict[float, float | None] = {}
    body: list[str] = []
    body.append("# Goal-A Maker-Mode Cost-Band Sensitivity\n")
    body.append(
        "Sensitivity sweep over maker-mode fees (bps per side). "
        "Negative = rebate; positive = fee. **Slippage assumed = 0** under "
        "maker execution (we post a limit at our chosen price; we do not cross "
        "the spread). This is the load-bearing simplification of this analysis. "
        "In reality, the equivalent risk under maker execution is **adverse "
        "selection** (we get filled when the model is wrong), which this "
        "first-cut model does not incorporate.\n"
    )
    body.append(
        "**Maker headroom** = (2p − 1) × |edge_bps| − 2 × maker_fee_bps. "
        f"**Fill-proxy** = fraction of windows with |edge_bps| ≥ "
        f"{fill_proxy_bps:g} bp (rough cross-symbol approximation of "
        "'mid traverses ≥1 tick within horizon'). A cell is alive iff "
        "median headroom > 0 AND frac_positive headroom > "
        f"{SURVIVOR_FRAC_POS_THRESHOLD:.2f}.\n"
    )
    body.append(
        "**Caveats explicitly NOT modelled**: queue position, partial fills, "
        "adverse selection, symbol-specific tick sizes. A real fill-rate "
        "study needs raw limit-order event data — which we do not have.\n"
    )

    for acc in accuracies:
        sub = sens_df.loc[sens_df["accuracy"] == acc].sort_values("maker_fee_bps")
        body.append(f"\n## Accuracy = {acc:.3f} ({acc*100:g}%)\n")
        body.append(
            "| maker_fee_bps | n_cells_alive | n_cells_alive (with fill-proxy) | "
            "top 5 cells by median headroom |"
        )
        body.append("|---:|---:|---:|---|")
        # Breakeven = the LARGEST maker_fee_bps at which n_cells_alive >= 1
        be: float | None = None
        for _, row in sub.iterrows():
            fee = float(row["maker_fee_bps"])
            n_alive = int(row["n_cells_alive"])
            n_alive_fp = int(row["n_cells_alive_with_fill_proxy"])
            top = str(row["top_5_cells_by_median_headroom"]) or "_(none)_"
            body.append(f"| {fee:+.1f} | {n_alive} | {n_alive_fp} | {top} |")
            if n_alive >= 1 and (be is None or fee > be):
                be = fee
        breakeven[acc] = be
        if be is None:
            body.append(
                f"\n_No cell is alive at any swept maker_fee_bps in "
                f"[{min(MAKER_FEES_BPS_SWEEP):+.1f}, {max(MAKER_FEES_BPS_SWEEP):+.1f}] "
                f"at accuracy {acc:.3f}._\n"
            )
        else:
            body.append(
                f"\n**Breakeven maker fee at {acc*100:g}% accuracy: "
                f"≤ {be:+.1f} bp/side** "
                "(highest swept fee at which ≥1 cell is alive on raw headroom).\n"
            )
    body.append("\n## Cross-reference\n")
    body.append(
        "See `survivors.md` for the taker-mode (6 bp/side + slip) verdict. "
        "Under taker execution, zero cells survive at any of "
        "55%/57.5%/60% accuracy.\n"
    )

    out_path.write_text("\n".join(body))
    return breakeven


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
# Adverse-selection simulator
# ---------------------------------------------------------------------------
#
# For each window, we simulate posting a symmetric pair of limit orders at
# `anchor_mid × (1 ± X/10000)` and ask: did either side get filled within the
# horizon, and conditional on a fill, what was the realized PnL exiting at
# `mid_at_horizon = anchor_mid × exp(forward_log_return)` with no slippage?
#
# Fills are detected by walking the OB snapshot grid in the time interval
# [anchor_ts, ts_at_event(anchor + H)]. A bid fills if any snapshot in that
# range has best_ask <= bid_price (a seller crossed our bid). Symmetrically
# for the ask side.
#
# Caveat: the OB cadence is ~24s. At H10 on liquid symbols the horizon may
# be sub-second, in which case the [anchor_ts, horizon_end_ts] window
# contains the at-anchor snapshot only (the first one we've already seen
# in the at-anchor mid lookup). At H100/H500 on liquid symbols we get
# multiple snapshots. Fill detection at short horizons is therefore
# conservative — we miss intra-snapshot trade-throughs. This is documented
# as a methodological flag in the verdict markdown.


def best_bid_ask_from_levels(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
) -> tuple[float, float]:
    """Best bid (highest price w/ qty>0) and best ask (lowest price w/ qty>0)
    from a single snapshot's L1-L10 vectors. Returns (NaN, NaN) if either side
    has no valid level.

    L1 IS the best level by parquet convention, but we defensively scan all
    10 levels in case L1 is missing (qty=0). This matches `simulate_taker_fill`.
    """
    bid_valid = (bid_prices > 0) & (bid_qtys > 0)
    ask_valid = (ask_prices > 0) & (ask_qtys > 0)
    if not bid_valid.any() or not ask_valid.any():
        return float("nan"), float("nan")
    # First valid level on each side IS the best (parquet pre-sorts).
    best_bid = float(bid_prices[bid_valid][0])
    best_ask = float(ask_prices[ask_valid][0])
    return best_bid, best_ask


def detect_fill_in_range(
    *,
    snap_best_bids: np.ndarray,
    snap_best_asks: np.ndarray,
    snap_ts: np.ndarray,
    anchor_ts: int,
    horizon_end_ts: int,
    bid_price: float,
    ask_price: float,
) -> tuple[bool, bool]:
    """Detect bid/ask fills within [anchor_ts, horizon_end_ts] using OB grid.

    Bid fills if any snapshot in the range has best_ask <= bid_price (a seller
    crossed our resting bid). Ask fills if any snapshot has best_bid >=
    ask_price. NaN snapshot bid/ask values are treated as not-crossing.
    """
    # searchsorted gives [lo, hi) over snap_ts s.t. anchor_ts <= snap_ts < hi
    lo = int(np.searchsorted(snap_ts, anchor_ts, side="left"))
    hi = int(np.searchsorted(snap_ts, horizon_end_ts, side="right"))
    if lo >= hi:
        return False, False
    asks_in_range = snap_best_asks[lo:hi]
    bids_in_range = snap_best_bids[lo:hi]
    # NaN-safe: treat NaN as "not crossing"
    bid_filled = bool(np.any(np.isfinite(asks_in_range) & (asks_in_range <= bid_price)))
    ask_filled = bool(np.any(np.isfinite(bids_in_range) & (bids_in_range >= ask_price)))
    return bid_filled, ask_filled


def model_accuracy_breakeven(
    *, expected_realized_pnl_bps: float, maker_fee_bps_per_side: float
) -> float:
    """Directional accuracy `p` at which expected per-round-trip PnL = 0.

    Math:
        (2p − 1) × E[realized | filled] − 2 × maker_fee_per_side = 0
        p = 0.5 + maker_fee_per_side / E[realized | filled]

    Edge cases:
      * E = 0 and fee = 0: equation 0 = 0; canonical break-even is 0.5.
      * E = 0 and fee != 0: equation has no solution; returns +inf (unreachable
        accuracy — the strategy is always cost-bound).
      * E < 0: breakeven < 0.5 means a long-bias model loses money even with
        perfect skill, since filled trades have negative expectation. Caller
        should flag the sign separately.
      * E very small positive (e.g. 0.1 bp) with fee=1.5: breakeven = 15.5,
        i.e. unreachable. Return value still meaningful as a magnitude flag.
    """
    if expected_realized_pnl_bps == 0.0:
        if maker_fee_bps_per_side == 0.0:
            return 0.5
        return float("inf")
    return 0.5 + maker_fee_bps_per_side / expected_realized_pnl_bps


def process_shard_adverse_selection(
    shard_path: Path,
    *,
    rng: np.random.Generator,
    horizons: Iterable[int] = DIRECTION_HORIZONS,
    offsets_bps: Iterable[float] = ADVERSE_SELECTION_OFFSETS_BPS,
    cap: int = ADVERSE_WINDOWS_PER_SHARD_CAP,
) -> list[dict]:
    """Per-window per-cell adverse-selection sim for one (sym, date) shard.

    Returns one row per (window, horizon, offset). Empty list if shard is in
    the April hold-out, has fewer than WINDOW_LEN events, or its OB parquet
    is missing.
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

    anchors = starts + WINDOW_LEN - 1
    anchor_ts = event_ts[anchors]
    log_returns = features[:, _LOG_RETURN_IDX].astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(log_returns)])

    horizon_list = list(horizons)
    offset_list = list(offsets_bps)

    # Forward returns + horizon-end timestamps per horizon
    fwd_by_h: dict[int, np.ndarray] = {}
    horizon_end_ts_by_h: dict[int, np.ndarray] = {}
    valid_by_h: dict[int, np.ndarray] = {}
    for h in horizon_list:
        end = anchors + h
        valid = end < n_events
        fwd = np.full(len(anchors), np.nan, dtype=np.float64)
        fwd[valid] = cum[end[valid] + 1] - cum[anchors[valid] + 1]
        h_end_ts = np.full(len(anchors), -1, dtype=np.int64)
        h_end_ts[valid] = event_ts[end[valid]]
        fwd_by_h[h] = fwd
        horizon_end_ts_by_h[h] = h_end_ts
        valid_by_h[h] = valid

    # Load raw OB and pre-extract best bid/ask per snapshot
    ob = load_ob_day(sym, date)
    if ob is None or len(ob) == 0:
        return []
    snap_ts = ob["ts_ms"].to_numpy(dtype=np.int64)

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

    # Per-snapshot best bid/ask. Vectorised: for each row, mask qty>0 then
    # take the min/max along axis 1. We do this with masked views.
    bid_valid_mask = (bid_prices_all > 0) & (bid_qtys_all > 0)
    ask_valid_mask = (ask_prices_all > 0) & (ask_qtys_all > 0)
    # best_bid = max of valid bid prices; best_ask = min of valid ask prices.
    # Use np.where to neutralise invalid entries.
    bid_for_max = np.where(bid_valid_mask, bid_prices_all, -np.inf)
    ask_for_min = np.where(ask_valid_mask, ask_prices_all, np.inf)
    snap_best_bids = bid_for_max.max(axis=1)
    snap_best_asks = ask_for_min.min(axis=1)
    snap_best_bids = np.where(np.isfinite(snap_best_bids), snap_best_bids, np.nan)
    snap_best_asks = np.where(np.isfinite(snap_best_asks), snap_best_asks, np.nan)
    snap_mid = 0.5 * (snap_best_bids + snap_best_asks)

    # Anchor mid: most-recent prior snapshot
    snap_idx = np.searchsorted(snap_ts, anchor_ts, side="right") - 1
    valid_anchor = snap_idx >= 0
    if not valid_anchor.any():
        return []

    rows: list[dict] = []
    for w_idx in range(len(anchors)):
        if not valid_anchor[w_idx]:
            continue
        s = int(snap_idx[w_idx])
        anchor_mid = float(snap_mid[s])
        if not np.isfinite(anchor_mid) or anchor_mid <= 0:
            continue
        a_ts = int(anchor_ts[w_idx])

        for h in horizon_list:
            if not valid_by_h[h][w_idx]:
                continue
            fr = fwd_by_h[h][w_idx]
            if not np.isfinite(fr):
                continue
            h_end_ts = int(horizon_end_ts_by_h[h][w_idx])
            mid_at_h = anchor_mid * float(np.exp(fr))

            for offset_bps in offset_list:
                bid_price = anchor_mid * (1.0 - offset_bps / 1e4)
                ask_price = anchor_mid * (1.0 + offset_bps / 1e4)
                bid_filled, ask_filled = detect_fill_in_range(
                    snap_best_bids=snap_best_bids,
                    snap_best_asks=snap_best_asks,
                    snap_ts=snap_ts,
                    anchor_ts=a_ts,
                    horizon_end_ts=h_end_ts,
                    bid_price=bid_price,
                    ask_price=ask_price,
                )
                # Realized PnL (bps), only meaningful when filled.
                if bid_filled and bid_price > 0:
                    realized_bid_pnl_bps = (mid_at_h - bid_price) / bid_price * 1e4
                else:
                    realized_bid_pnl_bps = float("nan")
                if ask_filled and ask_price > 0:
                    realized_ask_pnl_bps = (ask_price - mid_at_h) / ask_price * 1e4
                else:
                    realized_ask_pnl_bps = float("nan")

                rows.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "window_start": int(starts[w_idx]),
                        "anchor_ts": a_ts,
                        "anchor_mid": anchor_mid,
                        "horizon": int(h),
                        "offset_bps": float(offset_bps),
                        "bid_price": float(bid_price),
                        "ask_price": float(ask_price),
                        "bid_filled": bool(bid_filled),
                        "ask_filled": bool(ask_filled),
                        "fill_horizon_ts": h_end_ts,
                        "mid_at_horizon": float(mid_at_h),
                        "realized_bid_pnl_bps": (
                            float(realized_bid_pnl_bps)
                            if np.isfinite(realized_bid_pnl_bps)
                            else float("nan")
                        ),
                        "realized_ask_pnl_bps": (
                            float(realized_ask_pnl_bps)
                            if np.isfinite(realized_ask_pnl_bps)
                            else float("nan")
                        ),
                    }
                )
    return rows


def aggregate_adverse_selection(
    per_window_df: pd.DataFrame,
    *,
    maker_fee_bps_per_side: float = PACIFICA_MAKER_FEE_BPS,
) -> pd.DataFrame:
    """Per (symbol, horizon, offset) summary stats for the adverse-selection sim.

    Columns:
        symbol, horizon, offset_bps, n_windows,
        fill_rate_bid, fill_rate_ask, fill_rate_either,
        mean_pnl_bid_filled, median_pnl_bid_filled,
        q25_pnl_bid_filled, q75_pnl_bid_filled,
        mean_pnl_ask_filled, median_pnl_ask_filled,
        q25_pnl_ask_filled, q75_pnl_ask_filled,
        mean_pnl_either_filled, median_pnl_either_filled,
        unconditional_maker_pnl_bps,
        model_accuracy_breakeven
    """
    rows: list[dict] = []
    grouped = per_window_df.groupby(["symbol", "horizon", "offset_bps"], sort=True)
    for key, g in grouped:
        sym, h, offset = cast(tuple[str, int, float], key)
        n_total = int(len(g))
        bid_filled = g["bid_filled"].to_numpy(dtype=bool)
        ask_filled = g["ask_filled"].to_numpy(dtype=bool)
        either_filled = bid_filled | ask_filled

        fill_rate_bid = float(bid_filled.mean()) if n_total else float("nan")
        fill_rate_ask = float(ask_filled.mean()) if n_total else float("nan")
        fill_rate_either = float(either_filled.mean()) if n_total else float("nan")

        bid_pnl = g["realized_bid_pnl_bps"].to_numpy(dtype=np.float64)
        ask_pnl = g["realized_ask_pnl_bps"].to_numpy(dtype=np.float64)
        bid_pnl_f = bid_pnl[np.isfinite(bid_pnl)]
        ask_pnl_f = ask_pnl[np.isfinite(ask_pnl)]

        def _stat(arr: np.ndarray, fn) -> float:
            return float(fn(arr)) if arr.size else float("nan")

        mean_bid = _stat(bid_pnl_f, np.mean)
        median_bid = _stat(bid_pnl_f, np.median)
        q25_bid = (
            float(np.quantile(bid_pnl_f, 0.25)) if bid_pnl_f.size else float("nan")
        )
        q75_bid = (
            float(np.quantile(bid_pnl_f, 0.75)) if bid_pnl_f.size else float("nan")
        )
        mean_ask = _stat(ask_pnl_f, np.mean)
        median_ask = _stat(ask_pnl_f, np.median)
        q25_ask = (
            float(np.quantile(ask_pnl_f, 0.25)) if ask_pnl_f.size else float("nan")
        )
        q75_ask = (
            float(np.quantile(ask_pnl_f, 0.75)) if ask_pnl_f.size else float("nan")
        )

        # Combined "either filled" PnL: take whichever filled. If both filled
        # in the same window (counter-trend whipsaw — rare on the 1bp offset,
        # common on the 5bp+) we include both observations as separate samples.
        either_pnl = (
            np.concatenate([bid_pnl_f, ask_pnl_f])
            if (bid_pnl_f.size + ask_pnl_f.size) > 0
            else np.array([])
        )
        mean_either = _stat(either_pnl, np.mean)
        median_either = _stat(either_pnl, np.median)

        # Unconditional symmetric-limit PnL minus 2x maker fee. This is the
        # expected gross realised PnL of a strategy that posts both legs
        # every window with no model. The factor of 1/2 accounts for posting
        # *both* legs symmetrically; if only one fills, the other doesn't
        # contribute.
        # E[gross | bid filled] * P(bid filled) + E[gross | ask filled] * P(ask filled)
        # divided by 2 (we have two legs but expect ~half of windows in
        # productive flow). Then subtract 2 * maker_fee to model both legs
        # paying the fee on a round-trip.
        e_bid = mean_bid if np.isfinite(mean_bid) else 0.0
        e_ask = mean_ask if np.isfinite(mean_ask) else 0.0
        unconditional_pnl = (
            fill_rate_bid * e_bid + fill_rate_ask * e_ask
        ) / 2.0 - 2.0 * maker_fee_bps_per_side

        # Breakeven uses the median-cell expected PnL conditional on a fill
        # (combined either side) — the adverse-selection statistic.
        if np.isfinite(mean_either):
            breakeven = model_accuracy_breakeven(
                expected_realized_pnl_bps=mean_either,
                maker_fee_bps_per_side=maker_fee_bps_per_side,
            )
        else:
            breakeven = float("nan")

        rows.append(
            {
                "symbol": sym,
                "horizon": int(h),
                "offset_bps": float(offset),
                "n_windows": n_total,
                "fill_rate_bid": fill_rate_bid,
                "fill_rate_ask": fill_rate_ask,
                "fill_rate_either": fill_rate_either,
                "mean_pnl_bid_filled": mean_bid,
                "median_pnl_bid_filled": median_bid,
                "q25_pnl_bid_filled": q25_bid,
                "q75_pnl_bid_filled": q75_bid,
                "mean_pnl_ask_filled": mean_ask,
                "median_pnl_ask_filled": median_ask,
                "q25_pnl_ask_filled": q25_ask,
                "q75_pnl_ask_filled": q75_ask,
                "mean_pnl_either_filled": mean_either,
                "median_pnl_either_filled": median_either,
                "unconditional_maker_pnl_bps": unconditional_pnl,
                "model_accuracy_breakeven": breakeven,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["symbol", "horizon", "offset_bps"])
        .reset_index(drop=True)
    )


def _append_cell_rows_md(body: list[str], df: pd.DataFrame) -> None:
    """Append one markdown row per (symbol,horizon,offset) cell using tolist()
    extraction (avoids pandas Series typing issues with iterrows())."""
    symbols = df["symbol"].tolist()
    horizons = df["horizon"].tolist()
    offsets = df["offset_bps"].tolist()
    nwins = df["n_windows"].tolist()
    fillrates = df["fill_rate_either"].tolist()
    e_eithers = df["mean_pnl_either_filled"].tolist()
    breakevens = df["model_accuracy_breakeven"].tolist()
    for sym, h, off, nw, fr, e_e, be in zip(
        symbols, horizons, offsets, nwins, fillrates, e_eithers, breakevens, strict=True
    ):
        body.append(
            f"| {str(sym)} | H{int(float(h))} | "
            f"{float(off):+.1f} | {int(float(nw))} | "
            f"{float(fr):.1%} | "
            f"{float(e_e):+.3f} | "
            f"{float(be):.3f} |"
        )


def write_adverse_selection_md(
    cell_df: pd.DataFrame,
    out_path: Path,
    *,
    maker_fee_bps_per_side: float = PACIFICA_MAKER_FEE_BPS,
) -> None:
    """Emit `maker_adverse_selection.md` answering the four prompts:

    1. Is E[realized | filled] consistently negative across the universe?
    2. Universe-wide median model_accuracy_breakeven vs 51.4% / 60% thresholds.
    3. Per-symbol cells with breakeven < 55%.
    4. One-paragraph verdict.
    """
    body: list[str] = []
    body.append("# Goal-A Maker Adverse-Selection Sim — Empirical E[PnL | filled]\n")
    body.append(
        "This sim posts symmetric resting limits at "
        "`anchor_mid × (1 ± offset/1e4)` for every window in the cache and "
        "asks: did either side fill within the horizon, and what was the "
        "realized PnL exiting at `mid_at_horizon = anchor_mid × exp(forward_log_return)`? "
        "If E[realized | filled] is consistently negative, the Maker's Dilemma "
        "(Albers 2025) is empirically real on this universe, and the maker "
        "pivot's apparent cost-band advantage in `maker_sensitivity.md` is "
        f"partially or fully consumed by adverse selection at the actual "
        f"{maker_fee_bps_per_side:+.1f} bp/side maker fee.\n"
    )
    body.append(
        "**`model_accuracy_breakeven`** = directional accuracy `p` at which "
        f"`(2p − 1) × E[realized | filled] − 2 × {maker_fee_bps_per_side} bp = 0`. "
        "If E < 0, breakeven < 0.5: a long-bias model needs to be a contrarian "
        "to overcome the Dilemma — i.e. the maker pivot fails on filled trades. "
        "If E > 0, breakeven > 0.5; if breakeven < 55% the Dilemma is mild "
        "enough that a realistic model could overcome it.\n"
    )

    # Universe-wide stats
    n_cells = len(cell_df)
    medians_either = cell_df["mean_pnl_either_filled"].dropna().to_numpy()
    medians_bid = cell_df["mean_pnl_bid_filled"].dropna().to_numpy()
    medians_ask = cell_df["mean_pnl_ask_filled"].dropna().to_numpy()
    breakevens = cell_df["model_accuracy_breakeven"].dropna().to_numpy()

    universe_median_breakeven = (
        float(np.median(breakevens)) if breakevens.size else float("nan")
    )
    universe_median_e_either = (
        float(np.median(medians_either)) if medians_either.size else float("nan")
    )
    frac_negative_e = (
        float((medians_either < 0).mean()) if medians_either.size else float("nan")
    )
    # "Tradeable" = E[realized | filled] > 0 AND breakeven below threshold.
    # Cells with negative E have breakeven < 0.5 (contrarian zone, NOT tradeable
    # for a long-bias model — they confirm the Maker's Dilemma instead). We
    # require breakeven > 0.5 alongside the upper bound. Build a co-aligned
    # (E, breakeven) join via the cell_df's full row order; drop NaN breakeven
    # rows then test on the remaining 1-D arrays of equal length.
    paired = cell_df[["mean_pnl_either_filled", "model_accuracy_breakeven"]].dropna()
    if not paired.empty:
        e_arr = np.array(paired["mean_pnl_either_filled"].tolist(), dtype=np.float64)
        be_arr = np.array(paired["model_accuracy_breakeven"].tolist(), dtype=np.float64)
        tradeable_be = (e_arr > 0) & (be_arr > 0.5)
        cells_breakeven_below_55 = int((tradeable_be & (be_arr < 0.55)).sum())
        cells_breakeven_below_60 = int((tradeable_be & (be_arr < 0.60)).sum())
        cells_breakeven_below_5140 = int((tradeable_be & (be_arr < 0.514)).sum())
    else:
        cells_breakeven_below_55 = 0
        cells_breakeven_below_60 = 0
        cells_breakeven_below_5140 = 0

    body.append("## Universe-wide summary\n")
    body.append(f"* Total cells (symbol × horizon × offset): **{n_cells}**\n")
    body.append(
        f"* Median E[realized | either-side filled] across cells: "
        f"**{universe_median_e_either:+.3f} bp**\n"
    )
    body.append(
        f"* Fraction of cells with negative E[realized | filled]: "
        f"**{frac_negative_e:.1%}** (= Maker's Dilemma signal density)\n"
    )
    body.append(
        f"* Universe-wide median `model_accuracy_breakeven`: "
        f"**{universe_median_breakeven:.3f}** "
        f"(vs v1 demonstrated 51.4%, vs near-miss 60%)\n"
    )
    body.append(
        f"* Cells with breakeven < 51.4% (v1 ceiling): "
        f"**{cells_breakeven_below_5140}** of {n_cells}\n"
    )
    body.append(
        f"* Cells with breakeven < 55%: "
        f"**{cells_breakeven_below_55}** of {n_cells}\n"
    )
    body.append(
        f"* Cells with breakeven < 60%: "
        f"**{cells_breakeven_below_60}** of {n_cells}\n"
    )

    # Sign breakdown by horizon
    body.append("\n## E[realized | filled] sign by horizon\n")
    body.append(
        "| horizon | n_cells | median E[either] (bp) | frac negative | "
        "median breakeven |"
    )
    body.append("|---:|---:|---:|---:|---:|")
    for h_key, gh in cell_df.groupby("horizon", sort=True):
        h_int = int(cast(int, h_key))
        e_arr = gh["mean_pnl_either_filled"].dropna().to_numpy()
        be_arr = gh["model_accuracy_breakeven"].dropna().to_numpy()
        med_e = float(np.median(e_arr)) if e_arr.size else float("nan")
        frac_neg = float((e_arr < 0).mean()) if e_arr.size else float("nan")
        med_be = float(np.median(be_arr)) if be_arr.size else float("nan")
        body.append(
            f"| H{h_int} | {len(gh)} | {med_e:+.3f} | {frac_neg:.1%} | {med_be:.3f} |"
        )

    # By offset
    body.append("\n## E[realized | filled] sign by offset\n")
    body.append(
        "| offset (bp) | n_cells | median E[either] (bp) | frac negative | "
        "median breakeven |"
    )
    body.append("|---:|---:|---:|---:|---:|")
    for off_key, go in cell_df.groupby("offset_bps", sort=True):
        off_f = float(cast(float, off_key))
        e_arr = go["mean_pnl_either_filled"].dropna().to_numpy()
        be_arr = go["model_accuracy_breakeven"].dropna().to_numpy()
        med_e = float(np.median(e_arr)) if e_arr.size else float("nan")
        frac_neg = float((e_arr < 0).mean()) if e_arr.size else float("nan")
        med_be = float(np.median(be_arr)) if be_arr.size else float("nan")
        body.append(
            f"| {off_f:+.1f} | {len(go)} | {med_e:+.3f} | {frac_neg:.1%} | {med_be:.3f} |"
        )

    # Per-symbol "tradeable" cells: E[realized | filled] > 0 AND
    # 0.5 < breakeven < 0.55 (a long-bias model with a realistic edge can
    # overcome the Dilemma). Cells with E < 0 have breakeven < 0.5 by math,
    # which is the contrarian zone — NOT tradeable for a directional model.
    tradeable_mask = (
        (cell_df["mean_pnl_either_filled"] > 0)
        & (cell_df["model_accuracy_breakeven"] > 0.5)
        & (cell_df["model_accuracy_breakeven"] < 0.55)
    )
    tradeable = cell_df.loc[tradeable_mask].sort_values("model_accuracy_breakeven")
    body.append(
        f"\n## Tradeable cells (E[either] > 0 AND 50% < breakeven < 55%): "
        f"{len(tradeable)} of {n_cells}\n"
    )
    if tradeable.empty:
        body.append(
            "_No cells satisfy E[realized | filled] > 0 AND breakeven in (0.5, "
            "0.55). The Maker's Dilemma is uniformly severe — every cell either "
            "has negative E (contrarian zone) or requires accuracy above 55%._\n"
        )
    else:
        body.append(
            "| symbol | horizon | offset (bp) | n_windows | fill_rate_either | "
            "E[either] (bp) | breakeven |"
        )
        body.append("|---|---:|---:|---:|---:|---:|---:|")
        _append_cell_rows_md(body, tradeable)

    # Top 10 cells by lowest breakeven among ones where E > 0 (the meaningful
    # subset for a long-bias model). If no cells have E > 0, fall back to
    # showing the LEAST-NEGATIVE E cells with their breakeven (still in
    # the contrarian zone but informative).
    body.append("\n## Top 10 cells by lowest breakeven (E[either] > 0 only)\n")
    pos_e = cell_df.loc[cell_df["mean_pnl_either_filled"] > 0].dropna(
        subset=["model_accuracy_breakeven"]
    )
    if pos_e.empty:
        body.append(
            "_No cell has positive E[realized | filled]; falling back to "
            "least-negative E[either] cells (these are still in the "
            "contrarian zone — breakeven < 0.5)._\n"
        )
        top = (
            cell_df.dropna(subset=["mean_pnl_either_filled"])
            .sort_values("mean_pnl_either_filled", ascending=False)
            .head(10)
        )
    else:
        top = pos_e.sort_values("model_accuracy_breakeven").head(10)
    body.append(
        "| symbol | horizon | offset (bp) | n_windows | fill_rate_either | "
        "E[either] (bp) | breakeven |"
    )
    body.append("|---|---:|---:|---:|---:|---:|---:|")
    _append_cell_rows_md(body, top)

    # Verdict — the controlling question is E[realized | filled]. If E < 0
    # broadly, the Dilemma is real and a long-bias maker model cannot work
    # regardless of skill (the breakeven < 0.5 result means even infinite skill
    # would be paying fees on losing-trade fills). If E > 0 broadly but the
    # implied breakeven is above what the v1 model demonstrated, it's a reach.
    body.append("\n## Verdict\n")
    if frac_negative_e > 0.9:
        verdict = (
            "**The maker pivot fails the adverse-selection test.** "
            f"E[realized | filled] is negative in {frac_negative_e:.1%} of "
            f"cells (universe median **{universe_median_e_either:+.2f} bp**). "
            "Filled limits are systematically followed by adverse mid moves — "
            "the Maker's Dilemma pattern (Albers 2025) is empirically present "
            "across the entire universe. The implied universe-wide median "
            f"`model_accuracy_breakeven` of **{universe_median_breakeven:.3f}** "
            f"is below 0.5, which means a long-bias model needs to be "
            f"*contrarian* (predict against its own signal) to break even on "
            f"filled trades — i.e. the strategy fundamentally cannot rest "
            f"limits and profit on average. The cost-band advantage reported "
            f"in `maker_sensitivity.md` (~289/300 cells alive at "
            f"{maker_fee_bps_per_side:+.1f} bp/side under the slippage=0 "
            f"assumption) is illusory once adverse selection is incorporated. "
            f"Tradeable-cell count "
            f"(E > 0 AND 0.5 < breakeven < 0.55): "
            f"**{len(tradeable)} of {n_cells}**."
        )
    elif universe_median_breakeven < 0.55 and universe_median_e_either > 0:
        verdict = (
            f"**The maker pivot partially survives.** Universe-wide median "
            f"E[realized | filled] is **{universe_median_e_either:+.2f} bp** "
            f"(positive) and median breakeven is "
            f"**{universe_median_breakeven:.3f}** — below the 55% bar. "
            f"{len(tradeable)} of {n_cells} cells satisfy E > 0 AND breakeven "
            f"in (0.5, 0.55). The strategy is feasible if the model can be "
            f"deployed selectively to those cells."
        )
    elif universe_median_breakeven < 0.6 and universe_median_e_either > 0:
        verdict = (
            f"**The maker pivot is on the bubble.** Universe-wide median "
            f"E[realized | filled] is **{universe_median_e_either:+.2f} bp** "
            f"and median breakeven is **{universe_median_breakeven:.3f}** — "
            f"between the 55% and 60% thresholds. "
            f"{len(tradeable)} cells have a tradeable breakeven. Viable only "
            f"if the encoder can hit ~58-60% on the surviving subset."
        )
    else:
        verdict = (
            f"**The maker pivot fails on accuracy reach.** Universe-wide "
            f"median E[realized | filled] is "
            f"**{universe_median_e_either:+.2f} bp** and median breakeven is "
            f"**{universe_median_breakeven:.3f}** — outside any realistic "
            f"model's reach. The pivot needs a model that's never been "
            f"demonstrated on this data."
        )
    body.append(verdict + "\n")

    body.append("\n## Methodological flags\n")
    body.append(
        "* **OB cadence ~24 s.** At short horizons (H10) on liquid symbols the "
        "horizon may be sub-second, in which case the [anchor_ts, horizon_end_ts] "
        "range contains zero or one snapshots and fill detection is conservative "
        "(we miss intra-snapshot trade-throughs). Treat fill rates at H10 as a "
        "**lower bound**.\n"
    )
    body.append(
        "* **Symmetric-limit assumption.** This sim posts both bid and ask every "
        "window with no model conditioning. A directional model would post only "
        "the side it's predicting, which changes the conditional fill mix. The "
        "breakeven number assumes the realized PnL distribution under the "
        "symmetric strategy is representative of the model-conditional one — "
        "this is an approximation, not a derivation.\n"
    )
    body.append(
        "* **Mid-vs-VWAP exit.** `mid_at_horizon` is computed from "
        "`anchor_mid × exp(sum_log_return)` where `log_return` is per-event "
        "VWAP-to-VWAP. This is mid-anchored at the start, VWAP-anchored at the "
        "exit — a small bias for cross-side flow at the exit event.\n"
    )
    body.append(
        "* **No fill-time PnL.** Realized PnL marks to `mid_at_horizon` "
        "regardless of how early the fill happened within the horizon. A fill "
        "in the first second of a 30-second H100 window holds for the full 30 s; "
        "an earlier exit would have a different (and probably better) PnL. "
        "This biases the realized PnL estimate slightly negative for filled "
        "bids in down-trending windows (and vice-versa).\n"
    )

    out_path.write_text("\n".join(body))


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
    parser.add_argument(
        "--maker-sweep",
        action="store_true",
        help=(
            "Run the maker-mode cost-band sensitivity sweep on an existing "
            "per_window.parquet under --out-dir. Writes maker_sensitivity.csv "
            "and maker_sensitivity.md. Also patches survivors.md with a "
            "one-paragraph cross-reference (taker verdict left intact)."
        ),
    )
    parser.add_argument(
        "--adverse-selection",
        action="store_true",
        help=(
            "Run the maker-mode adverse-selection sim on the cached OB grid. "
            "For every window, posts symmetric limits at "
            "anchor_mid × (1 ± offset/1e4) for offset ∈ {1, 2, 5} bp, detects "
            "fills against [anchor_ts, ts_at_event(anchor+H)] OB snapshots, "
            "and computes realized PnL at mid_at_horizon. Writes "
            "maker_adverse_selection_per_window.parquet (gitignored), "
            "maker_adverse_selection_table.csv, and "
            "maker_adverse_selection.md."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.adverse_selection:
        cache_dir = args.cache_dir
        all_shards = sorted(cache_dir.glob("*.npz"))
        all_shards = [
            p for p in all_shards if p.stem.split("__")[1] < APRIL_HELDOUT_START
        ]
        if args.symbols:
            wanted = set(args.symbols)
            all_shards = [p for p in all_shards if p.stem.split("__")[0] in wanted]
        else:
            wanted = set(SYMBOLS)
            all_shards = [p for p in all_shards if p.stem.split("__")[0] in wanted]

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

        print(
            f"[adverse-selection] Processing {len(all_shards)} shards...",
            flush=True,
        )
        all_rows: list[dict] = []
        for i, p in enumerate(all_shards):
            if i % 50 == 0:
                print(f"  [{i}/{len(all_shards)}] {p.name}", flush=True)
            rows = process_shard_adverse_selection(p, rng=rng)
            all_rows.extend(rows)

        if not all_rows:
            raise RuntimeError(
                "No adverse-selection rows produced — check cache_dir and "
                "symbols filter."
            )

        per_window_df = pd.DataFrame(all_rows)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_window_path = args.out_dir / "maker_adverse_selection_per_window.parquet"
        per_window_df.to_parquet(per_window_path, index=False)
        print(
            f"Wrote {len(per_window_df)} per-window rows → {per_window_path}",
            flush=True,
        )

        cell_df = aggregate_adverse_selection(per_window_df)
        csv_path = args.out_dir / "maker_adverse_selection_table.csv"
        cell_df.to_csv(csv_path, index=False, float_format="%.6g")
        print(
            f"Wrote {len(cell_df)} (sym,horizon,offset) cells → {csv_path}",
            flush=True,
        )

        md_path = args.out_dir / "maker_adverse_selection.md"
        write_adverse_selection_md(cell_df, md_path)
        print(f"Wrote {md_path}", flush=True)
        return

    if args.maker_sweep:
        per_window_path = args.out_dir / "per_window.parquet"
        if not per_window_path.exists():
            raise FileNotFoundError(
                f"--maker-sweep requires {per_window_path} to exist. "
                "Run the full pipeline first."
            )
        print(f"Loading {per_window_path}...", flush=True)
        per_window_df = pd.read_parquet(per_window_path)
        print(
            f"  Loaded {len(per_window_df)} rows. Computing maker sensitivity "
            f"over fees {MAKER_FEES_BPS_SWEEP} × accuracies {ACCURACY_REGIMES}...",
            flush=True,
        )
        sens_df = compute_maker_sensitivity_table(per_window_df)
        csv_path = args.out_dir / "maker_sensitivity.csv"
        sens_df.to_csv(csv_path, index=False, float_format="%.6g")
        print(f"  Wrote {len(sens_df)} sensitivity rows → {csv_path}", flush=True)

        md_path = args.out_dir / "maker_sensitivity.md"
        breakeven = write_maker_sensitivity_md(sens_df, md_path)
        for acc, be in breakeven.items():
            be_str = f"{be:+.1f}" if be is not None else "no cells alive"
            print(f"  acc={acc:.3f}: breakeven maker_fee = {be_str}", flush=True)
        print(f"  Wrote {md_path}", flush=True)

        # Patch survivors.md with a one-paragraph cross-reference.
        survivors_path = args.out_dir / "survivors.md"
        if survivors_path.exists():
            current = survivors_path.read_text()
            xref_marker = "## Maker-mode cross-reference"
            if xref_marker not in current:
                xref = (
                    f"\n\n{xref_marker}\n\n"
                    "The taker verdict above (zero cells alive at 55%/57.5%/60% "
                    "accuracy under 6 bp/side fee + book-walk slippage) is "
                    "computed under taker execution. A complementary maker-mode "
                    "sweep (`maker_sensitivity.md`) parameterises the maker fee "
                    "from a 2 bp rebate to a 6 bp fee per side, with slippage "
                    "set to zero (the load-bearing simplification — adverse "
                    "selection is not modelled). The breakeven maker fee per "
                    "accuracy regime is reported there. The taker verdict in "
                    "this file is unchanged.\n"
                )
                survivors_path.write_text(current.rstrip() + xref)
                print(f"  Patched {survivors_path} with maker xref.", flush=True)
            else:
                print(
                    f"  {survivors_path} already has maker xref — no change.",
                    flush=True,
                )
        return

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

#!/usr/bin/env python
"""NaN contamination analysis for tape cache shards.

Investigates 5 representative contaminated shards + raw data to identify
root cause of NaN values in the feature tensor.

Usage:
    python scripts/analysis/analyze_nan_contamination.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/Users/diego/Dev/non-toxic/autoresearch-trading")
CACHE_DIR = REPO_ROOT / "data/cache"
RAW_DATA = REPO_ROOT / "data"

FEATURE_NAMES = (
    "log_return",  # 0
    "log_total_qty",  # 1
    "is_open",  # 2
    "time_delta",  # 3
    "num_fills",  # 4
    "book_walk",  # 5
    "effort_vs_result",  # 6
    "climax_score",  # 7
    "prev_seq_time_span",  # 8
    "log_spread",  # 9
    "imbalance_L1",  # 10
    "imbalance_L5",  # 11
    "depth_ratio",  # 12
    "trade_vs_mid",  # 13
    "delta_imbalance_L1",  # 14
    "kyle_lambda",  # 15
    "cum_ofi_5",  # 16
)

TARGET_SHARDS = [
    ("2Z", "2026-02-02"),
    ("KBONK", "2026-02-06"),
    ("CRV", "2026-02-03"),
    ("UNI", "2025-10-16"),
    ("WLFI", "2026-03-12"),
]

# ---------------------------------------------------------------------------
# Part 1: Per-shard NaN localization
# ---------------------------------------------------------------------------


def load_shard(symbol: str, date: str) -> np.ndarray | None:
    path = CACHE_DIR / f"{symbol}__{date}.npz"
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    with np.load(path, allow_pickle=False) as z:
        return z["features"].copy()


def analyze_shard(symbol: str, date: str) -> dict:
    print(f"\n{'='*60}")
    print(f"SHARD: {symbol}__{date}")
    print(f"{'='*60}")

    features = load_shard(symbol, date)
    if features is None:
        return {"symbol": symbol, "date": date, "error": "missing"}

    n_events, n_feats = features.shape
    print(f"Shape: ({n_events}, {n_feats})")

    # NaN mask per column
    nan_mask = np.isnan(features)
    inf_mask = np.isinf(features)
    bad_mask = nan_mask | inf_mask

    result = {
        "symbol": symbol,
        "date": date,
        "n_events": n_events,
        "nan_per_col": {},
        "nan_rows": {},
        "clustering": {},
    }

    if not bad_mask.any():
        print("  CLEAN — no NaN/Inf found")
        result["clean"] = True
        return result

    result["clean"] = False
    total_bad = int(bad_mask.sum())
    print(f"  Total bad cells: {total_bad} ({100*total_bad/(n_events*n_feats):.3f}%)")

    # Per-column NaN/Inf counts
    print(f"\n  {'Feature':<22} {'NaN':>6} {'Inf':>6} {'Total':>6}  Affected rows")
    print(f"  {'-'*70}")
    for i, fname in enumerate(FEATURE_NAMES):
        n_nan = int(nan_mask[:, i].sum())
        n_inf = int(inf_mask[:, i].sum())
        n_bad = n_nan + n_inf
        if n_bad > 0:
            bad_rows = np.where(bad_mask[:, i])[0]
            row_summary = f"rows {bad_rows[0]}..{bad_rows[-1]} ({len(bad_rows)} total)"
            print(f"  {fname:<22} {n_nan:>6} {n_inf:>6} {n_bad:>6}  {row_summary}")
            result["nan_per_col"][fname] = n_bad
            result["nan_rows"][fname] = bad_rows.tolist()

            # Clustering: is it the first N rows?
            is_prefix = bool(bad_rows[-1] < 200 and bad_rows[0] == 0)
            is_scattered = len(bad_rows) > 5 and (bad_rows[-1] - bad_rows[0]) > 500
            result["clustering"][fname] = (
                "prefix"
                if is_prefix
                else ("scattered" if is_scattered else "clustered")
            )

    # For columns with NaN: show adjacent values
    print(f"\n  Adjacent values around NaN rows:")
    for fname, bad_rows in result["nan_rows"].items():
        col_idx = FEATURE_NAMES.index(fname)
        col = features[:, col_idx]
        # Show context around first NaN occurrence
        first_bad = bad_rows[0]
        context_lo = max(0, first_bad - 3)
        context_hi = min(n_events - 1, first_bad + 3)
        context_vals = col[context_lo : context_hi + 1]
        context_str = ", ".join(
            f"[NaN]" if np.isnan(v) or np.isinf(v) else f"{v:.5f}" for v in context_vals
        )
        print(
            f"    {fname} @ row {first_bad}: [{context_lo}..{context_hi}] = {context_str}"
        )

        # Also check last NaN if different from first
        if len(bad_rows) > 1:
            last_bad = bad_rows[-1]
            ctx_lo = max(0, last_bad - 2)
            ctx_hi = min(n_events - 1, last_bad + 2)
            ctx_vals = col[ctx_lo : ctx_hi + 1]
            ctx_str = ", ".join(
                f"[NaN]" if np.isnan(v) or np.isinf(v) else f"{v:.5f}" for v in ctx_vals
            )
            print(
                f"    {fname} @ row {last_bad} (last): [{ctx_lo}..{ctx_hi}] = {ctx_str}"
            )

    return result


# ---------------------------------------------------------------------------
# Part 2: Raw data cross-reference
# ---------------------------------------------------------------------------


def check_raw_data(symbol: str, date: str) -> dict:
    print(f"\n  -- Raw data check: {symbol} {date} --")

    con = duckdb.connect()
    result = {"symbol": symbol, "date": date}

    # Check trades
    trades_glob = str(RAW_DATA / f"trades/symbol={symbol}/date={date}/*.parquet")
    try:
        df_t = con.execute(
            f"SELECT COUNT(*) as n FROM read_parquet('{trades_glob}')"
        ).fetchdf()
        n_trades = int(df_t["n"].iloc[0])
        print(f"    Trades: {n_trades} rows")
        result["n_trades"] = n_trades

        if n_trades > 0:
            # Check for anomalies
            info = (
                con.execute(
                    f"""
                SELECT
                    MIN(price) as min_price, MAX(price) as max_price,
                    MIN(qty) as min_qty, MAX(qty) as max_qty,
                    COUNT(DISTINCT side) as n_sides,
                    MIN(ts_ms) as ts_min, MAX(ts_ms) as ts_max
                FROM read_parquet('{trades_glob}')
            """
                )
                .fetchdf()
                .iloc[0]
            )
            print(f"    Price: [{info['min_price']:.4f}, {info['max_price']:.4f}]")
            print(f"    Qty:   [{info['min_qty']:.6f}, {info['max_qty']:.6f}]")
            print(f"    Sides: {info['n_sides']}")
            result.update(
                {
                    "trade_min_price": float(info["min_price"]),
                    "trade_max_price": float(info["max_price"]),
                    "trade_min_qty": float(info["min_qty"]),
                    "trade_max_qty": float(info["max_qty"]),
                }
            )
    except Exception as e:
        print(f"    Trades error: {e}")
        result["trades_error"] = str(e)

    # Check orderbook
    ob_glob = str(RAW_DATA / f"orderbook/symbol={symbol}/date={date}/*.parquet")
    try:
        df_ob = con.execute(
            f"SELECT COUNT(*) as n FROM read_parquet('{ob_glob}')"
        ).fetchdf()
        n_ob = int(df_ob["n"].iloc[0])
        print(f"    OB snapshots: {n_ob}")
        result["n_ob_snapshots"] = n_ob

        if n_ob > 0:
            # Check for gaps, zero prices, and spread anomalies
            ob_info = (
                con.execute(
                    f"""
                SELECT
                    MIN(ts_ms) as ts_min, MAX(ts_ms) as ts_max,
                    MIN(bid1_price) as min_bid1, MAX(ask1_price) as max_ask1,
                    MIN(bid1_qty) as min_bid1_qty, MAX(ask1_qty) as max_ask1_qty,
                    COUNT(*) as n_rows,
                    -- Check for zero/null L1
                    SUM(CASE WHEN bid1_price = 0 OR bid1_price IS NULL THEN 1 ELSE 0 END) as zero_bid1,
                    SUM(CASE WHEN ask1_price = 0 OR ask1_price IS NULL THEN 1 ELSE 0 END) as zero_ask1,
                    SUM(CASE WHEN bid1_qty = 0 OR bid1_qty IS NULL THEN 1 ELSE 0 END) as zero_bid1_qty,
                    SUM(CASE WHEN ask1_qty = 0 OR ask1_qty IS NULL THEN 1 ELSE 0 END) as zero_ask1_qty
                FROM read_parquet('{ob_glob}')
            """
                )
                .fetchdf()
                .iloc[0]
            )

            ts_span_h = (int(ob_info["ts_max"]) - int(ob_info["ts_min"])) / 3600000
            print(f"    OB time span: {ts_span_h:.2f}h")
            print(
                f"    Bid1 price range: [{ob_info['min_bid1']:.4f}, {ob_info['max_ask1']:.4f}]"
            )
            print(f"    Zero bid1 prices: {ob_info['zero_bid1']}")
            print(f"    Zero ask1 prices: {ob_info['zero_ask1']}")
            print(f"    Zero bid1 qty: {ob_info['zero_bid1_qty']}")
            print(f"    Zero ask1 qty: {ob_info['zero_ask1_qty']}")

            result.update(
                {
                    "ob_ts_span_h": float(ts_span_h),
                    "ob_zero_bid1": int(ob_info["zero_bid1"]),
                    "ob_zero_ask1": int(ob_info["zero_ask1"]),
                    "ob_zero_bid1_qty": int(ob_info["zero_bid1_qty"]),
                    "ob_zero_ask1_qty": int(ob_info["zero_ask1_qty"]),
                    "ob_min_bid1": float(ob_info["min_bid1"]),
                    "ob_max_ask1": float(ob_info["max_ask1"]),
                }
            )

            # Check for large timestamp gaps
            if n_ob > 1:
                gaps = (
                    con.execute(
                        f"""
                    SELECT MAX(gap_ms) as max_gap, SUM(CASE WHEN gap_ms > 300000 THEN 1 ELSE 0 END) as n_big_gaps
                    FROM (
                        SELECT ts_ms - LAG(ts_ms) OVER (ORDER BY ts_ms) as gap_ms
                        FROM read_parquet('{ob_glob}')
                    )
                    WHERE gap_ms IS NOT NULL
                """
                    )
                    .fetchdf()
                    .iloc[0]
                )
                max_gap_min = float(gaps["max_gap"]) / 60000
                n_big_gaps = int(gaps["n_big_gaps"])
                print(
                    f"    Max OB gap: {max_gap_min:.2f} min; gaps > 5min: {n_big_gaps}"
                )
                result["ob_max_gap_min"] = max_gap_min
                result["ob_n_big_gaps"] = n_big_gaps

            # KBONK-specific: check spread vs mid to verify log_spread = 0.074 claim
            if symbol == "KBONK":
                spread_check = (
                    con.execute(
                        f"""
                    SELECT
                        AVG((ask1_price - bid1_price) / ((ask1_price + bid1_price) / 2.0)) as avg_rel_spread,
                        MAX((ask1_price - bid1_price) / ((ask1_price + bid1_price) / 2.0)) as max_rel_spread,
                        SUM(CASE WHEN ask1_price > 2*bid1_price THEN 1 ELSE 0 END) as n_ask_gt_2x_bid,
                        SUM(CASE WHEN bid1_qty = 0 THEN 1 ELSE 0 END) as n_zero_bid_qty,
                        SUM(CASE WHEN ask1_qty = 0 THEN 1 ELSE 0 END) as n_zero_ask_qty
                    FROM read_parquet('{ob_glob}')
                    WHERE bid1_price > 0 AND ask1_price > 0
                """
                    )
                    .fetchdf()
                    .iloc[0]
                )
                print(
                    f"    KBONK avg rel-spread: {float(spread_check['avg_rel_spread']):.6f}"
                )
                print(
                    f"    KBONK max rel-spread: {float(spread_check['max_rel_spread']):.6f}"
                )
                print(f"    KBONK ask > 2x bid: {int(spread_check['n_ask_gt_2x_bid'])}")
                print(f"    KBONK zero bid qty: {int(spread_check['n_zero_bid_qty'])}")
                print(f"    KBONK zero ask qty: {int(spread_check['n_zero_ask_qty'])}")
                result["kbonk_avg_rel_spread"] = float(spread_check["avg_rel_spread"])
                result["kbonk_max_rel_spread"] = float(spread_check["max_rel_spread"])
                result["kbonk_n_ask_gt_2x_bid"] = int(spread_check["n_ask_gt_2x_bid"])
                result["kbonk_zero_bid_qty"] = int(spread_check["n_zero_bid_qty"])
                result["kbonk_zero_ask_qty"] = int(spread_check["n_zero_ask_qty"])

            # Check for L2+ levels being zero (one-sided book detection)
            level_check = (
                con.execute(
                    f"""
                SELECT
                    SUM(CASE WHEN bid10_qty = 0 OR bid10_qty IS NULL THEN 1 ELSE 0 END) as shallow_bid,
                    SUM(CASE WHEN ask10_qty = 0 OR ask10_qty IS NULL THEN 1 ELSE 0 END) as shallow_ask,
                    SUM(CASE WHEN bid1_qty = 0 AND bid2_qty = 0 THEN 1 ELSE 0 END) as empty_bid_book,
                    SUM(CASE WHEN ask1_qty = 0 AND ask2_qty = 0 THEN 1 ELSE 0 END) as empty_ask_book
                FROM read_parquet('{ob_glob}')
            """
                )
                .fetchdf()
                .iloc[0]
            )
            print(f"    Shallow bid (L10 qty=0): {int(level_check['shallow_bid'])}")
            print(f"    Shallow ask (L10 qty=0): {int(level_check['shallow_ask'])}")
            print(f"    Empty bid book (L1+L2=0): {int(level_check['empty_bid_book'])}")
            print(f"    Empty ask book (L1+L2=0): {int(level_check['empty_ask_book'])}")
            result.update(
                {
                    "ob_shallow_bid": int(level_check["shallow_bid"]),
                    "ob_shallow_ask": int(level_check["shallow_ask"]),
                    "ob_empty_bid": int(level_check["empty_bid_book"]),
                    "ob_empty_ask": int(level_check["empty_ask_book"]),
                }
            )

    except Exception as e:
        print(f"    OB error: {e}")
        result["ob_error"] = str(e)

    con.close()
    return result


# ---------------------------------------------------------------------------
# Part 3: Corpus-wide NaN feature attribution
# ---------------------------------------------------------------------------


def corpus_nan_feature_attribution(cache_dir: Path) -> pd.DataFrame:
    """Count NaN occurrences per feature column across ALL shards."""
    print(f"\n{'='*60}")
    print("CORPUS-WIDE NaN FEATURE ATTRIBUTION")
    print(f"{'='*60}")

    nan_counts = {fname: 0 for fname in FEATURE_NAMES}
    inf_counts = {fname: 0 for fname in FEATURE_NAMES}
    dirty_shards = []

    all_shards = sorted(cache_dir.glob("*.npz"))
    print(f"Scanning {len(all_shards)} shards...")

    for path in all_shards:
        try:
            with np.load(path, allow_pickle=False) as z:
                features = z["features"]
        except Exception:
            continue

        if not isinstance(features, np.ndarray) or features.ndim != 2:
            continue

        has_bad = False
        for i, fname in enumerate(FEATURE_NAMES):
            n_nan = int(np.isnan(features[:, i]).sum())
            n_inf = int(np.isinf(features[:, i]).sum())
            if n_nan > 0:
                nan_counts[fname] += n_nan
                has_bad = True
            if n_inf > 0:
                inf_counts[fname] += n_inf
                has_bad = True

        if has_bad:
            bad_row_idxs = np.where(~np.isfinite(features).all(axis=1))[0]
            n_bad = int((~np.isfinite(features)).sum())
            dirty_shards.append(
                {
                    "shard": path.name,
                    "n_events": features.shape[0],
                    "n_bad": n_bad,
                    "first_bad_row": int(bad_row_idxs[0]),
                    "last_bad_row": int(bad_row_idxs[-1]),
                    "prefix_only": bool(bad_row_idxs[-1] < 300),
                }
            )

    # Print per-feature breakdown
    df = pd.DataFrame(
        {
            "feature": list(FEATURE_NAMES),
            "idx": list(range(17)),
            "nan_count": [nan_counts[f] for f in FEATURE_NAMES],
            "inf_count": [inf_counts[f] for f in FEATURE_NAMES],
        }
    )
    df["total_bad"] = df["nan_count"] + df["inf_count"]
    df = df[df["total_bad"] > 0].sort_values("total_bad", ascending=False)

    print(f"\n{'Feature':<22} {'Col':>4} {'NaN':>8} {'Inf':>8} {'Total':>8}")
    print(f"{'-'*54}")
    for _, row in df.iterrows():
        print(
            f"  {row['feature']:<20} {int(row['idx']):>4} {int(row['nan_count']):>8} {int(row['inf_count']):>8} {int(row['total_bad']):>8}"
        )

    print(f"\nTotal dirty shards: {len(dirty_shards)}")
    print(
        f"\n{'Shard':<35} {'N_events':>9} {'N_bad':>7} {'FirstBad':>9} {'LastBad':>8} {'PrefixOnly':>11}"
    )
    print(f"{'-'*82}")

    # Show all dirty shards
    for s in dirty_shards:
        print(
            f"  {s['shard']:<33} {s['n_events']:>9,} {s['n_bad']:>7} {s['first_bad_row']:>9} {s['last_bad_row']:>8} {str(s['prefix_only']):>11}"
        )

    return df, dirty_shards


# ---------------------------------------------------------------------------
# Part 4: Deep dive — book_walk NaN trace
# ---------------------------------------------------------------------------


def trace_book_walk_nan(symbol: str, date: str) -> None:
    """Trace why book_walk might be NaN: spread=0, mid=0, or book_walk_abs=NaN."""
    print(f"\n  -- book_walk NaN trace: {symbol} {date} --")

    features = load_shard(symbol, date)
    if features is None:
        return

    col_idx = FEATURE_NAMES.index("book_walk")
    nan_rows = np.where(np.isnan(features[:, col_idx]))[0]

    if len(nan_rows) == 0:
        print("    book_walk: CLEAN in this shard")
        return

    print(f"    book_walk NaN at rows: {nan_rows[:20].tolist()}")

    # Look at spread and mid for those rows (cols 9=log_spread is available in cache)
    # We need to check if OB alignment returned NaN for pre-first-snapshot events
    log_spread_idx = FEATURE_NAMES.index("log_spread")
    trade_vs_mid_idx = FEATURE_NAMES.index("trade_vs_mid")

    print(f"    log_spread at NaN rows: {features[nan_rows[:5], log_spread_idx]}")
    print(f"    trade_vs_mid at NaN rows: {features[nan_rows[:5], trade_vs_mid_idx]}")
    print(f"    All OB features at first NaN row ({nan_rows[0]}):")
    for i in range(9, 17):
        print(f"      {FEATURE_NAMES[i]}: {features[nan_rows[0], i]:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("NaN CONTAMINATION ANALYSIS")
    print("=" * 60)

    # Part 1: Per-shard localization
    shard_results = []
    for symbol, date in TARGET_SHARDS:
        r = analyze_shard(symbol, date)
        shard_results.append(r)

    # Part 2: Raw data cross-reference for first 3 targets (Feb 2-6 shards)
    print(f"\n{'='*60}")
    print("RAW DATA CROSS-REFERENCE")
    print(f"{'='*60}")

    raw_checks = []
    for symbol, date in TARGET_SHARDS:
        r = check_raw_data(symbol, date)
        raw_checks.append(r)

    # Part 3: Corpus-wide attribution
    nan_df, dirty_shards = corpus_nan_feature_attribution(CACHE_DIR)

    # Part 4: Deep dive on book_walk specifically
    print(f"\n{'='*60}")
    print("BOOK_WALK NaN DEEP DIVE")
    print(f"{'='*60}")
    for symbol, date in TARGET_SHARDS:
        trace_book_walk_nan(symbol, date)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY FINDINGS")
    print(f"{'='*60}")

    print("\nPer-shard NaN column breakdown:")
    for r in shard_results:
        if r.get("clean"):
            print(f"  {r['symbol']}__{r['date']}: CLEAN")
        elif "error" in r:
            print(f"  {r['symbol']}__{r['date']}: ERROR ({r['error']})")
        else:
            print(f"  {r['symbol']}__{r['date']}: {dict(r['nan_per_col'])}")
            print(f"    Clustering: {dict(r.get('clustering', {}))}")

    print("\nRaw data summary:")
    for r in raw_checks:
        print(f"  {r['symbol']}__{r['date']}:")
        print(
            f"    Trades: {r.get('n_trades', 'ERR')}, OB: {r.get('n_ob_snapshots', 'ERR')} snaps"
        )
        if r.get("ob_max_gap_min"):
            print(
                f"    Max OB gap: {r['ob_max_gap_min']:.2f}min, gaps>5min: {r.get('ob_n_big_gaps', '?')}"
            )
        if r.get("ob_empty_bid") or r.get("ob_empty_ask"):
            print(
                f"    ALERT: empty bid={r.get('ob_empty_bid',0)}, empty ask={r.get('ob_empty_ask',0)}"
            )
        if r.get("ob_zero_bid1_qty") or r.get("ob_zero_ask1_qty"):
            print(
                f"    ALERT: zero bid1_qty={r.get('ob_zero_bid1_qty',0)}, zero ask1_qty={r.get('ob_zero_ask1_qty',0)}"
            )


if __name__ == "__main__":
    main()

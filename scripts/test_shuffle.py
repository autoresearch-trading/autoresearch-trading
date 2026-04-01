#!/usr/bin/env python3
"""Test 1.1: Does trade ordering within a batch matter?

Computes features normally vs with shuffled trades within each batch.
Compares feature-return correlations to determine if sequence order
carries predictive information beyond what summary statistics capture.

Uses raw trade data across all dates for 8 representative symbols.
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from prepare import (
    TEST_END,
    TRAIN_START,
    compute_features_v9,
    load_funding,
    load_orderbook,
    load_trades,
    normalize_features_v9,
)

SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "DOGE",
    "LINK",
    "HYPE",
    "KPEPE",
    "2Z",
]  # mix of liquidity levels
TRADE_BATCH = 100


def compute_correlations(features, prices):
    """Compute correlation between each feature and next-step return."""
    returns = np.zeros(len(prices))
    returns[1:] = np.diff(np.log(np.clip(prices, 1e-10, None)))
    next_returns = np.roll(returns, -1)
    next_returns[-1] = 0

    correlations = {}
    for i in range(features.shape[1]):
        feat = features[:, i]
        # Remove NaN/inf
        valid = np.isfinite(feat) & np.isfinite(next_returns)
        if valid.sum() > 100:
            correlations[i] = np.corrcoef(feat[valid], next_returns[valid])[0, 1]
        else:
            correlations[i] = 0.0
    return correlations


def shuffle_within_batches(trades_df, trade_batch):
    """Shuffle trades randomly within each batch, preserving batch boundaries."""
    df = trades_df.copy()
    n = len(df)
    num_batches = n // trade_batch
    indices = np.arange(num_batches * trade_batch)

    for b in range(num_batches):
        start = b * trade_batch
        end = start + trade_batch
        batch_idx = indices[start:end].copy()
        np.random.shuffle(batch_idx)
        indices[start:end] = batch_idx

    return df.iloc[indices].reset_index(drop=True)


def run_test():
    print("=" * 60)
    print("TEST 1.1: SHUFFLE TEST")
    print("Does trade ordering within a batch matter?")
    print("=" * 60)
    print()

    all_normal_corrs = []
    all_shuffled_corrs = []

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        trades = load_trades(symbol, TRAIN_START, TEST_END)
        orderbook = load_orderbook(symbol, TRAIN_START, TEST_END)
        funding = load_funding(symbol, TRAIN_START, TEST_END)

        if trades.empty:
            print(f"  No trades for {symbol}")
            continue

        # Normal features
        features_n, ts_n, prices_n, _, _ = compute_features_v9(
            trades, orderbook, funding, TRADE_BATCH
        )
        if len(features_n) == 0:
            continue
        features_n = normalize_features_v9(features_n)
        corrs_normal = compute_correlations(features_n, prices_n)

        # Shuffled features (3 runs, average to reduce noise)
        corrs_shuffled_runs = []
        for seed in range(3):
            np.random.seed(seed)
            trades_shuffled = shuffle_within_batches(trades, TRADE_BATCH)
            features_s, ts_s, prices_s, _, _ = compute_features_v9(
                trades_shuffled, orderbook, funding, TRADE_BATCH
            )
            if len(features_s) == 0:
                continue
            features_s = normalize_features_v9(features_s)
            corrs_shuffled_runs.append(compute_correlations(features_s, prices_s))

        if not corrs_shuffled_runs:
            continue

        # Average shuffled correlations
        corrs_shuffled = {}
        for i in corrs_normal:
            corrs_shuffled[i] = np.mean([r.get(i, 0) for r in corrs_shuffled_runs])

        feature_names = [
            "lambda_ofi",
            "dir_conviction",
            "vpin",
            "hawkes",
            "resv_price",
            "vol_of_vol",
            "utc_hour",
            "microprice",
            "delta_tfi",
            "mlofi",
            "buy_vwap",
            "arrival_rate",
            "r_20",
        ]

        print(
            f"  {'Feature':<18} {'Normal':>10} {'Shuffled':>10} {'Diff':>10} {'Signal?':>8}"
        )
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        symbol_normal = []
        symbol_shuffled = []
        for i in sorted(corrs_normal.keys()):
            n_corr = corrs_normal[i]
            s_corr = corrs_shuffled.get(i, 0)
            diff = abs(n_corr) - abs(s_corr)
            name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
            signal = "YES" if diff > 0.001 else "no"
            print(
                f"  {name:<18} {n_corr:>10.6f} {s_corr:>10.6f} {diff:>10.6f} {signal:>8}"
            )
            symbol_normal.append(abs(n_corr))
            symbol_shuffled.append(abs(s_corr))

        mean_normal = np.mean(symbol_normal)
        mean_shuffled = np.mean(symbol_shuffled)
        print(
            f"  {'MEAN |corr|':<18} {mean_normal:>10.6f} {mean_shuffled:>10.6f} {mean_normal - mean_shuffled:>10.6f}"
        )

        all_normal_corrs.append(mean_normal)
        all_shuffled_corrs.append(mean_shuffled)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL SYMBOLS")
    print("=" * 60)
    mean_n = np.mean(all_normal_corrs)
    mean_s = np.mean(all_shuffled_corrs)
    diff = mean_n - mean_s
    pct = (diff / mean_n * 100) if mean_n > 0 else 0

    print(f"  Mean |correlation| normal:   {mean_n:.6f}")
    print(f"  Mean |correlation| shuffled:  {mean_s:.6f}")
    print(f"  Difference:                   {diff:.6f} ({pct:.1f}%)")
    print()

    if diff > 0.002:
        print(
            "  VERDICT: SEQUENCE MATTERS — trade order carries predictive information"
        )
        print("           that current features partially capture.")
    elif diff > 0.0005:
        print("  VERDICT: MARGINAL — small signal from sequence order.")
        print("           May not justify sequential model complexity.")
    else:
        print("  VERDICT: SEQUENCE DOESN'T MATTER — shuffling has no effect.")
        print("           Current summary statistics capture all available signal.")
        print("           Sequential models will NOT help.")


if __name__ == "__main__":
    run_test()

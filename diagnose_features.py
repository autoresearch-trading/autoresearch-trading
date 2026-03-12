#!/usr/bin/env python3
"""Diagnose v3 feature quality: NaN/inf, normalization, predictive signal."""

import numpy as np
import pandas as pd

from prepare import (
    CACHE_DIR,
    DEFAULT_SYMBOLS,
    TRAIN_END,
    TRAIN_START,
    compute_features,
    load_cached,
    load_funding,
    load_orderbook,
    load_trades,
    normalize_features,
)

FEATURE_NAMES = [
    "returns",
    "r_5",
    "r_20",
    "r_100",
    "realvol_10",
    "bipower_var_20",
    "tfi",
    "volume_spike_ratio",
    "large_trade_share",
    "kyle_lambda_50",
    "amihud_illiq_50",
    "trade_arrival_rate",
    "spread_bps",
    "log_total_depth",
    "weighted_imbalance_5lvl",
    "microprice_dev",
    "ofi",
    "ob_slope_asym",
    "funding_zscore",
    "utc_hour_linear",
]


def diagnose_symbol(symbol: str, use_cache: bool = True):
    """Run full diagnostics on one symbol's train split."""
    print(f"\n{'='*60}")
    print(f"  DIAGNOSING: {symbol}")
    print(f"{'='*60}")

    # Load raw features (skip cache to get pre-normalization)
    if not use_cache:
        print("Loading raw data...")
        trades = load_trades(symbol, TRAIN_START, TRAIN_END)
        ob = load_orderbook(symbol, TRAIN_START, TRAIN_END)
        funding = load_funding(symbol, TRAIN_START, TRAIN_END)
        print(f"  trades={len(trades)}, ob={len(ob)}, funding={len(funding)}")

        raw_features, timestamps, prices = compute_features(trades, ob, funding)
        if len(raw_features) == 0:
            print("  NO DATA")
            return None
    else:
        # Load cached (already normalized)
        cached = load_cached(symbol, CACHE_DIR, TRAIN_START, TRAIN_END, 100)
        if cached is None:
            print("  NO CACHE - computing raw...")
            return diagnose_symbol(symbol, use_cache=False)
        raw_features, timestamps, prices = cached
        print(f"  Loaded from cache (already normalized): shape={raw_features.shape}")

    print(f"\n  Shape: {raw_features.shape}")
    print(f"  Dtype: {raw_features.dtype}")

    # --- Check 1: NaN / Inf ---
    print(f"\n  --- NaN/Inf Check ---")
    nan_count = np.isnan(raw_features).sum(axis=0)
    inf_count = np.isinf(raw_features).sum(axis=0)
    n = len(raw_features)
    for i, name in enumerate(FEATURE_NAMES):
        if i >= raw_features.shape[1]:
            break
        if nan_count[i] > 0 or inf_count[i] > 0:
            print(
                f"  *** {name}: {nan_count[i]} NaN, {inf_count[i]} Inf ({nan_count[i]/n*100:.1f}%)"
            )

    total_nan = np.isnan(raw_features).sum()
    total_inf = np.isinf(raw_features).sum()
    if total_nan == 0 and total_inf == 0:
        print(f"  OK - no NaN or Inf")
    else:
        print(f"  TOTAL: {total_nan} NaN, {total_inf} Inf")

    # --- Check 2: Feature Statistics ---
    print(f"\n  --- Feature Statistics ---")
    print(
        f"  {'Feature':<25s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'zeros%':>8s}"
    )
    for i, name in enumerate(FEATURE_NAMES):
        if i >= raw_features.shape[1]:
            break
        col = raw_features[:, i]
        zeros_pct = (col == 0).sum() / n * 100
        print(
            f"  {name:<25s} {col.mean():>10.6f} {col.std():>10.6f} {col.min():>10.4f} {col.max():>10.4f} {zeros_pct:>7.1f}%"
        )

    # --- Check 3: Constant or near-constant features ---
    print(f"\n  --- Degenerate Features ---")
    for i, name in enumerate(FEATURE_NAMES):
        if i >= raw_features.shape[1]:
            break
        col = raw_features[:, i]
        unique = len(np.unique(col[:1000]))  # sample first 1000
        if unique <= 3:
            print(f"  *** {name}: only {unique} unique values in first 1000 rows")
        if col.std() < 1e-10:
            print(f"  *** {name}: near-zero std ({col.std():.2e})")

    # --- Check 4: Predictive Signal (rank correlation with future 1-step return) ---
    print(f"\n  --- Predictive Signal (Spearman corr with next-step return) ---")
    # Future return = log(price[t+1] / price[t])
    future_ret = np.zeros(n)
    future_ret[:-1] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))

    # Also check 5-step and 20-step forward returns
    future_ret_5 = np.zeros(n)
    future_ret_20 = np.zeros(n)
    if n > 5:
        future_ret_5[:-5] = np.log(prices[5:] / np.maximum(prices[:-5], 1e-10))
    if n > 20:
        future_ret_20[:-20] = np.log(prices[20:] / np.maximum(prices[:-20], 1e-10))

    # Skip warmup (first 200 rows have rolling window artifacts)
    start = 200
    print(
        f"  {'Feature':<25s} {'corr_1':>8s} {'corr_5':>8s} {'corr_20':>8s} {'|corr_1|':>8s}"
    )
    signals = []
    for i, name in enumerate(FEATURE_NAMES):
        if i >= raw_features.shape[1]:
            break
        col = raw_features[start:, i]
        fr1 = future_ret[start:]
        fr5 = future_ret_5[start:]
        fr20 = future_ret_20[start:]

        # Spearman rank correlation
        from scipy.stats import spearmanr

        corr1, _ = spearmanr(col, fr1)
        corr5, _ = spearmanr(col, fr5)
        corr20, _ = spearmanr(col, fr20)
        corr1 = 0 if np.isnan(corr1) else corr1
        corr5 = 0 if np.isnan(corr5) else corr5
        corr20 = 0 if np.isnan(corr20) else corr20

        print(
            f"  {name:<25s} {corr1:>8.4f} {corr5:>8.4f} {corr20:>8.4f} {abs(corr1):>8.4f}"
        )
        signals.append((name, abs(corr1), abs(corr5), abs(corr20)))

    # Sort by absolute 1-step correlation
    signals.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  --- Top Features by |corr_1| ---")
    for name, c1, c5, c20 in signals[:10]:
        print(f"  {name:<25s} {c1:.4f}  (5-step: {c5:.4f}, 20-step: {c20:.4f})")

    return raw_features


def compare_normalization():
    """Check if normalization is destroying signal."""
    symbol = "BTC"
    print(f"\n{'='*60}")
    print(f"  NORMALIZATION CHECK: {symbol}")
    print(f"{'='*60}")

    # Load raw (uncached)
    trades = load_trades(symbol, TRAIN_START, TRAIN_END)
    ob = load_orderbook(symbol, TRAIN_START, TRAIN_END)
    funding = load_funding(symbol, TRAIN_START, TRAIN_END)
    raw, ts, prices = compute_features(trades, ob, funding)

    if len(raw) == 0:
        print("No data")
        return

    normed = normalize_features(raw)

    print(f"  Raw shape: {raw.shape}, Normalized shape: {normed.shape}")

    # Check NaN/inf after normalization
    nan_after = np.isnan(normed).sum()
    inf_after = np.isinf(normed).sum()
    print(f"  After normalization: {nan_after} NaN, {inf_after} Inf")

    # Compare predictive signal before and after normalization
    from scipy.stats import spearmanr

    n = len(raw)
    future_ret = np.zeros(n)
    future_ret[:-1] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))
    start = 200

    print(f"\n  {'Feature':<25s} {'raw_corr':>10s} {'norm_corr':>10s} {'delta':>8s}")
    for i, name in enumerate(FEATURE_NAMES):
        if i >= raw.shape[1]:
            break
        raw_corr, _ = spearmanr(raw[start:, i], future_ret[start:])
        norm_corr, _ = spearmanr(normed[start:, i], future_ret[start:])
        raw_corr = 0 if np.isnan(raw_corr) else raw_corr
        norm_corr = 0 if np.isnan(norm_corr) else norm_corr
        delta = norm_corr - raw_corr
        flag = " ***" if abs(delta) > 0.01 else ""
        print(f"  {name:<25s} {raw_corr:>10.4f} {norm_corr:>10.4f} {delta:>8.4f}{flag}")


if __name__ == "__main__":
    # 1. Diagnose cached (normalized) features for key symbols
    for sym in ["BTC", "ETH", "SOL", "DOGE", "HYPE"]:
        diagnose_symbol(sym, use_cache=True)

    # 2. Check normalization impact on BTC
    compare_normalization()

#!/usr/bin/env python3
"""T47: Measure signal autocorrelation to derive optimal window size.

Computes autocorrelation of features × forward returns to estimate
how quickly predictive signal decays. The optimal window captures
signal up to the decorrelation lag and no further (beyond that is noise).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, make_env

EXCLUDED = {"CRV", "XPL"}
# Representative symbols across liquidity tiers
SAMPLE_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "AAVE", "HYPE", "LINK", "SUI"]


def autocorrelation(x, max_lag=200):
    """Compute autocorrelation of x for lags 0..max_lag."""
    x = x - x.mean()
    n = len(x)
    var = np.var(x)
    if var < 1e-20:
        return np.zeros(max_lag + 1)
    acf = np.correlate(x, x, mode="full")
    acf = acf[n - 1 : n + max_lag]  # lags 0..max_lag-1
    return acf / (var * n)


def feature_return_correlation(features, prices, feature_idx, max_lag=200):
    """Cross-correlation between feature[t] and return[t+lag]."""
    returns = np.diff(prices) / prices[:-1]
    returns = returns[: len(features) - 1]
    feat = features[: len(returns), feature_idx]

    # Standardize
    feat = (feat - feat.mean()) / (feat.std() + 1e-10)
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)

    n = len(returns)
    xcorr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= n:
            break
        xcorr[lag] = np.mean(feat[: n - lag] * returns[lag:])
    return xcorr


def main():
    max_lag = 200
    n_features = None

    # Collect autocorrelation of features and feature-return cross-correlation
    all_feat_acf = []
    all_feat_ret_xcorr = []

    for sym in SAMPLE_SYMBOLS:
        if sym in EXCLUDED:
            continue
        print(f"Processing {sym}...")
        env = make_env(sym, "train", window_size=50, trade_batch=100, min_hold=1200)
        features = env.features
        prices = env.prices

        if n_features is None:
            n_features = features.shape[1]

        # Feature autocorrelation: how persistent is each feature?
        sym_acf = np.zeros((n_features, max_lag + 1))
        for fi in range(n_features):
            sym_acf[fi] = autocorrelation(features[:, fi], max_lag)
        all_feat_acf.append(sym_acf)

        # Feature-return cross-correlation: how long does predictive power last?
        sym_xcorr = np.zeros((n_features, max_lag))
        for fi in range(n_features):
            sym_xcorr[fi] = feature_return_correlation(features, prices, fi, max_lag)
        all_feat_ret_xcorr.append(sym_xcorr)

    # Average across symbols
    mean_acf = np.mean(all_feat_acf, axis=0)  # (n_features, max_lag+1)
    mean_xcorr = np.mean(all_feat_ret_xcorr, axis=0)  # (n_features, max_lag)

    # Feature names (from CLAUDE.md)
    feature_names = [
        "spread_bps",
        "log_depth",
        "w_imbal_5",
        "microprice_dev",
        "mlofi",
        "ob_slope_asym",
        "buy_vwap_dev",
        "sell_vwap_dev",
        "tfi",
        "vol_spike",
        "delta_tfi",
        "arrival_rate",
        "vpin",
    ]
    if n_features and len(feature_names) < n_features:
        feature_names.extend(
            [f"feat_{i}" for i in range(len(feature_names), n_features)]
        )

    print(f"\n{'='*80}")
    print("T47: SIGNAL AUTOCORRELATION ANALYSIS")
    print(f"{'='*80}")

    # 1. Feature persistence: at what lag does ACF drop below 0.5?
    print(f"\n--- Feature Persistence (lag where ACF < 0.5) ---")
    print(
        f"{'Feature':<18} {'Half-life':>10} {'ACF@10':>8} {'ACF@50':>8} {'ACF@100':>8}"
    )
    print("-" * 60)

    half_lives = []
    for fi in range(min(n_features, len(feature_names))):
        acf = mean_acf[fi]
        # Find half-life (first lag where ACF < 0.5)
        hl = max_lag
        for lag in range(1, max_lag + 1):
            if acf[lag] < 0.5:
                hl = lag
                break
        half_lives.append(hl)
        print(
            f"{feature_names[fi]:<18} {hl:>10} "
            f"{acf[10]:>8.3f} {acf[50]:>8.3f} {acf[min(100, max_lag)]:>8.3f}"
        )

    print(f"\nMedian feature half-life: {np.median(half_lives):.0f} steps")
    print(f"Mean feature half-life: {np.mean(half_lives):.0f} steps")

    # 2. Predictive power decay: at what lag does |xcorr| drop below noise?
    # Noise threshold: 2/sqrt(N) for 95% confidence
    N_approx = 50000  # approximate steps per symbol
    noise_threshold = 2 / np.sqrt(N_approx)

    print(
        f"\n--- Predictive Power Decay (lag where |xcorr| < noise={noise_threshold:.4f}) ---"
    )
    print(
        f"{'Feature':<18} {'Decay lag':>10} {'|xcorr|@1':>10} {'|xcorr|@10':>10} {'|xcorr|@50':>10}"
    )
    print("-" * 65)

    decay_lags = []
    peak_xcorrs = []
    for fi in range(min(n_features, len(feature_names))):
        xc = np.abs(mean_xcorr[fi])
        peak = xc.max()
        peak_xcorrs.append(peak)
        # Find decay lag (first lag after peak where xcorr < noise)
        decay = max_lag
        for lag in range(1, max_lag):
            if xc[lag] < noise_threshold:
                decay = lag
                break
        decay_lags.append(decay)
        print(
            f"{feature_names[fi]:<18} {decay:>10} "
            f"{xc[1]:>10.4f} {xc[min(10, max_lag-1)]:>10.4f} {xc[min(50, max_lag-1)]:>10.4f}"
        )

    print(f"\nMedian predictive decay lag: {np.median(decay_lags):.0f} steps")
    print(f"Mean predictive decay lag: {np.mean(decay_lags):.0f} steps")
    print(f"Max predictive decay lag: {np.max(decay_lags):.0f} steps")

    # 3. Optimal window size recommendation
    print(f"\n{'='*80}")
    print("T47: OPTIMAL WINDOW SIZE")
    print(f"{'='*80}")

    # The window should capture the predictive signal but not add noise
    # Optimal window ≈ max of (median feature half-life, median predictive decay lag)
    optimal_window = max(int(np.median(half_lives)), int(np.median(decay_lags)))
    signal_window = int(np.percentile(decay_lags, 75))  # 75th percentile of decay

    print(f"  Current window_size: 50")
    print(f"  Median feature half-life: {np.median(half_lives):.0f} steps")
    print(f"  Median predictive decay: {np.median(decay_lags):.0f} steps")
    print(f"  75th percentile decay: {signal_window} steps")
    print(f"  Recommended window (median-based): {optimal_window} steps")
    print(f"  Recommended window (75th pctl): {signal_window} steps")

    if optimal_window < 30:
        print(
            f"\n  CONCLUSION: Signal decays FAST. Window=50 may include too much noise."
        )
        print(f"  Try window_size={max(10, optimal_window)} (shorter = less noise)")
    elif optimal_window <= 70:
        print(f"\n  CONCLUSION: Window=50 is WELL-MATCHED to signal persistence.")
        print(f"  Current setting is near-optimal.")
    else:
        print(f"\n  CONCLUSION: Signal persists LONGER than window=50 captures.")
        print(f"  Try window_size={min(optimal_window, 200)} (longer = more signal)")

    # 4. Suggested sweep values
    sweep_values = sorted(
        set(
            [
                max(10, optimal_window // 2),
                optimal_window,
                min(200, int(signal_window * 1.5)),
                50,  # current
            ]
        )
    )
    print(f"\n  Suggested sweep values: {sweep_values}")


if __name__ == "__main__":
    main()

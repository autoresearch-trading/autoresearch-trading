#!/usr/bin/env python3
"""Measure autocorrelation of raw trade features to find natural sequence length.

Computes ACF of is_buy, log_qty, inter-trade time, and log_return at the
individual trade level. The lag where ACF hits zero tells us how long
microstructure patterns persist in the raw tape.
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from prepare import TEST_END, TRAIN_START, load_trades

SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "LINK", "HYPE", "KPEPE", "2Z"]
MAX_LAG = 500  # check up to 500 trades
SAMPLE_DAYS = 10  # use 10 days to keep it fast


def autocorrelation(x, max_lag):
    """Compute ACF for lags 1 to max_lag."""
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag)
    acf = np.correlate(x[:10000], x[:10000], mode="full")
    mid = len(acf) // 2
    acf = acf[mid : mid + max_lag + 1] / (var * len(x[:10000]))
    return acf[1:]  # skip lag 0


def find_zero_crossing(acf):
    """Find first lag where ACF crosses zero."""
    for i, v in enumerate(acf):
        if v <= 0:
            return i + 1
    return len(acf)


def find_half_life(acf):
    """Find lag where ACF drops below 0.5 * ACF[0]."""
    if len(acf) == 0:
        return 0
    threshold = acf[0] * 0.5
    for i, v in enumerate(acf):
        if v < threshold:
            return i + 1
    return len(acf)


def run():
    print("=" * 70)
    print("RAW TRADE AUTOCORRELATION ANALYSIS")
    print("Finding natural sequence length for tape reading")
    print("=" * 70)

    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        # Load a subset of dates for speed
        trades = load_trades(symbol, "2026-01-01", "2026-01-11")
        if trades.empty or len(trades) < 20000:
            print(f"  Not enough trades")
            continue

        # Compute raw features
        prices = trades["price"].values.astype(float)
        qtys = trades["qty"].values.astype(float)
        ts = trades["ts_ms"].values.astype(float)
        sides = trades["side"].str.lower().values

        is_buy = np.array(
            [1.0 if s in ("open_long", "close_short") else 0.0 for s in sides]
        )
        is_open = np.array([1.0 if s.startswith("open_") else 0.0 for s in sides])
        log_return = np.zeros(len(prices))
        log_return[1:] = np.diff(np.log(np.clip(prices, 1e-10, None)))
        log_qty = np.log(np.clip(qtys, 1e-10, None))
        time_delta = np.zeros(len(ts))
        time_delta[1:] = np.diff(ts)

        features = {
            "is_buy": is_buy,
            "is_open": is_open,
            "log_return": log_return,
            "log_qty": log_qty,
            "time_delta": time_delta,
        }

        print(
            f"  {'Feature':<15} {'Half-life':>10} {'Zero-cross':>12} {'ACF@10':>8} {'ACF@50':>8} {'ACF@100':>8} {'ACF@200':>8}"
        )
        print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        symbol_results = {}
        for name, data in features.items():
            acf = autocorrelation(data, MAX_LAG)
            hl = find_half_life(acf)
            zc = find_zero_crossing(acf)
            acf_10 = acf[9] if len(acf) > 9 else 0
            acf_50 = acf[49] if len(acf) > 49 else 0
            acf_100 = acf[99] if len(acf) > 99 else 0
            acf_200 = acf[199] if len(acf) > 199 else 0
            print(
                f"  {name:<15} {hl:>10} {zc:>12} {acf_10:>8.4f} {acf_50:>8.4f} {acf_100:>8.4f} {acf_200:>8.4f}"
            )
            symbol_results[name] = {"half_life": hl, "zero_cross": zc}

        all_results[symbol] = symbol_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RECOMMENDED SEQUENCE LENGTH")
    print("=" * 70)

    for feature in ["is_buy", "is_open", "log_return", "log_qty", "time_delta"]:
        half_lives = [
            r[feature]["half_life"] for r in all_results.values() if feature in r
        ]
        zero_crosses = [
            r[feature]["zero_cross"] for r in all_results.values() if feature in r
        ]
        if half_lives:
            print(
                f"  {feature:<15} half-life: mean={np.mean(half_lives):.0f}, median={np.median(half_lives):.0f}, max={np.max(half_lives)}"
            )
            print(
                f"  {'':<15} zero-cross: mean={np.mean(zero_crosses):.0f}, median={np.median(zero_crosses):.0f}, max={np.max(zero_crosses)}"
            )

    all_zero_crosses = []
    for r in all_results.values():
        for f in r.values():
            all_zero_crosses.append(f["zero_cross"])

    if all_zero_crosses:
        recommended = int(np.percentile(all_zero_crosses, 90))
        print(f"\n  RECOMMENDATION: sequence length = {recommended} trades")
        print(
            f"  (90th percentile of zero-crossing lags across all features and symbols)"
        )
        print(f"  This captures the natural microstructure event timescale.")


if __name__ == "__main__":
    run()

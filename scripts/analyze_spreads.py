#!/usr/bin/env python3
"""Analyze per-symbol spread data to determine profitability thresholds (T40).

For each symbol, compute:
- Median spread (bps) from orderbook data
- Total one-way cost = FEE_BPS + half_spread + 3 bps impact
- Round-trip cost
- Whether the symbol is tradeable given observed alpha
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from prepare import DEFAULT_SYMBOLS, FEE_BPS, make_env

IMPACT_BUFFER_BPS = 3.0


def main():
    print(
        f"{'Symbol':<12} {'Med Spread':>11} {'Half Sprd':>10} {'1-Way Cost':>11} {'RT Cost':>8} {'Tier':>8}"
    )
    print("-" * 70)

    results = []
    for sym in DEFAULT_SYMBOLS:
        try:
            env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=800)
            if env.spread_bps is not None and len(env.spread_bps) > 0:
                nonzero = env.spread_bps[env.spread_bps > 0]
                if len(nonzero) > 0:
                    median_spread = np.median(nonzero)
                    half_spread = median_spread / 2.0
                    one_way = FEE_BPS + half_spread + IMPACT_BUFFER_BPS
                    round_trip = 2 * one_way
                    tier = (
                        "TIGHT"
                        if median_spread < 10
                        else "MID" if median_spread < 25 else "WIDE"
                    )
                    results.append(
                        (sym, median_spread, half_spread, one_way, round_trip, tier)
                    )
                    print(
                        f"{sym:<12} {median_spread:>9.1f} bp {half_spread:>8.1f} bp {one_way:>9.1f} bp {round_trip:>6.1f} bp {tier:>8}"
                    )
                    continue
            print(
                f"{sym:<12} {'N/A':>11} {'N/A':>10} {'N/A':>11} {'N/A':>8} {'NO DATA':>8}"
            )
            results.append((sym, 999, 999, 999, 999, "NO DATA"))
        except Exception as e:
            print(f"{sym:<12} ERROR: {e}")
            results.append((sym, 999, 999, 999, 999, "ERROR"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tight = [r for r in results if r[5] == "TIGHT"]
    mid = [r for r in results if r[5] == "MID"]
    wide = [r for r in results if r[5] == "WIDE"]

    print(f"\nTIGHT (<10 bps spread, RT cost <16 bps): {len(tight)} symbols")
    for r in tight:
        print(f"  {r[0]}: spread={r[1]:.1f} bps, RT={r[4]:.1f} bps")

    print(f"\nMID (10-25 bps spread, RT cost 16-30 bps): {len(mid)} symbols")
    for r in mid:
        print(f"  {r[0]}: spread={r[1]:.1f} bps, RT={r[4]:.1f} bps")

    print(f"\nWIDE (>25 bps spread, RT cost >30 bps): {len(wide)} symbols")
    for r in wide:
        print(f"  {r[0]}: spread={r[1]:.1f} bps, RT={r[4]:.1f} bps")

    # T40 recommendation
    print("\n" + "=" * 70)
    print("T40 RECOMMENDATION")
    print("=" * 70)
    print("With observed alpha ~0.045 Sortino (thin edge), only symbols with")
    print("tight spreads have a realistic chance of profitability.")
    print(f"Recommended tradeable set: {[r[0] for r in tight + mid]}")
    print(f"Exclude (spread too wide): {[r[0] for r in wide]}")


if __name__ == "__main__":
    main()

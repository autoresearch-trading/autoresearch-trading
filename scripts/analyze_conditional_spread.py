#!/usr/bin/env python3
"""Measure E[spread | high activity] vs E[spread] from our data (T43 prep)."""
import sys

import numpy as np

sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, make_env

EXCLUDED = {"CRV", "XPL"}


def main():
    ratios = []
    for sym in [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED]:
        env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=1200)
        if env.spread_bps is None:
            continue
        spread = env.spread_bps
        nonzero = spread > 0

        # Use arrival rate feature (index 11) as activity proxy
        # High activity = top 25th percentile of arrival rate
        arrival = env.features[:, 11]  # trade_arrival_rate
        p75 = np.percentile(arrival[arrival != 0], 75) if (arrival != 0).any() else 0

        high_activity = arrival >= p75
        low_activity = ~high_activity

        spread_high = spread[high_activity & nonzero]
        spread_low = spread[low_activity & nonzero]
        spread_all = spread[nonzero]

        if len(spread_high) > 0 and len(spread_all) > 0:
            ratio = np.median(spread_high) / np.median(spread_all)
            ratios.append(ratio)
            print(
                f"{sym:<12} E[spread|high]={np.median(spread_high):.1f}bps  "
                f"E[spread]={np.median(spread_all):.1f}bps  ratio={ratio:.2f}x"
            )

    if ratios:
        print(f"\nMean conditional ratio: {np.mean(ratios):.2f}x")
        print(f"Median conditional ratio: {np.median(ratios):.2f}x")
        print(
            f"This means our slippage model should multiply spread by "
            f"~{np.mean(ratios):.1f}x during signal periods"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute correlation matrix of symbol returns, normal vs stress (T44 prep)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, make_env

EXCLUDED = {"CRV", "XPL"}


def main():
    returns = {}
    for sym in [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED]:
        env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=1200)
        px = env.prices
        ret = np.diff(np.log(np.clip(px, 1e-10, None)))
        returns[sym] = ret[: min(len(ret), 50000)]  # align lengths

    # Align to shortest
    min_len = min(len(v) for v in returns.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in returns.items()})

    # Full period correlation
    corr = df.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"Average pairwise correlation (full period): {avg_corr:.3f}")

    # Stress correlation (worst 5% of BTC returns)
    if "BTC" in df.columns:
        btc_ret = df["BTC"]
        stress_mask = btc_ret <= btc_ret.quantile(0.05)
        stress_corr = df[stress_mask].corr()
        avg_stress = stress_corr.values[
            np.triu_indices_from(stress_corr.values, k=1)
        ].mean()
        print(f"Average pairwise correlation (BTC worst 5%): {avg_stress:.3f}")

        # T44 formula: portfolio DD ≈ single DD × sqrt(1 + (n-1)*rho) / sqrt(n)
        n = len(returns)
        for rho, label in [(avg_corr, "normal"), (avg_stress, "stress")]:
            multiplier = np.sqrt(1 + (n - 1) * rho) / np.sqrt(n)
            print(f"\nT44 ({label}, rho={rho:.3f}, n={n}):")
            print(f"  DD multiplier vs independent: {multiplier:.2f}x")
            print(f"  If per-symbol DD = 20%: portfolio DD = {20 * multiplier:.1f}%")
            effective_n = n / (1 + (n - 1) * rho)
            print(f"  Effective independent bets: {effective_n:.1f}")


if __name__ == "__main__":
    main()

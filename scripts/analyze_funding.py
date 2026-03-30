#!/usr/bin/env python3
"""Analyze funding rate distributions from our Pacifica data (T42 prep).

Samples 5 days per symbol to avoid loading millions of tiny Parquet files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")
SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "HYPE", "FARTCOIN", "AAVE", "LTC"]
# Sample 5 days spread across the date range
SAMPLE_DATES = ["2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]


def load_funding_sample(symbol, dates):
    """Load funding from a few sample dates using pyarrow dataset."""
    dfs = []
    for date in dates:
        path = DATA_ROOT / "funding" / f"symbol={symbol}" / f"date={date}"
        if not path.exists():
            continue
        files = sorted(path.glob("*.parquet"))
        if not files:
            continue
        # Read all files in directory at once via pyarrow
        df = pd.read_parquet(path)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    all_rates = []
    for sym in SYMBOLS:
        df = load_funding_sample(sym, SAMPLE_DATES)
        if df.empty or "rate" not in df.columns:
            print(f"{sym:<12} (no funding data)")
            continue
        rates = df["rate"].values
        nonzero = rates[rates != 0]
        if len(nonzero) == 0:
            print(f"{sym:<12} (all zero)")
            continue
        median_bps = np.median(nonzero) * 10000
        p5_bps = np.percentile(nonzero, 5) * 10000
        p95_bps = np.percentile(nonzero, 95) * 10000
        pct_positive = (nonzero > 0).mean() * 100
        print(
            f"{sym:<12} median={median_bps:>7.2f}bps  pos={pct_positive:>5.1f}%  "
            f"p5={p5_bps:>7.2f}bps  p95={p95_bps:>7.2f}bps  n={len(nonzero)}"
        )
        all_rates.extend(nonzero.tolist())

    if not all_rates:
        print("No funding data found")
        return

    all_rates = np.array(all_rates)
    median_hourly_bps = np.median(all_rates) * 10000
    print(
        f"\nAGGREGATE: median={median_hourly_bps:.2f}bps  "
        f"mean={np.mean(all_rates)*10000:.2f}bps  "
        f"pos={(all_rates > 0).mean()*100:.1f}%"
    )
    print(f"\nFunding cost at various hold durations (median rate):")
    print(f"  Per hour:               {abs(median_hourly_bps):.2f} bps")
    print(f"  Per 1200-step hold (~30m): {abs(median_hourly_bps) * 0.5:.2f} bps")
    print(f"  Per 300-step hold (~8m):   {abs(median_hourly_bps) * 0.13:.2f} bps")
    print(f"  Fee barrier (fee_mult=8):  80.00 bps")
    print(f"  Funding/Fee ratio (1hr):   {abs(median_hourly_bps) / 80:.4f}")


if __name__ == "__main__":
    main()

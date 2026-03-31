#!/usr/bin/env python3
"""Migrate v11b caches to v12: append r_btc_lag1 as feature #13.

For non-BTC symbols: aligns BTC per-batch log returns to the symbol's
timestamps using nearest-prior matching.
For BTC: sets feature to 0 (self-prediction is meaningless).

Usage: uv run python scripts/migrate_v11b_to_v12.py
"""

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
OLD_VERSION = "v11b"
NEW_VERSION = "v12"

SYMBOLS = [
    "2Z",
    "AAVE",
    "ASTER",
    "AVAX",
    "BNB",
    "BTC",
    "CRV",
    "DOGE",
    "ENA",
    "ETH",
    "FARTCOIN",
    "HYPE",
    "KBONK",
    "KPEPE",
    "LDO",
    "LINK",
    "LTC",
    "PENGU",
    "PUMP",
    "SOL",
    "SUI",
    "UNI",
    "WLFI",
    "XPL",
    "XRP",
]

SPLITS = {
    "train": ("2025-10-16", "2026-01-23"),
    "val": ("2026-01-23", "2026-02-17"),
    "test": ("2026-02-17", "2026-03-25"),
}


def cache_key(symbol, start, end, version):
    key = f"{symbol}_{start}_{end}_100_{version}"
    return hashlib.md5(key.encode()).hexdigest()


def load_btc_returns(split_name, start, end):
    """Load BTC cache and compute per-batch log returns."""
    h = cache_key("BTC", start, end, OLD_VERSION)
    path = CACHE_DIR / f"BTC_{h}.npz"
    if not path.exists():
        print(f"  WARNING: BTC cache missing for {split_name} ({path})")
        return None, None
    data = np.load(path)
    prices = data["prices"]
    timestamps = data["timestamps"]
    returns = np.zeros(len(prices))
    returns[1:] = np.diff(np.log(np.clip(prices, 1e-10, None)))
    return timestamps, returns


def align_btc_returns(symbol_timestamps, btc_timestamps, btc_returns):
    """Align BTC returns to symbol timestamps using nearest-prior matching."""
    # For each symbol timestamp, find the index of the last BTC batch at or before it
    indices = np.searchsorted(btc_timestamps, symbol_timestamps, side="right") - 1
    # Clamp to valid range
    indices = np.clip(indices, 0, len(btc_returns) - 1)
    return btc_returns[indices]


def migrate():
    migrated = 0
    skipped = 0
    missing = 0

    for split_name, (start, end) in SPLITS.items():
        print(f"\n=== {split_name} ({start} to {end}) ===")

        btc_ts, btc_ret = load_btc_returns(split_name, start, end)
        if btc_ts is None:
            print(f"  Skipping {split_name} — no BTC cache")
            missing += 1
            continue

        for symbol in SYMBOLS:
            old_h = cache_key(symbol, start, end, OLD_VERSION)
            old_path = CACHE_DIR / f"{symbol}_{old_h}.npz"
            new_h = cache_key(symbol, start, end, NEW_VERSION)
            new_path = CACHE_DIR / f"{symbol}_{new_h}.npz"

            if new_path.exists():
                skipped += 1
                continue

            if not old_path.exists():
                print(f"  {symbol}: no v11b cache, skipping")
                missing += 1
                continue

            data = np.load(old_path)
            features = data["features"]
            timestamps = data["timestamps"]
            prices = data["prices"]
            raw_hawkes = data["raw_hawkes"] if "raw_hawkes" in data else None
            spread_bps = data["spread_bps"] if "spread_bps" in data else None

            if len(features) == 0:
                # Empty cache — just copy with new key
                save_dict = dict(
                    features=features, timestamps=timestamps, prices=prices
                )
                if raw_hawkes is not None:
                    save_dict["raw_hawkes"] = raw_hawkes
                if spread_bps is not None:
                    save_dict["spread_bps"] = spread_bps
                np.savez_compressed(new_path, **save_dict)
                migrated += 1
                continue

            # Compute r_btc_lag1
            if symbol == "BTC":
                r_btc = np.zeros(len(features))
            else:
                r_btc = align_btc_returns(timestamps, btc_ts, btc_ret)

            # Append as new column
            new_features = np.column_stack([features, r_btc])

            save_dict = dict(
                features=new_features, timestamps=timestamps, prices=prices
            )
            if raw_hawkes is not None:
                save_dict["raw_hawkes"] = raw_hawkes
            if spread_bps is not None:
                save_dict["spread_bps"] = spread_bps
            np.savez_compressed(new_path, **save_dict)
            print(f"  {symbol}: {features.shape} -> {new_features.shape}")
            migrated += 1

    print(f"\nDone: {migrated} migrated, {skipped} already existed, {missing} missing")


if __name__ == "__main__":
    migrate()

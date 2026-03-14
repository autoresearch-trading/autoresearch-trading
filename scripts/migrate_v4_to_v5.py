#!/usr/bin/env python3
"""Migrate v4 caches (25 features) to v5 (31 features) without raw Parquet.

Loads existing .npz caches, computes 6 new features from cached prices and
normalized feature columns, normalizes the new columns, and saves as v5.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare import (
    CACHE_DIR,
    DEFAULT_SYMBOLS,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
)

V4_VERSION = "v4"
V5_VERSION = "v5"


def _cache_key(symbol, start, end, trade_batch, version):
    key = f"{symbol}_{start}_{end}_{trade_batch}_{version}"
    return hashlib.md5(key.encode()).hexdigest()


def compute_new_features(features_25, prices):
    """Compute 6 new features from cached prices and normalized features."""
    n = len(prices)

    # Raw returns from cached prices
    returns = np.zeros(n)
    returns[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))

    # Raw realvol from raw returns
    realvol_10 = (
        pd.Series(returns).rolling(window=10, min_periods=1).std().fillna(0).values
    )

    # TFI from normalized feature column 6
    tfi_norm = features_25[:, 6]

    # 25: VPIN
    vpin = (
        pd.Series(np.abs(tfi_norm))
        .rolling(window=50, min_periods=1)
        .mean()
        .fillna(0)
        .values
    )

    # 26: delta_TFI
    delta_tfi = np.zeros(n)
    delta_tfi[1:] = tfi_norm[1:] - tfi_norm[:-1]

    # 27: Hurst exponent
    hurst = np.full(n, 0.5)
    hw = 200
    if n > hw:
        for i in range(hw, n):
            r = returns[i - hw : i]
            r_c = r - r.mean()
            cumdev = np.cumsum(r_c)
            R = cumdev.max() - cumdev.min()
            S = r.std()
            if S > 1e-10 and R > 0:
                hurst[i] = np.log(R / S) / np.log(hw)
    hurst = np.clip(hurst, 0, 1)

    # 28: realized skewness
    realized_skew = (
        pd.Series(returns).rolling(window=20, min_periods=5).skew().fillna(0).values
    )

    # 29: vol-of-vol
    vol_of_vol = (
        pd.Series(realvol_10).rolling(window=50, min_periods=10).std().fillna(0).values
    )

    # 30: sign autocorrelation
    ret_sign = np.sign(returns)
    sign_autocorr = np.zeros(n)
    if n > 1:
        sign_product = ret_sign[1:] * ret_sign[:-1]
        sa = (
            pd.Series(sign_product)
            .rolling(window=19, min_periods=5)
            .mean()
            .fillna(0)
            .values
        )
        sign_autocorr[1:] = sa

    return np.column_stack(
        [vpin, delta_tfi, hurst, realized_skew, vol_of_vol, sign_autocorr]
    )


def normalize_new_cols(new_raw, window=1000):
    """Normalize the 6 new columns using the same hybrid scheme as prepare.py."""
    robust_cols = {0, 4}  # vpin, vol_of_vol
    normalized = np.zeros_like(new_raw)

    for col in range(new_raw.shape[1]):
        series = pd.Series(new_raw[:, col])
        if col in robust_cols:
            med = series.rolling(window=window, min_periods=100).median()
            q75 = series.rolling(window=window, min_periods=100).quantile(0.75)
            q25 = series.rolling(window=window, min_periods=100).quantile(0.25)
            iqr = (q75 - q25).replace(0, 1)
            z = (series - med) / iqr
        else:
            rm = series.rolling(window=window, min_periods=100).mean()
            rs = series.rolling(window=window, min_periods=100).std()
            z = (series - rm) / rs.replace(0, 1)
        normalized[:, col] = z.fillna(0).values

    np.clip(normalized, -5, 5, out=normalized)
    return normalized


def migrate():
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }

    migrated = 0
    skipped = 0

    for sym in DEFAULT_SYMBOLS:
        for split_name, (start, end) in splits.items():
            v4_key = _cache_key(sym, start, end, 100, V4_VERSION)
            v4_path = CACHE_DIR / f"{sym}_{v4_key}.npz"

            v5_key = _cache_key(sym, start, end, 100, V5_VERSION)
            v5_path = CACHE_DIR / f"{sym}_{v5_key}.npz"

            if v5_path.exists():
                skipped += 1
                continue

            if not v4_path.exists():
                print(f"  SKIP {sym} {split_name} (no v4 cache)")
                skipped += 1
                continue

            data = np.load(v4_path)
            features_25 = data["features"]
            timestamps = data["timestamps"]
            prices = data["prices"]

            if len(features_25) == 0:
                skipped += 1
                continue

            new_raw = compute_new_features(features_25, prices)
            new_norm = normalize_new_cols(new_raw)
            features_31 = np.hstack([features_25, new_norm])

            np.savez_compressed(
                v5_path, features=features_31, timestamps=timestamps, prices=prices
            )
            print(f"  {sym} {split_name}: {features_25.shape} -> {features_31.shape}")
            migrated += 1

    print(f"\nMigrated: {migrated}, Skipped: {skipped}")


if __name__ == "__main__":
    migrate()

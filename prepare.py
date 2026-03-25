#!/usr/bin/env python3
"""Data loading, feature engineering, and Gym environment for autoresearch-trading."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import gymnasium
import numpy as np
import pandas as pd

# === CONSTANTS (do not modify) ===
TRAIN_BUDGET_SECONDS = 300  # 5-minute training budget
TRAIN_START = "2025-10-16"
TRAIN_END = "2026-01-23"
VAL_END = "2026-02-17"
TEST_END = "2026-03-25"
DEFAULT_SYMBOLS = [
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
FEE_BPS = 5  # Taker fee in basis points
USE_V9 = True  # v10: 9 features (5 Aristotle + 4 top permutation importance)

DATA_ROOT = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / ".cache"


# ============================================================
# Data Loading
# ============================================================


def discover_parquet_files(
    data_root: Path,
    data_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    """Find all Parquet files for a symbol within date range."""
    base = data_root / data_type / f"symbol={symbol}"
    if not base.exists():
        return []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    files = []
    for date_dir in sorted(base.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            dir_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if start_dt <= dir_date <= end_dt:
            files.extend(sorted(date_dir.glob("*.parquet")))

    return files


def _load_parquet_duckdb(
    data_type: str, symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load Hive-partitioned Parquet files using DuckDB for speed."""
    files = discover_parquet_files(DATA_ROOT, data_type, symbol, start_date, end_date)
    if not files:
        return pd.DataFrame()

    file_list = [str(f) for f in files]
    con = duckdb.connect()
    df = con.execute(
        "SELECT * FROM read_parquet($1) ORDER BY ts_ms",
        [file_list],
    ).fetchdf()
    con.close()
    return df


def load_trades(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load trade data from Parquet files, sorted by ts_ms."""
    return _load_parquet_duckdb("trades", symbol, start_date, end_date)


def load_orderbook(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load orderbook data from Parquet files, sorted by ts_ms."""
    return _load_parquet_duckdb("orderbook", symbol, start_date, end_date)


def load_funding(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load funding rate data from Parquet files, sorted by ts_ms."""
    return _load_parquet_duckdb("funding", symbol, start_date, end_date)


# ============================================================
# Feature Engineering
# ============================================================


def normalize_side(raw_side: str) -> str:
    """Convert perpetual trade sides to buy/sell.

    open_long/close_short = buy (lifting asks)
    open_short/close_long = sell (hitting bids)
    """
    raw = raw_side.lower()
    if raw in ("buy", "open_long", "close_short"):
        return "buy"
    return "sell"


def compute_features(
    trades_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    trade_batch: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 31 features from raw data.

    Returns: (features, timestamps, prices) where features has shape (num_batches, 31).

    Feature layout (0-19: tick-horizon, 20-24: longer-horizon, 25-30: cutting-edge):
      0: returns           - log return of batch VWAP
      1: r_5               - 5-batch log return
      2: r_20              - 20-batch log return
      3: r_100             - 100-batch log return
      4: realvol_10        - rolling std of returns (window=10)
      5: bipower_var_20    - (pi/2) * rolling mean(|r_t|*|r_{t-1}|) (window=20)
      6: tfi               - (buy_vol - sell_vol) / total_vol
      7: volume_spike_ratio - batch_notional / rolling_mean(notional, 20)
      8: large_trade_share - notional of trades > q95 / total notional
      9: kyle_lambda_50    - rolling price impact coefficient
     10: amihud_illiq_50   - rolling |return| / notional
     11: trade_arrival_rate - trades per second in batch
     12: spread_bps        - bid-ask spread in basis points
     13: log_total_depth   - log(bid_depth + ask_depth + 1)
     14: weighted_imbalance_5lvl - depth-weighted bid-ask imbalance
     15: microprice_dev    - microprice - mid
     16: ofi               - weighted multi-level order flow imbalance
     17: ob_slope_asym     - ask price-depth slope minus bid slope
     18: funding_zscore    - z-scored funding rate (event-aligned)
     19: utc_hour_linear   - hour_utc / 24
     20: r_500             - ~8h log return
     21: r_2800            - ~24h log return
     22: cum_tfi_100       - rolling sum of TFI over 100 batches (~1.5h)
     23: cum_tfi_500       - rolling sum of TFI over 500 batches (~8h)
     24: funding_rate_raw  - forward-filled raw funding rate
     25: vpin              - rolling mean of |TFI| (toxicity proxy)
     26: delta_tfi         - first difference of TFI (flow acceleration)
     27: hurst_exponent    - rolling R/S Hurst exponent (regime detection)
     28: realized_skew_20  - rolling skewness of returns
     29: vol_of_vol_50     - rolling std of realvol_10
     30: sign_autocorr_20  - return sign autocorrelation (trend persistence)
    """
    if trades_df.empty:
        return np.array([]), np.array([]), np.array([])

    # Normalize sides
    trades_df = trades_df.copy()
    trades_df["norm_side"] = trades_df["side"].apply(normalize_side)
    trades_df["is_buy"] = trades_df["norm_side"] == "buy"

    # Compute notional (price * qty for cross-symbol invariance)
    trades_df["notional"] = trades_df["price"] * trades_df["qty"]

    # Group into batches
    num_trades = len(trades_df)
    num_batches = num_trades // trade_batch
    if num_batches == 0:
        return np.array([]), np.array([]), np.array([])

    # Trim to exact multiple
    trades_df = trades_df.iloc[: num_batches * trade_batch]

    # Pre-extract arrays
    prices = trades_df["price"].values
    qtys = trades_df["qty"].values
    notionals = trades_df["notional"].values
    is_buy = trades_df["is_buy"].values
    ts_ms = trades_df["ts_ms"].values

    # Reshape into batches
    prices_batched = prices.reshape(num_batches, trade_batch)
    qtys_batched = qtys.reshape(num_batches, trade_batch)
    notionals_batched = notionals.reshape(num_batches, trade_batch)
    is_buy_batched = is_buy.reshape(num_batches, trade_batch)
    ts_batched = ts_ms.reshape(num_batches, trade_batch)

    # --- VWAP ---
    total_notional = notionals_batched.sum(axis=1)
    total_qty = qtys_batched.sum(axis=1)
    vwap = np.where(total_qty > 0, total_notional / total_qty, prices_batched[:, -1])

    # --- Feature 0: returns (1-step log return of VWAP) ---
    returns = np.zeros(num_batches)
    returns[1:] = np.log(vwap[1:] / np.maximum(vwap[:-1], 1e-10))

    # --- Features 1-3, 20-21: multi-horizon returns ---
    r_5 = np.zeros(num_batches)
    r_20 = np.zeros(num_batches)
    r_100 = np.zeros(num_batches)
    r_500 = np.zeros(num_batches)
    r_2800 = np.zeros(num_batches)
    for k, arr in [(5, r_5), (20, r_20), (100, r_100), (500, r_500), (2800, r_2800)]:
        if num_batches > k:
            arr[k:] = np.log(vwap[k:] / np.maximum(vwap[:-k], 1e-10))

    # --- Feature 4: realvol_10 ---
    returns_series = pd.Series(returns)
    realvol_10 = returns_series.rolling(window=10, min_periods=1).std().fillna(0).values

    # --- Feature 5: bipower_var_20 ---
    abs_returns = np.abs(returns)
    abs_ret_product = pd.Series(np.nan, index=range(num_batches))
    abs_ret_product.iloc[1:] = abs_returns[1:] * abs_returns[:-1]
    rolling_sum = abs_ret_product.rolling(window=20, min_periods=2).sum()
    rolling_count = abs_ret_product.rolling(window=20, min_periods=2).count()
    bipower_var_20 = np.where(
        rolling_count.values > 1,
        (np.pi / 2) * rolling_sum.values / (rolling_count.values - 1),
        0.0,
    )
    bipower_var_20 = np.nan_to_num(bipower_var_20)

    # --- Feature 27: hurst_exponent (rolling R/S analysis, regime detection) ---
    hurst = np.full(num_batches, 0.5)  # default = random walk
    hurst_window = 200
    if num_batches > hurst_window:
        for i in range(hurst_window, num_batches):
            r = returns[i - hurst_window : i]
            r_centered = r - r.mean()
            cumdev = np.cumsum(r_centered)
            R = cumdev.max() - cumdev.min()
            S = r.std()
            if S > 1e-10 and R > 0:
                hurst[i] = np.log(R / S) / np.log(hurst_window)
    hurst = np.clip(hurst, 0, 1)

    # --- Feature 28: realized_skew_20 (rolling skewness of returns) ---
    realized_skew = (
        pd.Series(returns).rolling(window=20, min_periods=5).skew().fillna(0).values
    )

    # --- Feature 29: vol_of_vol_50 (rolling std of realvol_10) ---
    vol_of_vol = (
        pd.Series(realvol_10).rolling(window=50, min_periods=10).std().fillna(0).values
    )

    # --- Feature 30: sign_autocorr_20 (return sign autocorrelation) ---
    ret_sign = np.sign(returns)
    sign_autocorr = np.zeros(num_batches)
    if num_batches > 1:
        sign_product = ret_sign[1:] * ret_sign[:-1]
        sa_rolling = (
            pd.Series(sign_product)
            .rolling(window=19, min_periods=5)
            .mean()
            .fillna(0)
            .values
        )
        sign_autocorr[1:] = sa_rolling

    # --- Feature 6: tfi ---
    buy_notional = (notionals_batched * is_buy_batched).sum(axis=1)
    sell_notional = (notionals_batched * ~is_buy_batched).sum(axis=1)
    total_batch_notional = buy_notional + sell_notional
    tfi = np.where(
        total_batch_notional > 0,
        (buy_notional - sell_notional) / total_batch_notional,
        0.0,
    )

    # --- Features 22-23: cumulative order flow (rolling sum of TFI) ---
    tfi_series = pd.Series(tfi)
    cum_tfi_100 = tfi_series.rolling(window=100, min_periods=1).sum().fillna(0).values
    cum_tfi_500 = tfi_series.rolling(window=500, min_periods=1).sum().fillna(0).values

    # --- Feature 25: VPIN (rolling mean of |TFI|, toxicity proxy) ---
    abs_tfi = np.abs(tfi)
    vpin = pd.Series(abs_tfi).rolling(window=50, min_periods=1).mean().fillna(0).values

    # --- Feature 26: delta_tfi (first difference of TFI, flow acceleration) ---
    delta_tfi = np.zeros(num_batches)
    delta_tfi[1:] = tfi[1:] - tfi[:-1]

    # --- Feature 7: volume_spike_ratio ---
    notional_series = pd.Series(total_batch_notional)
    rolling_mean_notional = (
        notional_series.shift(1).rolling(window=20, min_periods=1).mean()
    )
    volume_spike_ratio = np.where(
        rolling_mean_notional.values > 0,
        total_batch_notional / rolling_mean_notional.values,
        1.0,
    )

    # --- Feature 8: large_trade_share ---
    # Use rolling p95 over a fixed lookback window instead of O(n²) growing slice
    large_trade_share = np.zeros(num_batches)
    LOOKBACK_BATCHES = 50  # ~5000 trades lookback for p95 estimate
    flat_notionals = notionals_batched.ravel()
    for i in range(num_batches):
        batch_not = notionals_batched[i]
        if i > 0:
            start_idx = max(0, i - LOOKBACK_BATCHES) * trade_batch
            end_idx = i * trade_batch
            notional_95 = np.percentile(flat_notionals[start_idx:end_idx], 95)
        else:
            notional_95 = np.percentile(batch_not, 95)
        batch_sum = batch_not.sum()
        large_trade_share[i] = batch_not[batch_not > notional_95].sum() / max(
            batch_sum, 1e-10
        )

    # --- Feature 9: kyle_lambda_50 ---
    signed_notional = (notionals_batched * is_buy_batched).sum(axis=1) - (
        notionals_batched * ~is_buy_batched
    ).sum(axis=1)

    ret_s = pd.Series(returns)
    sn_s = pd.Series(signed_notional)
    rolling_cov = ret_s.rolling(window=50, min_periods=10).cov(sn_s)
    rolling_var = sn_s.rolling(window=50, min_periods=10).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        kyle_lambda = np.where(
            rolling_var.values > 1e-20,
            rolling_cov.values / rolling_var.values,
            0.0,
        )
    kyle_lambda = np.nan_to_num(kyle_lambda)

    # --- Feature 10: amihud_illiq_50 ---
    illiq_raw = np.where(
        total_batch_notional > 0, np.abs(returns) / total_batch_notional, 0.0
    )
    illiq_series = pd.Series(illiq_raw)
    amihud_illiq = (
        illiq_series.rolling(window=50, min_periods=1).mean().fillna(0).values
    )

    # --- Feature 11: trade_arrival_rate ---
    batch_first_ts = ts_batched[:, 0]
    batch_last_ts = ts_batched[:, -1]
    batch_duration_s = (batch_last_ts - batch_first_ts) / 1000.0
    trade_arrival_rate = np.where(
        batch_duration_s > 0, trade_batch / batch_duration_s, 0.0
    )

    # Batch timestamps and prices
    batch_timestamps = ts_batched[:, -1]
    batch_prices = vwap

    # === ORDERBOOK FEATURES (indices 12-17) ===
    ob_features = np.zeros((num_batches, 6))

    if not orderbook_df.empty:
        ob_ts = orderbook_df["ts_ms"].values
        ob_bids = orderbook_df["bids"].values
        ob_asks = orderbook_df["asks"].values

        ob_idx = 0
        prev_bid_vols = np.zeros(5)
        prev_ask_vols = np.zeros(5)
        prev_ob_valid = False
        weights_ofi = np.array([1.0, 0.5, 1 / 3, 0.25, 0.2])
        weights_imb = np.array([1.0, 0.5, 1 / 3, 0.25, 0.2])

        for i in range(num_batches):
            t = batch_timestamps[i]
            while ob_idx < len(ob_ts) - 1 and ob_ts[ob_idx + 1] <= t:
                ob_idx += 1

            if ob_idx < len(ob_ts) and ob_ts[ob_idx] <= t:
                bids = ob_bids[ob_idx]
                asks = ob_asks[ob_idx]

                if len(bids) > 0 and len(asks) > 0:
                    n_bid_lvls = min(5, len(bids))
                    n_ask_lvls = min(5, len(asks))

                    bid_prices_arr = np.array(
                        [
                            bids[l]["price"] if isinstance(bids[l], dict) else 0.0
                            for l in range(n_bid_lvls)
                        ]
                    )
                    bid_qtys_arr = np.array(
                        [
                            bids[l]["qty"] if isinstance(bids[l], dict) else 0.0
                            for l in range(n_bid_lvls)
                        ]
                    )
                    ask_prices_arr = np.array(
                        [
                            asks[l]["price"] if isinstance(asks[l], dict) else 0.0
                            for l in range(n_ask_lvls)
                        ]
                    )
                    ask_qtys_arr = np.array(
                        [
                            asks[l]["qty"] if isinstance(asks[l], dict) else 0.0
                            for l in range(n_ask_lvls)
                        ]
                    )

                    best_bid = bid_prices_arr[0] if n_bid_lvls > 0 else 0.0
                    best_ask = ask_prices_arr[0] if n_ask_lvls > 0 else 0.0
                    mid = (
                        (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 1.0
                    )

                    # Convert qty to notional for cross-symbol invariance
                    bid_notional_arr = bid_qtys_arr * bid_prices_arr
                    ask_notional_arr = ask_qtys_arr * ask_prices_arr

                    # Feature 12: spread_bps
                    ob_features[i, 0] = (best_ask - best_bid) / mid * 10000

                    # Feature 13: log_total_depth (notional-based)
                    bid_depth_not = bid_notional_arr.sum()
                    ask_depth_not = ask_notional_arr.sum()
                    ob_features[i, 1] = np.log(bid_depth_not + ask_depth_not + 1)

                    # Feature 14: weighted_imbalance_5lvl (notional-based)
                    w = weights_imb[:n_bid_lvls]
                    w_ask = weights_imb[:n_ask_lvls]
                    num_imb = (w * bid_notional_arr[:n_bid_lvls]).sum() - (
                        w_ask * ask_notional_arr[:n_ask_lvls]
                    ).sum()
                    den_imb = (w * bid_notional_arr[:n_bid_lvls]).sum() + (
                        w_ask * ask_notional_arr[:n_ask_lvls]
                    ).sum()
                    ob_features[i, 2] = num_imb / den_imb if den_imb > 0 else 0.0

                    # Feature 15: microprice_dev (price-based, uses raw qty)
                    best_bid_qty = bid_qtys_arr[0] if n_bid_lvls > 0 else 0.0
                    best_ask_qty = ask_qtys_arr[0] if n_ask_lvls > 0 else 0.0
                    total_best_qty = best_bid_qty + best_ask_qty
                    if total_best_qty > 0:
                        microprice = (
                            best_bid * best_ask_qty + best_ask * best_bid_qty
                        ) / total_best_qty
                    else:
                        microprice = mid
                    ob_features[i, 3] = microprice - mid

                    # Feature 16: ofi (notional-based)
                    curr_bid_vols = np.zeros(5)
                    curr_ask_vols = np.zeros(5)
                    curr_bid_vols[:n_bid_lvls] = bid_notional_arr
                    curr_ask_vols[:n_ask_lvls] = ask_notional_arr
                    if prev_ob_valid:
                        delta_bid = curr_bid_vols - prev_bid_vols
                        delta_ask = curr_ask_vols - prev_ask_vols
                        ob_features[i, 4] = (
                            weights_ofi * (delta_bid - delta_ask)
                        ).sum()
                    prev_bid_vols = curr_bid_vols.copy()
                    prev_ask_vols = curr_ask_vols.copy()
                    prev_ob_valid = True

                    # Feature 17: ob_slope_asymmetry (OLS price-depth slope)
                    if n_ask_lvls >= 2 and n_bid_lvls >= 2:
                        ask_cum_not = np.cumsum(ask_notional_arr)
                        bid_cum_not = np.cumsum(bid_notional_arr)

                        if np.var(ask_cum_not) > 1e-20:
                            ask_slope = np.polyfit(ask_cum_not, ask_prices_arr, 1)[0]
                        else:
                            ask_slope = 0.0
                        if np.var(bid_cum_not) > 1e-20:
                            bid_slope = np.polyfit(bid_cum_not, bid_prices_arr, 1)[0]
                        else:
                            bid_slope = 0.0
                        ob_features[i, 5] = ask_slope - bid_slope

    # === FUNDING + TIME FEATURES (indices 18-19) ===
    extra_features = np.zeros((num_batches, 2))

    # Feature 18: funding_zscore
    if not funding_df.empty:
        # Deduplicate: keep first row per unique ts_ms (raw data has ~143k dupes/day)
        funding_dedup = funding_df.drop_duplicates(subset="ts_ms", keep="first")
        fund_ts = funding_dedup["ts_ms"].values
        fund_rate = funding_dedup["rate"].values

        # Vectorized z-score over last 21 funding events
        fund_series = pd.Series(fund_rate)
        roll_mean = fund_series.rolling(window=21, min_periods=8).mean()
        roll_std = fund_series.rolling(window=21, min_periods=8).std()
        with np.errstate(invalid="ignore", divide="ignore"):
            fund_zscore = np.where(
                roll_std.values > 1e-12,
                (fund_rate - roll_mean.values) / roll_std.values,
                0.0,
            )
        fund_zscore = np.nan_to_num(fund_zscore)

        # Vectorized forward-fill to batches via searchsorted
        indices = np.searchsorted(fund_ts, batch_timestamps, side="right") - 1
        valid = indices >= 0
        extra_features[valid, 0] = fund_zscore[indices[valid]]

    # Feature 19: utc_hour_linear (vectorized)
    extra_features[:, 1] = ((batch_timestamps / 1000 / 3600) % 24) / 24.0

    # === LONGER-HORIZON FEATURES (indices 20-24) ===
    longer_features = np.zeros((num_batches, 5))
    longer_features[:, 0] = r_500  # 20
    longer_features[:, 1] = r_2800  # 21
    longer_features[:, 2] = cum_tfi_100  # 22
    longer_features[:, 3] = cum_tfi_500  # 23

    # Feature 24: funding_rate_raw (forward-filled, not z-scored)
    if not funding_df.empty:
        longer_features[valid, 4] = fund_rate[indices[valid]]

    # ── v6 tape-reading features ──────────────────────────────────
    # Feature 31-32: buy/sell run max (longest consecutive streak in batch)
    def _max_run_length(arr):
        """Max consecutive True count along last axis."""
        n = arr.shape[-1]
        result = np.zeros(arr.shape[0], dtype=np.float32)
        current = np.zeros(arr.shape[0], dtype=np.float32)
        for i in range(n):
            current = np.where(arr[:, i], current + 1, 0)
            result = np.maximum(result, current)
        return result

    buy_run_max = _max_run_length(is_buy_batched)
    sell_run_max = _max_run_length(~is_buy_batched)

    # Feature 33-34: large buy/sell share (directional large trade volume)
    # Pre-compute rolling p95 as array (feature 8 computes it per-batch inside a loop)
    # LOOKBACK_BATCHES and flat_notionals are already defined by feature 8 (line 329-330)
    rolling_p95 = np.zeros(num_batches)
    for i in range(num_batches):
        if i > 0:
            start_idx = max(0, i - LOOKBACK_BATCHES) * trade_batch
            end_idx = i * trade_batch
            rolling_p95[i] = np.percentile(flat_notionals[start_idx:end_idx], 95)
        else:
            rolling_p95[i] = np.percentile(notionals_batched[i], 95)

    is_large = notionals_batched > rolling_p95[:, np.newaxis]
    large_buy_notional = np.where(is_buy_batched & is_large, notionals_batched, 0).sum(
        axis=1
    )
    large_sell_notional = np.where(
        ~is_buy_batched & is_large, notionals_batched, 0
    ).sum(axis=1)
    # total_batch_notional already exists (line 295, computed for TFI)
    safe_total = np.maximum(total_batch_notional, 1e-10)
    large_buy_share = large_buy_notional / safe_total
    large_sell_share = large_sell_notional / safe_total

    # Feature 35: trade size entropy
    notional_probs = notionals_batched / np.maximum(
        notionals_batched.sum(axis=1, keepdims=True), 1e-10
    )
    # Clip to avoid log(0)
    notional_probs = np.clip(notional_probs, 1e-10, 1.0)
    trade_size_entropy = -np.sum(notional_probs * np.log(notional_probs), axis=1)

    # Feature 36: aggressor imbalance (count-based)
    buy_count = is_buy_batched.sum(axis=1).astype(np.float32)
    sell_count = trade_batch - buy_count
    aggressor_imbalance = (buy_count - sell_count) / trade_batch

    # Feature 37: price level absorption
    price_change = np.abs(np.diff(vwap, prepend=vwap[0]))
    price_change_safe = np.maximum(price_change, 1e-10)
    # total_batch_notional already exists (line 295, computed for TFI)
    price_level_absorption = total_batch_notional / price_change_safe
    # Zero out batches with negligible price change (batch 0 always, plus any identical-VWAP batches)
    # These produce ~1e14+ spikes from notional/1e-10; common for illiquid symbols
    tiny_move = price_change < 1e-8
    price_level_absorption[tiny_move] = 0.0

    # Feature 38: TFI acceleration (second difference)
    # delta_tfi is already computed for feature 26
    tfi_acceleration = np.diff(delta_tfi, prepend=0.0)

    # Combine all features
    trade_features = np.column_stack(
        [
            returns,  # 0
            r_5,  # 1
            r_20,  # 2
            r_100,  # 3
            realvol_10,  # 4
            bipower_var_20,  # 5
            tfi,  # 6
            volume_spike_ratio,  # 7
            large_trade_share,  # 8
            kyle_lambda,  # 9
            amihud_illiq,  # 10
            trade_arrival_rate,  # 11
        ]
    )

    # === CUTTING-EDGE FEATURES (indices 25-30) ===
    cutting_edge_features = np.column_stack(
        [
            vpin,  # 25
            delta_tfi,  # 26
            hurst,  # 27
            realized_skew,  # 28
            vol_of_vol,  # 29
            sign_autocorr,  # 30
        ]
    )

    # === TAPE READING FEATURES (indices 31-38) ===
    tape_reading_features = np.column_stack(
        [
            buy_run_max,  # 31
            sell_run_max,  # 32
            large_buy_share,  # 33
            large_sell_share,  # 34
            trade_size_entropy,  # 35
            aggressor_imbalance,  # 36
            price_level_absorption,  # 37
            tfi_acceleration,  # 38
        ]
    )

    features = np.hstack(
        [
            trade_features,
            ob_features,
            extra_features,
            longer_features,
            cutting_edge_features,
            tape_reading_features,
        ]
    )

    return features, batch_timestamps, batch_prices


# Indices of features that get robust (median/IQR) normalization.
# Tail-heavy: bipower_var, volume_spike_ratio, large_trade_share, kyle_lambda,
# amihud_illiq, trade_arrival_rate, spread_bps, log_total_depth, ofi, ob_slope_asym
# cum_tfi (22, 23), funding_rate_raw (24), vpin (25), vol_of_vol (29) are also tail-heavy
ROBUST_FEATURE_INDICES = {
    5,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    17,
    22,
    23,
    24,
    25,
    29,
    33,
    34,
    35,
    37,
}

# ============================================================
# v9 Aristotle-Proven Features (5 features)
# ============================================================

V9_FEATURE_NAMES = [
    "lambda_ofi",  # 0
    "directional_conviction",  # 1
    "vpin",  # 2
    "hawkes_branching",  # 3
    "reservation_price_dev",  # 4
    "vol_of_vol",  # 5
    "utc_hour_linear",  # 6
    "microprice_dev",  # 7
    "delta_tfi",  # 8
    "multi_level_ofi",  # 9  (v11, ablation: HELPS)
    "buy_vwap_dev",  # 10 (v11, ablation: HELPS)
    "trade_arrival_rate",  # 11 (v11, ablation: HELPS)
    "r_20",  # 12 (v11, ablation: HELPS)
]
# Dropped by ablation (HURTS): sell_vwap_dev, spread_bps, amihud_illiq, roll_measure

V9_NUM_FEATURES = 13
V9_ROBUST_FEATURE_INDICES = {
    4,
    5,
    11,
}  # reservation_price_dev, vol_of_vol, trade_arrival_rate


def compute_features_v9(
    trades_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    trade_batch: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 13 v11a features from trade/orderbook data.

    Returns: (features, timestamps, prices, raw_hawkes_branching)
    where features has shape (num_batches, 13).

    Feature layout (9 v10 + 4 ablation-validated new):
      0: lambda_ofi              - kyle_lambda * signed_notional
      1: directional_conviction  - TFI * |signed_notional|
      2: vpin                    - rolling mean of |TFI|
      3: hawkes_branching        - 1 - 1/sqrt(Var(R)/E[R])
      4: reservation_price_dev   - orderbook_imbalance * realvol^2
      5: vol_of_vol              - rolling std of realvol
      6: utc_hour_linear         - hour_utc / 24
      7: microprice_dev          - microprice - midprice
      8: delta_tfi               - first difference of TFI
      9: multi_level_ofi         - weighted depth changes across 5 levels (T30)
     10: buy_vwap_dev            - buy VWAP - VWAP (T31)
     11: trade_arrival_rate      - trades / second per batch (T34)
     12: r_20                    - 20-batch cumulative return (T35)

    Dropped by ablation (HURTS): sell_vwap_dev, spread_bps, amihud_illiq, roll_measure
    """
    if trades_df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # --- Reuse batching logic from compute_features ---
    trades_df = trades_df.copy()
    trades_df["norm_side"] = trades_df["side"].apply(normalize_side)
    trades_df["is_buy"] = trades_df["norm_side"] == "buy"
    trades_df["notional"] = trades_df["price"] * trades_df["qty"]

    num_trades = len(trades_df)
    num_batches = num_trades // trade_batch
    if num_batches == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    prices_arr = (
        trades_df["price"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
    )
    notionals_batched = (
        trades_df["notional"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
    )
    is_buy_batched = (
        trades_df["is_buy"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
    )
    ts_batched = (
        trades_df["ts_ms"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
    )

    # VWAP per batch
    total_batch_notional = notionals_batched.sum(axis=1)
    total_batch_qty = (
        trades_df["qty"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
        .sum(axis=1)
    )
    vwap = np.where(total_batch_qty > 0, total_batch_notional / total_batch_qty, 0)
    vwap = np.where(vwap == 0, prices_arr[:, -1], vwap)

    # Returns
    returns = np.zeros(num_batches)
    returns[1:] = np.diff(np.log(np.clip(vwap, 1e-10, None)))

    # --- Intermediate: TFI ---
    buy_vol = (notionals_batched * is_buy_batched).sum(axis=1)
    sell_vol = (notionals_batched * ~is_buy_batched).sum(axis=1)
    total_vol = buy_vol + sell_vol
    tfi = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0.0)

    # --- Intermediate: signed_notional ---
    signed_notional = buy_vol - sell_vol

    # --- Intermediate: kyle_lambda (rolling 50-batch) ---
    ret_s = pd.Series(returns)
    sn_s = pd.Series(signed_notional)
    rolling_cov = ret_s.rolling(window=50, min_periods=10).cov(sn_s)
    rolling_var = sn_s.rolling(window=50, min_periods=10).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        kyle_lambda = np.where(
            rolling_var.values > 1e-20,
            rolling_cov.values / rolling_var.values,
            0.0,
        )
    kyle_lambda = np.nan_to_num(kyle_lambda)

    # --- Intermediate: realvol (rolling 10-batch std) ---
    realvol = (
        pd.Series(returns).rolling(window=10, min_periods=2).std().fillna(0).values
    )

    # === Feature 0: lambda_ofi ===
    lambda_ofi = kyle_lambda * signed_notional

    # === Feature 1: directional_conviction ===
    directional_conviction = tfi * np.abs(signed_notional)

    # === Feature 2: vpin ===
    abs_tfi = np.abs(tfi)
    vpin = pd.Series(abs_tfi).rolling(window=50, min_periods=1).mean().fillna(0).values

    # === Feature 3: hawkes_branching ===
    batch_first_ts = ts_batched[:, 0]
    batch_last_ts = ts_batched[:, -1]
    batch_duration_s = (batch_last_ts - batch_first_ts) / 1000.0
    arrival_rates = np.where(
        batch_duration_s > 0.001, trade_batch / batch_duration_s, 0.0
    )

    hawkes_branching = np.zeros(num_batches)
    hawkes_window = 50
    for i in range(hawkes_window, num_batches):
        window_rates = arrival_rates[i - hawkes_window : i]
        window_rates = window_rates[window_rates > 0]
        if len(window_rates) < 10:
            continue
        mean_r = window_rates.mean()
        var_r = window_rates.var()
        if mean_r > 0:
            ratio = var_r / mean_r
            if ratio > 1.0:
                hawkes_branching[i] = 1.0 - 1.0 / np.sqrt(ratio)
    hawkes_branching = np.clip(hawkes_branching, 0.0, 0.99)
    raw_hawkes_branching = hawkes_branching.copy()

    # --- Orderbook features (unified helper) ---
    batch_timestamps_final = ts_batched[:, -1]
    microprice_dev, spread_bps_arr, mlofi, weighted_imbalance = (
        _compute_orderbook_features(orderbook_df, batch_timestamps_final, num_batches)
    )

    # === Feature 4: reservation_price_dev ===
    reservation_price_dev = weighted_imbalance * (realvol**2)

    # === Feature 5: vol_of_vol ===
    vol_of_vol = (
        pd.Series(realvol).rolling(window=50, min_periods=10).std().fillna(0).values
    )

    # === Feature 6: utc_hour_linear ===
    utc_hour_linear = ((batch_timestamps_final / 1000 / 3600) % 24) / 24.0

    # === Feature 7: microprice_dev (from unified OB helper) ===
    # (already computed above)

    # === Feature 8: delta_TFI ===
    delta_tfi = np.zeros(num_batches)
    delta_tfi[1:] = tfi[1:] - tfi[:-1]

    # === Feature 9: multi_level_ofi (T30) ===
    # (already computed by _compute_orderbook_features)

    # === Feature 10: buy_vwap_dev (T31) ===
    buy_notional = (notionals_batched * is_buy_batched).sum(axis=1)
    buy_qty = (
        trades_df["qty"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
        * is_buy_batched
    ).sum(axis=1)
    sell_notional = (notionals_batched * ~is_buy_batched).sum(axis=1)
    sell_qty = (
        trades_df["qty"]
        .values[: num_batches * trade_batch]
        .reshape(num_batches, trade_batch)
        * ~is_buy_batched
    ).sum(axis=1)
    buy_vwap = np.where(buy_qty > 0, buy_notional / buy_qty, vwap)
    sell_vwap = np.where(sell_qty > 0, sell_notional / sell_qty, vwap)
    buy_vwap_dev = buy_vwap - vwap
    # === Feature 11: sell_vwap_dev (T31) ===
    sell_vwap_dev = sell_vwap - vwap

    # === Feature 12: spread_bps (T32, from unified OB helper) ===
    # (already computed above)

    # === Feature 13: amihud_illiq (T32) ===
    amihud = np.where(
        total_batch_notional > 0, np.abs(returns) / total_batch_notional, 0.0
    )

    # === Feature 14: roll_measure (T32) ===
    roll_measure = np.zeros(num_batches)
    roll_window = 20
    for i in range(roll_window, num_batches):
        r_win = returns[i - roll_window : i]
        if len(r_win) > 1:
            autocov = np.cov(r_win[1:], r_win[:-1])[0, 1]
            roll_measure[i] = np.sqrt(max(-autocov, 0))

    # === Feature 15: trade_arrival_rate (T34) ===
    trade_arrival_rate = np.where(
        batch_duration_s > 0.001, trade_batch / batch_duration_s, 0.0
    )

    # === Feature 16: r_20 (T35) ===
    r_20 = pd.Series(returns).rolling(window=20, min_periods=1).sum().fillna(0).values

    # --- Stack features ---
    # Ablation-validated: 4 of 8 new features kept, 4 dropped (HURTS)
    # Dropped: sell_vwap_dev, spread_bps_arr, amihud, roll_measure
    features = np.column_stack(
        [
            lambda_ofi,  # 0
            directional_conviction,  # 1
            vpin,  # 2
            hawkes_branching,  # 3
            reservation_price_dev,  # 4
            vol_of_vol,  # 5
            utc_hour_linear,  # 6
            microprice_dev,  # 7
            delta_tfi,  # 8
            mlofi,  # 9  (HELPS -0.020)
            buy_vwap_dev,  # 10 (HELPS -0.029)
            trade_arrival_rate,  # 11 (HELPS -0.016)
            r_20,  # 12 (HELPS -0.017)
        ]
    )

    batch_prices = vwap

    return features, batch_timestamps_final, batch_prices, raw_hawkes_branching


def _compute_orderbook_features(
    orderbook_df: pd.DataFrame, batch_timestamps: np.ndarray, num_batches: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute orderbook-derived features aligned to batch timestamps.

    Returns: (microprice_dev, spread_bps, multi_level_ofi, weighted_imbalance)
    """
    microprice_dev = np.zeros(num_batches)
    spread_bps = np.zeros(num_batches)
    mlofi = np.zeros(num_batches)
    weighted_imbalance = np.zeros(num_batches)

    if orderbook_df.empty:
        return microprice_dev, spread_bps, mlofi, weighted_imbalance

    ob_ts = orderbook_df["ts_ms"].values
    ob_idx = 0
    prev_bid_depths = None
    prev_ask_depths = None
    weights = [1.0, 0.5, 1 / 3, 0.25, 0.2]

    for i in range(num_batches):
        while ob_idx < len(ob_ts) - 1 and ob_ts[ob_idx + 1] <= batch_timestamps[i]:
            ob_idx += 1
        if ob_idx >= len(ob_ts) or ob_ts[ob_idx] > batch_timestamps[i]:
            continue

        row = orderbook_df.iloc[ob_idx]
        bids = row.get("bids", [])
        asks = row.get("asks", [])
        if len(bids) == 0 or len(asks) == 0:
            continue

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2

        # Spread in bps (T32)
        if mid > 0:
            spread_bps[i] = (best_ask - best_bid) / mid * 10000

        # Microprice deviation (T33)
        best_bid_qty = bids[0]["qty"]
        best_ask_qty = asks[0]["qty"]
        total_qty = best_bid_qty + best_ask_qty
        if total_qty > 0:
            microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / total_qty
            microprice_dev[i] = microprice - mid

        # Multi-level OFI (T30): change in depth at each level
        n_levels = min(5, len(bids), len(asks))
        curr_bid_depths = np.array(
            [bids[lvl]["qty"] * bids[lvl]["price"] for lvl in range(n_levels)]
        )
        curr_ask_depths = np.array(
            [asks[lvl]["qty"] * asks[lvl]["price"] for lvl in range(n_levels)]
        )

        if prev_bid_depths is not None and len(prev_bid_depths) == n_levels:
            delta_bid = curr_bid_depths - prev_bid_depths
            delta_ask = curr_ask_depths - prev_ask_depths
            ofi_per_level = delta_bid - delta_ask
            level_weights = np.array(weights[:n_levels])
            mlofi[i] = np.sum(level_weights * ofi_per_level)

        prev_bid_depths = curr_bid_depths.copy()
        prev_ask_depths = curr_ask_depths.copy()

        # Weighted imbalance (for reservation_price_dev)
        bid_depth = sum(
            w * b["qty"] * b["price"]
            for w, b in zip(weights[: min(5, len(bids))], bids[:5])
        )
        ask_depth = sum(
            w * a["qty"] * a["price"]
            for w, a in zip(weights[: min(5, len(asks))], asks[:5])
        )
        total = bid_depth + ask_depth
        if total > 0:
            weighted_imbalance[i] = (bid_depth - ask_depth) / total

    return microprice_dev, spread_bps, mlofi, weighted_imbalance


def normalize_features_v9(features: np.ndarray, window: int = 1000) -> np.ndarray:
    """Normalize v9 features. IQR for feature 4, z-score for rest."""
    if features.ndim != 2 or len(features) == 0:
        return features

    normalized = np.zeros_like(features)
    for col in range(features.shape[1]):
        series = pd.Series(features[:, col])
        if col in V9_ROBUST_FEATURE_INDICES:
            rolling_median = series.rolling(window=window, min_periods=100).median()
            rolling_q75 = series.rolling(window=window, min_periods=100).quantile(0.75)
            rolling_q25 = series.rolling(window=window, min_periods=100).quantile(0.25)
            iqr = (rolling_q75 - rolling_q25).replace(0, 1)
            z = (series - rolling_median) / iqr
        else:
            rolling_mean = series.rolling(window=window, min_periods=100).mean()
            rolling_std = series.rolling(window=window, min_periods=100).std()
            z = (series - rolling_mean) / rolling_std.replace(0, 1)
        normalized[:, col] = z.fillna(0).values

    np.clip(normalized, -5, 5, out=normalized)
    return normalized


def normalize_features(features: np.ndarray, window: int = 1000) -> np.ndarray:
    """Hybrid rolling normalization.

    Robust scaling (median/IQR) for tail-heavy features.
    Rolling z-score (mean/std) for well-behaved features.
    """
    if features.ndim != 2 or len(features) == 0:
        return features

    normalized = np.zeros_like(features)
    num_features = features.shape[1]

    for col in range(num_features):
        series = pd.Series(features[:, col])

        if col in ROBUST_FEATURE_INDICES:
            # Robust scaling: (x - median) / IQR
            rolling_median = series.rolling(window=window, min_periods=100).median()
            rolling_q75 = series.rolling(window=window, min_periods=100).quantile(0.75)
            rolling_q25 = series.rolling(window=window, min_periods=100).quantile(0.25)
            iqr = (rolling_q75 - rolling_q25).replace(0, 1)
            z = (series - rolling_median) / iqr
        else:
            # Standard z-score
            rolling_mean = series.rolling(window=window, min_periods=100).mean()
            rolling_std = series.rolling(window=window, min_periods=100).std()
            z = (series - rolling_mean) / rolling_std.replace(0, 1)

        normalized[:, col] = z.fillna(0).values

    # Clip to prevent extreme outliers from poisoning gradients
    np.clip(normalized, -5, 5, out=normalized)

    return normalized


_FEATURE_VERSION = "v11a"  # v11a: 13 features (9 v10 + 4 ablation-validated)


def _cache_key(symbol: str, start: str, end: str, trade_batch: int) -> str:
    """Compute cache key from parameters."""
    key = f"{symbol}_{start}_{end}_{trade_batch}_{_FEATURE_VERSION}"
    return hashlib.md5(key.encode()).hexdigest()


def cache_features(
    symbol: str,
    features: np.ndarray,
    timestamps: np.ndarray,
    prices: np.ndarray,
    cache_dir: Path,
    start: str,
    end: str,
    trade_batch: int,
    raw_hawkes: np.ndarray | None = None,
) -> None:
    """Save features to .npz cache file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    save_dict = dict(features=features, timestamps=timestamps, prices=prices)
    if raw_hawkes is not None:
        save_dict["raw_hawkes"] = raw_hawkes
    np.savez_compressed(path, **save_dict)
    print(f"Cached {symbol} features to {path}")


def load_cached(
    symbol: str,
    cache_dir: Path,
    start: str,
    end: str,
    trade_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Load features from .npz cache if exists.

    Returns (features, timestamps, prices, raw_hawkes) or None.
    raw_hawkes is None for v8 caches that don't contain it.
    """
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    if path.exists():
        data = np.load(path)
        print(f"Loaded {symbol} features from cache ({path})")
        raw_hawkes = data["raw_hawkes"] if "raw_hawkes" in data else None
        return data["features"], data["timestamps"], data["prices"], raw_hawkes
    return None


# ============================================================
# Data Preparation Pipeline
# ============================================================


def prepare_data(
    symbols: list[str] | None = None,
    trade_batch: int = 100,
    force_recompute: bool = False,
) -> dict:
    """Prepare data for all symbols and splits.

    Returns dict of {symbol: {train: (features, timestamps, prices, raw_hawkes), ...}}
    raw_hawkes is None when using v8 features (USE_V9=False).
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }

    result = {}

    for symbol in symbols:
        result[symbol] = {}

        for split_name, (start, end) in splits.items():
            # Try cache first
            if not force_recompute:
                cached = load_cached(symbol, CACHE_DIR, start, end, trade_batch)
                if cached is not None:
                    features, timestamps, prices, raw_hawkes = cached
                    result[symbol][split_name] = (
                        features,
                        timestamps,
                        prices,
                        raw_hawkes,
                    )
                    continue

            print(f"Computing features for {symbol} {split_name} ({start} to {end})...")

            trades_df = load_trades(symbol, start, end)
            orderbook_df = load_orderbook(symbol, start, end)
            funding_df = load_funding(symbol, start, end)

            print(
                f"  Loaded {len(trades_df)} trades, {len(orderbook_df)} orderbook snapshots, {len(funding_df)} funding rates"
            )

            if USE_V9:
                features, timestamps, prices, raw_hawkes = compute_features_v9(
                    trades_df, orderbook_df, funding_df, trade_batch
                )
                if len(features) > 0:
                    features = normalize_features_v9(features)
            else:
                features, timestamps, prices = compute_features(
                    trades_df, orderbook_df, funding_df, trade_batch
                )
                raw_hawkes = None
                if len(features) > 0:
                    features = normalize_features(features)

            print(f"  Features shape: {features.shape}")

            # Cache
            cache_features(
                symbol,
                features,
                timestamps,
                prices,
                CACHE_DIR,
                start,
                end,
                trade_batch,
                raw_hawkes=raw_hawkes,
            )

            result[symbol][split_name] = (features, timestamps, prices, raw_hawkes)

    return result


# ============================================================
# Gym Environment
# ============================================================


class TradingEnv(gymnasium.Env):
    """Event-driven trading environment.

    Observations: window of normalized features (window_size, num_features)
    Actions: 0=flat, 1=long, 2=short
    Reward: computed externally by train.py's compute_reward()
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window_size: int = 50,
        fee_bps: float = 5,
        min_hold: int = 1,
    ):
        super().__init__()

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window_size = window_size
        self.fee_bps = fee_bps
        self.min_hold = min_hold
        self.num_steps = len(features)

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, features.shape[1]),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(3)  # flat, long, short

        # State
        self._idx = 0
        self._position = 0  # 0=flat, 1=long, 2=short
        self._equity = 1.0
        self._peak_equity = 1.0
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._hold_duration = 0
        self._steps_since_trade = 0
        self._episode_step = 0

    def _get_obs(self) -> np.ndarray:
        start = self._idx - self.window_size
        return self.features[start : self._idx].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options and options.get("sequential"):
            # For evaluation: start from the beginning
            self._idx = self.window_size
        else:
            # Random start for training
            max_start = self.num_steps - 2000
            if max_start <= self.window_size:
                max_start = self.window_size + 1
            self._idx = self.np_random.integers(self.window_size, max_start)

        self._position = 0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._hold_duration = 0
        self._steps_since_trade = self.min_hold  # allow first trade immediately
        self._episode_step = 0

        return self._get_obs(), {}

    def step(self, action: int):
        prev_position = self._position
        self._steps_since_trade += 1

        # Enforce min_hold: ignore position changes during hold period
        # Exception: entering from flat is always allowed
        if action != prev_position and prev_position != 0:
            if self._steps_since_trade < self.min_hold:
                action = prev_position  # override: keep current position

        prev_price = self.prices[self._idx - 1]
        curr_price = self.prices[self._idx]

        # Price change
        if prev_price > 0:
            price_return = (curr_price - prev_price) / prev_price
        else:
            price_return = 0.0

        # P&L from current position
        if prev_position == 1:  # long
            step_pnl = price_return
        elif prev_position == 2:  # short
            step_pnl = -price_return
        else:
            step_pnl = 0.0

        # Position change and transaction costs
        if action != prev_position:
            # Apply fee for closing old position
            if prev_position != 0:
                step_pnl -= self.fee_bps / 10000
            # Apply fee for opening new position
            if action != 0:
                step_pnl -= self.fee_bps / 10000
            self._trade_count += 1
            self._hold_duration = 0
            self._steps_since_trade = 0
            self._position = action
        else:
            if self._position != 0:
                self._hold_duration += 1

        # Update equity
        self._equity *= 1 + step_pnl
        self._realized_pnl += step_pnl
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (
            (self._peak_equity - self._equity) / self._peak_equity
            if self._peak_equity > 0
            else 0.0
        )

        self._idx += 1
        self._episode_step += 1

        done = self._idx >= self.num_steps
        truncated = self._episode_step >= 2000

        info = {
            "step_pnl": step_pnl,
            "position": self._position,
            "equity": self._equity,
            "drawdown": drawdown,
            "trade_count": self._trade_count,
            "hold_duration": self._hold_duration,
            "steps_since_trade": self._steps_since_trade,
            "realized_pnl": self._realized_pnl,
            "price": curr_price,
        }

        obs = (
            self._get_obs()
            if not (done or truncated)
            else np.zeros_like(
                self._get_obs()
                if self._idx < self.num_steps
                else self.features[: self.window_size]
            )
        )

        return obs, 0.0, done, truncated, info


# ============================================================
# Evaluation
# ============================================================


def evaluate(
    env_test: TradingEnv,
    policy_fn,
    min_trades: int = 50,
    max_drawdown: float = 0.20,
    r_min: float = 0.0,
    vpin_max_z: float = 0.0,
    fee_mult: float = 1.0,
) -> float:
    """Run policy on FULL test env, return Sortino ratio.

    policy_fn: callable(obs) -> action
    Returns Sortino ratio, or 0.0 if guardrails violated.

    Uses Sortino instead of Sharpe: only penalizes downside volatility.
    Runs the full test set (no 2000-step truncation).
    """
    obs, _ = env_test.reset(options={"sequential": True})
    step_returns = []
    max_dd = 0.0
    total_trades = 0

    # Trade-level tracking
    trade_pnls = []
    hold_durations = []
    entry_step = None
    entry_equity = 1.0
    prev_position = 0
    step_num = 0

    # Directional accuracy tracking
    directional_correct = 0
    directional_total = 0
    prev_action = 0

    # Run full test set — step directly, ignoring episode truncation
    while env_test._idx < env_test.num_steps:
        action = policy_fn(obs)
        # Regime gate 1: force flat when Hawkes branching below threshold
        if (
            r_min > 0
            and hasattr(env_test, "raw_hawkes")
            and env_test.raw_hawkes is not None
        ):
            if env_test.raw_hawkes[env_test._idx] < r_min:
                action = 0
        # Regime gate 2: force flat when VPIN too high (toxic flow, Theorem 13)
        # Uses z-scored VPIN from normalized features (index 2)
        if vpin_max_z > 0 and env_test._idx < len(env_test.features):
            vpin_z = env_test.features[env_test._idx, 2]  # normalized VPIN
            if vpin_z > vpin_max_z:
                action = 0
        obs, _, done, truncated, info = env_test.step(action)
        # Track directional accuracy: did PREVIOUS action's direction match this step's return?
        step_pnl = info.get("step_pnl", 0)
        if prev_action in (1, 2):
            directional_total += 1
            if (prev_action == 1 and step_pnl > 0) or (
                prev_action == 2 and step_pnl < 0
            ):
                directional_correct += 1
        prev_action = action
        step_returns.append(info["step_pnl"])
        max_dd = max(max_dd, info["drawdown"])
        total_trades = info["trade_count"]

        step_num += 1
        current_position = info["position"]
        current_equity = info["equity"]
        if prev_position == 0 and current_position != 0:
            # Entry
            entry_step = step_num
            entry_equity = current_equity
        elif prev_position != 0 and current_position == 0:
            # Exit back to flat
            trade_pnls.append(current_equity - entry_equity)
            if entry_step is not None:
                hold_durations.append(step_num - entry_step)
            entry_step = None
        elif (
            prev_position != 0
            and current_position != 0
            and prev_position != current_position
        ):
            # Flip (long→short or short→long): close + open
            trade_pnls.append(current_equity - entry_equity)
            if entry_step is not None:
                hold_durations.append(step_num - entry_step)
            entry_step = step_num
            entry_equity = current_equity
        prev_position = current_position

        # If truncated but not at end of data, reset episode counter and continue
        if truncated and env_test._idx < env_test.num_steps:
            env_test._episode_step = 0

    # Close any open position at end of data
    if prev_position != 0 and entry_step is not None:
        trade_pnls.append(info["equity"] - entry_equity)
        hold_durations.append(step_num - entry_step)

    returns = np.array(step_returns)

    # Guardrails
    if total_trades < min_trades:
        print(f"sortino: 0.000000 (only {total_trades} trades, min={min_trades})")
        print(f"num_trades: {total_trades}")
        print(f"max_drawdown: {max_dd:.4f}")
        return 0.0

    if max_dd > max_drawdown:
        print(f"sortino: 0.000000 (drawdown {max_dd:.4f} > {max_drawdown})")
        print(f"num_trades: {total_trades}")
        print(f"max_drawdown: {max_dd:.4f}")
        return 0.0

    # Sortino ratio (only penalizes downside vol)
    # Test period: use actual date range for annualization
    test_days = (
        datetime.strptime(TEST_END, "%Y-%m-%d") - datetime.strptime(VAL_END, "%Y-%m-%d")
    ).days
    steps_per_day = max(len(returns) / test_days, 1)

    mean_ret = returns.mean()
    downside_returns = np.minimum(returns, 0)
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(returns) > 0 else 1e-10

    if downside_std < 1e-10:
        sortino = 0.0
    else:
        sortino = mean_ret / downside_std * np.sqrt(steps_per_day)

    # Sharpe ratio (T27)
    std_ret = returns.std() if len(returns) > 1 else 1e-10
    sharpe = mean_ret / std_ret * np.sqrt(steps_per_day) if std_ret > 1e-10 else 0.0

    # Calmar ratio (T27: annualized return / max drawdown)
    annual_return = mean_ret * steps_per_day * 365
    calmar = annual_return / max_dd if max_dd > 1e-10 else 0.0

    # CVaR 95% (T28: mean of worst 5% of returns)
    sorted_returns = np.sort(returns)
    k = max(1, int(0.05 * len(returns)))
    cvar_95 = -np.mean(sorted_returns[:k])

    print(f"sortino: {sortino:.6f}")
    print(f"sharpe: {sharpe:.6f}")
    print(f"calmar: {calmar:.6f}")
    print(f"cvar_95: {cvar_95:.6f}")
    print(f"num_trades: {total_trades}")
    print(f"max_drawdown: {max_dd:.4f}")

    # Trade-level metrics
    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls)
        avg_profit = np.mean(trade_pnls)
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_wins / gross_losses
        avg_hold = np.mean(hold_durations) if hold_durations else 0
        print(f"win_rate: {win_rate:.4f}")
        print(f"avg_profit_per_trade: {avg_profit:.6f}")
        print(f"profit_factor: {profit_factor:.4f}")
        print(f"avg_hold_steps: {avg_hold:.0f}")

        # Kelly optimal fee_mult (Theorem 3)
        c = FEE_BPS / 10000
        p_w = len(wins) / len(trade_pnls)
        p_l = len(losses) / len(trade_pnls)
        if (p_w + p_l) > 0 and c > 0:
            f_kelly = (p_w - p_l) * (1 - c) / ((p_w + p_l) * c)
            print(f"f_opt_kelly: {f_kelly:.4f}")

    # Regime gate diagnostics
    alpha_min_val = 0.5 + 1.0 / (2.0 * fee_mult) if fee_mult > 0 else 1.0
    print(f"alpha_min: {alpha_min_val:.4f}")
    if directional_total > 0:
        print(f"empirical_accuracy: {directional_correct / directional_total:.4f}")
    if (
        r_min > 0
        and hasattr(env_test, "raw_hawkes")
        and env_test.raw_hawkes is not None
    ):
        filtered = np.sum(env_test.raw_hawkes[: env_test.num_steps] < r_min)
        print(f"regime_filter_rate: {filtered / env_test.num_steps:.4f}")
        print(f"hawkes_branching_mean: {np.mean(env_test.raw_hawkes):.4f}")

    return sortino


# ============================================================
# Helpers
# ============================================================


def make_env(
    symbol: str = "BTC",
    split: str = "train",
    window_size: int = 50,
    trade_batch: int = 100,
    min_hold: int = 1,
) -> TradingEnv:
    """Create a TradingEnv for the given symbol and data split."""
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }
    start, end = splits[split]

    # Try cache first
    cached = load_cached(symbol, CACHE_DIR, start, end, trade_batch)
    if cached is not None:
        features, timestamps, prices, raw_hawkes = cached
    else:
        # Need to compute - run prepare_data for this split
        print(f"Cache miss for {symbol} {split}, computing features...")
        trades_df = load_trades(symbol, start, end)
        orderbook_df = load_orderbook(symbol, start, end)
        funding_df = load_funding(symbol, start, end)

        if USE_V9:
            features, timestamps, prices, raw_hawkes = compute_features_v9(
                trades_df, orderbook_df, funding_df, trade_batch
            )
            if len(features) > 0:
                features = normalize_features_v9(features)
                cache_features(
                    symbol,
                    features,
                    timestamps,
                    prices,
                    CACHE_DIR,
                    start,
                    end,
                    trade_batch,
                    raw_hawkes=raw_hawkes,
                )
        else:
            features, timestamps, prices = compute_features(
                trades_df, orderbook_df, funding_df, trade_batch
            )
            raw_hawkes = None
            if len(features) > 0:
                features = normalize_features(features)
                cache_features(
                    symbol,
                    features,
                    timestamps,
                    prices,
                    CACHE_DIR,
                    start,
                    end,
                    trade_batch,
                )

    env = TradingEnv(
        features, prices, window_size=window_size, fee_bps=FEE_BPS, min_hold=min_hold
    )
    env.raw_hawkes = raw_hawkes  # for regime gate in evaluate()
    return env


if __name__ == "__main__":
    data = prepare_data(DEFAULT_SYMBOLS)
    for sym, splits in data.items():
        for split_name, (features, timestamps, prices, raw_hawkes) in splits.items():
            print(
                f"{sym} {split_name}: features={features.shape}, steps={len(timestamps)}"
            )

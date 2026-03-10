#!/usr/bin/env python3
"""Data loading, feature engineering, and Gym environment for autoresearch-trading."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy as np
import pandas as pd

# === CONSTANTS (do not modify) ===
TRAIN_BUDGET_SECONDS = 300  # 5-minute training budget
TRAIN_START = "2025-10-16"
TRAIN_END = "2026-01-23"
VAL_END = "2026-02-17"
TEST_END = "2026-03-09"
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


def load_trades(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load trade data from Parquet files, sorted by ts_ms."""
    files = discover_parquet_files(DATA_ROOT, "trades", symbol, start_date, end_date)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def load_orderbook(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load orderbook data from Parquet files, sorted by ts_ms."""
    files = discover_parquet_files(DATA_ROOT, "orderbook", symbol, start_date, end_date)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def load_funding(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load funding rate data from Parquet files, sorted by ts_ms."""
    files = discover_parquet_files(DATA_ROOT, "funding", symbol, start_date, end_date)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


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
    """Compute features from raw data.

    Returns: (features, timestamps, prices) where features has shape (num_batches, num_features).
    """
    if trades_df.empty:
        return np.array([]), np.array([]), np.array([])

    # Normalize sides
    trades_df = trades_df.copy()
    trades_df["norm_side"] = trades_df["side"].apply(normalize_side)
    trades_df["is_buy"] = trades_df["norm_side"] == "buy"

    # Compute 95th percentile for large trade detection
    qty_95 = trades_df["qty"].quantile(0.95)

    # Group into batches
    num_trades = len(trades_df)
    num_batches = num_trades // trade_batch
    if num_batches == 0:
        return np.array([]), np.array([]), np.array([])

    # Trim to exact multiple
    trades_df = trades_df.iloc[: num_batches * trade_batch]

    # Pre-extract arrays for vectorized ops
    prices = trades_df["price"].values
    qtys = trades_df["qty"].values
    is_buy = trades_df["is_buy"].values
    ts_ms = trades_df["ts_ms"].values

    # Reshape into batches
    prices_batched = prices[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )
    qtys_batched = qtys[: num_batches * trade_batch].reshape(num_batches, trade_batch)
    is_buy_batched = is_buy[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )
    ts_batched = ts_ms[: num_batches * trade_batch].reshape(num_batches, trade_batch)

    # Compute per-batch features
    # VWAP
    total_value = (prices_batched * qtys_batched).sum(axis=1)
    total_qty = qtys_batched.sum(axis=1)
    vwap = np.where(total_qty > 0, total_value / total_qty, prices_batched[:, -1])

    # Returns (log returns of VWAP)
    returns = np.zeros(num_batches)
    returns[1:] = np.log(vwap[1:] / np.maximum(vwap[:-1], 1e-10))

    # Buy/sell volumes
    buy_vol = (qtys_batched * is_buy_batched).sum(axis=1)
    sell_vol = (qtys_batched * ~is_buy_batched).sum(axis=1)
    net_volume = buy_vol - sell_vol

    # Trade count per batch (constant = trade_batch, but useful as feature)
    trade_count = np.full(num_batches, trade_batch, dtype=np.float64)

    # Buy ratio
    buy_ratio = is_buy_batched.sum(axis=1) / trade_batch

    # CVD delta (cumulative net volume change per batch)
    cvd_delta = net_volume  # Change in CVD over the batch

    # TFI: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    total_vol = buy_vol + sell_vol
    tfi = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0.0)

    # Large trade count
    large_trade_count = np.zeros(num_batches)
    for i in range(num_batches):
        batch_qtys = qtys_batched[i]
        large_trade_count[i] = (batch_qtys > qty_95).sum()

    # Liquidation cascade proxy
    price_accel = np.zeros(num_batches)
    price_accel[2:] = np.abs(
        returns[2:] - returns[1:-1]
    )  # need 2 returns for acceleration
    liq_cascade_magnitude = large_trade_count * price_accel
    liq_cascade_direction = np.sign(returns) * liq_cascade_magnitude

    # VPIN (flow toxicity): rolling mean of |TFI|
    abs_tfi = np.abs(tfi)
    abs_tfi_series = pd.Series(abs_tfi)
    vpin = abs_tfi_series.rolling(window=50, min_periods=1).mean().values

    # Multi-horizon realized volatility
    returns_series = pd.Series(returns)
    realvol_short = (
        returns_series.rolling(window=10, min_periods=1).std().fillna(0).values
    )
    realvol_med = (
        returns_series.rolling(window=50, min_periods=1).std().fillna(0).values
    )
    realvol_long = (
        returns_series.rolling(window=200, min_periods=1).std().fillna(0).values
    )

    # Batch timestamps (use last trade in batch)
    batch_timestamps = ts_batched[:, -1]
    batch_prices = vwap

    # === Orderbook features ===
    ob_features = np.zeros(
        (num_batches, 17)
    )  # bid_depth, ask_depth, imbalance, spread, 5 bid vols, 5 ask vols, microprice, microprice_dev, ofi

    if not orderbook_df.empty:
        ob_ts = orderbook_df["ts_ms"].values
        ob_bids = orderbook_df["bids"].values
        ob_asks = orderbook_df["asks"].values

        ob_idx = 0
        prev_bid_vols = np.zeros(5)
        prev_ask_vols = np.zeros(5)
        prev_ob_valid = False
        for i in range(num_batches):
            t = batch_timestamps[i]
            # Advance to most recent orderbook snapshot <= batch timestamp
            while ob_idx < len(ob_ts) - 1 and ob_ts[ob_idx + 1] <= t:
                ob_idx += 1

            if ob_idx < len(ob_ts) and ob_ts[ob_idx] <= t:
                bids = ob_bids[ob_idx]
                asks = ob_asks[ob_idx]

                if len(bids) > 0 and len(asks) > 0:
                    # Total depths
                    bid_depth = sum(
                        b["qty"] for b in bids if isinstance(b, dict) and "qty" in b
                    )
                    ask_depth = sum(
                        a["qty"] for a in asks if isinstance(a, dict) and "qty" in a
                    )

                    total_depth = bid_depth + ask_depth
                    imbalance = (
                        (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
                    )

                    # Spread
                    best_bid = bids[0]["price"] if isinstance(bids[0], dict) else 0
                    best_ask = asks[0]["price"] if isinstance(asks[0], dict) else 0
                    mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 1
                    spread_bps = (best_ask - best_bid) / mid * 10000

                    ob_features[i, 0] = bid_depth
                    ob_features[i, 1] = ask_depth
                    ob_features[i, 2] = imbalance
                    ob_features[i, 3] = spread_bps

                    # Per-level volumes (up to 5 levels)
                    for lvl in range(min(5, len(bids))):
                        if isinstance(bids[lvl], dict) and "qty" in bids[lvl]:
                            ob_features[i, 4 + lvl] = bids[lvl]["qty"]
                    for lvl in range(min(5, len(asks))):
                        if isinstance(asks[lvl], dict) and "qty" in asks[lvl]:
                            ob_features[i, 9 + lvl] = asks[lvl]["qty"]

                    # Microprice
                    best_bid_qty = bids[0]["qty"] if isinstance(bids[0], dict) else 0
                    best_ask_qty = asks[0]["qty"] if isinstance(asks[0], dict) else 0
                    total_best_qty = best_bid_qty + best_ask_qty
                    if total_best_qty > 0:
                        microprice = (
                            best_bid * best_ask_qty + best_ask * best_bid_qty
                        ) / total_best_qty
                    else:
                        microprice = mid
                    ob_features[i, 14] = microprice
                    ob_features[i, 15] = microprice - mid

                    # OFI (multi-level)
                    curr_bid_vols = np.array(
                        [
                            (
                                bids[lvl]["qty"]
                                if lvl < len(bids) and isinstance(bids[lvl], dict)
                                else 0.0
                            )
                            for lvl in range(5)
                        ]
                    )
                    curr_ask_vols = np.array(
                        [
                            (
                                asks[lvl]["qty"]
                                if lvl < len(asks) and isinstance(asks[lvl], dict)
                                else 0.0
                            )
                            for lvl in range(5)
                        ]
                    )
                    if prev_ob_valid:
                        weights = np.array([1.0, 0.5, 1 / 3, 0.25, 0.2])
                        delta_bid = curr_bid_vols - prev_bid_vols
                        delta_ask = curr_ask_vols - prev_ask_vols
                        ob_features[i, 16] = (weights * (delta_bid - delta_ask)).sum()
                    prev_bid_vols = curr_bid_vols.copy()
                    prev_ask_vols = curr_ask_vols.copy()
                    prev_ob_valid = True

    # === Funding features ===
    funding_features = np.zeros((num_batches, 2))  # rate, rate_change

    if not funding_df.empty:
        fund_ts = funding_df["ts_ms"].values
        fund_rate = funding_df["rate"].values

        fund_idx = 0
        prev_rate = 0.0
        for i in range(num_batches):
            t = batch_timestamps[i]
            while fund_idx < len(fund_ts) - 1 and fund_ts[fund_idx + 1] <= t:
                fund_idx += 1

            if fund_idx < len(fund_ts) and fund_ts[fund_idx] <= t:
                rate = fund_rate[fund_idx]
                funding_features[i, 0] = rate
                funding_features[i, 1] = rate - prev_rate
                prev_rate = rate

    # Combine all features
    trade_features = np.column_stack(
        [
            vwap,
            returns,
            net_volume,
            trade_count,
            buy_ratio,
            cvd_delta,
            tfi,
            large_trade_count,
            vpin,
            liq_cascade_magnitude,
            liq_cascade_direction,
            realvol_short,
            realvol_med,
            realvol_long,
        ]
    )

    features = np.hstack([trade_features, ob_features, funding_features])

    return features, batch_timestamps, batch_prices


# Indices of features that get robust (median/IQR) normalization.
# These are tail-heavy: net_volume, large_trade_count, vpin,
# liq_cascade_mag, liq_cascade_dir, bid_depth, ask_depth, microprice
ROBUST_FEATURE_INDICES = {2, 7, 8, 9, 10, 14, 15, 28}


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

    return normalized


_FEATURE_VERSION = "v2"  # bump when feature set changes


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
) -> None:
    """Save features to .npz cache file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    np.savez_compressed(path, features=features, timestamps=timestamps, prices=prices)
    print(f"Cached {symbol} features to {path}")


def load_cached(
    symbol: str,
    cache_dir: Path,
    start: str,
    end: str,
    trade_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load features from .npz cache if exists."""
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    if path.exists():
        data = np.load(path)
        print(f"Loaded {symbol} features from cache ({path})")
        return data["features"], data["timestamps"], data["prices"]
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

    Returns dict of {symbol: {train: (features, timestamps, prices), val: ..., test: ...}}
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
                    result[symbol][split_name] = cached
                    continue

            print(f"Computing features for {symbol} {split_name} ({start} to {end})...")

            trades_df = load_trades(symbol, start, end)
            orderbook_df = load_orderbook(symbol, start, end)
            funding_df = load_funding(symbol, start, end)

            print(
                f"  Loaded {len(trades_df)} trades, {len(orderbook_df)} orderbook snapshots, {len(funding_df)} funding rates"
            )

            features, timestamps, prices = compute_features(
                trades_df, orderbook_df, funding_df, trade_batch
            )

            if len(features) > 0:
                features = normalize_features(features)

            print(f"  Features shape: {features.shape}")

            # Cache
            cache_features(
                symbol, features, timestamps, prices, CACHE_DIR, start, end, trade_batch
            )

            result[symbol][split_name] = (features, timestamps, prices)

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
    ):
        super().__init__()

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window_size = window_size
        self.fee_bps = fee_bps
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
        self._episode_step = 0

        return self._get_obs(), {}

    def step(self, action: int):
        prev_position = self._position
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
    env_test: TradingEnv, policy_fn, min_trades: int = 50, max_drawdown: float = 0.20
) -> float:
    """Run policy on test env, return val_sharpe.

    policy_fn: callable(obs) -> action
    Returns sharpe ratio, or 0.0 if guardrails violated.
    """
    obs, _ = env_test.reset(options={"sequential": True})
    step_returns = []
    max_dd = 0.0
    total_trades = 0
    done = False
    truncated = False

    while not (done or truncated):
        action = policy_fn(obs)
        obs, _, done, truncated, info = env_test.step(action)
        step_returns.append(info["step_pnl"])
        max_dd = max(max_dd, info["drawdown"])
        total_trades = info["trade_count"]

    returns = np.array(step_returns)

    # Guardrails
    if total_trades < min_trades:
        print(f"val_sharpe: 0.000000 (only {total_trades} trades, min={min_trades})")
        print(f"num_trades: {total_trades}")
        print(f"max_drawdown: {max_dd:.4f}")
        return 0.0

    if max_dd > max_drawdown:
        print(f"val_sharpe: 0.000000 (drawdown {max_dd:.4f} > {max_drawdown})")
        print(f"num_trades: {total_trades}")
        print(f"max_drawdown: {max_dd:.4f}")
        return 0.0

    # Sharpe ratio
    # Estimate steps per day from data (trade batches per day)
    steps_per_day = len(returns) / max(1, (env_test.num_steps / len(returns)))
    if steps_per_day < 1:
        steps_per_day = 1000  # reasonable default

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret < 1e-10:
        sharpe = 0.0
    else:
        sharpe = mean_ret / std_ret * np.sqrt(steps_per_day)

    print(f"val_sharpe: {sharpe:.6f}")
    print(f"num_trades: {total_trades}")
    print(f"max_drawdown: {max_dd:.4f}")

    return sharpe


# ============================================================
# Helpers
# ============================================================


def make_env(
    symbol: str = "BTC",
    split: str = "train",
    window_size: int = 50,
    trade_batch: int = 100,
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
        features, timestamps, prices = cached
    else:
        # Need to compute - run prepare_data for this split
        print(f"Cache miss for {symbol} {split}, computing features...")
        trades_df = load_trades(symbol, start, end)
        orderbook_df = load_orderbook(symbol, start, end)
        funding_df = load_funding(symbol, start, end)

        features, timestamps, prices = compute_features(
            trades_df, orderbook_df, funding_df, trade_batch
        )

        if len(features) > 0:
            features = normalize_features(features)
            cache_features(
                symbol, features, timestamps, prices, CACHE_DIR, start, end, trade_batch
            )

    return TradingEnv(features, prices, window_size=window_size, fee_bps=FEE_BPS)


if __name__ == "__main__":
    data = prepare_data(DEFAULT_SYMBOLS)
    for sym, splits in data.items():
        for split_name, (features, timestamps, prices) in splits.items():
            print(
                f"{sym} {split_name}: features={features.shape}, steps={len(timestamps)}"
            )

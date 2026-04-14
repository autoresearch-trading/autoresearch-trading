# tape/constants.py
"""Canonical constants shared across the tape pipeline.

All values here are load-bearing — changing any one must be justified against the
spec at docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md and
the Step 0 measurements under docs/experiments/.
"""

from __future__ import annotations

# ---- Universe ----
SYMBOLS: tuple[str, ...] = (
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
)
HELD_OUT_SYMBOL: str = "AVAX"  # Gate 3 — gotcha #25
PRETRAINING_SYMBOLS: tuple[str, ...] = tuple(s for s in SYMBOLS if s != HELD_OUT_SYMBOL)
LIQUID_CONTRASTIVE_SYMBOLS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "LINK",
    "LTC",
)

# ---- Dates ----
PREAPRIL_START: str = "2025-10-16"
PREAPRIL_END: str = "2026-03-31"  # inclusive; April 1+ has different dedup rules
APRIL_START: str = "2026-04-01"
APRIL_HELDOUT_START: str = "2026-04-14"  # do NOT touch — gotcha #17

# ---- Windowing ----
WINDOW_LEN: int = 200
STRIDE_PRETRAIN: int = 50
STRIDE_EVAL: int = 200
ROLLING_WINDOW: int = 1000  # rolling median/std window for normalisation
KYLE_LAMBDA_WINDOW: int = 50  # snapshots (~20 min @ 24s cadence)
OFI_WINDOW: int = 5  # snapshots (~120s)
EMBARGO_EVENTS: int = 600  # gotcha #12

# ---- Horizons ----
DIRECTION_HORIZONS: tuple[int, ...] = (10, 50, 100, 500)

# ---- Wyckoff label parameters ----
# See docs/experiments/step0-falsifiability-prereqs.md for the calibration that
# produced these values.
SPRING_SIGMA_MULT: float = 3.0  # recalibrated from 2.0 — prereq #4
SPRING_LOOKBACK: int = 50
SPRING_PRIOR_LEN: int = 10
CLIMAX_Z_THRESHOLD: float = 2.0
STRESS_PCTL: float = 0.90
DEPTH_PCTL: float = 0.90
INFORMED_KYLE_PCTL: float = 0.75
INFORMED_OFI_PCTL: float = 0.50

# ---- MEM masking ----
MEM_MASK_FRACTION: float = 0.15
MEM_BLOCK_LEN: int = 5
# Features that use random-position masking instead of block masking because
# their lag-5 autocorrelation exceeds 0.8 (step0-falsifiability-prereqs.md #5).
MEM_RANDOM_MASK_FEATURES: tuple[str, ...] = ("prev_seq_time_span", "kyle_lambda")
# Features that are EXCLUDED from MEM reconstruction entirely — trivially
# copyable from neighbours (gotcha #22).
MEM_EXCLUDED_FEATURES: tuple[str, ...] = (
    "delta_imbalance_L1",
    "kyle_lambda",
    "cum_ofi_5",
)

# ---- Feature layout (order matters — this is the channel order in the tensor) ----
TRADE_FEATURES: tuple[str, ...] = (
    "log_return",
    "log_total_qty",
    "is_open",
    "time_delta",
    "num_fills",
    "book_walk",
    "effort_vs_result",
    "climax_score",
    "prev_seq_time_span",
)
OB_FEATURES: tuple[str, ...] = (
    "log_spread",
    "imbalance_L1",
    "imbalance_L5",
    "depth_ratio",
    "trade_vs_mid",
    "delta_imbalance_L1",
    "kyle_lambda",
    "cum_ofi_5",
)
FEATURE_NAMES: tuple[str, ...] = TRADE_FEATURES + OB_FEATURES
assert len(FEATURE_NAMES) == 17, "Feature count drift — see spec §Input Representation"

# ---- Data paths ----
DATA_ROOT: str = "data"
TRADES_GLOB: str = "data/trades/symbol={sym}/date={date}/*.parquet"
OB_GLOB: str = "data/orderbook/symbol={sym}/date={date}/*.parquet"
CACHE_ROOT: str = "data/cache/v1"
CACHE_SCHEMA_VERSION: int = 1

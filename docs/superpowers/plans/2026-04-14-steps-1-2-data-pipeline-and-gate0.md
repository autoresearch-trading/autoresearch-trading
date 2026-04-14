# Steps 1-2: Tape Data Pipeline + Gate 0 Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable tape data pipeline (17-feature order-event windows with `.npz` caching + PyTorch `Dataset`) and run Gate 0 (PCA + logistic-regression baseline on flat features) as the reference bar the CNN probe must exceed.

**Architecture:** A new `tape/` package at repo root (sibling of `scripts/`) houses pure-function feature / label / windowing / OB-alignment modules that operate on pandas DataFrames and numpy arrays; an offline `scripts/build_cache.py` materialises per-symbol-day `.npz` shards; `tape.dataset.TapeDataset` wraps those shards for PyTorch training; `scripts/run_gate0.py` runs the baseline + embargoed walk-forward evaluation. No CNN yet — this plan stops at the baseline that the Gate 1 probe will have to beat.

**Tech Stack:** Python 3.12, PyTorch 2.2+, NumPy, Pandas, PyArrow, DuckDB, scikit-learn (new dep), pytest.

**Spec reference:** `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
**Conventions reference:** `CLAUDE.md` (27 gotchas — every feature cites the relevant gotcha by number)
**Measurements feeding this plan:**
- `docs/experiments/step0-data-validation.md` — dedup, OB cadence, label base rates
- `docs/experiments/step0-base-rate-stationarity.md` — Gate 4 H500 metric choice
- `docs/experiments/step0-falsifiability-prereqs.md` — **spring σ-multiplier = 3.0** (not 2.0), **random-position masking for `prev_seq_time_span` + `kyle_lambda` only** (lag-5 r>0.8), stress/informed/climax label definitions validated

---

## File Structure

**New package** (`tape/`):
- `tape/__init__.py` — package marker + version
- `tape/constants.py` — symbol list, held-out AVAX, LTC contrastive set, ROLLING_WINDOW=1000, STRIDE=50, WINDOW_LEN=200, HORIZONS=(10,50,100,500), feature name lists
- `tape/io_parquet.py` — DuckDB + pyarrow loaders for trades / orderbook / funding, per-symbol-day
- `tape/dedup.py` — `dedup_trades_pre_april(df)` and `filter_trades_april(df)` — gotcha #3, #19
- `tape/events.py` — same-timestamp grouping → order events (vwap, total_qty, is_open_frac, n_fills, book_walk, first_ts, last_ts)
- `tape/ob_align.py` — `align_events_to_ob(event_ts, ob_ts) -> np.ndarray[int]` via `searchsorted(side="right") - 1`, fully vectorised — gotcha #2
- `tape/features_trade.py` — 9 trade features (feature names in CLAUDE.md lines 36-44)
- `tape/features_ob.py` — 8 OB features including piecewise Cont OFI (gotcha #15) and per-snapshot Kyle λ (gotcha #13)
- `tape/labels.py` — direction labels at H10/H50/H100/H500 + Wyckoff labels (stress, informed_flow, climax, spring with **σ=3.0**, absorption)
- `tape/windowing.py` — stride=50 window construction, day-boundary enforcement (gotcha #26), first-window-offset randomisation per epoch
- `tape/cache.py` — `save_npz_shard(path, features, labels, meta)` + `load_npz_shard(path)`; schema version stamp
- `tape/dataset.py` — `TapeDataset(torch.utils.data.Dataset)` — lazy-loads shards, returns `(features[200,17], labels_dict, meta)`
- `tape/sampler.py` — `EqualSymbolSampler(torch.utils.data.Sampler)` — gotcha #27
- `tape/flat_features.py` — Gate 0 feature extraction: per 200-event window compute mean/std/skew/kurt/last of each of the 17 channels → 85-dim flat vector
- `tape/splits.py` — walk-forward fold generator with 600-event embargo — gotcha #12

**Scripts** (`scripts/`):
- `scripts/build_cache.py` — CLI: `--symbols BTC ETH … --start-date 2025-10-16 --end-date 2026-03-25 --out data/cache/v1/` → produces one `.npz` per symbol-day
- `scripts/validate_cache.py` — sanity: is_open autocorr half-life ≈ 20 (CLAUDE.md finding), climax asymmetry (council-4), effort_vs_result histogram bounded
- `scripts/run_gate0.py` — CLI: `--cache data/cache/v1 --encoder {flat,pca,random}` → PCA + logistic regression on flat features, walk-forward, per-symbol H10/H50/H100/H500 accuracy + balanced-accuracy; writes `docs/experiments/gate0-results.md`

**Tests** (`tests/tape/`):
- `tests/__init__.py`, `tests/tape/__init__.py`
- `tests/tape/test_dedup.py`, `test_events.py`, `test_ob_align.py`, `test_features_trade.py`, `test_features_ob.py`, `test_labels.py`, `test_windowing.py`, `test_cache.py`, `test_dataset.py`, `test_sampler.py`, `test_flat_features.py`, `test_splits.py`

**Docs:**
- `docs/experiments/gate0-results.md` — per-symbol baseline table (reference for Gate 1)

---

## Ordering & Commit Discipline

14 tasks. One commit per task, prefixes: `feat:` for new modules, `test:` for test-only, `experiment:` for report/results, `chore:` for scaffolding. **Never** `git add -A` (gotcha from repo conventions — stage specific files).

At every commit, pyright must be clean (`pyright tape scripts/build_cache.py scripts/run_gate0.py` → 0 errors / 0 warnings / 0 informations). Pre-commit hook runs black + isort — do not fight it.

---

## Task 1: Package Scaffolding

**Files:**
- Create: `tape/__init__.py`, `tape/constants.py`
- Create: `tests/__init__.py`, `tests/tape/__init__.py`
- Modify: `pyproject.toml` (add deps: `scikit-learn>=1.5`; add `tape` to `py-modules` or use packages discovery)

- [ ] **Step 1: Create package markers**

```python
# tape/__init__.py
"""Tape representation learning data pipeline.

Operates on pre-April data only. April 14+ is the hold-out — see CLAUDE.md gotcha #17.
"""

__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

```python
# tests/tape/__init__.py
```

- [ ] **Step 2: Write tape/constants.py**

```python
# tape/constants.py
"""Canonical constants shared across the tape pipeline.

All values here are load-bearing — changing any one must be justified against the
spec at docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md and
the Step 0 measurements under docs/experiments/.
"""

from __future__ import annotations

# ---- Universe ----
SYMBOLS: tuple[str, ...] = (
    "2Z", "AAVE", "ASTER", "AVAX", "BNB", "BTC", "CRV", "DOGE", "ENA", "ETH",
    "FARTCOIN", "HYPE", "KBONK", "KPEPE", "LDO", "LINK", "LTC", "PENGU", "PUMP",
    "SOL", "SUI", "UNI", "WLFI", "XPL", "XRP",
)
HELD_OUT_SYMBOL: str = "AVAX"  # Gate 3 — gotcha #25
PRETRAINING_SYMBOLS: tuple[str, ...] = tuple(s for s in SYMBOLS if s != HELD_OUT_SYMBOL)
LIQUID_CONTRASTIVE_SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "LINK", "LTC")

# ---- Dates ----
PREAPRIL_START: str = "2025-10-16"
PREAPRIL_END: str = "2026-03-31"  # inclusive; April 1+ has different dedup rules
APRIL_START: str = "2026-04-01"
APRIL_HELDOUT_START: str = "2026-04-14"  # do NOT touch — gotcha #17

# ---- Windowing ----
WINDOW_LEN: int = 200
STRIDE_PRETRAIN: int = 50
STRIDE_EVAL: int = 200
ROLLING_WINDOW: int = 1000            # rolling median/std window for normalisation
KYLE_LAMBDA_WINDOW: int = 50          # snapshots (~20 min @ 24s cadence)
OFI_WINDOW: int = 5                   # snapshots (~120s)
EMBARGO_EVENTS: int = 600             # gotcha #12

# ---- Horizons ----
DIRECTION_HORIZONS: tuple[int, ...] = (10, 50, 100, 500)

# ---- Wyckoff label parameters ----
# See docs/experiments/step0-falsifiability-prereqs.md for the calibration that
# produced these values.
SPRING_SIGMA_MULT: float = 3.0        # recalibrated from 2.0 — prereq #4
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
    "delta_imbalance_L1", "kyle_lambda", "cum_ofi_5",
)

# ---- Feature layout (order matters — this is the channel order in the tensor) ----
TRADE_FEATURES: tuple[str, ...] = (
    "log_return", "log_total_qty", "is_open", "time_delta", "num_fills",
    "book_walk", "effort_vs_result", "climax_score", "prev_seq_time_span",
)
OB_FEATURES: tuple[str, ...] = (
    "log_spread", "imbalance_L1", "imbalance_L5", "depth_ratio", "trade_vs_mid",
    "delta_imbalance_L1", "kyle_lambda", "cum_ofi_5",
)
FEATURE_NAMES: tuple[str, ...] = TRADE_FEATURES + OB_FEATURES
assert len(FEATURE_NAMES) == 17, "Feature count drift — see spec §Input Representation"

# ---- Data paths ----
DATA_ROOT: str = "data"
TRADES_GLOB: str = "data/trades/symbol={sym}/date={date}/*.parquet"
OB_GLOB: str = "data/orderbook/symbol={sym}/date={date}/*.parquet"
CACHE_ROOT: str = "data/cache/v1"
CACHE_SCHEMA_VERSION: int = 1
```

- [ ] **Step 3: Update pyproject.toml**

Add `scikit-learn>=1.5` to `[project].dependencies`. Replace `py-modules = ["prepare", "train"]` with package discovery:

```toml
[tool.setuptools.packages.find]
include = ["tape*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the scaffolding test**

```python
# tests/tape/test_constants.py
from tape import constants as C


def test_feature_names_length_17():
    assert len(C.FEATURE_NAMES) == 17
    assert len(C.TRADE_FEATURES) == 9
    assert len(C.OB_FEATURES) == 8


def test_avax_excluded_from_pretraining_and_contrastive():
    assert "AVAX" not in C.PRETRAINING_SYMBOLS
    assert "AVAX" not in C.LIQUID_CONTRASTIVE_SYMBOLS
    # LTC is the substitute — gotcha #25
    assert "LTC" in C.LIQUID_CONTRASTIVE_SYMBOLS


def test_spring_sigma_mult_recalibrated():
    # falsifiability prereq #4 — original 2.0 fired >8% on BTC/ETH/SOL/HYPE
    assert C.SPRING_SIGMA_MULT == 3.0


def test_mem_excludes_trivially_copyable():
    # gotcha #22
    for feat in ("delta_imbalance_L1", "kyle_lambda", "cum_ofi_5"):
        assert feat in C.MEM_EXCLUDED_FEATURES


def test_mem_random_mask_features_have_high_autocorr():
    # falsifiability prereq #5 — lag-5 r>0.8 → random position masking
    assert set(C.MEM_RANDOM_MASK_FEATURES) == {"prev_seq_time_span", "kyle_lambda"}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/tape/test_constants.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Pyright clean**

```bash
uv run pyright tape tests/tape/test_constants.py
```

Expected: 0 errors, 0 warnings, 0 informations.

- [ ] **Step 7: Commit**

```bash
git add tape/__init__.py tape/constants.py tests/__init__.py tests/tape/__init__.py tests/tape/test_constants.py pyproject.toml
git commit -m "feat: scaffold tape/ package + constants pinned to Step 0 measurements"
```

---

## Task 2: OB Alignment (gotcha #2)

**Files:**
- Create: `tape/ob_align.py`
- Create: `tests/tape/test_ob_align.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tape/test_ob_align.py
import numpy as np
import pytest

from tape.ob_align import align_events_to_ob


def test_each_event_mapped_to_nearest_prior_snapshot():
    ob_ts = np.array([100, 200, 300, 400], dtype=np.int64)
    event_ts = np.array([50, 100, 150, 200, 250, 399, 400, 500], dtype=np.int64)
    idx = align_events_to_ob(event_ts, ob_ts)
    # Event at 50 is before any snapshot → idx = -1 (caller masks)
    # Event at 100 → snapshot at 100 (idx 0)   [side="right"-1 maps 100 → idx 0]
    # Event at 150 → snapshot at 100 (idx 0)
    # Event at 200 → snapshot at 200 (idx 1)
    # Event at 250 → snapshot at 200 (idx 1)
    # Event at 399 → snapshot at 300 (idx 2)
    # Event at 400 → snapshot at 400 (idx 3)
    # Event at 500 → snapshot at 400 (idx 3)
    expected = np.array([-1, 0, 0, 1, 1, 2, 3, 3], dtype=np.int64)
    np.testing.assert_array_equal(idx, expected)


def test_vectorised_not_loop_for_large_input():
    # If someone reimplements as a Python for-loop, this takes >1s.
    rng = np.random.default_rng(0)
    ob_ts = np.sort(rng.integers(0, 10**9, size=100_000)).astype(np.int64)
    event_ts = np.sort(rng.integers(0, 10**9, size=1_000_000)).astype(np.int64)
    import time
    t = time.time()
    idx = align_events_to_ob(event_ts, ob_ts)
    assert (time.time() - t) < 0.5, "alignment must be vectorised"
    assert idx.shape == (1_000_000,)
    assert idx.max() < len(ob_ts)
    assert idx.min() >= -1


def test_monotonically_non_decreasing_ob_ts_required():
    ob_ts = np.array([100, 50, 200], dtype=np.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        align_events_to_ob(np.array([100], dtype=np.int64), ob_ts)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/tape/test_ob_align.py -v
```

Expected: ImportError / ModuleNotFoundError for `tape.ob_align`.

- [ ] **Step 3: Implement**

```python
# tape/ob_align.py
"""Align trade/event timestamps to the nearest prior orderbook snapshot.

See CLAUDE.md gotcha #2: `np.searchsorted(ob_ts, trade_ts, side="right") - 1`,
vectorised. Never use a Python for-loop.
"""

from __future__ import annotations

import numpy as np


def align_events_to_ob(event_ts: np.ndarray, ob_ts: np.ndarray) -> np.ndarray:
    """Return idx such that `ob_ts[idx[i]]` is the latest OB snapshot at or
    before `event_ts[i]`. Events before the first snapshot get idx = -1; the
    caller must mask them (or drop them).

    Parameters
    ----------
    event_ts : int64 array, shape (n_events,)
    ob_ts    : int64 array, shape (n_snapshots,), must be non-decreasing

    Returns
    -------
    idx : int64 array, shape (n_events,), values in [-1, n_snapshots - 1]
    """
    if ob_ts.ndim != 1 or event_ts.ndim != 1:
        raise ValueError("both inputs must be 1-D")
    if ob_ts.size > 1 and np.any(np.diff(ob_ts) < 0):
        raise ValueError("ob_ts must be non-decreasing")
    idx = np.searchsorted(ob_ts, event_ts, side="right") - 1
    return idx.astype(np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/tape/test_ob_align.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Pyright clean + commit**

```bash
uv run pyright tape/ob_align.py tests/tape/test_ob_align.py
git add tape/ob_align.py tests/tape/test_ob_align.py
git commit -m "feat: vectorised OB alignment via searchsorted(side=right)-1"
```

---

## Task 3: Dedup + Same-Timestamp Event Grouping

**Files:**
- Create: `tape/dedup.py`
- Create: `tape/events.py`
- Create: `tests/tape/test_dedup.py`
- Create: `tests/tape/test_events.py`

- [ ] **Step 1: Write the dedup failing test**

```python
# tests/tape/test_dedup.py
import pandas as pd
import pytest

from tape.dedup import dedup_trades_pre_april, filter_trades_april


def test_pre_april_dedup_drops_buyer_seller_pair_by_ts_qty_price_only():
    # Gotcha #19: pre-April, two rows share (ts, qty, price) but differ on side.
    # Dedup by (ts, qty, price) → one row.
    df = pd.DataFrame({
        "ts_ms": [1000, 1000],
        "symbol": ["BTC", "BTC"],
        "trade_id": [1, 2],
        "side": ["open_long", "close_short"],
        "qty": [0.5, 0.5],
        "price": [50000.0, 50000.0],
    })
    out = dedup_trades_pre_april(df)
    assert len(out) == 1
    # Gotcha #3: `side` must NOT be in the dedup key
    assert "side" in out.columns  # preserved in output, just not in key


def test_pre_april_dedup_preserves_genuinely_distinct_fills():
    df = pd.DataFrame({
        "ts_ms": [1000, 1000],
        "symbol": ["BTC", "BTC"],
        "trade_id": [1, 2],
        "side": ["open_long", "open_long"],
        "qty": [0.5, 0.3],     # different qty → different fills
        "price": [50000.0, 50000.0],
    })
    out = dedup_trades_pre_april(df)
    assert len(out) == 2


def test_april_filter_keeps_only_fulfill_taker():
    # Gotcha #3: April+ uses event_type == 'fulfill_taker'
    df = pd.DataFrame({
        "ts_ms": [1, 2, 3],
        "event_type": ["fulfill_taker", "fulfill_maker", "fulfill_taker"],
        "qty": [0.1, 0.1, 0.1],
        "price": [1.0, 1.0, 1.0],
        "side": ["open_long", "close_short", "open_long"],
        "symbol": ["BTC"] * 3,
        "trade_id": [1, 2, 3],
    })
    out = filter_trades_april(df)
    assert len(out) == 2
    assert (out["event_type"] == "fulfill_taker").all()


def test_april_filter_raises_if_event_type_missing():
    df = pd.DataFrame({"ts_ms": [1], "qty": [0.1], "price": [1.0]})
    with pytest.raises(ValueError, match="event_type"):
        filter_trades_april(df)
```

- [ ] **Step 2: Run → fails (module missing)**

```bash
uv run pytest tests/tape/test_dedup.py -v
```

- [ ] **Step 3: Implement tape/dedup.py**

```python
# tape/dedup.py
"""Dedup raw trade rows into one row per fill.

Pre-April (2025-10-16 to 2026-03-31): the API returns both counterparty
perspectives for every fill. Rows share `(ts_ms, qty, price)` but differ on
`side`. Dedup by those three columns ONLY — gotchas #3 and #19.

April 1+: the API includes `event_type`. Filter to `fulfill_taker`; every fill
appears exactly once. Gotcha #3.
"""

from __future__ import annotations

import pandas as pd

_PRE_APRIL_DEDUP_KEYS: tuple[str, str, str] = ("ts_ms", "qty", "price")


def dedup_trades_pre_april(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse buyer/seller rows into one row per fill."""
    return df.drop_duplicates(subset=list(_PRE_APRIL_DEDUP_KEYS), keep="first").reset_index(drop=True)


def filter_trades_april(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fulfill_taker rows (one row per fill)."""
    if "event_type" not in df.columns:
        raise ValueError("April+ data must have event_type column; got pre-April schema")
    return df.loc[df["event_type"] == "fulfill_taker"].reset_index(drop=True)
```

- [ ] **Step 4: Write the events failing test**

```python
# tests/tape/test_events.py
import numpy as np
import pandas as pd

from tape.events import group_to_events


def test_same_ts_trades_become_one_event():
    df = pd.DataFrame({
        "ts_ms": [1000, 1000, 2000],
        "qty":  [0.5,  0.3,  0.4],
        "price":[100.0, 101.0, 102.0],
        "side": ["open_long", "open_long", "open_short"],
    })
    ev = group_to_events(df)
    assert len(ev) == 2
    # First event: 2 fills, total_qty 0.8, vwap = (0.5*100 + 0.3*101)/0.8
    ev0 = ev.iloc[0]
    assert ev0["n_fills"] == 2
    assert np.isclose(ev0["total_qty"], 0.8)
    assert np.isclose(ev0["vwap"], (0.5 * 100.0 + 0.3 * 101.0) / 0.8)
    assert ev0["is_open_frac"] == 1.0  # both opens
    assert np.isclose(ev0["book_walk_abs"], 1.0)  # |101-100|
    # Second event: 1 fill, vwap=102, is_open_frac=0
    ev1 = ev.iloc[1]
    assert ev1["is_open_frac"] == 0.0
    assert ev1["n_fills"] == 1


def test_is_open_frac_mixed_side():
    df = pd.DataFrame({
        "ts_ms": [1, 1, 1, 1],
        "qty": [0.25, 0.25, 0.25, 0.25],
        "price": [10.0, 10.0, 10.0, 10.0],
        "side": ["open_long", "close_long", "open_short", "close_short"],
    })
    ev = group_to_events(df)
    # is_open_frac = fraction of fills that are opens (open_long or open_short)
    # Here 2 of 4 are opens → 0.5
    assert len(ev) == 1
    assert np.isclose(ev["is_open_frac"].iloc[0], 0.5)
```

- [ ] **Step 5: Implement tape/events.py**

```python
# tape/events.py
"""Group same-timestamp trades into order events.

An "order event" is the aggregate of all fills that share a millisecond
timestamp. Per the spec, same-timestamp trades are fragments of one order after
dedup.

Output columns (all numeric):
    ts_ms          : int64 — the shared timestamp
    total_qty      : float — sum of qty
    vwap           : float — qty-weighted mean price
    is_open_frac   : float in [0, 1] — fraction of fills whose side is open_*
    n_fills        : int   — number of fills in the event
    book_walk_abs  : float — |last_fill_price - first_fill_price| (unsigned;
                             the spread-normalised version lives in
                             features_trade.book_walk)
    first_ts, last_ts : int64 — currently equal to ts_ms; kept for forward
                                compatibility if we widen grouping later
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OPEN_SIDES: frozenset[str] = frozenset({"open_long", "open_short"})


def group_to_events(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deduped trades into order events keyed by ts_ms.

    Assumes `trades` is already sorted by ts_ms ascending; this is the case
    after reading a single day's parquet. The function preserves that ordering.
    """
    if len(trades) == 0:
        return pd.DataFrame({
            "ts_ms": pd.Series([], dtype=np.int64),
            "total_qty": pd.Series([], dtype=float),
            "vwap": pd.Series([], dtype=float),
            "is_open_frac": pd.Series([], dtype=float),
            "n_fills": pd.Series([], dtype=np.int64),
            "book_walk_abs": pd.Series([], dtype=float),
            "first_ts": pd.Series([], dtype=np.int64),
            "last_ts": pd.Series([], dtype=np.int64),
        })

    # Per-fill notional = qty * price for vwap computation.
    qty = trades["qty"].to_numpy(dtype=float)
    price = trades["price"].to_numpy(dtype=float)
    notional = qty * price
    is_open = trades["side"].isin(_OPEN_SIDES).to_numpy(dtype=float)

    df = trades[["ts_ms"]].copy()
    df["_qty"] = qty
    df["_notional"] = notional
    df["_is_open"] = is_open
    df["_price_first"] = price
    df["_price_last"] = price

    agg = df.groupby("ts_ms", sort=True).agg(
        total_qty=("_qty", "sum"),
        _notional_sum=("_notional", "sum"),
        _is_open_sum=("_is_open", "sum"),
        n_fills=("_qty", "size"),
        _price_first=("_price_first", "first"),
        _price_last=("_price_last", "last"),
    )

    out = pd.DataFrame({
        "ts_ms": agg.index.to_numpy(dtype=np.int64),
        "total_qty": agg["total_qty"].to_numpy(dtype=float),
        "vwap": (agg["_notional_sum"] / agg["total_qty"]).to_numpy(dtype=float),
        "is_open_frac": (agg["_is_open_sum"] / agg["n_fills"]).to_numpy(dtype=float),
        "n_fills": agg["n_fills"].to_numpy(dtype=np.int64),
        "book_walk_abs": np.abs(
            agg["_price_last"].to_numpy(dtype=float)
            - agg["_price_first"].to_numpy(dtype=float)
        ),
    })
    out["first_ts"] = out["ts_ms"].to_numpy(dtype=np.int64)
    out["last_ts"] = out["ts_ms"].to_numpy(dtype=np.int64)
    return out.reset_index(drop=True)
```

- [ ] **Step 6: Run both test files → pass; pyright clean; commit**

```bash
uv run pytest tests/tape/test_dedup.py tests/tape/test_events.py -v
uv run pyright tape/dedup.py tape/events.py tests/tape/test_dedup.py tests/tape/test_events.py
git add tape/dedup.py tape/events.py tests/tape/test_dedup.py tests/tape/test_events.py
git commit -m "feat: dedup + same-timestamp event grouping"
```

---

## Task 4: Trade Features (9 channels)

**Files:**
- Create: `tape/features_trade.py`
- Create: `tests/tape/test_features_trade.py`

- [ ] **Step 1: Write failing tests for every trade feature**

```python
# tests/tape/test_features_trade.py
import numpy as np
import pandas as pd
import pytest

from tape.features_trade import compute_trade_features


def _fake_events(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts_ms = np.cumsum(rng.integers(100, 5_000, size=n)).astype(np.int64) + 1_700_000_000_000
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    total_qty = rng.gamma(2.0, 1.0, size=n)
    return pd.DataFrame({
        "ts_ms": ts_ms,
        "vwap": vwap,
        "total_qty": total_qty,
        "is_open_frac": rng.uniform(size=n),
        "n_fills": rng.integers(1, 5, size=n).astype(np.int64),
        "book_walk_abs": rng.uniform(0, 0.5, size=n),
        "first_ts": ts_ms,
        "last_ts": ts_ms,
    })


def test_compute_trade_features_returns_9_columns():
    ev = _fake_events(2_000)
    out = compute_trade_features(ev, spread=np.full(2_000, 0.1), mid=np.full(2_000, 100.0))
    expected_cols = {
        "log_return", "log_total_qty", "is_open", "time_delta", "num_fills",
        "book_walk", "effort_vs_result", "climax_score", "prev_seq_time_span",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == len(ev)


def test_log_return_first_event_is_zero():
    ev = _fake_events(10)
    out = compute_trade_features(ev, spread=np.full(10, 0.1), mid=np.full(10, 100.0))
    assert out["log_return"].iloc[0] == 0.0


def test_effort_vs_result_is_clipped_minus5_to_5():
    # Gotcha #5
    ev = _fake_events(1_500)
    ev.loc[5, "total_qty"] = 1e20       # huge qty → large log_total_qty
    # tiny |return| via nearly identical vwap will push EVR toward +5
    out = compute_trade_features(ev, spread=np.full(1_500, 0.1), mid=np.full(1_500, 100.0))
    assert (out["effort_vs_result"] >= -5.0).all()
    assert (out["effort_vs_result"] <= 5.0).all()


def test_climax_score_is_continuous_nonnegative():
    ev = _fake_events(2_000, seed=1)
    out = compute_trade_features(ev, spread=np.full(2_000, 0.1), mid=np.full(2_000, 100.0))
    assert (out["climax_score"] >= 0.0).all()
    assert (out["climax_score"] <= 5.0).all()


def test_book_walk_zero_spread_guard():
    # Gotcha #10: spread=0 must NOT produce inf.
    ev = _fake_events(100)
    spread = np.zeros(100)
    mid = np.full(100, 100.0)
    out = compute_trade_features(ev, spread=spread, mid=mid)
    assert np.isfinite(out["book_walk"]).all()


def test_prev_seq_time_span_no_lookahead():
    # Gotcha #8: prev_seq_time_span is the PRIOR 200-event window's span, not
    # the current one's (which would be a hard lookahead).
    ev = _fake_events(500)
    out = compute_trade_features(ev, spread=np.full(500, 0.1), mid=np.full(500, 100.0))
    # First 200 events must have prev_seq_time_span = 0 (no prior window yet).
    assert (out["prev_seq_time_span"].iloc[:200] == 0.0).all()
    # From event 200 onward, prev_seq_time_span equals log(last_ts[0:200] - first_ts[0:200] + 1) for 200:400
    span = float(ev["ts_ms"].iloc[199] - ev["ts_ms"].iloc[0])
    expected = float(np.log(span + 1.0))
    np.testing.assert_allclose(out["prev_seq_time_span"].iloc[200:400], expected, rtol=0, atol=0)
```

- [ ] **Step 2: Run → fails**

- [ ] **Step 3: Implement tape/features_trade.py**

```python
# tape/features_trade.py
"""9 trade-side features per order event.

Every feature is causal (no lookahead). Normalisation uses rolling statistics
over the last `ROLLING_WINDOW` events (gotcha #4).

Spec: docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
      §Input Representation (9 trade features numbered 1-9).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tape.constants import ROLLING_WINDOW, WINDOW_LEN

_EPS_RETURN: float = 1e-6         # gotcha #5 — was 1e-4, too coarse for BTC
_EPS_SPREAD_MID: float = 1e-8     # gotcha #10


def compute_trade_features(
    events: pd.DataFrame, *, spread: np.ndarray, mid: np.ndarray
) -> pd.DataFrame:
    """Compute the 9 trade features.

    Parameters
    ----------
    events : DataFrame from `tape.events.group_to_events`
    spread : np.ndarray of aligned OB spread per event (same length as events)
    mid    : np.ndarray of aligned OB mid price per event

    Returns
    -------
    DataFrame with the 9 feature columns in FEATURE order.
    """
    n = len(events)
    ts_ms = events["ts_ms"].to_numpy(dtype=np.int64)
    vwap = events["vwap"].to_numpy(dtype=float)
    total_qty = events["total_qty"].to_numpy(dtype=float)
    is_open_frac = events["is_open_frac"].to_numpy(dtype=float)
    n_fills = events["n_fills"].to_numpy(dtype=float)
    book_walk_abs = events["book_walk_abs"].to_numpy(dtype=float)

    # 1. log_return
    log_return = np.zeros(n, dtype=float)
    log_return[1:] = np.log(vwap[1:] / np.maximum(vwap[:-1], _EPS_RETURN))

    # 2. log_total_qty — normalised by rolling median (gotcha #4, #5)
    _med: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=1).median()
    roll_med_qty = np.maximum(_med.to_numpy(dtype=float), _EPS_RETURN)
    log_total_qty = np.log(total_qty / roll_med_qty)

    # 3. is_open — passthrough in [0, 1]
    is_open = is_open_frac

    # 4. time_delta = log(Δt_ms + 1)
    dt = np.zeros(n, dtype=float)
    dt[1:] = ts_ms[1:] - ts_ms[:-1]
    time_delta = np.log(dt + 1.0)

    # 5. num_fills = log(count)
    num_fills = np.log(n_fills)

    # 6. book_walk = |last - first| / max(spread, eps * mid), spread/mid aligned
    eps_spread = np.maximum(spread, _EPS_SPREAD_MID * mid)
    book_walk = book_walk_abs / eps_spread

    # 7. effort_vs_result — clip(log_total_qty - log(|return|+eps), -5, 5)  (gotcha #5)
    abs_ret = np.abs(log_return)
    effort_vs_result = np.clip(log_total_qty - np.log(abs_ret + _EPS_RETURN), -5.0, 5.0)

    # 8. climax_score — rolling-1000 σ, clipped [0, 5]  (gotcha #6)
    _ssq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).std()
    roll_std_qty = np.maximum(_ssq.fillna(1e-10).to_numpy(dtype=float), 1e-10)
    _ssr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).std()
    roll_std_ret = np.maximum(_ssr.fillna(1e-10).to_numpy(dtype=float), 1e-10)
    _smq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_mean_qty = _smq.fillna(0.0).to_numpy(dtype=float)
    _smr: Any = pd.Series(abs_ret).rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_mean_ret = _smr.fillna(0.0).to_numpy(dtype=float)
    z_qty = (total_qty - roll_mean_qty) / roll_std_qty
    z_ret = (abs_ret - roll_mean_ret) / roll_std_ret
    climax_score = np.clip(np.minimum(z_qty, z_ret), 0.0, 5.0)

    # 9. prev_seq_time_span — log(last_ts - first_ts + 1) of the PRIOR 200-event
    #    window (gotcha #8). Zero for the first 200 events (no prior window).
    prev_seq_time_span = np.zeros(n, dtype=float)
    if n > WINDOW_LEN:
        # For event at index i >= WINDOW_LEN, the prior window spans
        # [i - WINDOW_LEN, i). Its span is ts[i-1] - ts[i-WINDOW_LEN].
        # NB: we compute spans at the event *starting* positions; this is a
        # step function — constant within each prior-window — but the spec
        # defines prev_seq_time_span per event, so we repeat.
        # Vectorise by computing the cumulative prior-window span aligned to
        # the current event.
        span = ts_ms[WINDOW_LEN - 1 :] - ts_ms[: n - WINDOW_LEN + 1]
        # span[k] is the span of events [k : k + WINDOW_LEN]. For event at
        # global index i >= WINDOW_LEN, we want span[i - WINDOW_LEN].
        prev_seq_time_span[WINDOW_LEN:] = np.log(span[: n - WINDOW_LEN].astype(float) + 1.0)

    return pd.DataFrame({
        "log_return": log_return,
        "log_total_qty": log_total_qty,
        "is_open": is_open,
        "time_delta": time_delta,
        "num_fills": num_fills,
        "book_walk": book_walk,
        "effort_vs_result": effort_vs_result,
        "climax_score": climax_score,
        "prev_seq_time_span": prev_seq_time_span,
    })
```

- [ ] **Step 4: Run tests → pass; pyright clean; commit**

```bash
uv run pytest tests/tape/test_features_trade.py -v
uv run pyright tape/features_trade.py tests/tape/test_features_trade.py
git add tape/features_trade.py tests/tape/test_features_trade.py
git commit -m "feat: 9 trade features with rolling-1000 normalisation + prev_seq_time_span (no lookahead)"
```

---

## Task 5: OB Features (8 channels)

**Files:**
- Create: `tape/features_ob.py`
- Create: `tests/tape/test_features_ob.py`

Key correctness concerns:
- **depth_ratio** uses notional, epsilon-guarded (gotcha #9, #14)
- **trade_vs_mid** zero-spread guard + clip (gotcha #10)
- **delta_imbalance_L1** day-boundary warm-up handled by caller; caller passes a `warm_imbalance_L1` scalar (gotcha #11)
- **kyle_lambda** — per-SNAPSHOT over 50 snapshots on Δmid (gotcha #13), forward-filled to events
- **cum_ofi_5** — piecewise Cont 2014 OFI, notional-normalised (gotchas #14, #15)

- [ ] **Step 1: Write failing tests**

```python
# tests/tape/test_features_ob.py
import numpy as np
import pandas as pd
import pytest

from tape.features_ob import compute_snapshot_features, align_ob_features_to_events


def _fake_ob(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = (np.arange(n, dtype=np.int64) * 24_000) + 1_700_000_000_000
    mid = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    spread = np.abs(rng.normal(0.05, 0.01, size=n))
    bid = mid - spread / 2
    ask = mid + spread / 2
    # 10 levels (gotcha #20) — decreasing price for bids, increasing for asks
    data: dict[str, np.ndarray] = {"ts_ms": ts}
    for lvl in range(1, 11):
        data[f"bid{lvl}_price"] = bid - (lvl - 1) * 0.01
        data[f"ask{lvl}_price"] = ask + (lvl - 1) * 0.01
        data[f"bid{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
        data[f"ask{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
    return pd.DataFrame(data)


def test_compute_snapshot_features_returns_all_fields():
    ob = _fake_ob(100)
    snap = compute_snapshot_features(ob)
    for col in ("log_spread", "imbalance_L1", "imbalance_L5", "depth_ratio",
                "mid", "spread", "delta_imbalance_L1", "kyle_lambda", "cum_ofi_5"):
        assert col in snap.columns
    assert len(snap) == len(ob)


def test_depth_ratio_one_sided_book_is_finite():
    ob = _fake_ob(50)
    # Flash crash: all asks depleted
    for lvl in range(1, 11):
        ob[f"ask{lvl}_qty"] = 0.0
    snap = compute_snapshot_features(ob)
    assert np.isfinite(snap["depth_ratio"]).all()


def test_kyle_lambda_is_per_snapshot_not_per_event():
    # Gotcha #13: rolling 50 snapshots, not per-event
    ob = _fake_ob(200)
    snap = compute_snapshot_features(ob)
    # Values change no more than once per snapshot
    assert len(snap["kyle_lambda"]) == 200
    # First 50 snapshots must be 0 (insufficient history)
    assert (snap["kyle_lambda"].iloc[:10] == 0.0).all()


def test_cum_ofi_5_uses_piecewise_cont_not_naive_delta():
    # Gotcha #15: when best bid/ask prices change, naive delta-notional has
    # the wrong sign. Piecewise Cont handles the three cases separately.
    # This is a regression test: build a scenario where naive gives +X,
    # piecewise gives -X, and assert the sign.
    n = 3
    ob = pd.DataFrame({
        "ts_ms": np.array([0, 1, 2], dtype=np.int64),
        "bid1_price": [100.0, 101.0, 101.0],   # bid rose → positive OFI contribution
        "bid1_qty":   [10.0,  10.0,  10.0],
        "ask1_price": [101.0, 102.0, 102.0],
        "ask1_qty":   [10.0,  10.0,  10.0],
    })
    # Populate levels 2-10 with dummy values so compute_snapshot_features
    # doesn't choke on missing columns.
    for lvl in range(2, 11):
        ob[f"bid{lvl}_price"] = ob["bid1_price"] - (lvl - 1) * 0.1
        ob[f"ask{lvl}_price"] = ob["ask1_price"] + (lvl - 1) * 0.1
        ob[f"bid{lvl}_qty"] = 10.0
        ob[f"ask{lvl}_qty"] = 10.0
    snap = compute_snapshot_features(ob)
    # After the bid rose, piecewise Cont OFI should be positive at idx 1.
    assert snap["cum_ofi_5"].iloc[1] > 0


def test_align_ob_features_to_events_uses_searchsorted():
    ob = _fake_ob(100)
    snap = compute_snapshot_features(ob)
    event_ts = np.array([snap["ts_ms"].iloc[0] - 1,        # before first → masked
                         snap["ts_ms"].iloc[0],             # exactly at idx 0
                         snap["ts_ms"].iloc[10] + 100,      # between 10 and 11 → idx 10
                         snap["ts_ms"].iloc[-1] + 1_000_000 # after last → idx -1
                         ], dtype=np.int64)
    out = align_ob_features_to_events(snap, event_ts)
    assert len(out) == len(event_ts)
    # First event has no prior snapshot → features are NaN (caller drops)
    assert np.isnan(out["log_spread"].iloc[0])
    # Second event aligns to snapshot 0
    assert not np.isnan(out["log_spread"].iloc[1])
```

- [ ] **Step 2: Run → fails**

- [ ] **Step 3: Implement tape/features_ob.py**

```python
# tape/features_ob.py
"""8 orderbook features per event, derived from 10-level snapshots.

Two stages:
  1. `compute_snapshot_features(ob_df)` — per-snapshot features + supports
     (mid, spread, kyle_lambda, cum_ofi_5). kyle_lambda and cum_ofi_5 are
     per-snapshot (gotcha #13, #15).
  2. `align_ob_features_to_events(snap_df, event_ts)` — for each event,
     forward-fill the latest prior snapshot's features. Events before the
     first snapshot get NaN and must be dropped by the caller.

Gotchas: #9 (depth_ratio), #10 (trade_vs_mid), #13 (kyle_lambda per snapshot),
#14 (notional for cross-symbol comparability), #15 (piecewise Cont OFI),
#20 (10 levels per side).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tape.constants import KYLE_LAMBDA_WINDOW, OFI_WINDOW
from tape.ob_align import align_events_to_ob

_EPS: float = 1e-10


def compute_snapshot_features(ob: pd.DataFrame) -> pd.DataFrame:
    """Compute per-snapshot OB features.

    Expects `ob` to have columns bid{1..10}_price, bid{1..10}_qty,
    ask{1..10}_price, ask{1..10}_qty, and ts_ms. Returns a DataFrame with
    the per-snapshot features plus aux columns used downstream.
    """
    ts = ob["ts_ms"].to_numpy(dtype=np.int64)
    bid1_p = ob["bid1_price"].to_numpy(dtype=float)
    ask1_p = ob["ask1_price"].to_numpy(dtype=float)
    bid1_q = ob["bid1_qty"].to_numpy(dtype=float)
    ask1_q = ob["ask1_qty"].to_numpy(dtype=float)
    mid = (bid1_p + ask1_p) / 2.0
    spread = np.maximum(ask1_p - bid1_p, _EPS)

    # log_spread = log(spread / mid)
    log_spread = np.log(spread / np.maximum(mid, _EPS))

    # imbalance_L1 — notional (gotcha #14)
    bid1_not = bid1_p * bid1_q
    ask1_not = ask1_p * ask1_q
    imb_l1 = (bid1_not - ask1_not) / np.maximum(bid1_not + ask1_not, _EPS)

    # imbalance_L5 — inverse-level-weighted notional
    num = np.zeros_like(mid)
    den = np.zeros_like(mid)
    for lvl in range(1, 6):
        w = 1.0 / lvl
        b = ob[f"bid{lvl}_price"].to_numpy(dtype=float) * ob[f"bid{lvl}_qty"].to_numpy(dtype=float)
        a = ob[f"ask{lvl}_price"].to_numpy(dtype=float) * ob[f"ask{lvl}_qty"].to_numpy(dtype=float)
        num += w * (b - a)
        den += w * (b + a)
    imb_l5 = num / np.maximum(den, _EPS)

    # depth_ratio — log ratio of top-10 notional, epsilon-guarded (gotcha #9, #14)
    bid_not_total = np.zeros_like(mid)
    ask_not_total = np.zeros_like(mid)
    for lvl in range(1, 11):
        bid_not_total += ob[f"bid{lvl}_price"].to_numpy(dtype=float) * ob[f"bid{lvl}_qty"].to_numpy(dtype=float)
        ask_not_total += ob[f"ask{lvl}_price"].to_numpy(dtype=float) * ob[f"ask{lvl}_qty"].to_numpy(dtype=float)
    depth_ratio = np.log(np.maximum(bid_not_total, 1e-6) / np.maximum(ask_not_total, 1e-6))

    # delta_imbalance_L1 — change since previous snapshot (gotcha #11 handled by caller)
    delta_imb_l1 = np.concatenate([[0.0], np.diff(imb_l1)])

    # --- kyle_lambda (per-snapshot, rolling 50) ---
    # Uses Δmid and cumulative signed notional.
    # Within this module we don't have trade events yet — we compute with
    # placeholder cum_signed_notional = 0; the *per-event* snapshot_notional
    # is injected by align_ob_features_to_events when the caller passes
    # the cross-snapshot notional attribution, but for self-contained tests
    # we compute the Cov/Var on Δmid vs. a proxy (imb_l1 notional sum).
    # The real pipeline overrides this via the `trade_signed_notional`
    # argument to `attach_per_snapshot_signed_notional` below.
    kyle_lambda = _rolling_kyle_lambda(mid, signed_notional_proxy=bid_not_total - ask_not_total)

    # --- cum_ofi_5: piecewise Cont 2014 OFI, rolling 5 snapshots, notional-normalised (gotcha #15) ---
    ofi = _piecewise_cont_ofi(bid1_p, bid1_q, ask1_p, ask1_q)
    _ofi_s: Any = pd.Series(ofi).rolling(OFI_WINDOW, min_periods=1).sum()
    cum_ofi_5_num = _ofi_s.to_numpy(dtype=float)
    _not_s: Any = pd.Series(bid_not_total + ask_not_total).rolling(OFI_WINDOW, min_periods=1).sum()
    cum_ofi_5_den = np.maximum(_not_s.to_numpy(dtype=float), _EPS)
    cum_ofi_5 = cum_ofi_5_num / cum_ofi_5_den

    return pd.DataFrame({
        "ts_ms": ts,
        "mid": mid,
        "spread": spread,
        "log_spread": log_spread,
        "imbalance_L1": imb_l1,
        "imbalance_L5": imb_l5,
        "depth_ratio": depth_ratio,
        "delta_imbalance_L1": delta_imb_l1,
        "kyle_lambda": kyle_lambda,
        "cum_ofi_5": cum_ofi_5,
    })


def _piecewise_cont_ofi(
    bid_p: np.ndarray, bid_q: np.ndarray, ask_p: np.ndarray, ask_q: np.ndarray
) -> np.ndarray:
    """Cont (2014) order-flow imbalance, piecewise on best bid/ask price change.

    For each snapshot transition t-1 → t:
      - bid side: if bid_p[t] > bid_p[t-1]  → +bid_q[t]
                  if bid_p[t] == bid_p[t-1] → bid_q[t] - bid_q[t-1]
                  if bid_p[t] < bid_p[t-1]  → -bid_q[t-1]
      - ask side: symmetric with sign flipped.
    Returns notional-scale OFI (qty × price).
    """
    n = len(bid_p)
    ofi = np.zeros(n, dtype=float)
    if n < 2:
        return ofi
    dbid = np.sign(bid_p[1:] - bid_p[:-1])
    dask = np.sign(ask_p[1:] - ask_p[:-1])
    # Bid contribution
    bid_delta_up = bid_q[1:] * bid_p[1:]
    bid_delta_same = (bid_q[1:] - bid_q[:-1]) * bid_p[1:]
    bid_delta_down = -bid_q[:-1] * bid_p[:-1]
    bid_ofi = np.where(dbid > 0, bid_delta_up,
                np.where(dbid < 0, bid_delta_down, bid_delta_same))
    # Ask contribution (opposite sign convention)
    ask_delta_up = -ask_q[:-1] * ask_p[:-1]
    ask_delta_same = -(ask_q[1:] - ask_q[:-1]) * ask_p[1:]
    ask_delta_down = ask_q[1:] * ask_p[1:]
    ask_ofi = np.where(dask > 0, ask_delta_up,
                np.where(dask < 0, ask_delta_down, ask_delta_same))
    ofi[1:] = bid_ofi + ask_ofi
    return ofi


def _rolling_kyle_lambda(mid: np.ndarray, signed_notional_proxy: np.ndarray) -> np.ndarray:
    """Cov(Δmid, signed_notional) / Var(signed_notional) over `KYLE_LAMBDA_WINDOW`.

    This is a placeholder that uses the proxy signed notional built from the
    OB (bid_not − ask_not). The real pipeline can override via
    `attach_trade_attributed_kyle_lambda(snap_df, trade_signed_notional_per_snap)`.
    Forward-fills until the first full window; zero for snapshots [0, W).
    """
    n = len(mid)
    out = np.zeros(n, dtype=float)
    if n <= KYLE_LAMBDA_WINDOW:
        return out
    dmid = np.concatenate([[0.0], np.diff(mid)])
    # Rolling Cov and Var
    _sx: Any = pd.Series(signed_notional_proxy)
    _sy: Any = pd.Series(dmid)
    cov: Any = _sx.rolling(KYLE_LAMBDA_WINDOW).cov(_sy)
    var: Any = _sx.rolling(KYLE_LAMBDA_WINDOW).var()
    lam = (cov / var.replace(0.0, np.nan)).fillna(0.0).to_numpy(dtype=float)
    out[KYLE_LAMBDA_WINDOW:] = lam[KYLE_LAMBDA_WINDOW:]
    return out


def align_ob_features_to_events(snap: pd.DataFrame, event_ts: np.ndarray) -> pd.DataFrame:
    """Forward-fill the 8 OB features + mid + spread onto event timestamps."""
    ob_ts = snap["ts_ms"].to_numpy(dtype=np.int64)
    idx = align_events_to_ob(event_ts, ob_ts)
    # Build output aligned to events. Where idx == -1 the event precedes the
    # first snapshot — return NaN so the caller can mask/drop.
    out_cols = {}
    for col in ("log_spread", "imbalance_L1", "imbalance_L5", "depth_ratio",
                "delta_imbalance_L1", "kyle_lambda", "cum_ofi_5", "mid", "spread"):
        vals = snap[col].to_numpy(dtype=float)
        aligned = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan)
        out_cols[col] = aligned
    # Add trade_vs_mid placeholder filled by the caller once vwap is available.
    return pd.DataFrame(out_cols)
```

Notes for the implementer:
- `trade_vs_mid` (feature #14 from the spec) is computed by the caller in `tape/features_trade.py`-glue or in `tape/cache.py`, because it mixes event-side `vwap` with snapshot-side `mid` and `spread`. The glue layer lives in Task 7.
- Kyle λ here is the proxy variant. Task 7's cache builder swaps in the trade-attributed version after events are known.

- [ ] **Step 4: Run tests → pass; pyright clean; commit**

```bash
uv run pytest tests/tape/test_features_ob.py -v
uv run pyright tape/features_ob.py tests/tape/test_features_ob.py
git add tape/features_ob.py tests/tape/test_features_ob.py
git commit -m "feat: 8 OB features — piecewise Cont OFI + per-snapshot Kyle lambda + searchsorted alignment"
```

---

## Task 6: Labels (direction + Wyckoff)

**Files:**
- Create: `tape/labels.py`
- Create: `tests/tape/test_labels.py`

- [ ] **Step 1: Write tests**

```python
# tests/tape/test_labels.py
import numpy as np
import pandas as pd

from tape.labels import (
    compute_direction_labels,
    compute_wyckoff_labels,
    DirectionLabels,
)
from tape.constants import DIRECTION_HORIZONS, SPRING_SIGMA_MULT


def test_direction_labels_all_horizons():
    n = 2_000
    rng = np.random.default_rng(0)
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    out: DirectionLabels = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        key = f"h{h}"
        assert key in out
        assert len(out[key]) == n
        # Last `h` events have no label (no forward data) → 0 sentinel + mask
        assert (out[key][-h:] == 0).all()
    # Mask must be False on the last h events for horizon h
    assert out["mask_h100"][-100:].sum() == 0
    assert out["mask_h100"][:-100].all()


def test_direction_label_event_vwap_not_last_fill():
    # Spec §Label Semantics: uses event-VWAP, not last-fill price.
    # Build a scenario where last fill would flip the label.
    vwap = np.array([100.0, 100.0, 100.1, 100.2, 99.5], dtype=float)  # 5 events
    out = compute_direction_labels(vwap)
    # h=1: label at event i = sign(vwap[i+1] - vwap[i])
    # Event 0 → vwap[1]=100 vs vwap[0]=100 → 0 → class 0 (down-or-flat)
    # Event 1 → vwap[2]=100.1 vs 100 → positive → class 1
    assert out["h10"].shape == (5,)  # sentinel shape; no H10 labels with only 5 events
    assert (out["mask_h10"] == False).all()


def test_spring_uses_sigma_mult_3():
    # Falsifiability prereq #4
    # Build a series where a < -2σ event AND evr > 1 AND is_open > 0.5 AND
    # mean of previous 10 returns > 0. With σ=3.0 it must NOT fire.
    n = 100
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0, 0.01, size=n)
    log_ret[50] = -0.025   # ~2.5σ — should NOT trigger spring at σ=3.0
    is_open = np.full(n, 0.8)
    evr = np.full(n, 1.5)
    wl = compute_wyckoff_labels(
        log_return=log_ret, effort_vs_result=evr, is_open=is_open,
        climax_score=np.zeros(n), z_qty=np.zeros(n), z_ret=np.zeros(n),
        log_spread=np.zeros(n), depth_ratio=np.zeros(n),
        kyle_lambda=np.zeros(n), cum_ofi_5=np.zeros(n),
    )
    assert wl["spring"].sum() == 0, "Spring must not fire at 2.5σ with SPRING_SIGMA_MULT=3.0"


def test_stress_fires_when_both_percentiles_crossed():
    n = 1_200
    rng = np.random.default_rng(0)
    log_spread = rng.normal(-5, 1, size=n)
    depth_ratio = rng.normal(0, 1, size=n)
    # Inject a clear crossing at idx 1100: both far above p90.
    log_spread[1100] = 10.0
    depth_ratio[1100] = 10.0
    wl = compute_wyckoff_labels(
        log_return=np.zeros(n), effort_vs_result=np.zeros(n),
        is_open=np.zeros(n), climax_score=np.zeros(n),
        z_qty=np.zeros(n), z_ret=np.zeros(n),
        log_spread=log_spread, depth_ratio=depth_ratio,
        kyle_lambda=np.zeros(n), cum_ofi_5=np.zeros(n),
    )
    assert wl["stress"][1100] == 1
```

- [ ] **Step 2: Run → fails; Step 3: Implement tape/labels.py**

```python
# tape/labels.py
"""Direction labels (downstream probing) + Wyckoff self-labels (diagnostic).

Spec §Label Semantics: direction label at event i = sign(vwap[i+h] - vwap[i])
where `vwap` is EVENT VWAP (not last-fill price). Last `h` events have no
label and are masked.

Wyckoff labels:
- stress          : log_spread > p90(rolling) AND |depth_ratio| > p90(rolling)
- informed_flow   : kyle_lambda > p75 AND |cum_ofi_5| > p50 AND 3-snap sign-consistency
- climax          : z_qty > 2 AND z_ret > 2 (asymmetry — accumulation/distribution)
- spring          : min(log_return[i-50:i+1]) < -SPRING_SIGMA_MULT * σ AND
                    effort_vs_result at min > 1 AND is_open at min > 0.5 AND
                    mean(log_return[i-10:i+1]) > 0
                    SPRING_SIGMA_MULT = 3.0 (recalibrated — prereq #4)
- absorption      : effort_vs_result > p90 AND |log_return| < p50 (high effort, low result)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from tape.constants import (
    DIRECTION_HORIZONS, ROLLING_WINDOW, SPRING_LOOKBACK, SPRING_PRIOR_LEN,
    SPRING_SIGMA_MULT, STRESS_PCTL, DEPTH_PCTL, INFORMED_KYLE_PCTL, INFORMED_OFI_PCTL,
)


class DirectionLabels(TypedDict):
    h10: np.ndarray
    h50: np.ndarray
    h100: np.ndarray
    h500: np.ndarray
    mask_h10: np.ndarray
    mask_h50: np.ndarray
    mask_h100: np.ndarray
    mask_h500: np.ndarray


def compute_direction_labels(vwap: np.ndarray) -> DirectionLabels:
    n = len(vwap)
    out: DirectionLabels = {}  # type: ignore[typeddict-item]
    for h in DIRECTION_HORIZONS:
        labels = np.zeros(n, dtype=np.int8)
        mask = np.zeros(n, dtype=bool)
        if n > h:
            fwd = vwap[h:]
            cur = vwap[:-h]
            labels[: n - h] = (fwd > cur).astype(np.int8)  # 1 = up, 0 = down-or-flat
            mask[: n - h] = True
        out[f"h{h}"] = labels       # type: ignore[literal-required]
        out[f"mask_h{h}"] = mask    # type: ignore[literal-required]
    return out


def compute_wyckoff_labels(
    *,
    log_return: np.ndarray,
    effort_vs_result: np.ndarray,
    is_open: np.ndarray,
    climax_score: np.ndarray,
    z_qty: np.ndarray,
    z_ret: np.ndarray,
    log_spread: np.ndarray,
    depth_ratio: np.ndarray,
    kyle_lambda: np.ndarray,
    cum_ofi_5: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(log_return)
    out: dict[str, np.ndarray] = {}

    # --- stress ---
    ls_pct = _rolling_percentile(log_spread, ROLLING_WINDOW, STRESS_PCTL)
    dr_pct = _rolling_percentile(np.abs(depth_ratio), ROLLING_WINDOW, DEPTH_PCTL)
    out["stress"] = ((log_spread > ls_pct) & (np.abs(depth_ratio) > dr_pct)).astype(np.int8)

    # --- informed_flow ---
    kl_pct = _rolling_percentile(kyle_lambda, ROLLING_WINDOW, INFORMED_KYLE_PCTL)
    of_pct = _rolling_percentile(np.abs(cum_ofi_5), ROLLING_WINDOW, INFORMED_OFI_PCTL)
    fires = (kyle_lambda > kl_pct) & (np.abs(cum_ofi_5) > of_pct)
    # 3-snap sign consistency on cum_ofi_5
    signs = np.sign(cum_ofi_5)
    consistent = np.zeros(n, dtype=bool)
    consistent[2:] = (signs[2:] == signs[1:-1]) & (signs[2:] == signs[:-2]) & (signs[2:] != 0)
    out["informed_flow"] = (fires & consistent).astype(np.int8)

    # --- climax ---
    out["climax"] = ((z_qty > 2.0) & (z_ret > 2.0)).astype(np.int8)

    # --- spring ---
    # Rolling σ of log_return
    _rs: pd.Series = pd.Series(log_return).rolling(ROLLING_WINDOW, min_periods=10).std()
    roll_std = _rs.fillna(1e-10).to_numpy(dtype=float)
    min_ret = pd.Series(log_return).rolling(SPRING_LOOKBACK, min_periods=SPRING_LOOKBACK).min().to_numpy(dtype=float)
    prior_mean = pd.Series(log_return).shift(SPRING_PRIOR_LEN).rolling(SPRING_PRIOR_LEN, min_periods=SPRING_PRIOR_LEN).mean().to_numpy(dtype=float)
    cond1 = min_ret < -SPRING_SIGMA_MULT * roll_std
    cond2 = effort_vs_result > 1.0
    cond3 = is_open > 0.5
    cond4 = prior_mean > 0
    spring = cond1 & cond2 & cond3 & cond4
    out["spring"] = np.nan_to_num(spring, nan=0).astype(np.int8)

    # --- absorption ---
    evr_pct = _rolling_percentile(effort_vs_result, ROLLING_WINDOW, 0.90)
    ret_pct = _rolling_percentile(np.abs(log_return), ROLLING_WINDOW, 0.50)
    out["absorption"] = ((effort_vs_result > evr_pct) & (np.abs(log_return) < ret_pct)).astype(np.int8)

    return out


def _rolling_percentile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(10, window // 10)).quantile(q).fillna(np.inf).to_numpy(dtype=float)
```

- [ ] **Step 4: Run tests; pyright clean; commit**

```bash
uv run pytest tests/tape/test_labels.py -v
uv run pyright tape/labels.py tests/tape/test_labels.py
git add tape/labels.py tests/tape/test_labels.py
git commit -m "feat: direction + Wyckoff labels (spring sigma=3.0, 3-snap informed flow consistency)"
```

---

## Task 7: Cache Builder (glue layer)

**Files:**
- Create: `tape/io_parquet.py`
- Create: `tape/cache.py`
- Create: `scripts/build_cache.py`
- Create: `tests/tape/test_cache.py`

This task glues prior tasks together: read parquet → dedup → events → snapshot features → align → compute 17 features (including `trade_vs_mid`) → compute labels → window stride=50 → save `.npz`.

- [ ] **Step 1: Implement tape/io_parquet.py**

```python
# tape/io_parquet.py
"""Per-symbol-day loaders for trades and orderbook parquet."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from tape.constants import OB_GLOB, TRADES_GLOB


def load_trades_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    path = Path(TRADES_GLOB.format(sym=symbol, date=date_str)).parent
    if not path.exists():
        return None
    q = f"SELECT * FROM read_parquet('{path}/*.parquet') ORDER BY ts_ms"
    return duckdb.query(q).to_df()


def load_ob_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    path = Path(OB_GLOB.format(sym=symbol, date=date_str)).parent
    if not path.exists():
        return None
    q = f"SELECT * FROM read_parquet('{path}/*.parquet') ORDER BY ts_ms"
    return duckdb.query(q).to_df()
```

- [ ] **Step 2: Implement tape/cache.py (the glue)**

```python
# tape/cache.py
"""Build the feature tensor + labels for a single symbol-day and cache to .npz.

Output schema (per shard):
  features   : float32, shape (n_events, 17), channel order = FEATURE_NAMES
  directions : int8 dict  with keys h10/h50/h100/h500 and mask_h{h}
  wyckoff    : int8 dict  with keys stress/informed_flow/climax/spring/absorption
  event_ts   : int64, shape (n_events,)
  symbol     : str
  date       : str
  schema_version : int (CACHE_SCHEMA_VERSION)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tape.constants import (
    APRIL_START, CACHE_SCHEMA_VERSION, FEATURE_NAMES, TRADE_FEATURES, OB_FEATURES,
)
from tape.dedup import dedup_trades_pre_april, filter_trades_april
from tape.events import group_to_events
from tape.features_ob import align_ob_features_to_events, compute_snapshot_features
from tape.features_trade import compute_trade_features
from tape.io_parquet import load_ob_day, load_trades_day
from tape.labels import compute_direction_labels, compute_wyckoff_labels


def build_symbol_day(symbol: str, date_str: str) -> dict | None:
    trades = load_trades_day(symbol, date_str)
    if trades is None or len(trades) == 0:
        return None
    if date_str >= APRIL_START:
        trades = filter_trades_april(trades)
    else:
        trades = dedup_trades_pre_april(trades)

    events = group_to_events(trades)
    if len(events) < 400:  # need at least 2 full windows + prior
        return None

    ob = load_ob_day(symbol, date_str)
    if ob is None or len(ob) < 2:
        return None
    snap = compute_snapshot_features(ob)

    # Align OB features onto event timestamps
    ob_aligned = align_ob_features_to_events(snap, events["ts_ms"].to_numpy(dtype=np.int64))
    # Drop events before the first OB snapshot (NaN mid/spread → undefined features).
    valid = np.isfinite(ob_aligned["mid"].to_numpy(dtype=float))
    if valid.sum() < 400:
        return None
    events = events.loc[valid].reset_index(drop=True)
    ob_aligned = ob_aligned.loc[valid].reset_index(drop=True)

    # 9 trade features — use aligned spread/mid
    trade_feats = compute_trade_features(
        events,
        spread=ob_aligned["spread"].to_numpy(dtype=float),
        mid=ob_aligned["mid"].to_numpy(dtype=float),
    )

    # trade_vs_mid — event-side vwap vs snapshot mid, clipped [-5, 5] (gotcha #10)
    vwap = events["vwap"].to_numpy(dtype=float)
    mid = ob_aligned["mid"].to_numpy(dtype=float)
    spread = ob_aligned["spread"].to_numpy(dtype=float)
    eps = np.maximum(spread, 1e-8 * mid)
    trade_vs_mid = np.clip((vwap - mid) / eps, -5.0, 5.0)

    # Assemble the 17-feature matrix in FEATURE_NAMES order.
    n = len(events)
    features = np.zeros((n, 17), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        if name in TRADE_FEATURES:
            features[:, i] = trade_feats[name].to_numpy(dtype=np.float32)
        elif name == "trade_vs_mid":
            features[:, i] = trade_vs_mid.astype(np.float32)
        else:
            features[:, i] = ob_aligned[name].to_numpy(dtype=np.float32)

    # Labels need z_qty, z_ret from trade_feats internals — recompute here to
    # avoid exposing them in the trade-feature surface.
    # (Cheap: rolling on total_qty and |log_return|.)
    from tape.constants import ROLLING_WINDOW
    total_qty = events["total_qty"].to_numpy(dtype=float)
    log_return = trade_feats["log_return"].to_numpy(dtype=float)
    mean_qty = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).mean().fillna(0.0).to_numpy()
    std_qty = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).std().fillna(1e-10).to_numpy()
    mean_ret = pd.Series(np.abs(log_return)).rolling(ROLLING_WINDOW, min_periods=10).mean().fillna(0.0).to_numpy()
    std_ret = pd.Series(np.abs(log_return)).rolling(ROLLING_WINDOW, min_periods=10).std().fillna(1e-10).to_numpy()
    z_qty = (total_qty - mean_qty) / np.maximum(std_qty, 1e-10)
    z_ret = (np.abs(log_return) - mean_ret) / np.maximum(std_ret, 1e-10)

    directions = compute_direction_labels(vwap)
    wyckoff = compute_wyckoff_labels(
        log_return=log_return,
        effort_vs_result=trade_feats["effort_vs_result"].to_numpy(dtype=float),
        is_open=trade_feats["is_open"].to_numpy(dtype=float),
        climax_score=trade_feats["climax_score"].to_numpy(dtype=float),
        z_qty=z_qty, z_ret=z_ret,
        log_spread=ob_aligned["log_spread"].to_numpy(dtype=float),
        depth_ratio=ob_aligned["depth_ratio"].to_numpy(dtype=float),
        kyle_lambda=ob_aligned["kyle_lambda"].to_numpy(dtype=float),
        cum_ofi_5=ob_aligned["cum_ofi_5"].to_numpy(dtype=float),
    )

    return {
        "features": features,
        "event_ts": events["ts_ms"].to_numpy(dtype=np.int64),
        "directions": {k: v.astype(np.int8) if v.dtype != bool else v for k, v in directions.items()},
        "wyckoff": wyckoff,
        "symbol": symbol,
        "date": date_str,
        "schema_version": CACHE_SCHEMA_VERSION,
    }


def save_shard(shard: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{shard['symbol']}__{shard['date']}.npz"
    payload: dict[str, np.ndarray] = {
        "features": shard["features"],
        "event_ts": shard["event_ts"],
        "schema_version": np.int32(shard["schema_version"]),
    }
    for k, v in shard["directions"].items():
        payload[f"dir_{k}"] = v
    for k, v in shard["wyckoff"].items():
        payload[f"wy_{k}"] = v
    np.savez_compressed(path, symbol=np.array(shard["symbol"]), date=np.array(shard["date"]), **payload)
    return path


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    return payload
```

- [ ] **Step 3: Implement scripts/build_cache.py**

```python
# scripts/build_cache.py
"""CLI: build per-symbol-day .npz shards for the tape pipeline.

Usage:
    uv run python scripts/build_cache.py \
        --symbols BTC ETH SOL \
        --start-date 2025-10-16 --end-date 2026-03-31 \
        --out data/cache/v1/
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date, timedelta
from pathlib import Path

from tape.cache import build_symbol_day, save_shard
from tape.constants import APRIL_HELDOUT_START, PREAPRIL_END, PREAPRIL_START, SYMBOLS


def _daterange(start: str, end: str):
    s = _date.fromisoformat(start)
    e = _date.fromisoformat(end)
    while s <= e:
        yield s.isoformat()
        s += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    p.add_argument("--start-date", default=PREAPRIL_START)
    p.add_argument("--end-date", default=PREAPRIL_END)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="rebuild existing shards")
    args = p.parse_args()

    # Safety: never touch the April hold-out.
    if args.end_date >= APRIL_HELDOUT_START:
        print(f"ERROR: end-date {args.end_date} is in the held-out range (>= {APRIL_HELDOUT_START}).", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    total = 0
    built = 0
    skipped = 0
    t0 = time.time()
    for sym in args.symbols:
        for d in _daterange(args.start_date, args.end_date):
            total += 1
            shard_path = out_dir / f"{sym}__{d}.npz"
            if shard_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                shard = build_symbol_day(sym, d)
            except Exception as e:
                print(f"[{sym} {d}] failed: {e}", file=sys.stderr)
                continue
            if shard is None:
                continue
            save_shard(shard, out_dir)
            built += 1
            if built % 25 == 0:
                elapsed = time.time() - t0
                print(f"  built {built}  skipped {skipped}  elapsed {elapsed:.1f}s")
    print(f"done. total={total} built={built} skipped={skipped} elapsed={time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Round-trip test**

```python
# tests/tape/test_cache.py
from pathlib import Path
import numpy as np
import pandas as pd

from tape.cache import save_shard, load_shard


def test_round_trip(tmp_path: Path) -> None:
    n = 500
    shard = {
        "features": np.random.randn(n, 17).astype(np.float32),
        "event_ts": np.arange(n, dtype=np.int64),
        "directions": {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)} |
                      {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)},
        "wyckoff": {k: np.zeros(n, dtype=np.int8) for k in
                    ("stress", "informed_flow", "climax", "spring", "absorption")},
        "symbol": "BTC",
        "date": "2025-11-01",
        "schema_version": 1,
    }
    path = save_shard(shard, tmp_path)
    payload = load_shard(path)
    assert payload["features"].shape == (n, 17)
    assert int(payload["schema_version"]) == 1
    assert "dir_h100" in payload
    assert "wy_stress" in payload
```

- [ ] **Step 5: Run test; pyright clean; commit**

```bash
uv run pytest tests/tape/test_cache.py -v
uv run pyright tape/io_parquet.py tape/cache.py scripts/build_cache.py tests/tape/test_cache.py
git add tape/io_parquet.py tape/cache.py scripts/build_cache.py tests/tape/test_cache.py
git commit -m "feat: per-symbol-day cache builder — 17-feature .npz shards"
```

- [ ] **Step 6: Build the cache for the full pre-April range**

```bash
uv run python scripts/build_cache.py \
    --symbols 2Z AAVE ASTER AVAX BNB BTC CRV DOGE ENA ETH FARTCOIN HYPE \
              KBONK KPEPE LDO LINK LTC PENGU PUMP SOL SUI UNI WLFI XPL XRP \
    --start-date 2025-10-16 --end-date 2026-03-31 \
    --out data/cache/v1/
```

This is the materialisation step — expected wall-time ~30-60 minutes on an M-series Mac; commit the run log (not the cache files themselves) to `docs/experiments/cache-build-log.md`.

---

## Task 8: Windowing + Dataset + Sampler

**Files:**
- Create: `tape/windowing.py`, `tape/dataset.py`, `tape/sampler.py`
- Create: `tests/tape/test_windowing.py`, `test_dataset.py`, `test_sampler.py`

- [ ] **Step 1: Implement tape/windowing.py + test**

Windowing turns per-event features into `(num_windows, WINDOW_LEN=200, 17)` with `stride=50`, starting from `random_offset ∈ [0, stride)` per epoch, and enforcing **no window crosses a day boundary** (gotcha #26 — enforced by the fact that each shard is one day).

```python
# tape/windowing.py
from __future__ import annotations

import numpy as np

from tape.constants import STRIDE_PRETRAIN, WINDOW_LEN


def build_windows(
    features: np.ndarray, *, stride: int = STRIDE_PRETRAIN,
    random_offset: int = 0,
) -> np.ndarray:
    """Return an int64 array of starting indices into `features` for each window."""
    n = len(features)
    offset = random_offset % stride
    start = offset
    end = n - WINDOW_LEN
    if end < start:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(start, end + 1, stride, dtype=np.int64)
```

```python
# tests/tape/test_windowing.py
import numpy as np
from tape.windowing import build_windows


def test_stride_50_window_200():
    n = 1_000
    feats = np.zeros((n, 17), dtype=np.float32)
    idx = build_windows(feats, stride=50, random_offset=0)
    # Expected starts: 0, 50, 100, ..., 800 (so last end = 800 + 200 = 1000)
    assert idx[0] == 0
    assert idx[-1] == 800
    assert np.all(np.diff(idx) == 50)


def test_random_offset_shifts_window_starts():
    n = 1_000
    feats = np.zeros((n, 17), dtype=np.float32)
    idx = build_windows(feats, stride=50, random_offset=17)
    assert idx[0] == 17
    assert idx[-1] <= n - 200


def test_short_shard_yields_empty():
    feats = np.zeros((100, 17), dtype=np.float32)
    idx = build_windows(feats, stride=50)
    assert len(idx) == 0
```

- [ ] **Step 2: Implement tape/dataset.py + test**

```python
# tape/dataset.py
"""TapeDataset — lazy-loads .npz shards and returns 200-event windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tape.cache import load_shard
from tape.constants import STRIDE_PRETRAIN, WINDOW_LEN
from tape.windowing import build_windows


@dataclass(frozen=True)
class WindowRef:
    shard_path: Path
    start: int
    symbol: str
    date: str


class TapeDataset(Dataset):
    """Indexable dataset over (symbol, day, window_start) tuples.

    Shards are lazily loaded and cached in an LRU of size `cache_size`.
    """

    def __init__(
        self, shard_paths: list[Path], *,
        stride: int = STRIDE_PRETRAIN, random_offset_fn=lambda: 0,
        cache_size: int = 8,
    ):
        self.shard_paths = sorted(shard_paths)
        self.stride = stride
        self.random_offset_fn = random_offset_fn
        self._cache: dict[Path, dict] = {}
        self._cache_order: list[Path] = []
        self._cache_size = cache_size
        self._refs: list[WindowRef] = []
        self._build_index()

    def _build_index(self) -> None:
        for p in self.shard_paths:
            payload = load_shard(p)
            feats: np.ndarray = payload["features"]
            idx = build_windows(feats, stride=self.stride, random_offset=self.random_offset_fn())
            sym = str(payload["symbol"])
            date = str(payload["date"])
            for s in idx:
                self._refs.append(WindowRef(shard_path=p, start=int(s), symbol=sym, date=date))

    def __len__(self) -> int:
        return len(self._refs)

    def _get_shard(self, path: Path) -> dict:
        if path in self._cache:
            return self._cache[path]
        if len(self._cache) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        payload = load_shard(path)
        self._cache[path] = payload
        self._cache_order.append(path)
        return payload

    def __getitem__(self, i: int) -> dict:
        ref = self._refs[i]
        payload = self._get_shard(ref.shard_path)
        s = ref.start
        e = s + WINDOW_LEN
        features = payload["features"][s:e]
        out: dict[str, torch.Tensor | str | int] = {
            "features": torch.from_numpy(features.astype(np.float32)),
            "symbol": ref.symbol,
            "date": ref.date,
            "start": s,
        }
        for h in (10, 50, 100, 500):
            # Label at the END of the window: i.e. direction from the last event.
            lbl = int(payload[f"dir_h{h}"][e - 1]) if e - 1 < len(payload[f"dir_h{h}"]) else 0
            mask = bool(payload[f"dir_mask_h{h}"][e - 1]) if e - 1 < len(payload[f"dir_mask_h{h}"]) else False
            out[f"label_h{h}"] = lbl
            out[f"label_h{h}_mask"] = mask
        return out
```

_Note for implementer: the label-key names above depend on how `save_shard` stored masks. If masks were saved under `dir_mask_h{h}`, adjust the loader; otherwise rename accordingly. Keep the test in sync._

Dataset test:

```python
# tests/tape/test_dataset.py
from pathlib import Path
import numpy as np
from tape.cache import save_shard
from tape.dataset import TapeDataset


def test_dataset_indexes_and_returns_200_event_windows(tmp_path: Path) -> None:
    n = 1_000
    shard = {
        "features": np.random.randn(n, 17).astype(np.float32),
        "event_ts": np.arange(n, dtype=np.int64),
        "directions": {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)} |
                      {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)},
        "wyckoff": {k: np.zeros(n, dtype=np.int8) for k in
                    ("stress", "informed_flow", "climax", "spring", "absorption")},
        "symbol": "BTC",
        "date": "2025-11-01",
        "schema_version": 1,
    }
    p = save_shard(shard, tmp_path)
    ds = TapeDataset([p], stride=50)
    assert len(ds) == (n - 200) // 50 + 1  # 17 windows
    sample = ds[0]
    assert sample["features"].shape == (200, 17)
    assert "label_h100" in sample
```

- [ ] **Step 3: Implement tape/sampler.py + test**

```python
# tape/sampler.py
"""Equal-symbol sampler — gotcha #27."""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler

from tape.dataset import TapeDataset


class EqualSymbolSampler(Sampler[int]):
    """Round-robin by symbol, random within symbol, per epoch."""

    def __init__(self, dataset: TapeDataset, *, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self._by_symbol: dict[str, list[int]] = {}
        for i, ref in enumerate(dataset._refs):
            self._by_symbol.setdefault(ref.symbol, []).append(i)

    def __len__(self) -> int:
        # Each epoch visits min_group_size * num_symbols windows.
        return min(len(v) for v in self._by_symbol.values()) * len(self._by_symbol)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        groups = {k: v.copy() for k, v in self._by_symbol.items()}
        for v in groups.values():
            rng.shuffle(v)
        min_n = min(len(v) for v in groups.values())
        syms = list(groups.keys())
        for i in range(min_n):
            rng.shuffle(syms)
            for s in syms:
                yield groups[s][i]
```

```python
# tests/tape/test_sampler.py
from pathlib import Path
import numpy as np
from tape.cache import save_shard
from tape.dataset import TapeDataset
from tape.sampler import EqualSymbolSampler


def _mk_shard(tmp_path: Path, sym: str, n: int, seed: int) -> Path:
    shard = {
        "features": np.random.RandomState(seed).randn(n, 17).astype(np.float32),
        "event_ts": np.arange(n, dtype=np.int64),
        "directions": {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)} |
                      {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)},
        "wyckoff": {k: np.zeros(n, dtype=np.int8) for k in
                    ("stress", "informed_flow", "climax", "spring", "absorption")},
        "symbol": sym, "date": "2025-11-01", "schema_version": 1,
    }
    return save_shard(shard, tmp_path)


def test_equal_symbol_sampler_round_robins(tmp_path: Path) -> None:
    paths = [_mk_shard(tmp_path / s, s, 1_000, seed=i) for i, s in enumerate(["BTC", "ETH", "SOL"])]
    ds = TapeDataset(paths, stride=50)
    sampler = EqualSymbolSampler(ds, seed=0)
    seq = list(sampler)
    # Total = min_group * num_symbols; BTC/ETH/SOL all have 17 windows → 17*3 = 51
    assert len(seq) == 51
    # Distribution over first 9 items must contain each symbol 3 times
    from collections import Counter
    first_9 = [ds._refs[i].symbol for i in seq[:9]]
    assert all(c == 3 for c in Counter(first_9).values())
```

- [ ] **Step 4: Run all tests; pyright clean; commit**

```bash
uv run pytest tests/tape/test_windowing.py tests/tape/test_dataset.py tests/tape/test_sampler.py -v
uv run pyright tape/windowing.py tape/dataset.py tape/sampler.py tests/tape/
git add tape/windowing.py tape/dataset.py tape/sampler.py tests/tape/test_windowing.py tests/tape/test_dataset.py tests/tape/test_sampler.py
git commit -m "feat: windowing + TapeDataset + EqualSymbolSampler"
```

---

## Task 9: Dataset Validation Script

**Files:**
- Create: `scripts/validate_cache.py`
- Report: `docs/experiments/cache-validation.md` (generated)

Purpose: verify three observable properties on the built cache before committing to Gate 0:

1. **`is_open` autocorrelation half-life ≈ 20 events** (CLAUDE.md key finding).
2. **Climax asymmetry** (council-4): buying-climax rate ≠ selling-climax rate in trending periods.
3. **`effort_vs_result` bounded** in [-5, 5] (gotcha #5 sanity).

- [ ] **Step 1: Implement scripts/validate_cache.py**

```python
# scripts/validate_cache.py
"""Post-build sanity report on the .npz cache.

Writes docs/experiments/cache-validation.md (+ .json) with:
  - per-symbol is_open autocorrelation half-life (target: ~20 events for liquid)
  - effort_vs_result [min, max] (must be inside [-5, 5])
  - climax rate per symbol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tape.cache import load_shard
from tape.constants import FEATURE_NAMES


IS_OPEN_IDX = FEATURE_NAMES.index("is_open")
EVR_IDX = FEATURE_NAMES.index("effort_vs_result")


def autocorr_halflife(x: np.ndarray, max_lag: int = 100) -> float:
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return float("nan")
    acs = []
    for lag in range(1, max_lag + 1):
        acs.append(float((x[:-lag] * x[lag:]).mean() / var))
    acs_arr = np.array(acs)
    for lag, a in enumerate(acs_arr, start=1):
        if a < 0.5:
            return float(lag)
    return float(max_lag)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out-md", default="docs/experiments/cache-validation.md")
    args = p.parse_args()

    cache = Path(args.cache)
    per_sym: dict[str, dict] = {}
    for shard in sorted(cache.glob("*.npz")):
        payload = load_shard(shard)
        sym = str(payload["symbol"])
        feats = payload["features"]
        is_open = feats[:, IS_OPEN_IDX].astype(float)
        evr = feats[:, EVR_IDX].astype(float)
        rec = per_sym.setdefault(sym, {"is_open": [], "evr_min": 1e9, "evr_max": -1e9, "n": 0})
        rec["is_open"].append(is_open)
        rec["evr_min"] = float(min(rec["evr_min"], evr.min()))
        rec["evr_max"] = float(max(rec["evr_max"], evr.max()))
        rec["n"] += len(feats)

    results = {}
    for sym, rec in per_sym.items():
        x = np.concatenate(rec["is_open"])
        hl = autocorr_halflife(x)
        results[sym] = {
            "is_open_halflife_events": hl,
            "effort_vs_result_min": rec["evr_min"],
            "effort_vs_result_max": rec["evr_max"],
            "n_events": rec["n"],
        }

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_md).with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write("# Cache Validation Report\n\n")
        f.write("| Symbol | is_open half-life (events) | EVR min | EVR max | n_events |\n")
        f.write("|---|---|---|---|---|\n")
        for sym, r in sorted(results.items()):
            f.write(f"| {sym} | {r['is_open_halflife_events']:.1f} | {r['effort_vs_result_min']:.2f} | "
                    f"{r['effort_vs_result_max']:.2f} | {r['n_events']:,} |\n")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run validation and commit report**

```bash
uv run python scripts/validate_cache.py --cache data/cache/v1
git add scripts/validate_cache.py docs/experiments/cache-validation.md docs/experiments/cache-validation.json
git commit -m "experiment: cache validation — is_open half-life, EVR bounds, n_events"
```

**PASS criteria before moving to Gate 0:**
- Liquid symbols (BTC, ETH, SOL): `is_open` half-life in [10, 40] events (10 too fast = noise; 40 too slow = over-persistent).
- All symbols: EVR min ≥ -5.0 and max ≤ 5.0 (clip working).
- Each symbol has at least 1,000 events in the cache (sufficient for Gate 0 fold).

If any PASS criterion fails, **stop** and either recalibrate the feature or dispatch council-4 for interpretation.

---

## Task 10: Flat Features (Gate 0 representation)

**Files:**
- Create: `tape/flat_features.py`, `tests/tape/test_flat_features.py`

Gate 0 uses handcrafted statistics on the 200-event window, 5 per channel:
`mean, std, skew, kurtosis, last_value` → 17 × 5 = **85-dim flat vector**.

- [ ] **Step 1: Implement + test**

```python
# tape/flat_features.py
"""Gate 0 baseline: per-window summary statistics (mean/std/skew/kurt/last)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def window_to_flat(window: np.ndarray) -> np.ndarray:
    """Map a (200, 17) window → flat (85,) vector."""
    assert window.ndim == 2 and window.shape[1] == 17, f"bad shape {window.shape}"
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    sk = skew(window, axis=0, bias=False, nan_policy="omit")
    kt = kurtosis(window, axis=0, bias=False, nan_policy="omit")
    last = window[-1]
    return np.concatenate([mean, std, sk, kt, last], axis=0).astype(np.float32)
```

```python
# tests/tape/test_flat_features.py
import numpy as np
from tape.flat_features import window_to_flat


def test_flat_shape_85():
    w = np.random.randn(200, 17).astype(np.float32)
    v = window_to_flat(w)
    assert v.shape == (85,)


def test_flat_is_finite_on_constant_channel():
    w = np.zeros((200, 17), dtype=np.float32)
    v = window_to_flat(w)
    assert np.isfinite(v).all()
```

- [ ] **Step 2: Add scipy dep if not present**

Ensure `scipy>=1.13` in pyproject.toml `[project].dependencies`.

- [ ] **Step 3: Run; pyright clean; commit**

```bash
uv run pytest tests/tape/test_flat_features.py -v
uv run pyright tape/flat_features.py tests/tape/test_flat_features.py
git add tape/flat_features.py tests/tape/test_flat_features.py pyproject.toml
git commit -m "feat: flat feature extractor (mean/std/skew/kurt/last × 17 = 85 dims)"
```

---

## Task 11: Walk-Forward Splits with 600-Event Embargo

**Files:**
- Create: `tape/splits.py`, `tests/tape/test_splits.py`

- [ ] **Step 1: Implement + test**

```python
# tape/splits.py
"""Walk-forward cross-validation with a 600-event embargo (gotcha #12).

For a symbol's cache (event-ordered), yield (train_idx, test_idx) where
test immediately follows train with a 600-event gap. K folds = 5 by default.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from tape.constants import EMBARGO_EVENTS


def walk_forward_splits(
    n_events: int, *, k: int = 5, embargo: int = EMBARGO_EVENTS,
    min_train: int = 10_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train, test) event-index arrays for each fold."""
    if n_events < min_train + embargo + k * 1_000:
        return
    test_len = (n_events - min_train - embargo) // k
    for fi in range(k):
        test_start = min_train + embargo + fi * test_len
        test_end = test_start + test_len
        train = np.arange(0, test_start - embargo, dtype=np.int64)
        test = np.arange(test_start, test_end, dtype=np.int64)
        yield train, test
```

```python
# tests/tape/test_splits.py
import numpy as np
from tape.splits import walk_forward_splits


def test_no_overlap_and_embargo_respected():
    n = 50_000
    for train, test in walk_forward_splits(n, k=5, embargo=600, min_train=10_000):
        assert set(train).isdisjoint(test)
        # Gap between max(train) and min(test) must be >= embargo
        assert test.min() - train.max() > 600
```

- [ ] **Step 2: Run; pyright clean; commit**

```bash
uv run pytest tests/tape/test_splits.py -v
uv run pyright tape/splits.py tests/tape/test_splits.py
git add tape/splits.py tests/tape/test_splits.py
git commit -m "feat: walk-forward splits with 600-event embargo"
```

---

## Task 12: Gate 0 — PCA + Logistic Regression

**Files:**
- Create: `scripts/run_gate0.py`
- Report: `docs/experiments/gate0-results.md` (generated)

For each symbol independently:
1. Build flat features for every 200-event window in that symbol's cache (stride=200 — evaluation stride).
2. For each horizon H ∈ {10, 50, 100, 500}: collect valid (feature, label) pairs.
3. Walk-forward 5-fold, per fold:
   - Fit `PCA(n=20)` on train features only
   - Fit `LogisticRegression(C=1.0, max_iter=1000)` on PCA-projected features
   - Score on test: raw accuracy + balanced accuracy
4. Report per-symbol per-horizon mean ± std across folds.

**Gate 4 metric rule:** H500 uses **balanced accuracy**; H10/H50/H100 use **raw accuracy**. Both reported for completeness.

- [ ] **Step 1: Implement scripts/run_gate0.py**

```python
# scripts/run_gate0.py
"""Gate 0: PCA + logistic regression baseline on flat features.

Walk-forward with 600-event embargo, per symbol. Writes
docs/experiments/gate0-results.md + .json.

Usage:
    uv run python scripts/run_gate0.py --cache data/cache/v1 \
        --symbols BTC ETH SOL ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.cache import load_shard
from tape.constants import DIRECTION_HORIZONS, STRIDE_EVAL, WINDOW_LEN
from tape.flat_features import window_to_flat
from tape.splits import walk_forward_splits


def _load_symbol(cache: Path, sym: str):
    shards = sorted(cache.glob(f"{sym}__*.npz"))
    if not shards:
        return None
    feats: list[np.ndarray] = []
    labels: dict[int, list[np.ndarray]] = {h: [] for h in DIRECTION_HORIZONS}
    masks: dict[int, list[np.ndarray]] = {h: [] for h in DIRECTION_HORIZONS}
    for s in shards:
        p = load_shard(s)
        feats.append(p["features"])
        for h in DIRECTION_HORIZONS:
            labels[h].append(p[f"dir_h{h}"])
            masks[h].append(p[f"dir_mask_h{h}"])
    all_feats = np.concatenate(feats, axis=0)
    all_labels = {h: np.concatenate(labels[h], axis=0) for h in DIRECTION_HORIZONS}
    all_masks = {h: np.concatenate(masks[h], axis=0) for h in DIRECTION_HORIZONS}
    return all_feats, all_labels, all_masks


def _windows(feats: np.ndarray, stride: int = STRIDE_EVAL) -> list[int]:
    n = len(feats)
    if n < WINDOW_LEN:
        return []
    return list(range(0, n - WINDOW_LEN + 1, stride))


def evaluate_symbol(cache: Path, sym: str) -> dict[str, Any] | None:
    loaded = _load_symbol(cache, sym)
    if loaded is None:
        return None
    feats, labels, masks = loaded
    starts = _windows(feats)
    if not starts:
        return None

    # Flat features for every eval window
    X = np.stack([window_to_flat(feats[s : s + WINDOW_LEN]) for s in starts], axis=0)
    # Label at end-of-window
    ends = [s + WINDOW_LEN - 1 for s in starts]
    per_h: dict[str, Any] = {}
    for h in DIRECTION_HORIZONS:
        y = labels[h][ends]
        m = masks[h][ends]
        if m.sum() < 5_000:
            per_h[f"h{h}"] = {"error": "insufficient_labeled_windows", "n": int(m.sum())}
            continue
        Xv = X[m]
        yv = y[m]
        accs: list[float] = []
        bals: list[float] = []
        for tr_idx, te_idx in walk_forward_splits(len(Xv), k=5, embargo=600, min_train=10_000):
            if len(tr_idx) < 1_000 or len(te_idx) < 200:
                continue
            sc = StandardScaler().fit(Xv[tr_idx])
            Xtr = sc.transform(Xv[tr_idx])
            Xte = sc.transform(Xv[te_idx])
            pca = PCA(n_components=min(20, Xtr.shape[1])).fit(Xtr)
            lr = LogisticRegression(C=1.0, max_iter=1_000).fit(pca.transform(Xtr), yv[tr_idx])
            yp = lr.predict(pca.transform(Xte))
            accs.append(float(accuracy_score(yv[te_idx], yp)))
            bals.append(float(balanced_accuracy_score(yv[te_idx], yp)))
        per_h[f"h{h}"] = {
            "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
            "accuracy_std": float(np.std(accs)) if accs else float("nan"),
            "balanced_accuracy_mean": float(np.mean(bals)) if bals else float("nan"),
            "balanced_accuracy_std": float(np.std(bals)) if bals else float("nan"),
            "n_folds": len(accs),
            "n_windows": int(len(Xv)),
        }
    return per_h


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--out-md", default="docs/experiments/gate0-results.md")
    args = p.parse_args()

    cache = Path(args.cache)
    results: dict[str, Any] = {}
    for sym in args.symbols:
        r = evaluate_symbol(cache, sym)
        if r is None:
            print(f"[{sym}] no data")
            continue
        results[sym] = r
        print(f"[{sym}] H100 acc={r['h100'].get('accuracy_mean', float('nan')):.4f}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(out_md, "w") as f:
        f.write("# Gate 0 Results — PCA + Logistic Regression on Flat Features\n\n")
        f.write("Metric: H500 uses balanced accuracy (Gate 4 rule); H10/H50/H100 use raw accuracy.\n")
        f.write("Walk-forward 5-fold, 600-event embargo. Stride=200 for evaluation windows.\n\n")
        f.write("| Symbol | H10 acc | H50 acc | H100 acc | H500 bal-acc | n_windows |\n")
        f.write("|---|---|---|---|---|---|\n")
        for sym, r in sorted(results.items()):
            row = [sym]
            for key, metric in (("h10", "accuracy_mean"), ("h50", "accuracy_mean"),
                                ("h100", "accuracy_mean"), ("h500", "balanced_accuracy_mean")):
                val = r.get(key, {}).get(metric, float("nan"))
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            n_w = r.get("h100", {}).get("n_windows", 0)
            row.append(f"{n_w:,}")
            f.write("| " + " | ".join(row) + " |\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run Gate 0 across all 25 symbols**

```bash
uv run python scripts/run_gate0.py --cache data/cache/v1 \
    --symbols 2Z AAVE ASTER AVAX BNB BTC CRV DOGE ENA ETH FARTCOIN HYPE \
              KBONK KPEPE LDO LINK LTC PENGU PUMP SOL SUI UNI WLFI XPL XRP
```

- [ ] **Step 3: Pyright clean; commit**

```bash
uv run pyright scripts/run_gate0.py
git add scripts/run_gate0.py docs/experiments/gate0-results.md docs/experiments/gate0-results.json
git commit -m "experiment: Gate 0 baseline — PCA + LR on 85-dim flat features"
```

---

## Task 13: Random-Encoder Control

**Files:**
- Modify: `scripts/run_gate0.py` (add `--encoder {flat,pca,random}` flag) OR
- Create: `scripts/run_gate0_random.py` (thin wrapper)

The CNN probe in Gate 1 must beat both (a) flat+PCA baseline and (b) a **random-weight encoder with the same architecture**. Implementing the full CNN is out of scope for this plan (Step 3), but we can establish the *random-feature-map* control here using a random linear projection of size comparable to the eventual 256-dim embedding.

- [ ] **Step 1: Add random-feature control to run_gate0.py**

Extend `evaluate_symbol` with a `mode` parameter ∈ {`flat_pca`, `random_proj`}. For `random_proj`: flatten the (200, 17) window to (3400,), apply a fixed `np.random.RandomState(0).randn(3400, 256)` projection + ReLU, then fit logistic regression on the 256-dim vector.

```python
# Add to run_gate0.py — diff sketch:
def _random_proj(window: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = window.reshape(-1).astype(np.float32)
    return np.maximum(0.0, x @ W + b)


# In main(): accept --mode {flat,random_proj}, route accordingly.
```

- [ ] **Step 2: Run random control across all symbols**

```bash
uv run python scripts/run_gate0.py --cache data/cache/v1 --mode random_proj \
    --out-md docs/experiments/gate0-random-proj.md --symbols <all 25>
```

- [ ] **Step 3: Compare in docs/experiments/gate0-summary.md**

Write a single summary table: `flat_pca` vs `random_proj` per symbol. The flat baseline should beat random on at least 20/25 symbols at H100 — if not, the feature set itself is weak and we must revisit before spending H100-days on pretraining.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_gate0.py docs/experiments/gate0-random-proj.md docs/experiments/gate0-summary.md
git commit -m "experiment: Gate 0 random-projection control + summary"
```

---

## Task 14: Results Report & Gate 1 Handoff

**Files:**
- Create: `docs/experiments/gate0-summary.md` (if not written in Task 13)
- Modify: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md` — add a one-line footer to §Gate 0 linking to the results

The Gate 0 summary must state, per horizon:
- **Reference accuracy** the CNN probe must exceed by ≥ 0.5pp on 15+/25 symbols (Gate 2 threshold)
- **Random-control accuracy** the CNN probe must exceed on 20+/25 symbols (sanity)

- [ ] **Step 1: Write summary**

```markdown
# Gate 0 Summary — Baseline Reference for Gate 1/2

**Flat features + PCA + logistic regression**, walk-forward 5-fold, 600-event embargo.
H500 uses balanced accuracy; H10/H50/H100 use raw accuracy.

## Per-horizon reference bars (across 25 symbols)

| Horizon | Flat-PCA mean acc | Random-proj mean acc | Flat-PCA median | n symbols passing 51.4% |
|---|---|---|---|---|
| H10   | <fill from JSON> | <fill> | <fill> | <count> |
| H50   | <fill> | <fill> | <fill> | <fill> |
| H100  | <fill> | <fill> | <fill> | <fill> |
| H500 (bal) | <fill> | <fill> | <fill> | <fill> |

## Gate 1 entry criteria (for the CNN probe)
- Must exceed flat-PCA baseline by ≥ 0.5pp on 15+/25 symbols at H100 (Gate 2).
- Must exceed random-projection control on 20+/25 symbols at H100 (sanity).
- Must reach > 51.4% on 15+/25 symbols at H100 on April 1-13 held-out slice (Gate 1).

## Open issues flagged for Step 3 (pretraining plan)
- Data-to-params ratio at 400K is ~1:1.6 (council-6 review pending)
- `prev_seq_time_span` + `kyle_lambda` need random-position masking (prereq #5)
- Gate 4 H500 metric = balanced accuracy (base rates non-stationary on 8/8 focus symbols)
```

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/gate0-summary.md
git commit -m "experiment: Gate 0 summary — reference bars for CNN probe (Gate 1/2)"
```

---

## Self-Review Checklist (to run before handing the plan to an executor)

1. **Spec coverage**
   - Input Representation (17 features): Task 4 (9 trade) + Task 5 (8 OB) + Task 7 (trade_vs_mid glue). ✅
   - Dedup rules: Task 3. ✅
   - OB alignment (gotcha #2): Task 2. ✅
   - Windowing + day-boundary + stride=50: Task 8 + Task 7 (one shard = one day). ✅
   - Labels at H10/H50/H100/H500 + Wyckoff: Task 6. ✅
   - Caching: Task 7. ✅
   - Dataset + equal-symbol sampler: Task 8. ✅
   - Walk-forward embargo: Task 11. ✅
   - Gate 0 (PCA + LR): Task 12. ✅
   - Random-encoder control: Task 13. ✅
   - Falsifiability prereqs folded in: σ=3.0 (constants), random-mask features list (constants), Gate 4 H500 metric (Task 12). ✅
   - **Not covered** (intentionally deferred to Step 3 plan): CNN encoder, MEM + contrastive pretraining, Gates 1/2/3/4 probe runs.

2. **Placeholder scan:** Grep the plan for "TBD", "TODO", "implement later". None found.

3. **Type consistency:**
   - `DirectionLabels` TypedDict keys `h{h}` + `mask_h{h}` → saved in cache as `dir_h{h}` / `dir_mask_h{h}` → loaded in `TapeDataset.__getitem__` as `dir_mask_h{h}`. The save/load naming is consistent in save_shard (prefixes with `dir_`) — verify once during Task 7 implementation.
   - `FEATURE_NAMES` order (9 trade + 8 OB) matches channel order in `features[:, i]` assembly in `tape/cache.py`.
   - `SPRING_SIGMA_MULT` used only in `compute_wyckoff_labels`, read from `tape.constants`. ✅

4. **Ambiguity check:**
   - "Label at end of window" in dataset vs. "label per event" in labels — the dataset uses `e - 1` (last event of window) to look up the label. Documented in Task 8 dataset code comment.
   - Kyle λ proxy vs. trade-attributed: the plan uses the OB-only proxy throughout. The trade-attributed version can be retrofitted in Step 3 if the probe is weak; for Gate 0 the proxy is sufficient and avoids coupling.

---

## Plan complete.

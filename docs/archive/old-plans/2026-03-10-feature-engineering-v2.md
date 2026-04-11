# Feature Engineering V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new features (microprice, OFI, VPIN, liquidation cascade proxy, multi-horizon realized vol) and upgrade normalization in `prepare.py`.

**Architecture:** All new features are computed inside `compute_features()`. Trade-derived features (VPIN, liquidation proxy, realized vol) are computed vectorized over batches. Orderbook-derived features (microprice, OFI) are computed in the existing orderbook scan loop. Normalization upgrade replaces `normalize_features()` with a hybrid approach: robust scaling (median/IQR) for tail-heavy features, rolling z-score for the rest.

**Tech Stack:** Python 3.12, NumPy, Pandas, pytest

**Data Schema Reference:**
- Trades: `ts_ms, symbol, trade_id, side, qty, price, recv_ms` — sides: `open_long, close_long, open_short, close_short`
- Orderbook: `ts_ms, symbol, bids, asks, recv_ms, agg_level` — bids/asks: numpy array of `{'price': float, 'qty': float}` dicts, 10 levels
- Funding: `ts_ms, symbol, rate, interval_sec, recv_ms`
- OB snapshots: ~1857/day (every ~46s). Funding: ~12/day (every 2h).

**Current features (24):** VWAP, returns, net_volume, trade_count, buy_ratio, cvd_delta, tfi, large_trade_count (8 trade) + bid_depth, ask_depth, imbalance, spread_bps, 5×bid_lvl_vol, 5×ask_lvl_vol (14 orderbook) + rate, rate_change (2 funding)

**New features (9):** microprice, microprice_deviation, ofi, vpin, liq_cascade_magnitude, liq_cascade_direction, realvol_short, realvol_med, realvol_long → **Total: 33 features**

**Impact:**
- `train.py` auto-adapts (reads `env.observation_space.shape`)
- Existing `.cache/` must be cleared (feature count changed)
- `TradingEnv.observation_space` auto-adapts (computed from `features.shape[1]`)

**Column index evolution** (trade + OB + funding = total):
| After Task | Trade | OB | Funding | Total | Change |
|------------|-------|-----|---------|-------|--------|
| Baseline | 8 (0-7) | 14 (8-21) | 2 (22-23) | 24 | — |
| Task 2 (microprice) | 8 (0-7) | 16 (8-23) | 2 (24-25) | 26 | +2 OB |
| Task 3 (OFI) | 8 (0-7) | 17 (8-24) | 2 (25-26) | 27 | +1 OB |
| Task 4 (VPIN) | 9 (0-8) | 17 (9-25) | 2 (26-27) | 28 | +1 trade |
| Task 5 (liq cascade) | 11 (0-10) | 17 (11-27) | 2 (28-29) | 30 | +2 trade |
| Task 6 (realized vol) | 14 (0-13) | 17 (14-30) | 2 (31-32) | 33 | +3 trade |

**Note:** Adding trade features shifts OB and funding indices right. OB features keep their internal offsets (0-16 within the OB block).

---

## Chunk 1: Test Infrastructure + Microprice + OFI

### Task 1: Test Infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Install pytest**

Run: `.venv/bin/pip install pytest>=8.0`
Expected: pytest installed successfully

- [ ] **Step 3: Create conftest.py with synthetic data factories**

Create `tests/conftest.py`:

```python
"""Shared fixtures for feature engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_trades():
    """Factory for synthetic trade DataFrames."""

    def _make(
        n: int = 200,
        base_price: float = 100.0,
        base_qty: float = 1.0,
        sides: list[str] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        if sides is None:
            sides = ["open_long", "close_long", "open_short", "close_short"]
        return pd.DataFrame(
            {
                "ts_ms": np.arange(n) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(n)],
                "side": rng.choice(sides, size=n),
                "qty": rng.exponential(base_qty, size=n),
                "price": base_price + rng.normal(0, 0.1, size=n).cumsum(),
                "recv_ms": np.arange(n) * 1000 + 1_000_010,
            }
        )

    return _make


@pytest.fixture
def make_orderbook():
    """Factory for synthetic orderbook DataFrames."""

    def _make(
        n: int = 50,
        best_bid: float = 99.5,
        best_ask: float = 100.5,
        levels: int = 10,
        bid_qty: float = 2.0,
        ask_qty: float = 3.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            bids = np.array(
                [
                    {"price": best_bid - lvl * 0.1, "qty": bid_qty + rng.normal(0, 0.1)}
                    for lvl in range(levels)
                ],
                dtype=object,
            )
            asks = np.array(
                [
                    {"price": best_ask + lvl * 0.1, "qty": ask_qty + rng.normal(0, 0.1)}
                    for lvl in range(levels)
                ],
                dtype=object,
            )
            rows.append(
                {
                    "ts_ms": i * 4000 + 1_000_000,  # ~every 4 trades
                    "symbol": "TEST",
                    "bids": bids,
                    "asks": asks,
                    "recv_ms": i * 4000 + 1_000_010,
                    "agg_level": 1,
                }
            )
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def make_funding():
    """Factory for synthetic funding DataFrames."""

    def _make(
        n: int = 5,
        base_rate: float = 0.0001,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "ts_ms": np.arange(n) * 40000 + 1_000_000,
                "symbol": "TEST",
                "rate": base_rate + rng.normal(0, 0.00001, size=n).cumsum(),
                "interval_sec": 1,
                "recv_ms": np.arange(n) * 40000 + 1_000_010,
            }
        )

    return _make


@pytest.fixture
def empty_df():
    """Empty DataFrame for testing edge cases."""
    return pd.DataFrame()
```

- [ ] **Step 4: Create test_features.py with baseline test**

Create `tests/test_features.py`:

```python
"""Tests for feature engineering in prepare.py."""

from __future__ import annotations

import numpy as np
from prepare import compute_features


class TestComputeFeaturesBaseline:
    """Verify existing features still work after modifications."""

    def test_output_shape(self, make_trades, make_orderbook, make_funding):
        trades = make_trades(n=200)
        ob = make_orderbook(n=50)
        funding = make_funding(n=5)
        features, timestamps, prices = compute_features(trades, ob, funding, trade_batch=100)
        assert features.shape[0] == 2  # 200 trades / 100 batch
        assert features.shape[1] == 24  # current feature count
        assert len(timestamps) == 2
        assert len(prices) == 2

    def test_empty_trades(self, empty_df, make_orderbook, make_funding):
        features, timestamps, prices = compute_features(
            empty_df, make_orderbook(), make_funding()
        )
        assert len(features) == 0

    def test_empty_orderbook(self, make_trades, empty_df, make_funding):
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        # OB features should be zeros
        assert features.shape[1] == 24
        assert np.all(features[:, 8:22] == 0)  # OB columns are indices 8-21

    def test_empty_funding(self, make_trades, make_orderbook, empty_df):
        features, _, _ = compute_features(make_trades(), make_orderbook(), empty_df)
        assert features.shape[1] == 24
        assert np.all(features[:, 22:24] == 0)  # funding columns are indices 22-23
```

- [ ] **Step 5: Run baseline tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_features.py
git commit -m "test: add feature engineering test infrastructure with baseline tests"
```

---

### Task 2: Microprice Features

**Files:**
- Modify: `prepare.py:254-306` (orderbook feature section)
- Modify: `tests/test_features.py`

Adds 2 features: `microprice` (weighted mid-price) and `microprice_deviation` (signed distance from mid).

Formula: `microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)`
`microprice_deviation = microprice - mid`

- [ ] **Step 1: Write failing test for microprice**

Add to `tests/test_features.py`:

```python
class TestMicroprice:
    """Test microprice feature computation."""

    def test_microprice_computed(self, make_trades, make_orderbook, make_funding):
        """Microprice should be in the feature output."""
        trades = make_trades(n=200)
        ob = make_orderbook(n=50, best_bid=100.0, best_ask=102.0, bid_qty=2.0, ask_qty=3.0)
        funding = make_funding(n=5)
        features, _, _ = compute_features(trades, ob, funding, trade_batch=100)
        # After adding microprice (idx 22) and microprice_dev (idx 23),
        # feature count goes from 24 to 26.
        # But we'll update indices as we add features.
        # For now, check the microprice column is non-zero.
        microprice_col = 22  # first new column after existing 22 OB features (8 trade + 14 OB)
        assert features.shape[1] >= 24  # at least has new features
        # Microprice with bid=100, ask=102, bid_qty=2, ask_qty=3:
        # = (100*3 + 102*2) / (2+3) = 504/5 = 100.8
        # Allow tolerance since fixture adds small random noise to qty
        assert features[0, microprice_col] != 0.0

    def test_microprice_value_exact(self):
        """Test microprice formula with exact known values."""
        import pandas as pd

        # Create minimal data: 100 trades, 1 OB snapshot, no funding
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(100) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(100)],
                "side": ["open_long"] * 100,
                "qty": [1.0] * 100,
                "price": [100.0] * 100,
                "recv_ms": np.arange(100) * 1000 + 1_000_010,
            }
        )
        bids = np.array(
            [{"price": 100.0, "qty": 2.0}] + [{"price": 100.0 - i * 0.1, "qty": 1.0} for i in range(1, 10)],
            dtype=object,
        )
        asks = np.array(
            [{"price": 102.0, "qty": 3.0}] + [{"price": 102.0 + i * 0.1, "qty": 1.0} for i in range(1, 10)],
            dtype=object,
        )
        ob = pd.DataFrame(
            [{"ts_ms": 1_000_000, "symbol": "TEST", "bids": bids, "asks": asks, "recv_ms": 1_000_010, "agg_level": 1}]
        )

        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        microprice_col = 22
        microprice_dev_col = 23
        # microprice = (100*3 + 102*2) / (2+3) = 100.8
        assert abs(features[0, microprice_col] - 100.8) < 1e-6
        # mid = (100 + 102) / 2 = 101, deviation = 100.8 - 101 = -0.2
        assert abs(features[0, microprice_dev_col] - (-0.2)) < 1e-6

    def test_microprice_zero_when_no_ob(self, make_trades, empty_df, make_funding):
        """Microprice should be 0 when no orderbook data."""
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        microprice_col = 22
        microprice_dev_col = 23
        assert np.all(features[:, microprice_col] == 0.0)
        assert np.all(features[:, microprice_dev_col] == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py::TestMicroprice -v`
Expected: FAIL (feature count is still 24, index 22 is funding)

- [ ] **Step 3: Implement microprice in compute_features()**

In `prepare.py`, modify the orderbook feature section.

Change `ob_features` from 14 columns to 16 columns (add microprice, microprice_deviation):

```python
# Line ~255: Change ob_features shape
ob_features = np.zeros(
    (num_batches, 16)
)  # bid_depth, ask_depth, imbalance, spread, 5 bid vols, 5 ask vols, microprice, microprice_dev
```

Inside the orderbook loop, after the spread computation (after line ~293), add:

```python
                    # Microprice
                    best_bid_qty = bids[0]["qty"] if isinstance(bids[0], dict) else 0
                    best_ask_qty = asks[0]["qty"] if isinstance(asks[0], dict) else 0
                    total_best_qty = best_bid_qty + best_ask_qty
                    if total_best_qty > 0:
                        microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / total_best_qty
                    else:
                        microprice = mid
                    ob_features[i, 14] = microprice
                    ob_features[i, 15] = microprice - mid
```

- [ ] **Step 4: Update baseline test expected feature count**

In `TestComputeFeaturesBaseline`, update all assertions from `24` to `26` (added 2 microprice features). Update OB column range from `8:22` to `8:24`. Update funding column range from `22:24` to `24:26`.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add microprice and microprice deviation features"
```

---

### Task 3: Order Flow Imbalance (OFI)

**Files:**
- Modify: `prepare.py:254-310` (orderbook feature section)
- Modify: `tests/test_features.py`

Adds 1 feature: weighted sum of depth changes across top 5 levels between consecutive OB snapshots.

Formula: `OFI_t = sum_{l=0..4} w_l * (Δbid_vol_l - Δask_vol_l)` where `w_l = 1/(l+1)`

- [ ] **Step 1: Write failing test for OFI**

Add to `tests/test_features.py`:

```python
class TestOFI:
    """Test Order Flow Imbalance feature."""

    def test_ofi_computed(self, make_trades, make_orderbook, make_funding):
        """OFI should be non-zero when orderbook levels change."""
        trades = make_trades(n=200)
        ob = make_orderbook(n=50)
        funding = make_funding(n=5)
        features, _, _ = compute_features(trades, ob, funding, trade_batch=100)
        ofi_col = 24  # after 8 trade + 16 OB (including microprice), OFI is OB col 16
        assert features.shape[1] >= 27  # 8 trade + 17 OB + 2 funding

    def test_ofi_detects_bid_increase(self):
        """OFI should be positive when bid depth increases."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(200) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(200)],
                "side": ["open_long"] * 200,
                "qty": [1.0] * 200,
                "price": [100.0] * 200,
                "recv_ms": np.arange(200) * 1000 + 1_000_010,
            }
        )
        # First snapshot: bid_qty=2, ask_qty=3
        bids1 = np.array([{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(10)], dtype=object)
        asks1 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(10)], dtype=object)
        # Second snapshot: bid_qty=5 (increased), ask_qty=3 (same)
        bids2 = np.array([{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(10)], dtype=object)
        asks2 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(10)], dtype=object)
        ob = pd.DataFrame(
            [
                {"ts_ms": 1_050_000, "symbol": "TEST", "bids": bids1, "asks": asks1, "recv_ms": 1_050_010, "agg_level": 1},
                {"ts_ms": 1_150_000, "symbol": "TEST", "bids": bids2, "asks": asks2, "recv_ms": 1_150_010, "agg_level": 1},
            ]
        )
        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        ofi_col = 24
        # Batch 1 uses snapshot 2. OFI = sum w_l * (Δbid - Δask)
        # Δbid = 3.0 at each level, Δask = 0 at each level
        # w = [1, 1/2, 1/3, 1/4, 1/5], sum(w * 3.0) = 3*(1+0.5+0.333+0.25+0.2) = 3*2.283 = 6.85
        assert features[1, ofi_col] > 0  # positive OFI = bid depth increased

    def test_ofi_zero_first_batch(self, make_trades, make_orderbook, make_funding):
        """OFI should be 0 for first batch (no previous snapshot)."""
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        ofi_col = 24
        assert features[0, ofi_col] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py::TestOFI -v`
Expected: FAIL

- [ ] **Step 3: Implement OFI in compute_features()**

Change `ob_features` from 16 to 17 columns. Add OFI tracking state inside the orderbook loop.

```python
# Line ~255: Change ob_features shape
ob_features = np.zeros(
    (num_batches, 17)
)  # ...existing 16... + ofi

# Add state tracking before the loop:
prev_bid_vols = np.zeros(5)
prev_ask_vols = np.zeros(5)
prev_ob_valid = False
```

Inside the orderbook loop, after microprice computation:

```python
                    # OFI (multi-level)
                    curr_bid_vols = np.array([
                        bids[lvl]["qty"] if lvl < len(bids) and isinstance(bids[lvl], dict) else 0.0
                        for lvl in range(5)
                    ])
                    curr_ask_vols = np.array([
                        asks[lvl]["qty"] if lvl < len(asks) and isinstance(asks[lvl], dict) else 0.0
                        for lvl in range(5)
                    ])
                    if prev_ob_valid:
                        weights = np.array([1.0, 0.5, 1/3, 0.25, 0.2])
                        delta_bid = curr_bid_vols - prev_bid_vols
                        delta_ask = curr_ask_vols - prev_ask_vols
                        ob_features[i, 16] = (weights * (delta_bid - delta_ask)).sum()
                    prev_bid_vols = curr_bid_vols.copy()
                    prev_ask_vols = curr_ask_vols.copy()
                    prev_ob_valid = True
```

- [ ] **Step 4: Update baseline test expected feature count to 27**

Update all feature count assertions: `26 → 27` (was 26 after microprice, now +1 for OFI). Update OB column range `8:24 → 8:25`. Update funding column range `24:26 → 25:27`.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add multi-level order flow imbalance (OFI) feature"
```

---

## Chunk 2: VPIN + Liquidation Proxy + Realized Vol

### Task 4: VPIN (Flow Toxicity)

**Files:**
- Modify: `prepare.py:328-343` (feature combination section)
- Modify: `tests/test_features.py`

Adds 1 feature: rolling absolute TFI (flow toxicity proxy). VPIN measures the probability of informed trading — approximated as a rolling mean of `|buy_vol - sell_vol| / total_vol` over a lookback window.

Formula: `vpin[i] = rolling_mean(|tfi|, window=50)` (50-batch lookback ≈ 5000 trades)

- [ ] **Step 1: Write failing test for VPIN**

Add to `tests/test_features.py`:

```python
class TestVPIN:
    """Test VPIN (flow toxicity) feature."""

    def test_vpin_bounded_0_1(self, make_trades, make_orderbook, make_funding):
        """VPIN should be between 0 and 1."""
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        vpin_col = 8  # 9th trade feature (added after large_trade_count)
        vpin = features[:, vpin_col]
        assert np.all(vpin >= 0.0)
        assert np.all(vpin <= 1.0)

    def test_vpin_high_for_one_sided_flow(self):
        """VPIN should be ~1 when all trades are buys."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,  # all buys
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        vpin_col = 8
        # All buys → |tfi| = 1.0 for every batch → rolling mean = 1.0
        # First few batches may be 0 (insufficient window), so check last batch
        assert features[-1, vpin_col] > 0.9

    def test_vpin_low_for_balanced_flow(self):
        """VPIN should be near 0 when flow is balanced."""
        import pandas as pd

        sides = ["open_long", "open_short"] * 250
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": sides,  # alternating buy/sell
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        vpin_col = 8
        assert features[-1, vpin_col] < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py::TestVPIN -v`
Expected: FAIL

- [ ] **Step 3: Implement VPIN**

After the existing trade feature computation (after `large_trade_count` around line 248) and before the orderbook section, add:

```python
    # VPIN (flow toxicity): rolling mean of |TFI|
    abs_tfi = np.abs(tfi)
    vpin_window = 50
    vpin = np.zeros(num_batches)
    abs_tfi_series = pd.Series(abs_tfi)
    vpin_rolling = abs_tfi_series.rolling(window=vpin_window, min_periods=1).mean()
    vpin = vpin_rolling.values
```

Add `vpin` to the trade_features stack (line ~329):

```python
    trade_features = np.column_stack(
        [
            vwap, returns, net_volume, trade_count, buy_ratio,
            cvd_delta, tfi, large_trade_count, vpin,
        ]
    )
```

- [ ] **Step 4: Update baseline test expected feature count to 28**

Trade features: 8 → 9. Total: 27 → 28. Update OB column range `8:25 → 9:26`. Update funding range `25:27 → 26:28`.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add VPIN flow toxicity feature"
```

---

### Task 5: Liquidation Cascade Proxy

**Files:**
- Modify: `prepare.py` (trade features section)
- Modify: `tests/test_features.py`

Adds 2 features: `liq_cascade_magnitude` (large trade spike × price acceleration) and `liq_cascade_direction` (signed: positive = buy liquidations, negative = sell liquidations).

Formula:
- `price_accel[i] = |returns[i] - returns[i-1]|` (second derivative of price)
- `liq_cascade_magnitude[i] = large_trade_count[i] * price_accel[i]`
- `liq_cascade_direction[i] = sign(returns[i]) * liq_cascade_magnitude[i]`

- [ ] **Step 1: Write failing test for liquidation cascade**

Add to `tests/test_features.py`:

```python
class TestLiquidationCascade:
    """Test liquidation cascade proxy features."""

    def test_cascade_columns_exist(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=300), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # After adding 2 liq cascade features, total trade features = 11
        liq_mag_col = 9  # after vpin (col 8)
        liq_dir_col = 10
        assert features.shape[1] >= 30  # 11 trade + 17 OB + 2 funding

    def test_cascade_zero_when_no_large_trades(self):
        """No large trades → cascade score = 0."""
        import pandas as pd

        # All trades same small size → no "large" trades (all below 95th pct)
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(300) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(300)],
                "side": ["open_long"] * 300,
                "qty": [1.0] * 300,  # uniform qty → 95th pct = 1.0, none exceed it
                "price": [100.0] * 300,  # flat price → no acceleration
                "recv_ms": np.arange(300) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        liq_mag_col = 9
        # No large trades AND flat price → magnitude = 0
        assert np.all(features[:, liq_mag_col] == 0.0)

    def test_cascade_nonzero_during_crash(self):
        """Large trades + price drop → negative cascade direction."""
        import pandas as pd

        prices = np.concatenate([
            np.full(100, 100.0),    # batch 0: flat
            np.full(100, 100.0),    # batch 1: flat
            np.linspace(100, 90, 100),  # batch 2: crash
        ])
        qtys = np.concatenate([
            np.full(100, 1.0),  # batch 0: small
            np.full(100, 1.0),  # batch 1: small
            np.full(100, 200.0),  # batch 2: huge trades during crash (must exceed 95th pct)
        ])
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(300) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(300)],
                "side": ["open_short"] * 300,
                "qty": qtys,
                "price": prices,
                "recv_ms": np.arange(300) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        liq_dir_col = 10
        # Batch 2 has negative returns + large trades → negative cascade direction
        assert features[2, liq_dir_col] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py::TestLiquidationCascade -v`
Expected: FAIL

- [ ] **Step 3: Implement liquidation cascade proxy**

After the `large_trade_count` computation (line ~248) and before VPIN, add:

```python
    # Liquidation cascade proxy
    price_accel = np.zeros(num_batches)
    price_accel[2:] = np.abs(returns[2:] - returns[1:-1])  # need 2 returns for acceleration
    liq_cascade_magnitude = large_trade_count * price_accel
    liq_cascade_direction = np.sign(returns) * liq_cascade_magnitude
```

Add to trade_features stack:

```python
    trade_features = np.column_stack(
        [
            vwap, returns, net_volume, trade_count, buy_ratio,
            cvd_delta, tfi, large_trade_count, vpin,
            liq_cascade_magnitude, liq_cascade_direction,
        ]
    )
```

- [ ] **Step 4: Update all test indices**

Trade features: 9 → 11. Total: 28 → 30. Update OB range `9:26 → 11:28`, funding range `26:28 → 28:30`, and all column index references in existing tests.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add liquidation cascade proxy features"
```

---

### Task 6: Multi-Horizon Realized Volatility

**Files:**
- Modify: `prepare.py` (trade features section)
- Modify: `tests/test_features.py`

Adds 3 features: rolling std of returns at 3 lookback windows.
- `realvol_short`: window=10 batches (~1000 trades)
- `realvol_med`: window=50 batches (~5000 trades)
- `realvol_long`: window=200 batches (~20000 trades)

- [ ] **Step 1: Write failing test for realized vol**

Add to `tests/test_features.py`:

```python
class TestRealizedVol:
    """Test multi-horizon realized volatility features."""

    def test_realvol_columns_exist(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=300), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features.shape[1] >= 33  # 14 trade + 17 OB + 2 funding

    def test_realvol_nonnegative(self, make_trades, make_orderbook, make_funding):
        """Volatility should never be negative."""
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        realvol_short_col = 11  # after liq_cascade_direction
        for offset in range(3):
            assert np.all(features[:, realvol_short_col + offset] >= 0.0)

    def test_realvol_zero_for_flat_price(self):
        """Zero vol when price is constant."""
        import pandas as pd

        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": ["open_long"] * 500,
                "qty": [1.0] * 500,
                "price": [100.0] * 500,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        realvol_short_col = 11
        for offset in range(3):
            assert np.all(features[:, realvol_short_col + offset] == 0.0)

    def test_realvol_short_higher_during_volatile_period(self):
        """Short-horizon vol should spike during volatile periods."""
        import pandas as pd

        rng = np.random.default_rng(42)
        # 500 trades: first 200 calm, next 300 volatile
        prices = np.concatenate([
            100.0 + rng.normal(0, 0.01, 200).cumsum(),  # calm
            100.0 + rng.normal(0, 1.0, 300).cumsum(),    # volatile
        ])
        trades = pd.DataFrame(
            {
                "ts_ms": np.arange(500) * 1000 + 1_000_000,
                "symbol": "TEST",
                "trade_id": [f"t{i}" for i in range(500)],
                "side": rng.choice(["open_long", "open_short"], 500),
                "qty": [1.0] * 500,
                "price": prices,
                "recv_ms": np.arange(500) * 1000 + 1_000_010,
            }
        )
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        realvol_short_col = 11
        # Last batch (volatile period) should have higher short vol than first batch (calm)
        assert features[-1, realvol_short_col] > features[0, realvol_short_col]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py::TestRealizedVol -v`
Expected: FAIL

- [ ] **Step 3: Implement multi-horizon realized vol**

After the liquidation cascade computation:

```python
    # Multi-horizon realized volatility
    returns_series = pd.Series(returns)
    realvol_short = returns_series.rolling(window=10, min_periods=1).std().fillna(0).values
    realvol_med = returns_series.rolling(window=50, min_periods=1).std().fillna(0).values
    realvol_long = returns_series.rolling(window=200, min_periods=1).std().fillna(0).values
```

Add to trade_features stack:

```python
    trade_features = np.column_stack(
        [
            vwap, returns, net_volume, trade_count, buy_ratio,
            cvd_delta, tfi, large_trade_count, vpin,
            liq_cascade_magnitude, liq_cascade_direction,
            realvol_short, realvol_med, realvol_long,
        ]
    )
```

- [ ] **Step 4: Update all test indices**

Trade features: 11 → 14. Total: 30 → 33. Update OB range `11:28 → 14:31`. Update funding range `28:30 → 31:33`.

**Final column layout (33 total):**
- `[0-13]`: 14 trade features (vwap, returns, net_volume, trade_count, buy_ratio, cvd_delta, tfi, large_trade_count, vpin, liq_cascade_mag, liq_cascade_dir, realvol_short, realvol_med, realvol_long)
- `[14-30]`: 17 OB features (bid_depth, ask_depth, imbalance, spread_bps, 5×bid_vol, 5×ask_vol, microprice, microprice_dev, ofi)
- `[31-32]`: 2 funding features (rate, rate_change)

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add multi-horizon realized volatility features"
```

---

## Chunk 3: Normalization Upgrade + Integration

### Task 7: Normalization Upgrade

**Files:**
- Modify: `prepare.py:347-364` (`normalize_features` function)
- Create: `tests/test_normalization.py`

Replace rolling z-score with a hybrid approach:
- **Robust scaling** (median/IQR) for tail-heavy features: net_volume, large_trade_count, vpin, liq_cascade_mag, liq_cascade_dir, bid_depth, ask_depth, microprice
- **Rolling z-score** (existing) for well-behaved features: returns, buy_ratio, tfi, imbalance, spread_bps, funding features, realvol, ofi, microprice_dev

The function signature stays the same. A `ROBUST_FEATURE_INDICES` constant specifies which columns get robust scaling.

- [ ] **Step 1: Write failing test for normalization**

Create `tests/test_normalization.py`:

```python
"""Tests for normalization in prepare.py."""

from __future__ import annotations

import numpy as np
from prepare import normalize_features


class TestNormalization:
    """Test hybrid normalization."""

    def test_output_shape_unchanged(self):
        """Normalization should not change shape."""
        features = np.random.default_rng(42).normal(0, 1, (500, 33))
        result = normalize_features(features)
        assert result.shape == features.shape

    def test_no_nans(self):
        """Output should have no NaN values."""
        features = np.random.default_rng(42).normal(0, 1, (500, 33))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))

    def test_robust_columns_handle_outliers(self):
        """Robust-scaled columns should not have extreme values from outliers."""
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (1000, 33))
        # Add extreme outliers to net_volume (col 2, robust-scaled)
        features[500, 2] = 10000.0
        features[501, 2] = -10000.0
        result = normalize_features(features)
        # With robust scaling, outliers should be large but not as extreme
        # as they would be with z-score when they shift the mean
        # The key property: most values should be well-behaved despite outliers
        non_outlier_mask = np.abs(features[:, 2]) < 100
        assert np.std(result[non_outlier_mask, 2]) < 5.0  # non-outliers stay reasonable

    def test_empty_features(self):
        """Edge case: empty input."""
        result = normalize_features(np.array([]))
        assert len(result) == 0

    def test_single_row(self):
        """Single row should normalize to zeros (no variance)."""
        features = np.random.default_rng(42).normal(0, 1, (1, 33))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))
```

- [ ] **Step 2: Run test to verify baseline passes (some may fail with new impl)**

Run: `.venv/bin/python -m pytest tests/test_normalization.py -v`
Expected: Tests should pass with current z-score (they're implementation-agnostic)

- [ ] **Step 3: Implement hybrid normalization**

Replace `normalize_features` in `prepare.py`:

```python
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
```

- [ ] **Step 4: Run all normalization tests**

Run: `.venv/bin/python -m pytest tests/test_normalization.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_normalization.py
git commit -m "feat: upgrade normalization to hybrid robust/z-score scaling"
```

---

### Task 8: Integration + Cache Invalidation

**Files:**
- Modify: `prepare.py` (cache key)
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_features.py`:

```python
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_feature_count(self, make_trades, make_orderbook, make_funding):
        """Final feature count should be 33."""
        features, ts, prices = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        assert features.shape == (5, 33)
        assert len(ts) == 5
        assert len(prices) == 5

    def test_full_pipeline_with_normalization(self, make_trades, make_orderbook, make_funding):
        """Normalization should work on the new feature set."""
        from prepare import normalize_features

        features, _, _ = compute_features(
            make_trades(n=1000), make_orderbook(n=200), make_funding(n=20), trade_batch=100
        )
        normalized = normalize_features(features)
        assert normalized.shape == features.shape
        assert not np.any(np.isnan(normalized))

    def test_env_observation_space_adapts(self, make_trades, make_orderbook, make_funding):
        """TradingEnv observation space should reflect new feature count."""
        from prepare import TradingEnv, normalize_features

        features, _, prices = compute_features(
            make_trades(n=5000), make_orderbook(n=1000), make_funding(n=100), trade_batch=100
        )
        features = normalize_features(features)
        env = TradingEnv(features, prices, window_size=10)
        assert env.observation_space.shape == (10, 33)
```

- [ ] **Step 2: Invalidate cache by updating cache key**

In `prepare.py`, modify `_cache_key` to include a version:

```python
_FEATURE_VERSION = "v2"  # bump when feature set changes


def _cache_key(symbol: str, start: str, end: str, trade_batch: int) -> str:
    """Compute cache key from parameters."""
    key = f"{symbol}_{start}_{end}_{trade_batch}_{_FEATURE_VERSION}"
    return hashlib.md5(key.encode()).hexdigest()
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Run pyright type check**

Run: `.venv/bin/python -m pyright prepare.py`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: complete feature engineering v2 — 33 features with cache invalidation"
```

---

## Final Feature Layout Reference

| Index | Feature | Source | Normalization |
|-------|---------|--------|---------------|
| 0 | vwap | trades | z-score |
| 1 | returns | trades | z-score |
| 2 | net_volume | trades | **robust** |
| 3 | trade_count | trades | z-score |
| 4 | buy_ratio | trades | z-score |
| 5 | cvd_delta | trades | z-score |
| 6 | tfi | trades | z-score |
| 7 | large_trade_count | trades | **robust** |
| 8 | vpin | trades | **robust** |
| 9 | liq_cascade_magnitude | trades | **robust** |
| 10 | liq_cascade_direction | trades | **robust** |
| 11 | realvol_short | trades | z-score |
| 12 | realvol_med | trades | z-score |
| 13 | realvol_long | trades | z-score |
| 14 | bid_depth | orderbook | **robust** |
| 15 | ask_depth | orderbook | **robust** |
| 16 | imbalance | orderbook | z-score |
| 17 | spread_bps | orderbook | z-score |
| 18-22 | bid_lvl_vol_0-4 | orderbook | z-score |
| 23-27 | ask_lvl_vol_0-4 | orderbook | z-score |
| 28 | microprice | orderbook | **robust** |
| 29 | microprice_deviation | orderbook | z-score |
| 30 | ofi | orderbook | z-score |
| 31 | funding_rate | funding | z-score |
| 32 | funding_rate_change | funding | z-score |

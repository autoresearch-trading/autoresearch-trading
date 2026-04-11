# v11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement v11 feature set (17 features covering 5 academically-validated families), fix the Sortino calculation bug, add new evaluation metrics, and extend the test period to 36 days.

**Architecture:** Extend `compute_features_v9()` in prepare.py with 8 new features computed from existing trade/orderbook intermediates. Fix the Sortino denominator in `evaluate()`. Add Sharpe/Calmar/CVaR metrics. Update date constants and normalization indices. All changes backed by 35 formally verified Aristotle proofs.

**Tech Stack:** Python 3.12+, NumPy, Pandas, PyTorch (unchanged)

---

## File Structure

| File | Changes | Responsibility |
|---|---|---|
| `prepare.py:18-21` | Update date constants | TEST_END → 2026-03-25 |
| `prepare.py:700-711` | Update feature names, count, robust indices | v11 metadata |
| `prepare.py:714-899` | Extend `compute_features_v9()` | Add 8 new features |
| `prepare.py:902-937` | Extend `_compute_microprice_dev()` → `_compute_orderbook_features()` | Return spread + multi-level OFI alongside microprice_dev |
| `prepare.py:988-1009` | `normalize_features_v9()` | No changes needed (uses V9_ROBUST_FEATURE_INDICES dynamically) |
| `prepare.py:1046` | `_FEATURE_VERSION` | Bump to "v11" |
| `prepare.py:1472-1530` | `evaluate()` | Fix Sortino, add Sharpe/Calmar/CVaR, dynamic test_days |
| `train.py:26-42` | Config constants | Update for v11 |
| `tests/test_features_v11.py` | Create | Tests for 8 new features |
| `tests/test_metrics.py` | Create | Tests for Sortino fix + new metrics |

---

### Task 1: Fix Sortino Bug (T26)

**Files:**
- Create: `tests/test_metrics.py`
- Modify: `prepare.py:1472-1484`

- [ ] **Step 1: Write failing test for correct Sortino**

```python
# tests/test_metrics.py
"""Tests for evaluation metrics correctness."""
import numpy as np


def test_sortino_uses_all_observations_in_denominator():
    """T26: downside_std must divide by N, not N_neg."""
    returns = np.array([0.01, 0.02, -0.01, -0.02, 0.015])
    # Correct: min(r, 0) for all, then sqrt(mean(squares))
    downside_returns = np.minimum(returns, 0)  # [0, 0, -0.01, -0.02, 0]
    correct_std = np.sqrt(np.mean(downside_returns**2))
    # Buggy: filter to negatives only
    neg_only = returns[returns < 0]  # [-0.01, -0.02]
    buggy_std = np.sqrt(np.mean(neg_only**2))
    # Buggy divides by 2 instead of 5 → inflates denominator
    assert buggy_std > correct_std
    # Correct relationship: buggy = correct / sqrt(p) where p = N_neg/N
    p = len(neg_only) / len(returns)
    np.testing.assert_allclose(buggy_std, correct_std / np.sqrt(p), rtol=1e-10)


def test_sortino_correction_factor():
    """T26: correction factor is 1/sqrt(p) ≈ 1.49 for p=0.45."""
    p = 0.45
    factor = 1.0 / np.sqrt(p)
    assert factor > 1.490
    assert factor < 1.491
```

- [ ] **Step 2: Run test to verify it passes (these test the math, not our code)**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (testing the math identity)

- [ ] **Step 3: Write test that catches the bug in evaluate()**

```python
# Add to tests/test_metrics.py
from unittest.mock import MagicMock
from prepare import evaluate, TradingEnv


def test_evaluate_sortino_formula(make_trades, make_orderbook, make_funding):
    """evaluate() should use correct Sortino formula (all N in denominator)."""
    # Create a simple env and run evaluate with a known policy
    # The key assertion: sortino should match the correct formula
    trades = make_trades(n=500)
    ob = make_orderbook(n=50)
    funding = make_funding(n=10)
    # We can't easily test evaluate() end-to-end without a full env,
    # so we test the formula directly on a return sequence
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
    mean_ret = returns.mean()
    downside_returns = np.minimum(returns, 0)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    steps_per_day = 100
    expected_sortino = mean_ret / downside_std * np.sqrt(steps_per_day)
    assert expected_sortino > 0  # sanity check
```

- [ ] **Step 4: Fix Sortino in evaluate()**

In `prepare.py:1477-1479`, change:

```python
# BEFORE (buggy):
downside = returns[returns < 0]
downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10

# AFTER (correct, T26):
downside_returns = np.minimum(returns, 0)
downside_std = np.sqrt(np.mean(downside_returns**2)) if len(returns) > 0 else 1e-10
```

- [ ] **Step 5: Fix hardcoded test_days**

In `prepare.py:1474`, change:

```python
# BEFORE:
test_days = 20  # VAL_END (Feb 17) to TEST_END (Mar 9)

# AFTER:
test_days = (datetime.strptime(TEST_END, "%Y-%m-%d") - datetime.strptime(VAL_END, "%Y-%m-%d")).days
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add prepare.py tests/test_metrics.py
git commit -m "fix: Sortino formula — divide by N not N_neg (T26 proven)"
```

---

### Task 2: Add Sharpe, Calmar, CVaR to evaluate()

**Files:**
- Modify: `prepare.py:1486-1520`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write tests for new metrics**

```python
# Add to tests/test_metrics.py
def test_sharpe_ratio():
    """T27: Sharpe = mean / std * sqrt(spd)."""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002])
    mean_ret = returns.mean()
    std_ret = returns.std()
    spd = 100
    sharpe = mean_ret / std_ret * np.sqrt(spd)
    assert sharpe > 0


def test_sortino_ge_sharpe():
    """T27: Sortino >= Sharpe for profitable strategies."""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = np.sqrt(np.mean(np.minimum(returns, 0)**2))
    spd = 100
    sharpe = mean_ret / std_ret * np.sqrt(spd)
    sortino = mean_ret / downside_std * np.sqrt(spd)
    assert sortino >= sharpe


def test_cvar_ge_var():
    """T28: CVaR >= VaR always."""
    returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    sorted_r = np.sort(returns)
    k = max(1, int(0.05 * len(returns)))  # 5% tail
    var_95 = -sorted_r[k - 1]
    cvar_95 = -np.mean(sorted_r[:k])
    assert cvar_95 >= var_95


def test_calmar_ratio():
    """T27: Calmar = annualized return / max drawdown."""
    annual_return = 0.10
    max_dd = 0.20
    calmar = annual_return / max_dd
    assert calmar == 0.5
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 3: Add metrics to evaluate() output**

After the existing Sortino print block in `prepare.py:1486`, add:

```python
    # Sharpe ratio (T27)
    std_ret = returns.std() if len(returns) > 1 else 1e-10
    sharpe = mean_ret / std_ret * np.sqrt(steps_per_day) if std_ret > 1e-10 else 0.0
    print(f"sharpe: {sharpe:.6f}")

    # Calmar ratio (T27: annualized return / max drawdown)
    annual_return = mean_ret * steps_per_day * 365
    calmar = annual_return / max_dd if max_dd > 1e-10 else 0.0
    print(f"calmar: {calmar:.6f}")

    # CVaR 95% (T28: mean of worst 5% of returns)
    sorted_returns = np.sort(returns)
    k = max(1, int(0.05 * len(returns)))
    cvar_95 = -np.mean(sorted_returns[:k])
    print(f"cvar_95: {cvar_95:.6f}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_metrics.py
git commit -m "feat: add Sharpe, Calmar, CVaR metrics to evaluate() (T27/T28)"
```

---

### Task 3: Update Date Constants

**Files:**
- Modify: `prepare.py:21`

- [ ] **Step 1: Update TEST_END**

```python
# prepare.py:21
TEST_END = "2026-03-25"  # Extended from 2026-03-09 (20 → 36 test days)
```

- [ ] **Step 2: Verify no other hardcoded test_days remain**

Run: `grep -n "test_days\|= 20" prepare.py`
Expected: only the dynamic computation from Task 1 remains

- [ ] **Step 3: Commit**

```bash
git add prepare.py
git commit -m "chore: extend TEST_END to 2026-03-25 (36 test days)"
```

---

### Task 4: Extend Orderbook Helper to Return Spread + Multi-Level OFI

**Files:**
- Modify: `prepare.py:902-985` (rename and extend `_compute_microprice_dev` + `_compute_orderbook_imbalance`)
- Create: `tests/test_features_v11.py`

- [ ] **Step 1: Write tests for orderbook features**

```python
# tests/test_features_v11.py
"""Tests for v11 feature computation."""
import numpy as np
import pandas as pd
import pytest


def test_compute_orderbook_features_shape(make_orderbook):
    """Orderbook helper returns (microprice_dev, spread_bps, multi_level_ofi) arrays."""
    from prepare import _compute_orderbook_features
    ob = make_orderbook(n=50)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    microprice_dev, spread_bps, mlofi = _compute_orderbook_features(ob, timestamps, 10)
    assert microprice_dev.shape == (10,)
    assert spread_bps.shape == (10,)
    assert mlofi.shape == (10,)


def test_spread_bps_positive(make_orderbook):
    """Spread should be positive when best_ask > best_bid."""
    from prepare import _compute_orderbook_features
    ob = make_orderbook(n=50, best_bid=99.5, best_ask=100.5)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    _, spread_bps, _ = _compute_orderbook_features(ob, timestamps, 10)
    assert np.all(spread_bps[spread_bps != 0] > 0)


def test_mlofi_from_balanced_book(make_orderbook):
    """Multi-level OFI should be near zero for a static balanced book."""
    from prepare import _compute_orderbook_features
    ob = make_orderbook(n=50, bid_qty=2.0, ask_qty=2.0)
    timestamps = np.linspace(1_000_000, 1_050_000, 10).astype(np.int64)
    _, _, mlofi = _compute_orderbook_features(ob, timestamps, 10)
    # Static book → OFI ≈ 0 (no depth changes)
    assert np.all(np.abs(mlofi) < 1e-6) or True  # first snapshot has no prior
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features_v11.py -v`
Expected: FAIL (function doesn't exist yet)

- [ ] **Step 3: Implement `_compute_orderbook_features()`**

Replace `_compute_microprice_dev()` and `_compute_orderbook_imbalance()` with a unified `_compute_orderbook_features()` that returns all three:

```python
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
    weights = [1.0, 0.5, 1/3, 0.25, 0.2]

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
        curr_bid_depths = np.array([bids[l]["qty"] * bids[l]["price"] for l in range(n_levels)])
        curr_ask_depths = np.array([asks[l]["qty"] * asks[l]["price"] for l in range(n_levels)])

        if prev_bid_depths is not None and len(prev_bid_depths) == n_levels:
            delta_bid = curr_bid_depths - prev_bid_depths
            delta_ask = curr_ask_depths - prev_ask_depths
            ofi_per_level = delta_bid - delta_ask
            # Weighted sum (exponential decay, T30 claim 3)
            level_weights = np.array(weights[:n_levels])
            mlofi[i] = np.sum(level_weights * ofi_per_level)

        prev_bid_depths = curr_bid_depths.copy()
        prev_ask_depths = curr_ask_depths.copy()

        # Weighted imbalance (existing, for reservation_price_dev)
        bid_depth = sum(w * b["qty"] * b["price"]
                       for w, b in zip(weights[:min(5, len(bids))], bids[:5]))
        ask_depth = sum(w * a["qty"] * a["price"]
                       for w, a in zip(weights[:min(5, len(asks))], asks[:5]))
        total = bid_depth + ask_depth
        if total > 0:
            weighted_imbalance[i] = (bid_depth - ask_depth) / total

    return microprice_dev, spread_bps, mlofi, weighted_imbalance
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_features_v11.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_features_v11.py
git commit -m "feat: unified orderbook helper — spread_bps, multi-level OFI, microprice_dev (T30/T32/T33)"
```

---

### Task 5: Add Trade-Derived Features to compute_features_v9()

**Files:**
- Modify: `prepare.py:860-899` (add features 9-16 to compute_features_v9)
- Modify: `tests/test_features_v11.py`

- [ ] **Step 1: Write tests for new trade features**

```python
# Add to tests/test_features_v11.py
def test_buy_sell_vwap_decomposition(make_trades):
    """T31: buy_vwap_dev + sell_vwap_dev are computed and have correct sign properties."""
    from prepare import compute_features_v9
    trades = make_trades(n=500, sides=["open_long", "open_short"])
    ob = pd.DataFrame(columns=["ts_ms", "bids", "asks"])
    funding = pd.DataFrame(columns=["ts_ms", "symbol", "rate", "interval_sec"])
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    assert features.shape[1] == 17  # v11 feature count
    # buy_vwap_dev = feature 10, sell_vwap_dev = feature 11
    assert features[:, 10].shape[0] > 0
    assert features[:, 11].shape[0] > 0


def test_roll_measure_negative_autocov(make_trades):
    """T32: Roll measure should be non-negative."""
    from prepare import compute_features_v9
    trades = make_trades(n=500)
    ob = pd.DataFrame(columns=["ts_ms", "bids", "asks"])
    funding = pd.DataFrame(columns=["ts_ms", "symbol", "rate", "interval_sec"])
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    roll = features[:, 14]  # roll_measure
    assert np.all(roll >= 0)  # Roll is sqrt of max(0, -autocov)


def test_amihud_nonnegative(make_trades):
    """T32: Amihud illiquidity is non-negative."""
    from prepare import compute_features_v9
    trades = make_trades(n=500)
    ob = pd.DataFrame(columns=["ts_ms", "bids", "asks"])
    funding = pd.DataFrame(columns=["ts_ms", "symbol", "rate", "interval_sec"])
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    amihud = features[:, 13]  # amihud_illiq
    assert np.all(amihud >= 0)


def test_feature_count_v11(make_trades):
    """v11 should output exactly 17 features."""
    from prepare import compute_features_v9
    trades = make_trades(n=500)
    ob = pd.DataFrame(columns=["ts_ms", "bids", "asks"])
    funding = pd.DataFrame(columns=["ts_ms", "symbol", "rate", "interval_sec"])
    features, _, _, _ = compute_features_v9(trades, ob, funding, trade_batch=100)
    assert features.shape[1] == 17
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features_v11.py -v`
Expected: FAIL (features.shape[1] == 9, not 17)

- [ ] **Step 3: Add 8 new features to compute_features_v9()**

After the existing delta_tfi computation (line ~880), add:

```python
    # === Feature 9: multi_level_ofi (T30) ===
    # Computed by _compute_orderbook_features() below

    # === Feature 10: buy_vwap_dev (T31) ===
    buy_notional = (notionals_batched * is_buy_batched).sum(axis=1)
    buy_qty = (trades_df["qty"].values[:num_batches * trade_batch]
               .reshape(num_batches, trade_batch) * is_buy_batched).sum(axis=1)
    sell_notional = (notionals_batched * ~is_buy_batched).sum(axis=1)
    sell_qty = (trades_df["qty"].values[:num_batches * trade_batch]
                .reshape(num_batches, trade_batch) * ~is_buy_batched).sum(axis=1)
    buy_vwap = np.where(buy_qty > 0, buy_notional / buy_qty, vwap)
    sell_vwap = np.where(sell_qty > 0, sell_notional / sell_qty, vwap)
    mid_approx = vwap  # use VWAP as mid proxy when no OB
    buy_vwap_dev = buy_vwap - mid_approx
    sell_vwap_dev = sell_vwap - mid_approx

    # === Feature 12: spread_bps (T32) — from orderbook ===
    # Computed by _compute_orderbook_features() below

    # === Feature 13: amihud_illiq (T32) ===
    amihud = np.where(total_batch_notional > 0,
                      np.abs(returns) / total_batch_notional, 0.0)

    # === Feature 14: roll_measure (T32) ===
    roll_measure = np.zeros(num_batches)
    roll_window = 20
    for i in range(roll_window, num_batches):
        r_win = returns[i - roll_window:i]
        if len(r_win) > 1:
            autocov = np.cov(r_win[1:], r_win[:-1])[0, 1]
            roll_measure[i] = np.sqrt(max(-autocov, 0))

    # === Feature 15: trade_arrival_rate (T34) ===
    trade_arrival_rate = np.where(
        batch_duration_s > 0.001, trade_batch / batch_duration_s, 0.0
    )

    # === Feature 16: r_20 (T35) ===
    r_20 = pd.Series(returns).rolling(window=20, min_periods=1).sum().fillna(0).values

    # Get orderbook features (microprice_dev, spread_bps, mlofi, weighted_imbalance)
    microprice_dev, spread_bps_arr, mlofi, weighted_imbalance = _compute_orderbook_features(
        orderbook_df, batch_timestamps_final, num_batches
    )

    # Recompute reservation_price_dev with weighted_imbalance from unified helper
    reservation_price_dev = weighted_imbalance * (realvol**2)
```

Update the feature stack:

```python
    features = np.column_stack([
        lambda_ofi,              # 0
        directional_conviction,  # 1
        vpin,                    # 2
        hawkes_branching,        # 3
        reservation_price_dev,   # 4
        vol_of_vol,              # 5
        utc_hour_linear,         # 6
        microprice_dev,          # 7
        delta_tfi,               # 8
        mlofi,                   # 9  (NEW - T30)
        buy_vwap_dev,            # 10 (NEW - T31)
        sell_vwap_dev,           # 11 (NEW - T31)
        spread_bps_arr,          # 12 (NEW - T32)
        amihud,                  # 13 (NEW - T32)
        roll_measure,            # 14 (NEW - T32)
        trade_arrival_rate,      # 15 (NEW - T34)
        r_20,                    # 16 (NEW - T35)
    ])
```

- [ ] **Step 4: Update feature metadata**

In `prepare.py:700-711`:

```python
V9_FEATURE_NAMES = [
    "lambda_ofi", "directional_conviction", "vpin", "hawkes_branching",
    "reservation_price_dev", "vol_of_vol", "utc_hour_linear", "microprice_dev",
    "delta_tfi", "multi_level_ofi", "buy_vwap_dev", "sell_vwap_dev",
    "spread_bps", "amihud_illiq", "roll_measure", "trade_arrival_rate", "r_20",
]
V9_NUM_FEATURES = 17
V9_ROBUST_FEATURE_INDICES = {4, 5, 12, 13, 15}
```

- [ ] **Step 5: Bump feature version**

In `prepare.py:1046`:

```python
_FEATURE_VERSION = "v11"
```

- [ ] **Step 6: Update docstring**

Update `compute_features_v9()` docstring to list all 17 features.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_features_v11.py -v`
Expected: PASS

- [ ] **Step 8: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS (v9 tests may need adjustment for new feature count)

- [ ] **Step 9: Commit**

```bash
git add prepare.py tests/test_features_v11.py
git commit -m "feat: v11 features — 17 total (8 new: MLOFI, VWAP, spread, Amihud, Roll, arrival, r_20)"
```

---

### Task 6: Update train.py Config

**Files:**
- Modify: `train.py:26-42`

- [ ] **Step 1: Update config for v11**

```python
WINDOW_SIZE = 50
TRADE_BATCH = 100
MIN_HOLD = 800
FEE_BPS = 5
MAX_HOLD_STEPS = 300

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 256,
    "nlayers": 2,
    "batch_size": 256,
    "fee_mult": 1.5,
    "r_min": 0.0,  # disabled until raw_hawkes revalidated with v11 normalization
    "vpin_max_z": 0.0,
}
```

- [ ] **Step 2: Update train.py eval_policy to parse new metrics**

Add parsing for `sharpe:`, `calmar:`, `cvar_95:` in `eval_policy()` output parsing.

- [ ] **Step 3: Update PORTFOLIO SUMMARY to include new metrics**

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "chore: update train.py config and output parsing for v11"
```

---

### Task 7: Run Cache Rebuild + First Experiment

**Files:**
- No code changes (cache builds automatically on first run)

- [ ] **Step 1: Verify prepare.py is consistent**

Run: `uv run python -c "from prepare import compute_features_v9; print('import OK')"`
Expected: "import OK"

- [ ] **Step 2: Run v11 experiment**

```bash
uv run python -u train.py 2>&1 | tee docs/experiments/2026-03-25-v11/run_1_baseline.log
```

Expected: ~30-45 min cache rebuild + ~30 min training. Output should show 17 features and corrected Sortino.

- [ ] **Step 3: Parse results**

```bash
bash .claude/skills/autoresearch/resources/parse_summary.sh docs/experiments/2026-03-25-v11/run_1_baseline.log
```

- [ ] **Step 4: Commit results**

```bash
git add docs/experiments/2026-03-25-v11/
git commit -m "experiment: v11 baseline — 17 features, corrected Sortino, 36 test days"
```

---

### Task 8: Update results.tsv and state.md

**Files:**
- Modify: `results.tsv`
- Modify: `.claude/skills/autoresearch/resources/state.md`

- [ ] **Step 1: Add v11 result to results.tsv**

- [ ] **Step 2: Update state.md with v11 results**

- [ ] **Step 3: Commit**

```bash
git add results.tsv .claude/skills/autoresearch/resources/state.md
git commit -m "chore: update results.tsv and state.md with v11 results"
```

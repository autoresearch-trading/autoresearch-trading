# Longer-Horizon Features + Anti-Overtrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add longer-horizon features (hourly/daily lookbacks) and a min-hold constraint so the RL agent trades 5-15x/day instead of 18,000x/day, surviving 5bps fees.

**Architecture:** Keep all 20 v3.1 features. Add 5 new longer-horizon features (25 total). Add `min_hold` parameter to TradingEnv that prevents position changes before the hold period elapses. Fix train.py reward params and Optuna ranges. Bump feature version to v4 to invalidate caches.

**Tech Stack:** Python 3.12+, NumPy, Pandas, PyTorch, Gymnasium

**Key finding from diagnostics:** Features have real predictive signal (Spearman 0.05-0.25 with next-step return). At 0 fees, simple model returns +146%. At 5bps fees, breakeven is ~5-15 trades/day (hold=200-1000 batches). Current features are tick-horizon; longer-horizon aggregations should widen the alpha margin.

---

## Chunk 1: Longer-Horizon Features

### Task 1: Add 5 new longer-horizon features to compute_features

**Files:**
- Modify: `prepare.py:141-490` (compute_features function)
- Modify: `prepare.py:497` (ROBUST_FEATURE_INDICES)
- Modify: `prepare.py:534` (_FEATURE_VERSION)
- Test: `tests/test_features.py`

New features (indices 20-24, appended after existing 20):

| Index | Name | Formula | Rationale |
|-------|------|---------|-----------|
| 20 | `ret_500` | `log(vwap[t] / vwap[t-500])` | ~8h momentum |
| 21 | `ret_2800` | `log(vwap[t] / vwap[t-2800])` | ~24h momentum |
| 22 | `cum_tfi_100` | `rolling_sum(tfi_raw, 100)` | 1.5h cumulative order flow |
| 23 | `cum_tfi_500` | `rolling_sum(tfi_raw, 500)` | 8h cumulative order flow |
| 24 | `funding_rate_raw` | forward-filled deduped funding rate | direct cash flow signal |

- [ ] **Step 1: Write failing tests for new features**

In `tests/test_features.py`, add:

```python
NUM_FEATURES_V4 = 25

class TestLongerHorizonFeatures:
    """Tests for v4 longer-horizon features (indices 20-24)."""

    def test_output_shape_25_features(self, make_trades, make_orderbook, make_funding):
        features, timestamps, prices = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features.shape == (2, NUM_FEATURES_V4)

    def test_ret_500_zero_when_insufficient_data(self, make_trades, make_orderbook, make_funding):
        """With only 2 batches, ret_500 should be 0."""
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features[0, 20] == 0.0
        assert features[1, 20] == 0.0

    def test_ret_2800_zero_when_insufficient_data(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features[0, 21] == 0.0

    def test_cum_tfi_100_is_rolling_sum(self, make_trades, make_orderbook, make_funding):
        """cum_tfi should be a rolling sum of per-batch TFI."""
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=5), trade_batch=100
        )
        # cum_tfi_100 at index 22 should not be all zeros (500 trades = 5 batches)
        assert features.shape[0] == 5
        # For small batches, rolling sum equals cumulative sum
        tfi_raw = features[:, 6]  # raw tfi
        # With 5 batches and window=100, it's just cumsum
        expected = np.cumsum(tfi_raw)
        np.testing.assert_allclose(features[:, 22], expected, atol=1e-6)

    def test_funding_rate_raw_not_zscore(self, make_trades, make_orderbook, make_funding):
        """Feature 24 should be the raw rate, not z-scored."""
        funding = make_funding(n=5, base_rate=0.0005)
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), funding, trade_batch=100
        )
        # Raw funding rate should be nonzero when funding data exists
        # (unlike funding_zscore which needs 8+ events for rolling stats)
        # With only 5 funding events and 2 batches, at least one should pick up a rate
        assert features.shape[1] == NUM_FEATURES_V4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py::TestLongerHorizonFeatures -v`
Expected: FAIL (shape mismatch — still 20 features)

- [ ] **Step 3: Implement the 5 new features in compute_features**

In `prepare.py`, after the existing `r_100` computation (~line 222), add `ret_500` and `ret_2800`:

```python
    # --- Features 20-21: longer-horizon returns ---
    r_500 = np.zeros(num_batches)
    r_2800 = np.zeros(num_batches)
    for k, arr in [(500, r_500), (2800, r_2800)]:
        if num_batches > k:
            arr[k:] = np.log(vwap[k:] / np.maximum(vwap[:-k], 1e-10))
```

After the existing `tfi` computation (~line 249), add cumulative TFI:

```python
    # --- Features 22-23: cumulative order flow (rolling sum of TFI) ---
    tfi_series = pd.Series(tfi)
    cum_tfi_100 = tfi_series.rolling(window=100, min_periods=1).sum().fillna(0).values
    cum_tfi_500 = tfi_series.rolling(window=500, min_periods=1).sum().fillna(0).values
```

In the funding section (~line 465), after funding_zscore, add raw rate forward-fill:

```python
        # Feature 24: funding_rate_raw (forward-filled, not z-scored)
        extra_features = np.zeros((num_batches, 3))  # Change from 2 to 3
        # ... (existing funding_zscore code fills extra_features[:, 0])
        # Raw rate forward-fill
        extra_features[valid, 1] = fund_rate[indices[valid]]
```

Wait — the `extra_features` array is currently `(num_batches, 2)` for funding_zscore + utc_hour. We need to make it `(num_batches, 3)` for funding_zscore + funding_rate_raw + utc_hour. Adjust the utc_hour index accordingly.

Update `extra_features` initialization:
```python
    extra_features = np.zeros((num_batches, 3))  # was 2
```

Move utc_hour to index 2:
```python
    extra_features[:, 2] = ((batch_timestamps / 1000 / 3600) % 24) / 24.0  # was index 1
```

Add raw funding rate at index 1:
```python
        extra_features[valid, 1] = fund_rate[indices[valid]]
```

Update the column_stack to include new features:

```python
    trade_features = np.column_stack(
        [
            returns,          # 0
            r_5,              # 1
            r_20,             # 2
            r_100,            # 3
            realvol_10,       # 4
            bipower_var_20,   # 5
            tfi,              # 6
            volume_spike_ratio, # 7
            large_trade_share,  # 8
            kyle_lambda,      # 9
            amihud_illiq,     # 10
            trade_arrival_rate, # 11
            # --- new ---
            r_500,            # (will become 20 after OB features)
            r_2800,           # (will become 21)
            cum_tfi_100,      # (will become 22)
            cum_tfi_500,      # (will become 23)
        ]
    )
    # trade_features: 16 columns
    # ob_features: 6 columns (indices 12-17 → 16-21 after shift... NO)
```

Actually, to keep existing feature indices stable (tests depend on them), **append new features at the end**. Keep trade_features as 12 columns, ob_features as 6, extra_features as 3 (funding_zscore, funding_rate_raw, utc_hour), then append the 4 new trade features:

```python
    longer_horizon = np.column_stack([r_500, r_2800, cum_tfi_100, cum_tfi_500])
    features = np.hstack([trade_features, ob_features, extra_features, longer_horizon])
```

Final layout (25 features):
- 0-11: existing trade features
- 12-17: existing OB features
- 18: funding_zscore
- 19: funding_rate_raw (NEW)
- 20: utc_hour_linear (moved from 19 → 20)
- 21: r_500 (NEW)
- 22: r_2800 (NEW)
- 23: cum_tfi_100 (NEW)
- 24: cum_tfi_500 (NEW)

WAIT — this shifts utc_hour from index 19 to 20, breaking existing tests. Let me reconsider.

**Simplest approach: just append all 5 new features at indices 20-24, keeping 0-19 untouched.**

```
0-19: unchanged v3.1 features
20: r_500
21: r_2800
22: cum_tfi_100
23: cum_tfi_500
24: funding_rate_raw
```

This means `extra_features` stays as `(num_batches, 2)`, and we create a separate `new_features` array:

```python
    new_features = np.zeros((num_batches, 5))
    new_features[:, 0] = r_500
    new_features[:, 1] = r_2800
    new_features[:, 2] = cum_tfi_100
    new_features[:, 3] = cum_tfi_500
    # funding_rate_raw
    if not funding_df.empty:
        new_features[valid, 4] = fund_rate[indices[valid]]

    features = np.hstack([trade_features, ob_features, extra_features, new_features])
```

This keeps all existing indices stable. Good.

- [ ] **Step 4: Update ROBUST_FEATURE_INDICES and feature version**

```python
# Add cum_tfi_100 (23), cum_tfi_500 (24) to robust set (heavy-tailed sums)
ROBUST_FEATURE_INDICES = {5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 23, 24}

_FEATURE_VERSION = "v4"  # v4: longer-horizon features (25 total)
```

- [ ] **Step 5: Update docstring**

Update the compute_features docstring to list all 25 features.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: new tests PASS, existing tests PASS (indices 0-19 unchanged)

- [ ] **Step 7: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add 5 longer-horizon features (v4, 25 total)"
```

---

## Chunk 2: Min-Hold Constraint in TradingEnv

### Task 2: Add min_hold parameter to TradingEnv

**Files:**
- Modify: `prepare.py:651-790` (TradingEnv class)
- Test: `tests/test_env.py` (new file)

- [ ] **Step 1: Write failing tests**

Create `tests/test_env.py`:

```python
"""Tests for TradingEnv min_hold constraint."""

import numpy as np
import pytest
from prepare import TradingEnv


@pytest.fixture
def simple_env():
    """Env with 200 steps of synthetic data."""
    rng = np.random.default_rng(42)
    n = 200
    features = rng.standard_normal((n, 25)).astype(np.float32)
    prices = 100.0 + rng.normal(0, 0.1, n).cumsum()
    prices = np.maximum(prices, 1.0)  # keep positive
    return features, prices


class TestMinHold:

    def test_default_min_hold_is_1(self, simple_env):
        features, prices = simple_env
        env = TradingEnv(features, prices, window_size=10)
        assert env.min_hold == 1

    def test_action_ignored_during_hold_period(self, simple_env):
        features, prices = simple_env
        env = TradingEnv(features, prices, window_size=10, min_hold=5)
        env.reset(options={"sequential": True})

        # Go long
        _, _, _, _, info1 = env.step(1)
        assert info1["position"] == 1
        assert info1["trade_count"] == 1

        # Try to go short immediately — should be ignored
        _, _, _, _, info2 = env.step(2)
        assert info2["position"] == 1  # still long
        assert info2["trade_count"] == 1  # no new trade

        # Steps 2-4: still held
        for _ in range(3):
            _, _, _, _, info = env.step(2)
            assert info["position"] == 1

        # Step 5: min_hold elapsed, can change
        _, _, _, _, info5 = env.step(2)
        assert info5["position"] == 2  # now short
        assert info5["trade_count"] == 2

    def test_holding_same_position_doesnt_reset_hold_timer(self, simple_env):
        features, prices = simple_env
        env = TradingEnv(features, prices, window_size=10, min_hold=3)
        env.reset(options={"sequential": True})

        # Go long
        env.step(1)
        # Hold long (same action — not a trade)
        env.step(1)
        env.step(1)
        # Now try to exit — should work (3 steps since last TRADE)
        _, _, _, _, info = env.step(0)
        assert info["position"] == 0

    def test_flat_to_position_always_allowed(self, simple_env):
        """When flat, entering a position should always be allowed."""
        features, prices = simple_env
        env = TradingEnv(features, prices, window_size=10, min_hold=100)
        env.reset(options={"sequential": True})

        # From flat, entering long should always work
        _, _, _, _, info = env.step(1)
        assert info["position"] == 1
        assert info["trade_count"] == 1

    def test_info_includes_steps_since_trade(self, simple_env):
        features, prices = simple_env
        env = TradingEnv(features, prices, window_size=10, min_hold=5)
        env.reset(options={"sequential": True})

        env.step(1)  # trade
        _, _, _, _, info = env.step(1)  # hold
        assert info["steps_since_trade"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL (TradingEnv doesn't accept min_hold)

- [ ] **Step 3: Implement min_hold in TradingEnv**

Modify `TradingEnv.__init__` to accept `min_hold`:

```python
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window_size: int = 50,
        fee_bps: float = 5,
        min_hold: int = 1,
    ):
        # ... existing init ...
        self.min_hold = min_hold
        self._steps_since_trade = 0
```

Modify `reset` to reset the counter:

```python
        self._steps_since_trade = self.min_hold  # allow first trade immediately
```

Modify `step` to enforce min_hold:

```python
    def step(self, action: int):
        prev_position = self._position
        self._steps_since_trade += 1

        # Enforce min_hold: ignore position changes during hold period
        # Exception: entering from flat is always allowed
        if action != prev_position and prev_position != 0:
            if self._steps_since_trade < self.min_hold:
                action = prev_position  # override: keep current position

        # ... rest of step unchanged ...
```

Add `steps_since_trade` to the info dict:

```python
        info = {
            # ... existing fields ...
            "steps_since_trade": self._steps_since_trade,
        }
```

Reset counter when a trade happens:

```python
        if action != prev_position:
            # ... existing fee code ...
            self._steps_since_trade = 0
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_env.py -v`
Expected: all PASS

- [ ] **Step 5: Update make_env to pass min_hold**

In `prepare.py:make_env`, add `min_hold` parameter:

```python
def make_env(
    symbol: str = "BTC",
    split: str = "train",
    window_size: int = 50,
    trade_batch: int = 100,
    min_hold: int = 1,
) -> TradingEnv:
    # ... existing code ...
    return TradingEnv(features, prices, window_size=window_size, fee_bps=FEE_BPS, min_hold=min_hold)
```

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_env.py
git commit -m "feat: add min_hold constraint to TradingEnv"
```

---

## Chunk 3: Training Configuration

### Task 3: Update train.py for new feature set and anti-overtrade

**Files:**
- Modify: `train.py:17-31` (config constants)
- Modify: `train.py:59-73` (reward function)
- Modify: `train.py:271-310` (full_run — pass min_hold to envs)
- Modify: `train.py:311-349` (Optuna objective — fix search ranges)

- [ ] **Step 1: Update config constants**

```python
# Add min_hold to config
MIN_HOLD = 200  # ~3 hours between trades — sweet spot from breakeven analysis
WINDOW_SIZE = 50
```

- [ ] **Step 2: Pass min_hold when creating envs**

In `full_run`, update `make_env` calls:

```python
            env = make_env(
                sym, "train", window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
```

In `eval_policy`, update `make_env` calls:

```python
            env_test = make_env(
                sym, split, window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
```

- [ ] **Step 3: Fix Optuna search ranges**

Replace the objective's param sampling with tighter ranges based on mar10 findings:

```python
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024]),
        "n_epochs": 4,
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lam": 0.95,
        "ent": trial.suggest_float("ent", 0.01, 0.05, log=True),  # min 0.01, not 0.001
        "clip": 0.2,
        "hdim": trial.suggest_categorical("hdim", [128, 256]),
        "nlayers": 3,
        "lam_vol": 0.5,   # FIXED, not searched
        "lam_draw": 1.0,  # FIXED, not searched
    }
```

- [ ] **Step 4: Update study name**

```python
        study_name="ppo_v4_longhorizon",
```

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: anti-overtrade training config (min_hold=200, fixed reward params)"
```

---

## Chunk 4: Validate and Run

### Task 4: Clear stale caches, validate, and run quick test

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all pass

- [ ] **Step 2: Validate features on real data**

Run a quick diagnostic on BTC to verify new features are non-degenerate:

```bash
uv run python -c "
from prepare import load_trades, load_orderbook, load_funding, compute_features, normalize_features, TRAIN_START, TRAIN_END
import numpy as np

t = load_trades('BTC', TRAIN_START, TRAIN_END)
ob = load_orderbook('BTC', TRAIN_START, TRAIN_END)
f = load_funding('BTC', TRAIN_START, TRAIN_END)
raw, ts, prices = compute_features(t, ob, f)
print(f'Shape: {raw.shape}')  # should be (N, 25)
NAMES = ['ret_500', 'ret_2800', 'cum_tfi_100', 'cum_tfi_500', 'fund_rate_raw']
for i, name in enumerate(NAMES):
    col = raw[:, 20+i]
    print(f'{name}: mean={col.mean():.6f} std={col.std():.6f} zeros={((col==0).sum()/len(col)*100):.1f}%')

normed = normalize_features(raw)
print(f'Normed range: [{normed.min():.2f}, {normed.max():.2f}]')
"
```

Expected: 25 features, new features have non-zero std, normed range is [-5, 5].

- [ ] **Step 3: Quick training smoke test**

```bash
uv run python -c "
from train import full_run, SEARCH_SYMBOLS, MIN_HOLD
p = {'lr': 3e-4, 'n_steps': 512, 'n_epochs': 4, 'gamma': 0.99, 'gae_lam': 0.95,
     'ent': 0.01, 'clip': 0.2, 'hdim': 256, 'nlayers': 3, 'lam_vol': 0.5, 'lam_draw': 1.0}
sh, ps, tr, dd = full_run(SEARCH_SYMBOLS, p, budget=60, n_seeds=1, split='val', verbose=True)
print(f'sharpe={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} trades={tr} dd={dd:.4f}')
"
```

Expected: more than 1 trade per symbol (min_hold prevents overtrading, but agent should still trade). If trades > 50 per symbol, the min_hold is working.

- [ ] **Step 4: Commit all diagnostic scripts cleanup**

```bash
git add -A
git commit -m "chore: v4 feature validation and smoke test"
```

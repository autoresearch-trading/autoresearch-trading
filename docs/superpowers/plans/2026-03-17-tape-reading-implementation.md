# Tape Reading Pivot — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 tape reading features, shorten labeling horizon to 2.5 min, add trade-level eval metrics, and run locally on CPU.

**Architecture:** Flat MLP (v5 DirectionClassifier) with 39 features (31 existing + 8 new tape reading), forward_horizon=150, min_hold=100. Setup detection: flat by default, long/short only on high-conviction setups.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

**Spec:** `docs/superpowers/specs/2026-03-17-tape-reading-pivot-design.md`

---

## Chunk 1: New Features in prepare.py

### Task 1: Validate class distribution at horizon=150

Before implementing anything, check that the shorter horizon produces a reasonable class distribution. If flat > 80%, we need to lower fee_mult.

**Files:**
- None modified — exploratory script only

- [ ] **Step 1: Run empirical check on BTC cached data**

```bash
uv run python -c "
from prepare import make_env
import numpy as np

for sym in ['BTC', 'ETH', 'SOL', 'LTC', 'UNI']:
    env = make_env(sym, 'train', window_size=50, trade_batch=100, min_hold=100)
    prices = env.prices
    n = len(prices)
    horizon = 150
    fee_threshold = 2 * 5 / 10000 * 1.5  # 15 bps

    valid = np.arange(50, n - horizon)
    mask = prices[valid] > 0
    valid = valid[mask]
    fwd = (prices[valid + horizon] - prices[valid]) / prices[valid]
    n_long = (fwd > fee_threshold).sum()
    n_short = (fwd < -fee_threshold).sum()
    n_flat = len(fwd) - n_long - n_short
    total = len(fwd)
    print(f'{sym:6s}: flat={100*n_flat/total:.1f}%  long={100*n_long/total:.1f}%  short={100*n_short/total:.1f}%  (n={total})')
"
```

Expected: flat ~40-60% for volatile symbols, ~60-80% for quiet ones. If flat > 80% across the board, lower fee_mult to 1.0 in subsequent steps.

- [ ] **Step 2: Record results and decide fee_mult**

Note the results. If flat > 80% for most symbols, use `fee_mult=1.0` instead of 1.5 throughout the rest of the plan.

---

### Task 2: Add 8 tape reading features to compute_features()

**Files:**
- Modify: `prepare.py` — insert before `# Combine all features` at line 547 (after ALL existing features are computed: trade 0-11, orderbook 12-17, funding 18-19, longer-horizon 20-24, cutting-edge 25-30)
- Modify: `prepare.py:594` (ROBUST_FEATURE_INDICES)
- Modify: `prepare.py:633` (_FEATURE_VERSION)

**Insertion point:** All 8 tape reading features go right before `# Combine all features` (line 547). This is critical — features 7-11 are computed at lines 315-376 and `LOOKBACK_BATCHES`/`flat_notionals` (needed by Step 2) are defined at lines 329-330. Inserting earlier causes NameError.

- [ ] **Step 1: Add features 31-32 (buy_run_max, sell_run_max)**

Add before `# Combine all features` at line 547. These count the longest consecutive buy/sell streak in each 100-trade batch.

```python
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
```

- [ ] **Step 2: Add features 33-34 (large_buy_share, large_sell_share)**

Uses same 50-batch rolling 95th percentile as existing feature 8 (large_trade_share). Feature 8 (line 326-342) computes the threshold inside a per-batch loop — we need to pre-compute it as a vectorized array first. `LOOKBACK_BATCHES` and `flat_notionals` are already defined by feature 8 — reuse them.

```python
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
large_buy_notional = np.where(is_buy_batched & is_large, notionals_batched, 0).sum(axis=1)
large_sell_notional = np.where(~is_buy_batched & is_large, notionals_batched, 0).sum(axis=1)
# total_batch_notional already exists (line 295, computed for TFI)
safe_total = np.maximum(total_batch_notional, 1e-10)
large_buy_share = large_buy_notional / safe_total
large_sell_share = large_sell_notional / safe_total
```

- [ ] **Step 3: Add feature 35 (trade_size_entropy)**

Shannon entropy of trade size distribution within each batch. Low = algorithmic (equal clips), high = mixed participants.

```python
# Feature 35: trade size entropy
notional_probs = notionals_batched / np.maximum(notionals_batched.sum(axis=1, keepdims=True), 1e-10)
# Clip to avoid log(0)
notional_probs = np.clip(notional_probs, 1e-10, 1.0)
trade_size_entropy = -np.sum(notional_probs * np.log(notional_probs), axis=1)
```

- [ ] **Step 4: Add feature 36 (aggressor_imbalance)**

Count-based imbalance (unlike TFI which is volume-weighted).

```python
# Feature 36: aggressor imbalance (count-based)
buy_count = is_buy_batched.sum(axis=1).astype(np.float32)
sell_count = trade_batch - buy_count
aggressor_imbalance = (buy_count - sell_count) / trade_batch
```

- [ ] **Step 5: Add feature 37 (price_level_absorption)**

High volume + small price change = someone absorbing.

```python
# Feature 37: price level absorption
price_change = np.abs(np.diff(vwap, prepend=vwap[0]))
price_change_safe = np.maximum(price_change, 1e-10)
# total_batch_notional already exists (line 295, computed for TFI)
price_level_absorption = total_batch_notional / price_change_safe
# First batch has price_change=0 (diff with itself), producing ~1e15 spike — zero it out
price_level_absorption[0] = 0.0
```

- [ ] **Step 6: Add feature 38 (tfi_acceleration)**

Second finite difference of TFI. Note: `tfi` and `delta_tfi` are already computed for features 6 and 26.

```python
# Feature 38: TFI acceleration (second difference)
# delta_tfi is already computed for feature 26
tfi_acceleration = np.diff(delta_tfi, prepend=0.0)
```

- [ ] **Step 7: Include new features in the feature array**

At `prepare.py:565-585`. Add a new `np.column_stack` for tape reading features, then include it in the `np.hstack` at line 577.

```python
# === TAPE READING FEATURES (indices 31-38) ===
tape_reading_features = np.column_stack(
    [
        buy_run_max,            # 31
        sell_run_max,           # 32
        large_buy_share,        # 33
        large_sell_share,       # 34
        trade_size_entropy,     # 35
        aggressor_imbalance,    # 36
        price_level_absorption, # 37
        tfi_acceleration,       # 38
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
```

This replaces the existing `np.hstack` at line 577-585.

- [ ] **Step 8: Update ROBUST_FEATURE_INDICES**

At `prepare.py:594`:

```python
ROBUST_FEATURE_INDICES = {5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 24, 25, 29, 33, 34, 35, 37}
```

Added: `{33, 34, 35, 37}` (large_buy_share, large_sell_share, entropy, absorption).

- [ ] **Step 9: Bump _FEATURE_VERSION**

At `prepare.py:633`:

```python
_FEATURE_VERSION = "v6"  # v6: 39 features (v5 + 8 tape reading features)
```

- [ ] **Step 10: Commit**

```bash
git add prepare.py
git commit -m "feat: add 8 tape reading features (v6, 39 total)"
```

---

### Task 3: Update tests for 39 features

**Files:**
- Modify: `tests/test_features.py`

- [ ] **Step 1: Update NUM_FEATURES constant**

At `tests/test_features.py:10`:

```python
NUM_FEATURES = 39
```

- [ ] **Step 2: Add tests for new tape reading features**

Add a new test class after existing tests:

```python
class TestTapeReadingFeatures:
    """Tests for v6 tape reading features (indices 31-38)."""

    def test_buy_sell_run_max_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Run lengths bounded by batch size (100)
        assert np.all(features[:, 31] >= 0) and np.all(features[:, 31] <= 100)
        assert np.all(features[:, 32] >= 0) and np.all(features[:, 32] <= 100)

    def test_large_buy_sell_share_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Share ratios in [0, 1]
        assert np.all(features[:, 33] >= 0) and np.all(features[:, 33] <= 1)
        assert np.all(features[:, 34] >= 0) and np.all(features[:, 34] <= 1)

    def test_entropy_non_negative(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert np.all(features[:, 35] >= 0)

    def test_aggressor_imbalance_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Bounded [-1, 1]
        assert np.all(features[:, 36] >= -1) and np.all(features[:, 36] <= 1)

    def test_absorption_non_negative(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert np.all(features[:, 37] >= 0)

    def test_tfi_acceleration_finite(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Second difference of TFI — symmetric, bounded roughly by [-2, 2]
        assert np.all(np.isfinite(features[:, 38]))
        assert np.all(features[:, 38] >= -4) and np.all(features[:, 38] <= 4)
```

- [ ] **Step 3: Run all tests**

```bash
uv run pytest tests/test_features.py -v
```

Expected: All tests pass, including new ones and existing ones with updated feature count.

- [ ] **Step 4: Commit**

```bash
git add tests/test_features.py
git commit -m "test: update feature count to 39, add tape reading feature tests"
```

---

## Chunk 2: Evaluation Metrics + Train Config

### Task 4: Add trade-level metrics to evaluate()

**Files:**
- Modify: `prepare.py:912-972` (evaluate function)

- [ ] **Step 1: Add trade tracking to evaluate()**

Inside `evaluate()`, add accumulators for per-trade PnL and hold duration. Track when position changes (entry/exit).

The env's `step()` returns `info` dict with `"position"` (0=flat, 1=long, 2=short) and `"equity"` (cumulative equity). Use these — do NOT access `env._position` directly (it's private).

After the existing loop variables are initialized (line 924-926), add:

```python
trade_pnls = []
hold_durations = []
entry_step = None
entry_equity = 1.0
prev_position = 0
step_num = 0
```

Inside the main step loop (line 929-937), after the existing `step_returns.append(info["step_pnl"])`, add trade detection:

```python
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
elif prev_position != 0 and current_position != 0 and prev_position != current_position:
    # Flip (long→short or short→long): close + open
    trade_pnls.append(current_equity - entry_equity)
    if entry_step is not None:
        hold_durations.append(step_num - entry_step)
    entry_step = step_num
    entry_equity = current_equity
prev_position = current_position
```

After the main loop, before the guardrails section (before line 941), compute and print the new metrics:

```python
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
```

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add prepare.py
git commit -m "feat: add trade-level metrics to evaluate() (win_rate, profit_factor, avg_hold)"
```

---

### Task 5: Update train.py config for tape reading

**Files:**
- Modify: `train.py:19-39` (configuration section)

- [ ] **Step 1: Update horizon and min_hold constants**

```python
# ── Configuration ──────────────────────────────────────────────
SEARCH_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "CRV"]
SEARCH_BUDGET = 90
SEARCH_SEEDS = 2
SEARCH_TRIALS = 20
FINAL_SEEDS = 5
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = 50
TRADE_BATCH = 100
MIN_HOLD = 100       # ~1.5 min between trades (tape reading scale)
FEE_BPS = 5
FORWARD_HORIZON = 150  # ~2.5 min (setup plays out in minutes)

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 256,
    "nlayers": 2,
    "batch_size": 256,
    "fee_mult": 1.5,  # adjust based on Task 1 class distribution check
}
```

Only 2 lines changed: `MIN_HOLD` (800→100) and `FORWARD_HORIZON` (800→150).

- [ ] **Step 2: Fix sharpe→sortino naming throughout**

In `prepare.py` — the `evaluate()` function prints `val_sharpe:`. Change all three:

```python
# Line 943:
print(f"sortino: 0.000000 (only {total_trades} trades, min={min_trades})")
# Line 949:
print(f"sortino: 0.000000 (drawdown {max_dd:.4f} > {max_drawdown})")
# Line 968:
print(f"sortino: {sortino:.6f}")
```

In `train.py` — change all `sharpe` references to `sortino`:

```python
# Line 220 (eval_policy docstring):
"""Run policy_fn on all symbols. Returns (sortino, passing, trades, dd)."""
# Line 250 (per-symbol output):
print(f"  {sym}: sortino={sh:.4f} trades={t} dd={d:.4f} [{tag}]")
# Line 350 (Optuna trial output):
f"  => sortino={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
# Line 395 (Optuna top trials):
print(f"  #{t.number}: sortino={t.value:.4f}  {t.params}")
# Line 423 (PORTFOLIO SUMMARY):
print(f"sortino: {sh:.6f}")
```

- [ ] **Step 3: Commit**

```bash
git add train.py prepare.py
git commit -m "config: horizon=150, min_hold=100, fix sharpe→sortino naming"
```

---

### Task 6: Smoke test the full pipeline

- [ ] **Step 1: Run on single symbol to verify pipeline**

```bash
uv run python -c "
from prepare import make_env, evaluate
from train import WINDOW_SIZE, TRADE_BATCH, MIN_HOLD, FORWARD_HORIZON, FEE_BPS
import numpy as np

print(f'Config: window={WINDOW_SIZE}, horizon={FORWARD_HORIZON}, min_hold={MIN_HOLD}')

env = make_env('BTC', 'train', window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH, min_hold=MIN_HOLD)
print(f'Features shape: {env.features.shape}')
assert env.features.shape[1] == 39, f'Expected 39 features, got {env.features.shape[1]}'

# Check class distribution
prices = env.prices
n = len(prices)
from train import BEST_PARAMS
fee_threshold = (2 * FEE_BPS / 10000) * BEST_PARAMS['fee_mult']
valid = np.arange(WINDOW_SIZE, n - FORWARD_HORIZON)
mask = prices[valid] > 0
valid = valid[mask]
fwd = (prices[valid + FORWARD_HORIZON] - prices[valid]) / prices[valid]
n_flat = ((fwd >= -fee_threshold) & (fwd <= fee_threshold)).sum()
print(f'BTC class dist: flat={100*n_flat/len(fwd):.1f}% (n={len(fwd)})')

print('PIPELINE SMOKE TEST PASSED')
"
```

- [ ] **Step 2: Quick training run on 1 symbol**

```bash
uv run python -c "
from train import full_run, BEST_PARAMS, FINAL_BUDGET
so, ps, tr, dd, steps, updates = full_run(['BTC'], BEST_PARAMS, FINAL_BUDGET, 1, split='val', verbose=True)
print(f'sortino={so:.4f} passing={ps}/1 trades={tr} dd={dd:.4f}')
print('QUICK TRAIN TEST PASSED')
"
```

- [ ] **Step 3: Commit any fixes**

```bash
git add prepare.py train.py
git commit -m "fix: pipeline smoke test fixes"
```

---

## Chunk 3: Full Run and Results

### Task 7: Run full training and evaluation

- [ ] **Step 1: Run full pipeline**

```bash
uv run python train.py 2>&1 | tee run_tape_v1.log
```

Expected output: training 5 seeds on 25 symbols, eval with new trade-level metrics. ~10-15 min on CPU.

- [ ] **Step 2: Record results**

```bash
grep -E "^(sortino|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor|avg_hold)" run_tape_v1.log
```

- [ ] **Step 3: Compare to v5 baseline and commit**

```bash
SORTINO=$(grep '^sortino:' run_tape_v1.log | tail -1 | awk '{print $2}')
PASSING=$(grep '^symbols_passing:' run_tape_v1.log | awk '{print $2}')
TRADES=$(grep '^num_trades:' run_tape_v1.log | awk '{print $2}')
DRAWDOWN=$(grep '^max_drawdown:' run_tape_v1.log | awk '{print $2}')
echo -e "$(git rev-parse --short HEAD)\t$SORTINO\t$TRADES\t$DRAWDOWN\t$PASSING\tkept/discarded\ttape-v1: 39 features, horizon=150, min_hold=100" >> results.tsv
git add results.tsv
git commit -m "experiment: tape reading v1 (sortino=$SORTINO, passing=$PASSING)"
```

- [ ] **Step 4: If flat-dominant (< 10 trades most symbols), retry with fee_mult=1.0**

Update `BEST_PARAMS["fee_mult"]` to 1.0 in train.py, re-run, compare.

---

## Experiment Order

1. **Task 1**: Validate class distribution (5 min, no code changes)
2. **Task 2**: Add 8 tape reading features to prepare.py
3. **Task 3**: Update tests
4. **Task 4**: Add trade-level eval metrics
5. **Task 5**: Update train.py config
6. **Task 6**: Smoke test pipeline
7. **Task 7**: Full run and results

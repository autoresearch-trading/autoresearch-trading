# Tape Reading Pivot — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 tape reading features, replace fixed-horizon labeling with Triple Barrier (TP/SL/timeout), add trade-level eval metrics, and run locally on CPU.

**Architecture:** Flat MLP (v5 DirectionClassifier) with 39 features (31 existing + 8 new tape reading), Triple Barrier labeling (max_hold=300, tp=sl=15 bps), min_hold=100. Setup detection: flat by default, long/short only on high-conviction setups.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

**Spec:** `docs/superpowers/specs/2026-03-17-tape-reading-pivot-design.md`

---

## Chunk 1: New Features in prepare.py

### Task 1: Validate Triple Barrier class distribution

Before implementing anything, check that Triple Barrier labeling with max_hold=300 and fee_threshold=15 bps produces a reasonable class distribution. If flat > 80%, we need to lower fee_mult or increase max_hold.

**Files:**
- None modified — exploratory script only

- [ ] **Step 1: Run Triple Barrier distribution check**

```bash
uv run python -c "
from prepare import make_env
import numpy as np

max_hold = 300
fee_threshold = 2 * 5 / 10000 * 1.5  # 15 bps (symmetric TP/SL)

for sym in ['BTC', 'ETH', 'SOL', 'LTC', 'UNI']:
    env = make_env(sym, 'train', window_size=50, trade_batch=100, min_hold=100)
    prices = env.prices
    n = len(prices)

    valid = np.arange(50, n - max_hold)
    valid = valid[prices[valid] > 0]
    # Subsample to cap memory: (50K, 300) float64 ≈ 120 MB per symbol
    if len(valid) > 50000:
        valid = np.sort(np.random.default_rng(42).choice(valid, 50000, replace=False))

    # Vectorized Triple Barrier
    offsets = np.arange(1, max_hold + 1)
    future_prices = prices[valid[:, np.newaxis] + offsets[np.newaxis, :]]
    entry_prices = prices[valid]
    fwd = (future_prices - entry_prices[:, np.newaxis]) / entry_prices[:, np.newaxis]

    hit_tp = fwd >= fee_threshold
    hit_sl = fwd <= -fee_threshold
    tp_first = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), max_hold)
    sl_first = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), max_hold)

    n_long = (tp_first < sl_first).sum()
    n_short = (sl_first < tp_first).sum()
    n_flat = len(valid) - n_long - n_short
    total = len(valid)
    # Median steps to first barrier hit (excluding timeouts)
    barrier_steps = np.minimum(tp_first, sl_first)
    hit_mask = barrier_steps < max_hold
    med_steps = int(np.median(barrier_steps[hit_mask])) if hit_mask.any() else -1
    print(f'{sym:6s}: flat={100*n_flat/total:.1f}%  long={100*n_long/total:.1f}%  short={100*n_short/total:.1f}%  med_hit={med_steps} steps  (n={total})')
"
```

Expected: flat ~30-50% for volatile symbols (BTC, ETH, SOL), ~50-70% for quiet ones. The `med_hit` shows typical setup duration. If flat > 80% across the board, lower fee_mult to 1.0 or increase max_hold to 500.

- [ ] **Step 2: Record results and decide parameters**

Note the results. Key decisions:
- If flat > 80% for most symbols → lower `fee_mult` to 1.0 or 1.2
- If med_hit > 200 for most symbols → increase `max_hold` to 500
- If flat < 20% for most symbols → raise `fee_mult` to 2.0 (barriers too easy to hit)

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
# Zero out batches with negligible price change (batch 0 always, plus any identical-VWAP batches)
# These produce ~1e14+ spikes from notional/1e-10; common for illiquid symbols
tiny_move = price_change < 1e-8
price_level_absorption[tiny_move] = 0.0
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

After the main loop ends, close any position still open at end of data, then compute and print the new metrics (before the guardrails section, before line 941):

```python
# Close any open position at end of data
if prev_position != 0 and entry_step is not None:
    trade_pnls.append(info["equity"] - entry_equity)
    hold_durations.append(step_num - entry_step)
```

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

### Task 5: Replace labeling + update train.py config + forward trade metrics

**Files:**
- Modify: `train.py:19-39` (configuration section)
- Modify: `train.py:72-124` (replace `make_labeled_dataset` with Triple Barrier)
- Modify: `train.py:141` (update caller to pass tp/sl thresholds)
- Modify: `train.py:241-250` (eval_policy — parse and forward trade metrics)
- Modify: `train.py:420-428` (main — aggregate trade metrics in PORTFOLIO SUMMARY)

- [ ] **Step 0: Replace make_labeled_dataset with Triple Barrier**

Replace the entire `make_labeled_dataset` function (train.py:72-124) with the vectorized Triple Barrier implementation from the spec:

```python
# ── Data labeling ──────────────────────────────────────────────
def make_labeled_dataset(env, max_hold, tp_threshold, sl_threshold, max_samples=10000):
    """Extract (obs, label) pairs using Triple Barrier labeling.

    For each sample, scan forward up to max_hold steps:
    - TP barrier hit first (return >= +tp_threshold) → long (1)
    - SL barrier hit first (return <= -sl_threshold) → short (2)
    - Neither hit within max_hold → flat (0) (timeout)

    Vectorized via numpy broadcasting. Memory: O(n_samples × max_hold).
    """
    features = env.features
    prices = env.prices
    n = len(features)
    window = env.window_size

    valid_start = window
    valid_end = n - max_hold
    if valid_end <= valid_start:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    all_idx = np.arange(valid_start, valid_end)
    if len(all_idx) > max_samples:
        idx = np.random.choice(all_idx, max_samples, replace=False)
        idx.sort()
    else:
        idx = all_idx

    # Filter zero prices
    idx = idx[prices[idx] > 0]
    if len(idx) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Forward return matrix: (n_samples, max_hold)
    offsets = np.arange(1, max_hold + 1)
    future_prices = prices[idx[:, np.newaxis] + offsets[np.newaxis, :]]
    entry_prices = prices[idx]
    fwd_returns = (future_prices - entry_prices[:, np.newaxis]) / entry_prices[:, np.newaxis]

    # First barrier hit per sample
    hit_tp = fwd_returns >= tp_threshold
    hit_sl = fwd_returns <= -sl_threshold
    tp_any = hit_tp.any(axis=1)
    sl_any = hit_sl.any(axis=1)
    tp_first = np.where(tp_any, hit_tp.argmax(axis=1), max_hold)
    sl_first = np.where(sl_any, hit_sl.argmax(axis=1), max_hold)

    # Label by which barrier hit first
    labels = np.zeros(len(idx), dtype=np.int64)   # 0 = flat (timeout)
    labels[tp_first < sl_first] = 1                # long: TP hit first
    labels[sl_first < tp_first] = 2                # short: SL hit first
    # tp_first == sl_first: both never hit (flat) or both hit same step (ambiguous → flat)

    obs = np.array([features[i - window:i] for i in idx], dtype=np.float32)
    return obs, labels, idx
```

Then update the caller in `train_one_model` (train.py:141):

```python
# Old:
obs, labels, indices = make_labeled_dataset(env, FORWARD_HORIZON, fee_threshold)
# New:
obs, labels, indices = make_labeled_dataset(env, MAX_HOLD_STEPS, fee_threshold, fee_threshold)
```

- [ ] **Step 1: Update config constants**

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
MIN_HOLD = 100            # ~1.5 min between trades (tape reading scale)
FEE_BPS = 5
MAX_HOLD_STEPS = 300      # Triple Barrier timeout: ~5 min (setup window)

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 256,
    "nlayers": 2,
    "batch_size": 256,
    "fee_mult": 1.5,  # adjust based on Task 1 class distribution check
}
```

Changes: `MIN_HOLD` (800→100), `FORWARD_HORIZON` removed, `MAX_HOLD_STEPS = 300` added.

- [ ] **Step 2: Forward trade-level metrics through eval_policy**

`eval_policy()` (train.py:234-238) captures all of `evaluate()`'s stdout into a StringIO and only parses `num_trades:` and `max_drawdown:`. The new trade metrics from Task 4 are silently discarded. Fix by parsing `win_rate` and `profit_factor`, extending the return to a 6-tuple, and threading `wr`/`pf` through `full_run()` (8-tuple), `objective()`, and `main()`.

**Note:** `avg_profit_per_trade` and `avg_hold_steps` are intentionally print-only (visible in per-symbol stdout during verbose runs). Only `win_rate` and `profit_factor` are forwarded to the portfolio summary to keep the return tuple manageable.

In `eval_policy()`, add accumulators before the symbol loop (after line 223):

```python
all_win_rates = []
all_profit_factors = []
```

Expand the parsing loop at `train.py:241-246`:

```python
t, d = 0, 0.0
wr, pf = 0.0, 0.0
for ln in out.strip().split("\n"):
    if ln.startswith("num_trades:"):
        t = int(ln.split()[1])
    elif ln.startswith("max_drawdown:"):
        d = float(ln.split()[1])
    elif ln.startswith("win_rate:"):
        wr = float(ln.split()[1])
    elif ln.startswith("profit_factor:"):
        pf = float(ln.split()[1])
```

Replace the per-symbol output block (train.py:248-254) with the **complete** modified version below. This replaces the print, preserves the existing accumulation logic, and adds wr/pf tracking:

```python
            passed = (t >= 10 and d <= 0.20) if t > 0 else False
            tag = "PASS" if passed else "FAIL"
            extra = f" wr={wr:.2f} pf={pf:.2f}" if wr > 0 else ""
            print(f"  {sym}: sortino={sh:.4f} trades={t} dd={d:.4f}{extra} [{tag}]")
            if passed:
                passing.append(sh)
            if passed and wr > 0:
                all_win_rates.append(wr)
                all_profit_factors.append(pf)
            trades_all += t
            worst_dd = max(worst_dd, d)
```

Change `eval_policy()` return to include trade metrics (line 258):

```python
    mean_wr = float(np.mean(all_win_rates)) if all_win_rates else 0.0
    mean_pf = float(np.mean(all_profit_factors)) if all_profit_factors else 0.0

    return (
        float(np.mean(passing)) if passing else 0.0,
        len(passing),
        trades_all,
        worst_dd,
        mean_wr,
        mean_pf,
    )
```

Update `full_run()` to forward the new return values. At line 324:

```python
    sh, ps, tr, dd, wr, pf = eval_policy(ensemble_fn, symbols, split=split)
    return sh, ps, tr, dd, total_steps_all, total_updates_all, wr, pf
```

Update `objective()` to ignore the new values. At line 345:

```python
        sh, ps, tr, dd, _, _, _, _ = full_run(
```

Update `main()` to destructure and print trade metrics. At line 416:

```python
    sh, ps, tr, dd, total_steps, total_updates, wr, pf = full_run(
```

And add to the PORTFOLIO SUMMARY block (after `max_drawdown` line):

```python
    if wr > 0:
        print(f"win_rate: {wr:.4f}")
        print(f"profit_factor: {pf:.4f}")
```

- [ ] **Step 3: Fix sharpe→sortino naming throughout**

In `prepare.py` — the `evaluate()` function prints `val_sharpe:`. Change all three:

```python
# Line 943:
print(f"sortino: 0.000000 (only {total_trades} trades, min={min_trades})")
# Line 949:
print(f"sortino: 0.000000 (drawdown {max_dd:.4f} > {max_drawdown})")
# Line 968:
print(f"sortino: {sortino:.6f}")
```

In `train.py` — change remaining `sharpe` references to `sortino` (line 250 already handled by Step 2):

```python
# Line 220 (eval_policy docstring):
"""Run policy_fn on all symbols. Returns (sortino, passing, trades, dd, win_rate, profit_factor)."""
# Line 350 (Optuna trial output):
f"  => sortino={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
# Line 395 (Optuna top trials):
print(f"  #{t.number}: sortino={t.value:.4f}  {t.params}")
# Line 423 (PORTFOLIO SUMMARY):
print(f"sortino: {sh:.6f}")
```

- [ ] **Step 4: Commit**

```bash
git add train.py prepare.py
git commit -m "feat: triple barrier labeling, min_hold=100, forward trade metrics, fix naming"
```

---

### Task 6: Smoke test the full pipeline

- [ ] **Step 1: Run on single symbol to verify pipeline**

```bash
uv run python -c "
from prepare import make_env, evaluate
from train import WINDOW_SIZE, TRADE_BATCH, MIN_HOLD, MAX_HOLD_STEPS, FEE_BPS, make_labeled_dataset
import numpy as np

print(f'Config: window={WINDOW_SIZE}, max_hold={MAX_HOLD_STEPS}, min_hold={MIN_HOLD}')

env = make_env('BTC', 'train', window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH, min_hold=MIN_HOLD)
print(f'Features shape: {env.features.shape}')
assert env.features.shape[1] == 39, f'Expected 39 features, got {env.features.shape[1]}'

# Test Triple Barrier labeling
from train import BEST_PARAMS
fee_threshold = (2 * FEE_BPS / 10000) * BEST_PARAMS['fee_mult']
obs, labels, indices = make_labeled_dataset(env, MAX_HOLD_STEPS, fee_threshold, fee_threshold, max_samples=5000)
unique, counts = np.unique(labels, return_counts=True)
total = len(labels)
for cls, cnt in zip(unique, counts):
    tag = {0: 'flat', 1: 'long', 2: 'short'}[cls]
    print(f'  {tag}: {100*cnt/total:.1f}% ({cnt})')
print(f'BTC Triple Barrier: {total} samples, obs shape {obs.shape}')

print('PIPELINE SMOKE TEST PASSED')
"
```

- [ ] **Step 2: Quick training run on 1 symbol**

```bash
uv run python -c "
from train import full_run, BEST_PARAMS, FINAL_BUDGET
so, ps, tr, dd, steps, updates, wr, pf = full_run(['BTC'], BEST_PARAMS, FINAL_BUDGET, 1, split='val', verbose=True)
print(f'sortino={so:.4f} passing={ps}/1 trades={tr} dd={dd:.4f} wr={wr:.2f} pf={pf:.2f}')
print('QUICK TRAIN+EVAL TEST PASSED')
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
grep -E "^(sortino|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor)" run_tape_v1.log
```

- [ ] **Step 3: Compare to v5 baseline and commit**

```bash
SORTINO=$(grep '^sortino:' run_tape_v1.log | tail -1 | awk '{print $2}')
PASSING=$(grep '^symbols_passing:' run_tape_v1.log | awk '{print $2}')
TRADES=$(grep '^num_trades:' run_tape_v1.log | awk '{print $2}')
DRAWDOWN=$(grep '^max_drawdown:' run_tape_v1.log | awk '{print $2}')
echo -e "$(git rev-parse --short HEAD)\t$SORTINO\t$TRADES\t$DRAWDOWN\t$PASSING\tkept/discarded\ttape-v1: 39 features, triple barrier max_hold=300, min_hold=100" >> results.tsv
git add results.tsv
git commit -m "experiment: tape reading v1 (sortino=$SORTINO, passing=$PASSING)"
```

- [ ] **Step 4: If flat-dominant (< 10 trades most symbols), adjust barriers**

Options: lower `BEST_PARAMS["fee_mult"]` to 1.0 (narrower barriers), or increase `MAX_HOLD_STEPS` to 500 (longer timeout). Re-run, compare.

---

### Task 8: Update documentation

- [ ] **Step 1: Update CLAUDE.md**

Update the feature table (31→39), architecture section (flat_dim 1,612→2,028, features 31→39, ~0.5M→~0.6M params), and add new features 31-38 to the table.

Update the "Current Best" section comment to note this is v5 baseline, and the tape reading section to reflect completed state.

- [ ] **Step 2: Update program.md**

Reflect the new state: 39 features, Triple Barrier labeling (max_hold=300), min_hold=100, new trade-level metrics available. Update any references to v5 feature count or fixed-horizon labeling.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md program.md
git commit -m "docs: update feature count, architecture, and experiment state for v6"
```

---

## Experiment Order

1. **Task 1**: Validate class distribution (5 min, no code changes)
2. **Task 2**: Add 8 tape reading features to prepare.py
3. **Task 3**: Update tests
4. **Task 4**: Add trade-level eval metrics
5. **Task 5**: Replace labeling with Triple Barrier + update train.py config + forward trade metrics
6. **Task 6**: Smoke test pipeline (features + labeling + training)
7. **Task 7**: Full run and results
8. **Task 8**: Update documentation (CLAUDE.md, program.md)

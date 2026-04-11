# Tape Reading Pivot — Design Spec

## Goal

Pivot from long-horizon directional prediction (12h holds) to **setup detection** inspired by tape reading. The model stays flat by default and only enters when it detects a high-conviction setup from order flow patterns. Setups play out in 10-30 minutes.

## Motivation

- v5 flat MLP (window=50, horizon=800, min_hold=800) achieved Sortino=0.230, 18/25 passing
- v7 attention (window=2000, same horizon) overfit badly: Sortino=0.061, 11/25 passing
- The 800-step forward horizon (~12 hours) doesn't match tape reading signal, which lives at the minute scale
- Current features already capture tape reading signals (TFI, VPIN, kyle_lambda, etc.) but labeling asks the wrong question

## Architecture

**Model:** Flat MLP (v5 `DirectionClassifier` from git history)

Revert to the DirectionClassifier from commit `04cfa49` (v5 baseline on `autoresearch/v5-features` branch). The model flattens the window and appends per-feature mean and std summary stats:

- Window: 50 steps
- Features: 39 (31 existing + 8 new tape reading features)
- flat_dim = 50 × 39 + 2 × 39 = 2,028 (window flattened + temporal mean + temporal std)
- Hidden: 256, 2 layers, ReLU, orthogonal init
- Output: 3 classes (flat / long setup / short setup)
- ~0.6M parameters
- Device: CPU (local)

**Training:**
- 25 epochs, batch_size=256, AdamW, focal loss
- 10K samples per symbol, 5 seeds, ensemble
- Class weights + recency weighting (same as v5)

## New Features (8)

Added to `compute_features()` in prepare.py. All computed within existing batch=100 framework.

| Index | Feature | Formula | Description |
|-------|---------|---------|-------------|
| 31 | buy_run_max | Max consecutive count where `side == buy` in the batch | Longest sustained buying streak |
| 32 | sell_run_max | Max consecutive count where `side == sell` in the batch | Longest sustained selling streak |
| 33 | large_buy_share | `sum(notional[is_buy & is_large]) / sum(notional)` where `is_large = notional > rolling_p95` (same 50-batch rolling lookback as feature 8) | Large buy volume as share of total |
| 34 | large_sell_share | `sum(notional[is_sell & is_large]) / sum(notional)` (same rolling lookback) | Large sell volume as share of total |
| 35 | trade_size_entropy | `-sum(p * log(p))` where `p = notional_i / sum(notional)` over the 100 trades in the batch | Low = algo (equal clips), high = mixed participants |
| 36 | aggressor_imbalance | `(buy_count - sell_count) / total_count` | Count-based imbalance (vs TFI which is volume-weighted) |
| 37 | price_level_absorption | `sum(notional) / max(abs(vwap[t] - vwap[t-1]), 1e-10)` | High volume + small price change = absorption. Division-by-zero guarded. Default 0.0 when no price data. |
| 38 | tfi_acceleration | `delta_tfi[t] - delta_tfi[t-1]` (where `delta_tfi = tfi[t] - tfi[t-1]`, i.e. second finite difference of TFI) | Is buy/sell pressure accelerating or decelerating? |

**Data requirements:** All use existing `side`, `qty`, `price` fields from trade data. Feature 37 uses batch VWAP (already computed). No new data sources.

**Normalization:** Features 33, 34 (share ratios with outlier potential), 35 (entropy), 37 (absorption — unbounded ratio) use robust IQR scaling. Features 31, 32 (run lengths — bounded 0-100), 36 (imbalance — bounded [-1,1]), 38 (acceleration — symmetric) use z-score. Updated `ROBUST_FEATURE_INDICES` adds `{33, 34, 35, 37}`.

## Labeling Changes — Triple Barrier

The fixed-horizon label asks "where is price in exactly N steps?" A tape reader asks "does price hit my target before my stop?" Triple Barrier labeling captures this by racing three exit conditions against each other.

| Parameter | v5 (old) | tape-v1 (new) | Rationale |
|-----------|----------|---------------|-----------|
| labeling | fixed-horizon forward return | Triple Barrier (TP/SL/timeout) | Matches tape reader's actual decision process |
| max_hold | N/A (forward_horizon=800) | 300 (~5 min BTC) | Timeout — setup didn't play out, label flat |
| tp_threshold | N/A | fee_threshold (15 bps) | Take-profit barrier, must exceed round-trip fees |
| sl_threshold | N/A | fee_threshold (15 bps) | Stop-loss barrier, symmetric with TP for v1 |
| min_hold | 800 (~12h) | 100 (~1.5 min) | Env constraint: min steps between position changes |
| fee_mult | 1.5 | 1.5 | Controls barrier width: `(2 * FEE_BPS / 10000) * fee_mult` |
| window_size | 50 | 50 | Keep same observation window |

### Label logic (Triple Barrier)

For each sample at step `i`, scan forward through `prices[i+1] ... prices[i+max_hold]`:

1. Compute running return: `r[k] = (prices[i+k] - prices[i]) / prices[i]` for `k = 1..max_hold`
2. Find first step where `r[k] >= +tp_threshold` (TP hit)
3. Find first step where `r[k] <= -sl_threshold` (SL hit)
4. Label by which barrier is hit first:
   - TP hit first → **long (1)** — price reached take-profit before stop-loss
   - SL hit first → **short (2)** — price reached stop-loss before take-profit
   - Neither hit within max_hold → **flat (0)** — setup timed out, no clear move

**Why this is better than fixed-horizon:**

| Scenario | Fixed-horizon (150 steps) | Triple Barrier (300 steps) |
|----------|--------------------------|---------------------------|
| Price spikes +20 bps at step 60, reverts to 0 by step 150 | **flat** (return ≈ 0 at step 150) | **long** (TP hit at step 60) |
| Price drifts slowly to +15 bps over 250 steps | **flat** (only +8 bps at step 150) | **long** (TP hit at step 250) |
| Price oscillates ±5 bps for 300 steps | flat | **flat** (timeout, correct) |
| Price drops -20 bps at step 30 | **short** (also short at step 150) | **short** (SL hit at step 30) |

The first two rows are the critical improvement: fixed-horizon misses moves that happen at the "wrong" time or take longer to develop.

### Parameters

**`max_hold = 300`** (~5 min for BTC, ~15 min for quiet symbols): Long enough for tape reading setups to play out. The spec says setups resolve in 10-30 minutes; 300 steps is at the lower end for BTC, which is correct since BTC is the most liquid and moves resolve fastest.

**`tp_threshold = sl_threshold = fee_threshold`** (symmetric, 15 bps): Symmetric barriers for v1. Asymmetric tuning (e.g., TP=20 bps, SL=10 bps for 2:1 reward:risk) is a natural Optuna search dimension for later. The fee_mult parameter already controls barrier width.

**Fee threshold:** `2 * 5/10000 * 1.5 = 0.0015` (15 bps). With Triple Barrier, a 15 bps move within 300 steps is more common than a 15 bps move at exactly step 150. Expect fewer flat labels than fixed-horizon at the same threshold. **Validate empirically before implementation.**

**Expected class distribution:** ~30-50% flat for volatile symbols (BTC, ETH, SOL), ~50-70% for quiet ones. More balanced than fixed-horizon because the barrier can be hit at any point in the 300-step window, not just at one specific step.

### Implementation (vectorized)

Replaces `make_labeled_dataset` in `train.py`. Uses numpy broadcasting — no Python loop over the forward path:

```python
def make_labeled_dataset(env, max_hold, tp_threshold, sl_threshold, max_samples=10000):
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

**Memory:** O(n_samples × max_hold) = 10K × 300 = 3M floats ≈ 24 MB. Fine for CPU.

**Caller change** in `train_one_model`:
```python
# Old: make_labeled_dataset(env, FORWARD_HORIZON, fee_threshold)
# New: make_labeled_dataset(env, MAX_HOLD_STEPS, fee_threshold, fee_threshold)
```

## Evaluation

**Existing metrics (unchanged):**
- Sortino ratio (annualized, downside vol only)
- symbols_passing: count passing min_trades ≥ 10 AND max_drawdown ≤ 20%
- Portfolio Sortino: mean across passing symbols

**New trade-level metrics (added to evaluate() output):**
- `win_rate`: % of trades with positive PnL
- `avg_profit_per_trade`: mean PnL per trade (after fees)
- `profit_factor`: sum(winning PnL) / abs(sum(losing PnL))
- `avg_hold_steps`: mean steps between position entry and exit

These are informational only — no change to pass/fail logic. Implementation requires a small accumulator in `evaluate()` to track per-trade PnL and hold durations (entry step, exit step, PnL per trade).

## Cache Invalidation

- Bump `_FEATURE_VERSION`: `"v5"` → `"v6"`
- Existing v5 caches remain on disk (keyed by version), new caches computed on first run
- ~36GB raw data → ~240 new cache files, computed lazily per symbol/date range

## File Changes

| File | Change |
|------|--------|
| prepare.py | Add 8 features to `compute_features()`, bump feature version to `"v6"`, update `ROBUST_FEATURE_INDICES` (add `{33, 34, 35, 37}`), add trade-level metrics to `evaluate()` |
| train.py | Replace `make_labeled_dataset` with Triple Barrier labeling, rename `FORWARD_HORIZON` → `MAX_HOLD_STEPS=300`, update config (min_hold=100, window=50), fix 2-tuple return bug, keep DEVICE=cpu |
| tests/test_features.py | Update expected feature count 31→39, add tests for new features |

## Success Criteria

**Minimum:** Match v5 baseline — Sortino ≥ 0.230, ≥ 18/25 passing

**Target:** Better Sortino or more passing symbols, with win_rate > 50% and profit_factor > 1.0

**Failure signal:** If flat-dominant class distribution causes the model to almost never trade (< 10 trades per symbol), the fee_threshold is too high for the shorter horizon. Fix: lower fee_mult to 1.0 or 1.2.

## Risks

1. **Barrier width vs fees.** A round trip costs 10 bps. TP barrier at 15 bps means net profit of only 5 bps per winning trade. If win_rate < 67%, the strategy loses money. Mitigated by: (a) asymmetric barriers in later tuning (wider TP, tighter SL), (b) Optuna search over fee_mult.

2. **Cache rebuild time.** First run will recompute all features (~20-30 min for 25 symbols). Subsequent runs use cache.

3. **More trades = more fee drag.** With min_hold=100 instead of 800, the model can trade 8x more often. Watch total fee costs in eval output.

4. **Class imbalance risk.** If fee_mult=1.5 produces >80% flat labels (barriers rarely hit within max_hold), the model may learn to always predict flat. Mitigated by: (a) validate distribution before training, (b) focal loss + class weights, (c) lower fee_mult or increase max_hold if needed.

5. **Memory for vectorized labeling.** Forward return matrix is O(n_samples × max_hold). At 10K × 300 = 24 MB per symbol, this is fine. But if max_samples is increased significantly, watch memory.

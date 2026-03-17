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

## Labeling Changes

| Parameter | v5 (old) | tape-v1 (new) | Rationale |
|-----------|----------|---------------|-----------|
| forward_horizon | 800 (~12h) | 150 (~2.5 min) | Setup plays out in minutes, not hours |
| min_hold | 800 (~12h) | 100 (~1.5 min) | Allow exit when setup completes |
| fee_mult | 1.5 | 1.5 | Keep same fee threshold initially |
| window_size | 50 | 50 | Keep same observation window |

**Label logic (unchanged):**
- forward return > fee_threshold → long (1)
- forward return < -fee_threshold → short (2)
- otherwise → flat (0)

**Fee threshold:** `2 * 5/10000 * 1.5 = 0.0015` (15 bps). At horizon=150 (~2.5 min for BTC), this should be exceeded ~30-40% of the time for volatile symbols (BTC, ETH, SOL) and ~10-15% for quieter ones (LTC, UNI). **Validate empirically before implementation** — if flat > 80% for most symbols, lower fee_mult to 1.0 or 1.2.

**Expected class distribution:** ~40-60% flat (vs 5% in v7). Flat-dominant is correct for setup detection. Validate with a quick data check during implementation.

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
| train.py | Revert to flat MLP from commit `04cfa49` (`DirectionClassifier` + `make_labeled_dataset`), update config (horizon=150, min_hold=100, window=50), keep DEVICE=cpu |
| tests/test_features.py | Update expected feature count 31→39, add tests for new features |

## Success Criteria

**Minimum:** Match v5 baseline — Sortino ≥ 0.230, ≥ 18/25 passing

**Target:** Better Sortino or more passing symbols, with win_rate > 50% and profit_factor > 1.0

**Failure signal:** If flat-dominant class distribution causes the model to almost never trade (< 10 trades per symbol), the fee_threshold is too high for the shorter horizon. Fix: lower fee_mult to 1.0 or 1.2.

## Risks

1. **Shorter horizon = smaller moves = fees eat profits.** A round trip costs 10 bps. A 2.5-minute move needs to exceed 15 bps to be labeled non-flat. Validate empirically on cached data before committing.

2. **Cache rebuild time.** First run will recompute all features (~20-30 min for 25 symbols). Subsequent runs use cache.

3. **More trades = more fee drag.** With min_hold=100 instead of 800, the model can trade 8x more often. Watch total fee costs in eval output.

4. **Class imbalance risk.** If fee_mult=1.5 produces >80% flat labels at horizon=150, the model may learn to always predict flat. Mitigated by: (a) validate distribution before training, (b) focal loss + class weights, (c) lower fee_mult if needed.

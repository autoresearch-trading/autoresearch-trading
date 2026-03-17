# Repository Guidelines

## Codebase Overview

**DEX perpetual futures trading research.** Supervised classification models trained on ~36GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols from Pacifica API.

**Current approach**: Supervised forward-return classifier with flat MLP (DirectionClassifier), pivoting to tape reading (setup detection with short-horizon labeling).

**Stack**: Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, DuckDB, Optuna

## Structure

```
prepare.py              — Data loading, feature engineering, TradingEnv, evaluate()
train.py                — Flat MLP model, training loop, Optuna search
program.md              — Experiment loop instructions
tests/                  — Feature engineering + env tests (test prepare.py, not train.py)
  conftest.py           — Shared fixtures (synthetic trades/orderbook/funding)
  test_features.py      — Feature shape, bounds, integration tests
  test_normalization.py — Hybrid normalization tests
  test_env.py           — TradingEnv min_hold constraint tests
scripts/
  sync_cloud_data.sh    — Fly.io -> R2
  fetch_cloud_data.sh   — R2 -> local
data/                   — Parquet: {trades,orderbook,funding}/symbol={SYM}/date={DATE}/ (gitignored)
.cache/                 — Cached .npz feature files, ~240 files (gitignored)
docs/superpowers/
  specs/                — Architecture design specs
  plans/                — Implementation plans
.github/workflows/
  daily_sync.yml        — Daily Fly.io -> R2 sync at 2AM UTC
```

## Key Exports from prepare.py

- `make_env(symbol, split, window_size, trade_batch, min_hold)` — creates TradingEnv
- `evaluate(env, policy_fn, min_trades)` — runs policy on full test set, returns Sortino ratio
- `compute_features(trades, orderbook, funding, trade_batch)` — features per step
- `normalize_features(features, window)` — hybrid z-score + robust scaling
- `TradingEnv` — Gymnasium env, 3 actions (flat/long/short), obs shape (window, num_features)
- `DEFAULT_SYMBOLS` — 25 crypto symbols
- `TRAIN_BUDGET_SECONDS = 300` (legacy, not used by current epoch-based training)
- Date constants: `TRAIN_START` (2025-10-16), `TRAIN_END` (2026-01-23), `VAL_END` (2026-02-17), `TEST_END` (2026-03-09)

## Features (31, v5)

Each step = 100 consecutive trades (~1-2 seconds for BTC).

| # | Feature | Source |
|---|---------|--------|
| 0-3 | returns, r_5, r_20, r_100 | trade |
| 4-5 | realvol_10, bipower_var_20 | trade |
| 6-8 | tfi, volume_spike_ratio, large_trade_share | trade |
| 9-11 | kyle_lambda_50, amihud_illiq_50, trade_arrival_rate | trade |
| 12-17 | spread_bps, log_total_depth, weighted_imbalance_5lvl, microprice_dev, ofi, ob_slope_asym | orderbook |
| 18-19 | funding_zscore, utc_hour_linear | funding/time |
| 20-24 | r_500, r_2800, cum_tfi_100, cum_tfi_500, funding_rate_raw | longer-horizon |
| 25-26 | VPIN, delta_TFI | v5 flow |
| 27-30 | Hurst, realized_skew, vol_of_vol, sign_autocorr | v5 higher-order |

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-09 (~145 days)
- **Splits**: Train (100d) / Val (25d) / Test (20d)
- **Sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`
- **Cache**: v5, keyed on `(symbol, start, end, trade_batch, _FEATURE_VERSION)`

## Evaluation

- **Metric**: Sortino ratio (downside vol only) on full test set (28K steps/symbol, no truncation)
- **Guardrails**: >= 10 trades, <= 20% drawdown per symbol
- **Portfolio**: mean Sortino across all passing symbols
- **Annualization**: `steps_per_day = total_steps / 20` (20 test days)

## Architecture (v5 — Flat MLP)

```
Input: (batch, window=50, features=31)

  → Flatten: 50 × 31 = 1,550
  → Append: per-feature mean(31) + std(31) = 62
  → flat_dim = 1,612
  → MLP: 1612 → 256 → 256 → 3 (flat/long/short)
  → ReLU, orthogonal init

Ensemble: 5 seeds, logit sum argmax
~0.5M parameters
Device: CPU
```

### Current Best (v5)
- Sortino=0.230, 18/25 passing, 923 trades, max_dd=0.367
- Config: lr=1e-3, hdim=256, nlayers=2, AdamW wd=5e-4, 25 epochs, 5 seeds

### Tape Reading Pivot (v8 — in progress)
- **Spec**: `docs/superpowers/specs/2026-03-17-tape-reading-pivot-design.md`
- **Plan**: `docs/superpowers/plans/2026-03-17-tape-reading-implementation.md`
- Add 8 tape reading features (39 total), shorten horizon to 150 steps (~2.5 min), min_hold=100
- Same flat MLP architecture, runs on CPU

## Conventions

- **Commit style**: `feat:`, `fix:`, `chore:`, `experiment:`, `spec:`, `plan:`
- **Git safety**: Only stage specific files, never `git add -A`
- **Experiment tracking**: `results.tsv` (commit, sortino, trades, dd, passing, status, description)
- **Output format**: Greppable `key: value` lines in PORTFOLIO SUMMARY

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone — R2 returns 1999-12-31 for all file timestamps
2. **Cache invalidation**: Bump `_FEATURE_VERSION` in prepare.py to invalidate. Currently `"v5"`
3. **Fee model**: Switching positions (long->short) pays 2x fees (close + open)
4. **ROBUST_FEATURE_INDICES**: `{5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 24, 25, 29}` — these use IQR-based robust scaling instead of z-score
5. **`evaluate()` uses test split**: Despite being called during training runs, it evaluates on test data (2026-02-17 to 2026-03-09)
6. **Full-test eval is ground truth**: Old 2000-step truncated eval was hiding failures (25/25 was an illusion, reality is 18/25)
7. **v7 attention overfit**: 2D attention (window=2000, RunPod H100) scored Sortino=0.061, 11/25 — worse than v5 flat MLP. Learnings: temporal architectures need much more data/compute; flat MLP is surprisingly strong

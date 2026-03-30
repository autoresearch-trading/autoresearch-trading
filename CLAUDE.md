# Repository Guidelines

## Codebase Overview

**DEX perpetual futures trading research.** Supervised classification models trained on ~36GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols from Pacifica API.

**Current approach**: Supervised forward-return classifier with flat MLP (DirectionClassifier). Microstructure-informed direction classifier on 13 features, fully tuned.

**Stack**: Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, DuckDB, Optuna

## Structure

```
prepare.py              — Data loading, feature engineering, TradingEnv, evaluate()
train.py                — Flat MLP model, training loop, Optuna search
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
.claude/skills/autoresearch/
  SKILL.md              — Autonomous research loop protocol
  resources/
    parse_summary.sh    — Extract PORTFOLIO SUMMARY → key=value
    state.md            — Current research state (updated each cycle)
docs/experiments/       — Experiment plans, logs, results, reports
docs/superpowers/
  specs/                — Architecture design specs
  plans/                — Implementation plans
.github/workflows/
  daily_sync.yml        — Daily Fly.io -> R2 sync at 2AM UTC
```

## Key Exports from prepare.py

- `make_env(symbol, split, window_size, trade_batch, min_hold, include_funding, date_range)` — creates TradingEnv
- `evaluate(env, policy_fn, min_trades)` — runs policy on full test set, returns Sortino ratio
- `compute_features(trades, orderbook, funding, trade_batch)` — features per step
- `normalize_features(features, window)` — hybrid z-score + robust scaling
- `TradingEnv` — Gymnasium env, 3 actions (flat/long/short), obs shape (window, num_features)
- `DEFAULT_SYMBOLS` — 25 crypto symbols
- `TRAIN_BUDGET_SECONDS = 300` (legacy, not used by current epoch-based training)
- Date constants: `TRAIN_START` (2025-10-16), `TRAIN_END` (2026-01-23), `VAL_END` (2026-02-17), `TEST_END` (2026-03-25)

## Features (39, v6)

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
| 31-32 | buy_run_max, sell_run_max | v6 tape reading |
| 33-34 | large_buy_share, large_sell_share | v6 tape reading |
| 35 | trade_size_entropy | v6 tape reading |
| 36 | aggressor_imbalance | v6 tape reading |
| 37 | price_level_absorption | v6 tape reading |
| 38 | tfi_acceleration | v6 tape reading |

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-09 (~145 days)
- **Splits**: Train (100d) / Val (25d) / Test (20d)
- **Sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`
- **Cache**: v6, keyed on `(symbol, start, end, trade_batch, _FEATURE_VERSION)`

## Evaluation

- **Metric**: Sortino ratio (downside vol only) on full test set (28K steps/symbol, no truncation)
- **Guardrails**: >= 10 trades, <= 20% drawdown per symbol
- **Portfolio**: mean Sortino across all passing symbols
- **Annualization**: `steps_per_day = total_steps / test_days` (dynamically computed)

## Architecture (v11b — Flat MLP)

```
Input: (batch, window=50, features=13)

  → Flatten: 50 × 13 = 650
  → Append: per-feature mean(13) + std(13) = 26
  → flat_dim = 676
  → MLP: 676 → 64 → 64 → 64 → 3 (flat/long/short)
  → ReLU, orthogonal init

Ensemble: 5 seeds, logit sum argmax
~160K parameters
Device: CPU
```

**Labeling:** Triple Barrier (TP/SL/timeout) with `MAX_HOLD_STEPS=300`, `fee_mult=11.0`, `MIN_HOLD=1200`

**Trade-level metrics:** evaluate() reports `win_rate`, `profit_factor`, `avg_profit_per_trade`, `avg_hold_steps`

### Current Best (v11b)
- Sortino=0.353 (fixed test), walk-forward mean=0.261 (4 folds, all positive)
- 9/23 passing, 1269 trades, WR=55%, PF=1.71
- Config: lr=1e-3, hdim=64, nlayers=3, batch=256, wd=0.0, fee_mult=11.0, min_hold=1200, window=50, 5 seeds, 25 epochs
- All hyperparameters exhaustively swept — model is at local optimum

### Historical Baselines
- **v5**: Sortino=0.230, 18/25, hdim=256, nlayers=2, fixed-horizon labeling
- **v6 tape reading**: Sortino=0.057, 8/25 — regression from multiple simultaneous changes
- **v7 attention**: Sortino=0.061, 11/25 — temporal architectures need much more data/compute

## Workflow

Two distinct modes of work:

1. **Superpowers workflow** (spec → plan → execute) — for structural changes: new features in prepare.py, architecture pivots, eval metric changes, anything that needs design review before code. Specs go in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`.

2. **Autoresearch skill** — for autonomous experimentation. Claude reads current state, forms hypotheses, designs experiments, runs them, draws conclusions, and repeats. Invoked by asking Claude to "run experiments", "investigate", "improve the model", etc. See `.claude/skills/autoresearch/SKILL.md`.

The handoff: execute the superpowers plan to build new infrastructure, then autoresearch takes over for tuning and experimentation.

## Conventions

- **Commit style**: `feat:`, `fix:`, `chore:`, `experiment:`, `spec:`, `plan:`
- **Git safety**: Only stage specific files, never `git add -A`
- **Experiment tracking**: `results.tsv` (commit, sortino, trades, dd, passing, status, description)
- **Output format**: Greppable `key: value` lines in PORTFOLIO SUMMARY

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone — R2 returns 1999-12-31 for all file timestamps
2. **Cache invalidation**: Bump `_FEATURE_VERSION` in prepare.py to invalidate. Currently `"v11b"`
3. **Fee model**: Switching positions (long->short) pays 2x fees (close + open)
4. **ROBUST_FEATURE_INDICES**: `{5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 24, 25, 29, 33, 34, 35, 37}` — these use IQR-based robust scaling instead of z-score
5. **`evaluate()` uses test split**: Despite being called during training runs, it evaluates on test data (2026-02-17 to 2026-03-25)
6. **Full-test eval is ground truth**: Old 2000-step truncated eval was hiding failures (25/25 was an illusion, reality is 18/25)
7. **v7 attention overfit**: 2D attention (window=2000, RunPod H100) scored Sortino=0.061, 11/25 — worse than v5 flat MLP. Learnings: temporal architectures need much more data/compute; flat MLP is surprisingly strong
8. **Walk-forward validated**: 4-fold rolling (80d train/20d test), all positive. Mean Sortino=0.261, std=0.220. T46 proved variance is sampling noise, not regime shifts.
9. **Fair comparison protocol**: When swapping loss functions, must re-tune lr (can vary 500x between losses). Keeping lr fixed = invalid comparison.
10. **Funding loading is slow**: `include_funding=True` in `make_env()` loads 200K+ tiny Parquet files per symbol. Use sparingly — proven negligible (T42).

## Key Discoveries

1. **Fee structure is the binding constraint** — alpha exists but is thin per trade. Barrier width (fee_mult) is the most sensitive parameter.
2. **One change at a time** — multiple arch/config changes simultaneously = regression. Ablation is the only way to attribute improvements.
3. **MLP beats XGBoost** (18/25 vs 8/25) — temporal pattern extraction from windowed features matters.
4. **Recency weighting helps** — decay=1.0, recent samples ~2.7x weight of oldest.
5. **Smaller network generalizes better** — hdim=64 > 128 > 256. The 676→64 bottleneck is a feature.
6. **Window=50 captures nonlinear temporal patterns** — T47 proved linear signal decays at lag 1, but MLP learns trajectory shapes (mean reversion, regime transitions) that linear cross-correlation misses.
7. **Focal loss resists "improvements"** — logit bias, curriculum learning, and UACE loss all made things worse, even with proper lr re-tuning for UACE. The focal+class_weights+recency setup is a strong local optimum.
8. **Realism assumptions are sound** — T42-T45 verified: funding negligible (0.16% of fee barrier), no conditional spread widening at tick scale, correlation rho=0.28 at tick level, latency captured by spread model. Sortino is not inflated.
9. **All hyperparameters exhaustively swept** — 16 variables tested, all confirmed at current values. Next improvements require new features, more data, or structural changes.

## Aristotle Proofs (T0-T47, 206+ theorems, 0 sorry)

Formal Lean 4 proofs via Harmonic's Aristotle prover. Submit via `aristotle formalize proofs/inputs/theoremNN-name.txt`.

- T0-T15: Math foundations (sufficient statistics, Kelly, Hawkes, gates, diversification)
- T16-T22: Experiment-backed (optimal trade count, gate paradox, frequency, complexity, min_hold)
- T23-T29: Metrics validation (optimal features, drawdown bounds, Sortino bug, statistical significance)
- T30-T38: Feature/implementation validation (OFI, VWAP, spread estimators, normalization, alignment)
- T42-T45: Realism (funding costs, conditional slippage, correlated drawdown, execution latency)
- T46: Sortino variance bound (walk-forward fold variance = sampling noise)
- T47: Optimal window from signal decay (linear signal at lag 0, but MLP finds nonlinear patterns)

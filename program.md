# autoresearch-trading

Supervised classification research for DEX perpetual futures trading. You are an AI researcher experimenting with trading models. Your goal is to maximize out-of-sample portfolio Sortino ratio.

**Git note:** Never use blind `git add -A` or `git add .`. Only stage the files you changed.

**Compute:** Local M4 MacBook Pro CPU. All training runs execute locally.

## Setup

1. Read this file completely
2. Read `prepare.py` to understand the environment and features
3. Read `train.py` to understand the current model
4. Verify data caches exist:
   ```bash
   ls .cache/*.npz | wc -l
   # Expected: ~240 (v4 + v5 caches)
   ```

## Data

**36GB of DEX perpetual futures data** from 25 crypto symbols, pre-computed as cached `.npz` feature files (~1.1GB).

25 symbols: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP

### Splits
- **Train**: 2025-10-16 to 2026-01-23 (100 days)
- **Validation**: 2026-01-23 to 2026-02-17 (25 days)
- **Test**: 2026-02-17 to 2026-03-09 (20 days) — used by evaluate()

### Feature Channels (31 features per step, v5)

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
| 25-26 | VPIN, delta_TFI | flow |
| 27-30 | Hurst, realized_skew, vol_of_vol, sign_autocorr | higher-order |

## Current Approach

**Supervised forward-return classifier** with recency-weighted focal loss.

- Labels: forward return over configurable horizon -> long/flat/short based on fee threshold
- Model: flat MLP (DirectionClassifier) — window flattened + temporal summary stats
- Ensemble: multi-seed, logit-sum argmax
- Evaluation: Sortino ratio on full test set (28K steps/symbol), no truncation

### Current Best (v5 flat MLP)
- Sortino=0.230, 18/25 passing, 923 trades, max_dd=0.367
- Config: lr=1e-3, hdim=256, nlayers=2, AdamW wd=5e-4, 25 epochs, 5 seeds

## What You CAN Modify

- `train.py` — model architecture, training loop, hyperparameters, ensemble strategy
- `prepare.py` — features, evaluation (if needed for research)

## Goal

**Maximize portfolio Sortino ratio** — mean Sortino across all symbols that pass guardrails.

### Guardrails (per symbol, enforced by evaluate())
- **Minimum 10 trades**
- **Maximum 20% drawdown**
- Violating either -> that symbol excluded from portfolio mean

## Experiment Loop

### 1. Form hypothesis
### 2. Implement change
### 3. Commit
```bash
git add train.py
git commit -m "experiment: <brief description>"
```
### 4. Run training
```bash
uv run train.py 2>&1 | tee run.log
```
### 5. Record results in results.tsv
### 6. Keep or discard

## Output Format

`train.py` MUST print these lines in the PORTFOLIO SUMMARY section:
```
symbols_passing: N/25
sortino: X.XXXXXX
num_trades: N
max_drawdown: X.XXXX
total_steps: N
num_updates: N
```

## Logging

`results.tsv` columns:
```
commit	sortino	num_trades	max_drawdown	symbols_passing	status	description
```

## Key Discoveries

1. **Fee structure is the binding constraint** — alpha exists but is thin per trade
2. **One change at a time** — multiple arch changes simultaneously = regression
3. **Full-test eval is ground truth** — 2000-step truncation was hiding failures
4. **MLP beats XGBoost** (18/25 vs 8/25) — temporal pattern extraction matters
5. **v7 attention overfit** — 2D attention (Sortino=0.061, 11/25) lost badly to flat MLP
6. **Recency weighting helps** — decay=1.0, recent samples ~2.7x weight of oldest

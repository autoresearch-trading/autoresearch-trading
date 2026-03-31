# Autoresearch Trading

DEX perpetual futures trading research. Supervised classification models trained on ~40GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols from Pacifica API.

## Quick Start

```bash
# Install dependencies
uv sync

# Sync data from Cloudflare R2
rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only

# Run tests
uv run pytest tests/

# Train with current best config
uv run python train.py

# Optuna hyperparameter search
uv run python train.py --search
```

## Architecture

Flat MLP classifier on 13 microstructure features, trained with focal loss and Triple Barrier labeling.

```
Raw Parquet (trades + orderbook)
  -> compute_features_v9()     # 100-trade batches -> 13 features
  -> normalize_features_v9()   # rolling z-score/IQR, clip +/-5
  -> TradingEnv                # Gymnasium: obs=(50,13), actions={flat,long,short}
  -> make_labeled_dataset()    # Triple Barrier: 300-step horizon, cost-adjusted
  -> DirectionClassifier       # 676 -> 64 -> 64 -> 64 -> 3, ~52K params
  -> 5-seed ensemble           # sum logits -> argmax
  -> evaluate()                # Sortino ratio on full test set
```

**Current best (v11b):** Sortino=0.353, 9/23 symbols passing, 1269 trades, WR=55%, PF=1.71

## Features (13, v11b)

| # | Feature | Source |
|---|---------|--------|
| 0 | lambda_ofi | trade+orderbook |
| 1 | directional_conviction | trade |
| 2 | vpin | trade |
| 3 | hawkes_branching | trade |
| 4 | reservation_price_dev | orderbook |
| 5 | vol_of_vol | trade |
| 6 | utc_hour_linear | time |
| 7 | microprice_dev | orderbook |
| 8 | delta_tfi | trade |
| 9 | multi_level_ofi | orderbook |
| 10 | buy_vwap_dev | trade |
| 11 | trade_arrival_rate | trade |
| 12 | r_20 | trade |

## Project Structure

```
prepare.py              — Data loading, feature engineering, TradingEnv, evaluate()
train.py                — DirectionClassifier, focal loss training, Optuna search
tests/                  — 100 tests across 10 files
scripts/                — Data sync + analysis scripts (walk-forward, ablation, T42-T47)
data/                   — ~40GB Hive-partitioned Parquet (gitignored)
.cache/                 — Cached v11b .npz feature files (gitignored)
docs/experiments/       — Experiment plans, reports, results
docs/superpowers/       — Design specs and implementation plans
proofs/                 — Aristotle Lean 4 formal proofs (T0-T47)
```

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-25 (~160 days)
- **Splits**: Train (99d) / Val (25d) / Test (36d)
- **Pipeline**: Fly.io collector -> GitHub Actions daily sync -> Cloudflare R2 -> local

## Evaluation

- **Metric**: Sortino ratio (annualized, downside deviation only)
- **Guardrails**: >= 10 trades, <= 20% max drawdown per symbol
- **Portfolio score**: mean Sortino across passing symbols
- **Walk-forward validated**: 4-fold rolling, all positive, mean=0.261

## Stack

Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, DuckDB, Optuna

See [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for detailed architecture documentation.

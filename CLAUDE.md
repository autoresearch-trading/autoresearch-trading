# Repository Guidelines — tape-reading branch

## Project Overview

**DEX perpetual futures tape reading research.** Training a sequential model directly on raw trade data (40GB, 160 days, 25 crypto symbols from Pacifica API) to learn universal microstructure patterns. No handcrafted features — the model learns its own representations from the tape.

**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`

**Stack**: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

**Agent system:** Start with `claude --agent lead-0`. Lead-0 orchestrates council (council-1 through council-7) and workers (builder-8, analyst-9, reviewer-10, validator-11).

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-25 (~160 days)
- **Sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`

### Raw data schema

**Trades** (`data/trades/symbol={SYM}/date={DATE}/*.parquet`):
- `ts_ms`, `symbol`, `trade_id`, `side` (open_long/close_long/open_short/close_short), `qty`, `price`, `recv_ms`
- New fields (from Apr 1 only): `cause` (normal/market_liquidation/backstop_liquidation), `event_type` (fulfill_taker/fulfill_maker)

**Orderbook** (`data/orderbook/symbol={SYM}/date={DATE}/*.parquet`):
- 25 levels per side, sampled every ~3 seconds

**Funding** (`data/funding/symbol={SYM}/date={DATE}/*.parquet`):
- `ts_ms`, `symbol`, `rate`, `interval_sec`, `recv_ms`

**Prices** (`data/prices/symbol={SYM}/date={DATE}/*.parquet` — from Apr 1 only):
- `open_interest`, `volume_24h`, `mark`, `oracle`, `funding`, `next_funding`

## Input Representation (16 features per order event)

**Order events:** Same-timestamp trades grouped into single events.

**Trade features (10):**
1. `log_return` — log(vwap / prev_vwap)
2. `log_total_qty` — log(total_qty / rolling_median_qty)
3. `is_buy` — 1 if buy, 0 if sell
4. `is_open` — fraction of fills that are opens [0,1]
5. `time_delta` — log(ts - prev_ts + 1)
6. `num_fills` — log(fill count)
7. `price_impact` — (last_fill - first_fill) / mid
8. `effort_vs_result` — clip(log_qty - log(|return| + 1e-4), -5, 5)
9. `is_climax` — 1 if qty > 2σ_rolling AND |return| > 2σ_rolling
10. `seq_time_span` — log(last_ts - first_ts + 1) for the full sequence

**Orderbook features (6, aligned by nearest prior snapshot):**
11. `log_spread` — log(best_ask - best_bid)
12. `imbalance_L1` — (bid_L1 - ask_L1) / (bid_L1 + ask_L1)
13. `imbalance_L5` — top 5 levels
14. `depth_ratio` — log(total_bid / total_ask)
15. `trade_vs_mid` — (vwap - mid) / spread
16. `delta_imbalance_L1` — change since previous event

## Architecture

Dilated 1D CNN (first try, ~65K params):
```
Input: (batch, 200, 16) — 200 order events, 16 features
BatchNorm → Conv1d(16→32, k=5, d=1) → Conv1d(32→64, k=5, d=2) → Conv1d(64→64, k=5, d=4) → GlobalAvgPool → Linear(64→4)
Output: 4 sigmoid heads (direction at 10/50/100/500 events forward)
```

## Evaluation

- **Primary metric:** accuracy on ALL 25 symbols (universal, not cherry-picked)
- **Target:** > 52% accuracy, < 2% std across symbols, > 20/25 symbols above 51%
- **Mandatory linear baseline:** logistic regression must beat 50.5% before any neural network

## Implementation Steps

0. Label validation — compute base rate (go/no-go gate)
1. Data pipeline — `tape_dataset.py` (order event grouping + orderbook alignment + caching)
1.5. Linear baseline — logistic regression (go/no-go gate)
2. Prototype — 1D CNN on 1 symbol, 1 day (verify pipeline)
3. Full training — RunPod H100, all symbols, all days
4. Analysis — per-symbol accuracy, feature attribution

## Key Findings from Previous Work

- `is_open` has autocorrelation half-life of 20 trades — the strongest persistent signal in raw trades
- `is_buy` has half-life of 1 — no persistence, essentially random
- Shuffling trades within batches reduces feature-return correlations by 37.6% — sequence order matters
- The flat MLP classifier (main branch) hit Sortino 0.353 on 9/23 symbols — every incremental change made it worse
- 100-trade batching destroys tape signals — this branch works with raw trades

## Conventions

- **Commit style**: `feat:`, `fix:`, `chore:`, `experiment:`, `spec:`, `analysis:`
- **Git safety**: Only stage specific files, never `git add -A`
- **Branch**: `tape-reading` (from main)

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone
2. **Orderbook alignment**: use `np.searchsorted(ob_ts, trade_ts, side="right") - 1` — each trade gets the most recent prior snapshot
3. **Order event grouping**: same-timestamp trades are fragments of one order — group before feeding to model
4. **Rolling medians for normalization**: never use global statistics (lookahead bias)
5. **`effort_vs_result` explosion**: clip to [-5, 5] — near-zero returns cause log explosion
6. **`is_climax` σ**: must be rolling 1000-event σ, not global
7. **Test set contamination**: the Mar 5-25 window was used for 20+ experiments on main branch — treat with skepticism. Prefer April data when available.

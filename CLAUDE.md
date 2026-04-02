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
- 10 levels per side, sampled every ~24 seconds

**Funding** (`data/funding/symbol={SYM}/date={DATE}/*.parquet`):
- `ts_ms`, `symbol`, `rate`, `interval_sec`, `recv_ms`

**Prices** (`data/prices/symbol={SYM}/date={DATE}/*.parquet` — from Apr 1 only):
- `open_interest`, `volume_24h`, `mark`, `oracle`, `funding`, `next_funding`

## Input Representation (17 features per order event)

**Order events:** Same-timestamp trades grouped into single events after dedup (validated in Step 0).
- Pre-April dedup: `drop_duplicates(subset=['ts_ms', 'qty', 'price'])` — without `side`
- April+ dedup: filter to `event_type == 'fulfill_taker'`

**Trade features (9):**
1. `log_return` — log(vwap / prev_vwap)
2. `log_total_qty` — log(total_qty / rolling_median_qty_1000) — rolling 1000-event median, causal
3. `is_open` — fraction of fills that are opens [0,1]
4. `time_delta` — log(ts - prev_ts + 1)
5. `num_fills` — log(fill count)
6. `book_walk` — abs(last_fill - first_fill) / max(spread, 1e-8*mid) — unsigned, spread-normalized
7. `effort_vs_result` — clip(log_total_qty - log(|return| + 1e-6), -5, 5) — uses median-normalized log_total_qty
8. `climax_score` — clip(min(z_qty, z_return), 0, 5) — continuous intensity, rolling 1000-event σ
9. `prev_seq_time_span` — log(last_ts - first_ts + 1) for PREVIOUS 200-event window (no lookahead)

**Orderbook features (8, aligned by nearest prior snapshot, ~24s cadence, 10 levels):**
10. `log_spread` — log((ask - bid) / mid) — mid-normalized
11. `imbalance_L1` — notional imbalance at L1
12. `imbalance_L5` — inverse-level-weighted notional imbalance L1:5
13. `depth_ratio` — log(max(bid_notional, 1e-6) / max(ask_notional, 1e-6)) — notional, epsilon-guarded
14. `trade_vs_mid` — clip((vwap - mid) / max(spread, 1e-8*mid), -5, 5) — zero-spread guarded, doubles as direction proxy
15. `delta_imbalance_L1` — change since previous snapshot, carry-forward between snapshots
16. `kyle_lambda` — per-SNAPSHOT Cov(Δmid, cum_signed_notional)/Var(cum_signed_notional) over 50 snapshots (~20 min), forward-filled
17. `cum_ofi_5` — rolling sum of OFI (piecewise Cont 2014) over last 5 book snapshots (~120s), normalized by rolling notional volume

## Architecture

Dilated 1D CNN (~91K params, 6 layers, RF=253):
```
Input: (batch, 200, 17) — 200 order events, 17 features
BatchNorm(17) → [Conv1d + LayerNorm + ReLU + Dropout(0.1)] × 2 → [Conv1d + LayerNorm + ReLU + residual] × 4
Dilations: 1, 2, 4, 8, 16, 32 — receptive field covers full 200-event window
concat[GlobalAvgPool, last_position] → Linear(128→64) + ReLU → [Linear(64→1)] × 4
Output: 4 sigmoid heads (direction at 10/50/100/500 events forward)
Loss: BCEWithLogits + label smoothing (ε=0.10/0.08/0.05/0.05), weights 0.10/0.20/0.35/0.35
Optimizer: AdamW + OneCycleLR(max_lr=3e-4, 30% warmup)
```

## Evaluation

- **Primary metric:** accuracy at **primary horizon (100 events)** on ALL 25 symbols
- **Target:** > 52% accuracy, < 2% std across symbols, > 20/25 symbols above 51%
- **Hold-out symbol test:** 1 symbol excluded from training, tested for universality
- **Multiple testing:** Holm-Bonferroni correction across all trials (~1,600 including sweeps); maintain trial_log.csv
- **Mandatory linear baseline:** logistic regression (C sweep) — 15+/25 symbols must exceed 51.4% at primary horizon
- **April hold-out:** April 14+ designated untouched (2026-04-02). April 1-13 for dev validation.
- **Embargo:** 600-event gap between train/test folds

## Implementation Steps

0. Label + data validation — compute base rate, validate same-timestamp grouping (go/no-go gate)
1. Data pipeline — `tape_dataset.py` (order event grouping + OB alignment + caching)
1.5. Linear baseline — logistic regression with C sweep (go/no-go gate)
2. Prototype — 1D CNN on 1 symbol, 1 day (verify pipeline)
3. Full training — RunPod H100, all symbols, all days
4. Analysis — per-symbol accuracy, feature attribution, DSR

## Key Findings from Previous Work

- `is_open` has autocorrelation half-life of 20 trades — the strongest persistent signal in raw trades
- `is_buy` has half-life of 1 — no persistence, essentially random. **Dropped in council round 5** (59% direction ambiguity + no persistence = useless feature)
- Shuffling trades within batches reduces feature-return correlations by 37.6% — sequence order matters
- The flat MLP classifier (main branch) hit Sortino 0.353 on 9/23 symbols — every incremental change made it worse
- 100-trade batching destroys tape signals — this branch works with raw trades

## Conventions

- **Commit style**: `feat:`, `fix:`, `chore:`, `experiment:`, `spec:`, `analysis:`
- **Git safety**: Only stage specific files, never `git add -A`
- **Branch**: `tape-reading` (from main)

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone
2. **Orderbook alignment**: use `np.searchsorted(ob_ts, trade_ts, side="right") - 1` — vectorize over all events, not Python for-loop
3. **Order event grouping**: same-timestamp trades are fragments of one order — **dedup first**, then group. Pre-April: `drop_duplicates(subset=['ts_ms', 'qty', 'price'])` WITHOUT `side`. April+: filter to `event_type == 'fulfill_taker'`. 59% of events have mixed buy/sell fills (exchange mechanic, not error).
4. **Rolling medians for normalization**: never use global statistics (lookahead bias). Rolling 1000-event, causal.
5. **`effort_vs_result` explosion**: clip to [-5, 5], epsilon = 1e-6 (not 1e-4 — too coarse for BTC tick-level). Uses median-normalized log_total_qty, not raw log(qty).
6. **`climax_score` σ**: must be rolling 1000-event σ, not global. Continuous score, not binary.
7. **Test set contamination**: the Mar 5-25 window was used for 20+ experiments on main branch — treat with skepticism. Prefer April data when available.
8. **`prev_seq_time_span` not `seq_time_span`**: original was hard lookahead — used 200th event's timestamp for event 1. Use prior window's time span.
9. **`depth_ratio` log(0)**: one-sided book during flash crashes → epsilon guard required
10. **`trade_vs_mid` div-by-zero**: spread can be 0 in snapshots → guard with max(spread, 1e-8*mid) and clip
11. **`delta_imbalance_L1` day boundaries**: first event of day has no prior → pre-warm from prior day (committed — no masking except first calendar day per symbol)
12. **Walk-forward embargo**: 600-event gap between train/test folds — label lookahead at boundaries
13. **kyle_lambda is per-SNAPSHOT, not per-event**: event-level had ~2 effective observations per 50-event window at 24s OB cadence. Per-snapshot over 50 snapshots (~20 min), forward-filled. Uses Δmid, not Δvwap.
14. **depth_ratio, kyle_lambda, cum_ofi_5**: must use notional (qty × price) for cross-symbol comparability
15. **cum_ofi_5 uses piecewise Cont 2014 OFI**: naive delta-notional has wrong sign when best bid/ask price changes between snapshots (60-80% of the time at 24s cadence). Must check price level changes.
16. **Stride=200**: non-overlapping input windows. First window offset randomized per epoch.
17. **April hold-out**: April 14+ is untouched — do not view, even for data quality checks
18. **BatchNorm at inference**: must use `model.eval()` for entire test pass — otherwise running stats contaminated. Single-sample = NaN.
19. **Dedup key must NOT include `side`**: buyer/seller pairs differ on `side`, so including it in dedup removes nothing. Use `(ts_ms, qty, price)` only.
20. **OB has 10 levels, not 25**: all symbols measured at 10 bid + 10 ask levels. ~24s cadence, not ~3s.
21. **Training samples ~400-560K, not 1.2M**: after dedup + grouping. ~28K events/day on BTC (was 140K raw trades).

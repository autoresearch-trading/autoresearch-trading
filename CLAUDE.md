# Repository Guidelines — representation-learning branch

## Project Overview

**DEX perpetual futures tape representation learning.** Self-supervised model trained on 40GB of raw trade data (160 days, 25 crypto symbols from Pacifica API) to learn meaningful tape representations — the way a human tape reader develops intuition from watching millions of order events. Direction prediction is a downstream probing task, not the primary objective.

**Spec:** `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`

**Stack**: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

**Agent system:** Start with `claude --agent lead-0`. Lead-0 orchestrates council (council-1 through council-6) and workers (runpod-7 through researcher-14, excluding data-eng-13 — merged into builder-8).

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-25 (~160 days)
- **Held-out symbol**: AVAX (pre-designated, excluded from pretraining)
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

**Three load-bearing features (Wyckoff):**
- `effort_vs_result` — the master signal: absorption (high) vs ease-of-movement (low)
- `climax_score` — phase transition markers (buying/selling climax)
- `is_open` — DEX-specific Composite Operator footprint (no equivalent in traditional markets)

## Architecture

Self-supervised encoder (~400K params, 7 layers, RF=253):
```
Input: (batch, 200, 17) — 200 order events, 17 features
BatchNorm(17) → [Conv1d + LayerNorm + ReLU + Dropout(0.1)] × 2 → [Conv1d + LayerNorm + ReLU + residual] × 4
Channels: 17→64→128→128→128→128→128, kernel=5, dilations: 1, 2, 4, 8, 16, 32
Global embedding: concat[GlobalAvgPool(128), last_position(128)] → 256-dim

Pretraining heads (discarded after):
  MEM decoder: Linear(128→17) at masked positions → MSE (14 of 17 features)
  Projection: Linear(256→256) + ReLU + Linear(256→128) → L2-norm → NT-Xent

Fine-tuning heads (added after):
  Linear(256→64) + ReLU → [Linear(64→1)] × 4, sigmoid — per-horizon direction
```

## Training

**Self-supervised pretraining** (primary):
- Masked Event Modeling (block masking, 5-event blocks, 15% of events) — weight 0.70
- SimCLR contrastive on global embeddings — weight 0.30
- Direction labels NOT used during pretraining
- Stride=50 (4× data), equal-symbol sampling per epoch
- Exclude delta_imbalance_L1, kyle_lambda, cum_ofi_5 from MEM reconstruction (trivial copy)
- AdamW + OneCycleLR(max_lr=1e-3, 20% warmup)
- Compute cap: 1 H100-day before evaluation gates

**Fine-tuning** (after Gate 1 passes):
- Freeze encoder 5 epochs → unfreeze at lr=5e-5
- Loss weights: 0.10/0.20/0.50/0.20 for 10/50/100/500 event horizons
- Walk-forward validation with 600-event embargo

## Evaluation Gates (pre-registered, sequential)

| Gate | Test | Threshold |
|------|------|-----------|
| **0** | PCA + logistic regression baseline | Reference (CNN probe must exceed) |
| **1** | Linear probe on frozen embeddings, 100-event direction | > 51.4% on 15+/25 symbols (April 1-13) |
| **2** | Fine-tuned CNN vs logistic regression on flat features | Exceed by ≥ 0.5pp on 15+ symbols |
| **3** | Held-out symbol (AVAX) accuracy | > 51.4% at primary horizon |
| **4** | Temporal stability (months 1-4 vs 5-6) | < 3pp drop on 10+ symbols |

**Representation quality diagnostics:**
- Symbol identity probe < 20% accuracy (embeddings must NOT encode symbol)
- CKA > 0.7 between seed-varied runs
- Wyckoff label probes (absorption, climax, informed flow, stress)

## Implementation Steps

0. Label + data validation — base rate, same-timestamp grouping, dedup validation
1. Data pipeline — `tape_dataset.py` (order events + OB alignment + caching + stride=50)
2. Baselines — Gate 0: PCA + logistic regression, random encoder
3. Pretraining — MEM + contrastive on RunPod H100
4. Evaluation — Gates 1-4, symbol probe, cluster analysis, Wyckoff probes
5. Fine-tuning — conditional on Gate 1 pass
6. Interpretation — feature attribution, embedding trajectories, market state vocabulary

## Key Findings from Previous Work

- `is_open` has autocorrelation half-life of 20 trades — the strongest persistent signal in raw trades
- `is_buy` has half-life of 1 — no persistence, essentially random. **Dropped in council round 5**
- Shuffling trades within batches reduces feature-return correlations by 37.6% — sequence order matters
- The flat MLP classifier (main branch) hit Sortino 0.353 on 9/23 symbols — every incremental change made it worse
- 100-trade batching destroys tape signals — this branch works with raw trades
- Council data sufficiency review: 40GB is massive for representation learning, marginal for proving a 2% trading edge

## Conventions

- **Commit style**: `feat:`, `fix:`, `chore:`, `experiment:`, `spec:`, `analysis:`
- **Git safety**: Only stage specific files, never `git add -A`
- **Branch**: `main`

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone
2. **Orderbook alignment**: use `np.searchsorted(ob_ts, trade_ts, side="right") - 1` — vectorize over all events, not Python for-loop
3. **Order event grouping**: same-timestamp trades are fragments of one order — **dedup first**, then group. Pre-April: `drop_duplicates(subset=['ts_ms', 'qty', 'price'])` WITHOUT `side`. April+: filter to `event_type == 'fulfill_taker'`. After correct dedup, **3-16% of events are mixed-side** (median ~3%; liquid symbols higher). The earlier "59%" figure was an artifact of measuring on undeduped data where each fill appears twice (buyer + seller perspective).
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
16. **Stride**: 50 for pretraining, 200 for evaluation probes. First window offset randomized per epoch.
17. **April hold-out**: April 14+ is untouched — do not view, even for data quality checks
18. **BatchNorm at inference**: must use `model.eval()` for entire test pass — otherwise running stats contaminated. Single-sample = NaN.
19. **Dedup key must NOT include `side`**: buyer/seller pairs share identical `(ts_ms, qty, price)` but differ on `side` — including `side` in the dedup key PRESERVES both counterparty records (doubling event counts); excluding `side` collapses them to one record per fill (correct). Use `(ts_ms, qty, price)` only.
20. **OB has 10 levels, not 25**: all symbols measured at 10 bid + 10 ask levels. ~24s cadence, not ~3s.
21. **Training windows ~627K at stride=50** (Step 0 measurement): pre-April, 25 symbols, 161 days. BTC alone contributes ~133K; all other symbols average ~2K/symbol. BTC events/day median is ~20,563 (not ~28K). 200-event window on BTC is ~5.1 min (not ~10 min). Data-to-400K-params ratio is ~1:1.6 — tight; model size is a live question for Step 3 (council-6 review pending).
22. **MEM reconstruction targets**: exclude delta_imbalance_L1, kyle_lambda, cum_ofi_5 (trivially copyable from neighbors)
23. **MEM loss space**: compute in BatchNorm-normalized space, not raw feature space
24. **Embedding collapse**: monitor per-batch embedding std. If → 0, pretraining has collapsed.
25. **Cross-symbol contrastive**: only for liquid symbols (BTC, ETH, SOL, BNB, LINK, LTC). **AVAX is the Gate 3 held-out symbol and must NOT appear in contrastive pairs.** LTC substitutes for AVAX in the 6-symbol anchor set. Do NOT force invariance with memecoins.
26. **Day boundaries**: do not construct windows crossing day boundaries
27. **Symbol sampling**: equal-symbol sampling per epoch to prevent BTC dominance

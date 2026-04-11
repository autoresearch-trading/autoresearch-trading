# Direct Tape Reading: Specification

## Goal

Train a sequential model directly on raw trade data (40GB, 160 days, 25 symbols) to learn universal microstructure patterns. No handcrafted features — the model learns its own representations from the tape.

## Why This Is Different

**Current approach:**
```
Raw trades (40GB) → 13 summary stats per 100 trades → flat MLP → direction
```
Throws away 99.9% of information. The 13 features are human-designed bottleneck.

**Tape reading approach:**
```
Raw trades (40GB) → order events → sequential model → learned representations → direction
```
The model decides what matters. 40GB is actually substantial training data for a sequence model.

## Raw Trade Schema

Each trade from Pacifica:
```
ts_ms:   1775044557458        (timestamp in ms)
price:   68686.0              (execution price)
qty:     0.0019               (size)
side:    open_long             (open_long/close_long/open_short/close_short)
cause:   normal                (normal/market_liquidation/backstop_liquidation — new, collecting from Apr 1)
event_type: fulfill_taker      (fulfill_taker/fulfill_maker — new, collecting from Apr 1)
```

## Pre-Step: Validate Label Signal

**Before building anything, compute the base rate of the binary label.**

Script: compute next-N-event direction for all symbols across all dates. Report:
- Up/down split per horizon (if 50.1%/49.9%, the label is pure noise)
- Mean absolute return per horizon (if < 2 bps over the full horizon, moves are too small to be meaningful after fees)
- **Conditional return distribution:** top decile and top quartile of absolute returns per horizon. If the top decile of 100-event returns exceeds 15 bps, a model achieving 54% accuracy on those events is economically viable even if overall accuracy is only 51%. This connects accuracy targets to the fee structure (fee_mult=11.0, ~11 bps round-trip). (Kyle fix, round 3)
- Compare horizons: 10, 50, 100, 500 events forward
- **Verify label computation does not use any data from the label period itself** — add explicit assertion in `test_label_signal.py` (Lopez de Prado fix, round 3)

**Stop if EITHER condition is true** (per horizon, evaluated per-symbol, not aggregate — Lopez de Prado fix, round 4):
- Base rate is within 50 ± 0.5% for more than 10/25 symbols at that horizon, **OR**
- 90th-percentile absolute return is below fee_mult (11 bps) at the primary horizon (100 events) — even the best 10% of events cannot cover round-trip costs.

Also report per-symbol base rates — any symbol outside 50 ± 1% at the primary horizon requires investigation before inclusion in training. Consider longer horizons or different labels (e.g., threshold-based: up > 5 bps, down < -5 bps, flat otherwise).

## Order Event Grouping

**Trades with the same timestamp are fragments of one order being filled across price levels.** Group them before feeding to the model.

```
Raw trades:                          Order events:
ts=100, buy,  0.01 @ 68686         → Event 1: buy, total_qty=0.03, vwap=68687,
ts=100, buy,  0.01 @ 68687            fills=3, price_range=2
ts=100, buy,  0.01 @ 68688
ts=105, sell, 0.02 @ 68685         → Event 2: sell, total_qty=0.02, vwap=68685,
                                       fills=1, price_range=0
```

Per order event (preliminary features computed during grouping):
```
1. log_return:      log(vwap / prev_event_vwap)
2. log_total_qty:   log(total_qty / rolling_median_event_qty)  — rolling 1000-event median, causal
3. is_open:         fraction of fills that are opens (0 to 1)
4. time_delta:      log(ts - prev_event_ts + 1)
5. num_fills:       log(number of fills in this event)
6. book_walk:       abs(last_fill_price - first_fill_price) / max(spread, 1e-8 * mid) — unsigned, spread-normalized
```

**`is_buy` dropped (council round 5):** 59% of pre-April events have ambiguous direction (exchange reports both counterparties at same timestamp). `is_buy` had half-life of 1 event (no persistence). Directional info is already in `log_return` (signed price) and `trade_vs_mid` (execution location vs midpoint). Keeping it would create a pre/post-April distributional discontinuity. See `docs/council-reviews/2026-04-02-council-4-tape-viability.md`.

**Dedup required before grouping (council round 5):** The exchange reports both counterparties of each fill as separate rows. 30-74% of raw rows are duplicated buyer/seller pairs. `trade_id` is always empty.
- **Pre-April data:** `df.drop_duplicates(subset=['ts_ms', 'qty', 'price'], keep='first')` — dedup on `(ts_ms, qty, price)` WITHOUT `side` (buyer/seller pairs differ on `side`, so including it removes nothing). See `docs/council-reviews/2026-04-02-council-5-dedup-direction.md`.
- **April+ data:** Filter to `event_type == 'fulfill_taker'` — handles both dedup and direction identification in one step. Verify `event_type` is non-null for >99% of rows before using this filter.

**Validation required (Step 0):** The same-timestamp = same-order assumption must be empirically verified on Pacifica data before building the pipeline. Check for mixed-side fills at same timestamp and extreme fill counts. After dedup, 59% of events still contain both buy+sell fills (exchange mechanic, not data quality issue). See Step 0 for details.

This reduces noise (one order = one event, not multiple rows) and captures order-level signals (an order that fills across 5 price levels is more aggressive than one that fills at a single price).

**Sequence length adjustment:** With order events, 200 events covers more clock time than 200 raw trades. This is better — each event is a real decision, not a matching engine artifact.

## Input Representation

Per order event, 17 features from 2 data sources (trades + orderbook):

**From trade events (9 features):**
```
1.  log_return:        log(vwap / prev_event_vwap)                — relative price change
2.  log_total_qty:     log(total_qty / rolling_median_event_qty)   — relative order size (rolling 1000-event median, causal)
3.  is_open:           fraction of fills that are opens [0,1]      — position flow (Wyckoff's Composite Operator)
4.  time_delta:        log(ts - prev_event_ts + 1)                 — urgency
5.  num_fills:         log(fill count)                              — order complexity
6.  book_walk:         abs(last_fill - first_fill) / max(spread, 1e-8 * mid) — how much the order walked the book (unsigned, spread-normalized; renamed from price_impact to avoid collision with the theoretical concept of permanent price impact — Kyle fix, round 4)
7.  effort_vs_result:  clip(log_total_qty - log(abs(return) + 1e-6), -5, 5) — Wyckoff: high = absorption, low = breakout (uses median-normalized log_total_qty from feature 2, not raw log(qty) — practitioner fix, round 4)
8.  climax_score:      clip(min(qty_zscore, return_zscore), 0, 5)  — Wyckoff: buying/selling climax intensity (rolling 1000-event σ)
9.  prev_seq_time_span: log(last_ts - first_ts + 1) for the PREVIOUS 200-event window — context: fast or slow tape (no lookahead)
```

**Notes on council fixes (rounds 1-5):**
- `is_buy` dropped (council round 5): 59% of pre-April events have ambiguous direction (exchange reports both counterparties). Half-life of 1 = no persistence. Info already in `log_return` + `trade_vs_mid`. See council-4 and council-5 reviews.
- `log_total_qty` normalized by rolling 1000-event median (not global/daily — prevents lookahead). First 1000 events of each day pre-warmed from prior day or masked from training.
- `book_walk` (renamed from `price_impact`, Kyle fix round 4) is unsigned and spread-normalized for cross-symbol comparability (Cont fix, round 3). Guard against zero spread with `max(spread, 1e-8 * mid)`.
- `effort_vs_result` clipped to [-5, 5]; epsilon tightened from 1e-4 to 1e-6 to preserve tick-level absorption detection (Wyckoff fix, round 3). At 1e-4, BTC absorption events below 6.8 points were masked.
- `climax_score` replaced binary `is_climax` — continuous intensity score provides gradient signal for near-climax events and captures climax clustering (Kyle/Wyckoff fix, round 3). Uses rolling 1000-event σ (not global) to prevent lookahead bias.
- `prev_seq_time_span` replaced `seq_time_span` — original used the 200th event's timestamp, which is hard lookahead when computing features for earlier events in the window (practitioner fix, round 3). Now uses the preceding window's time span, which is always available.

**From orderbook (aligned by nearest prior snapshot, ~24s cadence, 10 levels per side; 8 features):**
```
10. log_spread:        log((best_ask - best_bid) / mid + 1e-10)           — relative spread (mid-normalized for cross-symbol comparability)
11. imbalance_L1:      (bid_notional_L1 - ask_notional_L1) / (bid_notional_L1 + ask_notional_L1)  — top of book imbalance (notional, not raw qty)
12. imbalance_L5:      weighted_sum(bid_notional_L1:5 - ask_notional_L1:5) / sum — near book imbalance (inverse-level weighted: 1.0, 0.5, 0.33, 0.25, 0.2; uses up to 5 of 10 available levels)
13. depth_ratio:       log(max(total_bid_notional, 1e-6) / max(total_ask_notional, 1e-6)) — deep book asymmetry in dollar terms (epsilon-guarded, Cont fix round 4: notional not raw qty for cross-symbol comparability)
14. trade_vs_mid:      clip((event_vwap - mid) / max(spread, 1e-8 * mid), -5, 5) — where in spread this event executed (clipped, zero-spread guarded). Also serves as continuous direction proxy (positive = executed at ask = buy-side; negative = at bid = sell-side)
15. delta_imbalance_L1: imbalance_L1 - prev_event_imbalance_L1            — book motion between events (carry-forward between snapshots; ~90% zero at 24s cadence with ~10.6 events/snapshot — sparsity is informative; Cont fix)
16. kyle_lambda:       Cov(Δmid_snapshot, cum_signed_notional_snapshot) / Var(cum_signed_notional_snapshot) over rolling 50 OB snapshots (~20 min), forward-filled to events — market maker's price updating coefficient (information regime indicator). Per-SNAPSHOT computation, not per-event (council round 5: event-level had ~2 effective observations per window)
17. cum_ofi_5:         rolling sum of OFI (Cont 2014, piecewise formula) over last 5 book snapshots (~120s at 24s cadence), normalized by rolling 5-snapshot total notional volume, forward-filled to events — dimensionless rolling order flow imbalance. Sweep {3, 5, 10} after baseline.
```

**Notes on orderbook features (rounds 3-5):**
- **OB cadence is ~24 seconds** (measured), not ~3s. 10 levels per side (measured), not 25. ~10.6 order events per snapshot on BTC. (data-eng-13 fix, round 5)
- `log_spread` normalized by mid price for cross-symbol/cross-regime comparability (Cont fix).
- `imbalance_L1/L5` use notional (qty × price) not raw qty — a 1 BTC bid ≠ 100 SOL bid in dollar terms. L5 uses inverse-level weighting (L1 is 5-10x more predictive than L5). Uses up to 5 of 10 available levels.
- `depth_ratio` epsilon-guarded — one-sided book during flash crashes produces log(0) = -inf (practitioner fix). **Round 4 fix:** uses notional (qty × price), not raw qty (Cont fix).
- `trade_vs_mid` clipped to [-5, 5] and zero-spread guarded (practitioner fix). **Round 5:** now serves double duty as continuous direction proxy (replaces dropped `is_buy`). Positive = executed above mid (buy-side), negative = below mid (sell-side). This is the Lee-Ready classification in continuous form.
- `delta_imbalance_L1` will be ~90% zero due to 24s snapshot cadence vs ~2.3s event rate (~10.6 events/snapshot). Fix: carry forward the last non-zero delta value for all events between two snapshots. The block structure (10 identical non-zero deltas followed by zeros) is learnable by the CNN's dilated architecture. Day boundaries: pre-warm from prior day or mask first event (practitioner fix).
- `kyle_lambda` **redesigned (council round 5):** switched from per-event to **per-snapshot** computation. Event-level had ~2 effective Δmid observations per 50-event window (24s cadence + 59% direction ambiguity = statistically meaningless). Per-snapshot uses `Cov(Δmid_snapshot, cum_signed_notional_snapshot) / Var(cum_signed_notional_snapshot)` over rolling 50 snapshots (~20 min). All 50 observations have potentially non-zero Δmid, giving full effective degrees of freedom. `cum_signed_notional_snapshot` = sum of signed notional across all events between consecutive snapshots (net long vs short sides per snapshot period). Forward-filled to events. Guard: `where(var > 1e-20, cov/var, 0)`. See `docs/council-reviews/2026-04-02-council-3-kyle-lambda.md`.
- `cum_ofi_5` **reduced from 20 to 5 snapshots (council round 5):** at 24s cadence, 20 snapshots = 480s (~8 min), which exceeds the primary 100-event prediction horizon (~300s). 5 snapshots = ~120s, matching the Cont (2014) principle that OFI lookback should bracket the prediction horizon. Sweep {3, 5, 10} after baseline. **Must use piecewise Cont 2014 OFI formula** — at 24s cadence, best bid/ask price changes in 60-80% of snapshot pairs. The naive `delta_notional` formula has the **wrong sign during trending markets**. Correct formula:
  - If `best_bid_price_t > best_bid_price_{t-1}`: `delta_bid = +bid_notional_L1_t`
  - If `best_bid_price_t == best_bid_price_{t-1}`: `delta_bid = bid_notional_L1_t - bid_notional_L1_{t-1}`
  - If `best_bid_price_t < best_bid_price_{t-1}`: `delta_bid = -bid_notional_L1_{t-1}`
  - Mirror for ask side. `OFI_t = delta_bid - delta_ask`.
  See `docs/council-reviews/2026-04-02-council-2-ob-cadence.md`.

**Why these 17:**
- Features 1-6 capture the order event itself (what happened, how aggressively — feature 6 `book_walk` measures intra-order execution range, NOT permanent price impact)
- Feature 7 (effort_vs_result) is Wyckoff's core: volume/price divergence = absorption = reversal
- Feature 8 (climax_score) measures climax intensity — continuous signal for phase transitions (rolling σ, no lookahead)
- Feature 9 (prev_seq_time_span) tells the model the speed of the tape from recent past (no lookahead)
- Features 10-14 capture the static market context (liquidity landscape when it happened). Feature 14 (trade_vs_mid) doubles as continuous direction proxy.
- Feature 15 (delta_imbalance_L1) captures book dynamics — the CHANGE in imbalance is more predictive than the level (Cont)
- Feature 16 (kyle_lambda) captures the information regime — are market makers facing informed or noise flow? Per-snapshot, ~20 min window. (Kyle)
- Feature 17 (cum_ofi_5) captures the rolling pressure from order flow over 5 book updates (~120s). (Cont 2014)

## Sequence Length

**200 order events** (not raw trades).

Determined empirically from autocorrelation analysis of raw trade features:
- `is_open` has half-life of 20 trades (median), persists up to 500
- `log_qty` has half-life of 18 trades (median), persists up to 500
- 200 events captures ~10x the median half-life of the persistent features
- After order event grouping, 200 events covers more clock time than 200 raw trades

Sweep {100, 200, 500} once the pipeline works.

## Label

**Multi-horizon direction** — predict simultaneously at multiple forward horizons:
- 10 events forward (very short term, ~30 sec at measured BTC event rate)
- 50 events forward (short term, ~2.5 min)
- 100 events forward (medium term, ~5 min) — **primary horizon**
- 500 events forward (longer term, ~25 min)

Binary per horizon: did price go up or down?

**Multi-task loss:** weighted sum of binary cross-entropy across 4 horizons (weights: 0.10/0.20/0.35/0.35 for 10/50/100/500 events). This forces the model to learn representations that work at multiple timescales while prioritizing the more signal-rich longer horizons — the 10-event horizon is dominated by bid-ask bounce noise.

**Output:** 4 sigmoid heads, one per horizon.

## Mandatory Linear Baseline

**Before training ANY neural network, fit logistic regression on the same data.**

Flatten the (200, 17) input to 3400 features, fit logistic regression per horizon. **Sweep regularization C ∈ {0.001, 0.01, 0.1, 1.0}** — at 3400 features, default C=1.0 will overfit. Report best C per horizon, and report both train AND test accuracy separately (if train >> test, the baseline itself is overfitting). (Lopez de Prado/practitioner fix, round 3)

**Per-symbol go/no-go gate at the primary horizon (100 events)** (Lopez de Prado fix, round 4):

At N≈6,000 samples per symbol per test fold, the Bonferroni-corrected (4 horizons) significance threshold is ~51.4%. The previous 50.5% threshold was inside the null band and would pass pure noise.

- If fewer than **15/25 symbols** achieve logistic regression accuracy above **51.4%** at the primary horizon: **STOP.** The signal does not exist broadly enough, and a neural network will just overfit to a few symbols.
- If 15-19 symbols exceed 51.4%: signal exists but is not universal. Proceed cautiously, investigate failing symbols.
- If 20+ symbols exceed 51.4%: clear broad signal, neural network should improve further.

Report both train AND test accuracy separately per symbol per horizon (if train >> test, the baseline itself is overfitting).

## Architecture

### Option A: Dilated 1D CNN (simplest, try first)

```
Input: (batch, seq_len=200, 17)       — 200 order events, 17 features each

BatchNorm1d(17)                        — normalize input features (keeps running stats for eval)
Conv1d(17 → 32, kernel=5, dilation=1)  — local patterns (RF=5)
LayerNorm + ReLU + Dropout(0.1)
Conv1d(32 → 64, kernel=5, dilation=2)  — RF=13 cumulative
LayerNorm + ReLU + Dropout(0.1)
Conv1d(64 → 64, kernel=5, dilation=4)  — RF=29 cumulative       + residual
LayerNorm + ReLU
Conv1d(64 → 64, kernel=5, dilation=8)  — RF=61 cumulative       + residual
LayerNorm + ReLU
Conv1d(64 → 64, kernel=5, dilation=16) — RF=125 cumulative      + residual
LayerNorm + ReLU
Conv1d(64 → 64, kernel=5, dilation=32) — RF=253 cumulative      + residual
LayerNorm + ReLU
concat[GlobalAvgPool, last_position]    — (batch, 128): GAP preserves gradient flow,
                                          last_position has full RF=253 (recency signal)
Linear(128 → 64) + ReLU                — shared neck
[Linear(64 → 1)] × 4, sigmoid          — per-horizon predictions

~91K parameters (was ~94K with 18 features)
```

**Round 3 architecture fixes:**
- **6 dilated layers** (not 3): original spec had RF=29, only 14.5% of the 200-event window. The model literally could not see the full sequence. Dilations 1,2,4,8,16,32 give RF=253 ≥ 200. (DL Researcher fix)
- **LayerNorm** replaces BatchNorm in conv body: BN accumulates running statistics during training; if market regime shifts between train and test, those statistics are stale → silent degradation. LayerNorm normalizes per-sample, making it regime-invariant. Input BatchNorm1d(17) is kept because it handles feature-scale heterogeneity. (DL Researcher fix)
- **Dropout(0.1)** on first two conv layers: binary direction labels are noisy; without dropout the model memorizes noise. Only on first two layers — deep layers see more abstract features that need less regularization. (DL Researcher fix)
- **17 input features** (round 5): `is_buy` dropped, reducing input dim from 18 to 17. ~91K parameters (was ~94K).

**Round 4 architecture fixes (DL Researcher):**
- **concat[GAP, last_position]** replaces pure GlobalAvgPool as the default (not upgrade path). The last position already has RF=253 (sees all 200 events) — it is the CNN's richest summary. GAP preserves gradient flow to all positions; last_position preserves recency. Cost: +256 params.
- **Residual connections for layers 3-6** (same 64-channel): zero extra parameters, preserves local patterns from early layers through large-dilation layers. No residual for layers 1-2 (channel dimension is changing 17→32→64).
- **Shared neck** `Linear(128→64) + ReLU` before per-horizon heads: adds non-linear feature combinations before projecting to each horizon. 4 independent `Linear(64→1)` heads replace single `Linear(64→4)`.
- **Upgrade path:** Attention pooling in iteration 2.

Why dilated instead of strided: strided convolutions downsample and lose temporal resolution. Dilated convolutions increase the receptive field (how far back the model can see) without throwing away events. Each layer sees a wider context while keeping all 200 positions.

### Option B: Transformer (if CNN shows signal at > 51%)

```
Input: (batch, seq_len=200, 17)

Linear(17 → 128)                      — project to model dim (64 was too small: 16 dim/head → noisy attention)
RoPE(128)                             — rotary positional embeddings (relative position > absolute for financial sequences)
Pre-norm TransformerEncoder(128, 4 heads, 2 layers)  — pre-norm = LayerNorm before attention, better gradient flow
CLS token pooling → Linear(128 → 4, sigmoid)

~350K parameters
```

**Do NOT build until CNN validates signal.** The v7 Transformer (window=2000, H100) achieved Sortino=0.061 — temporal architectures do not reliably outperform simpler models at this data scale.

### Option C: GRU (baseline sequential)

```
Input: (batch, seq_len=200, 17)

GRU(17 → 64, 2 layers, bidirectional=False)
Last hidden state → Linear(64 → 4, sigmoid)

~60K parameters
```

**GRU replaces LSTM** (round 3 fix): ~30% fewer parameters, faster training, comparable performance on sequences < 500 steps. bidirectional=False is correct (causal for online prediction). Note: 2-layer GRU on seq_len=200 has gradient flow challenges for early positions — CNN avoids this by construction.

## Training Strategy

### Data Loading

- Group raw trades into order events (same-timestamp trades = one event)
- **Dedup raw trades** before grouping (see Order Event Grouping section)
- Each training sample: 200 consecutive order events with aligned orderbook context
- Label: direction at 4 forward horizons (10, 50, 100, 500 events)
- Samples drawn from ALL symbols — no per-symbol distinction during training
- **Stride = 200** (non-overlapping input windows). The first window's start position is randomized per epoch to avoid alignment artifacts. Adjacent non-overlapping samples from the same symbol-day have bounded 500-event label correlation (60% overlap at the 500-event horizon) — acknowledged and does not affect walk-forward evaluation. (Practitioner fix, round 4)
- Shuffle across symbols and dates each epoch
- Orderbook alignment: precompute per-symbol per-day to avoid repeated joins

### Scale

Per symbol per day: ~140K trades → ~28K order events (after dedup + grouping; 3.8-8.6x grouping ratio varies by symbol) → ~141 non-overlapping samples of 200 events (BTC measured)
Total: 25 symbols × 160 days × ~140 samples ≈ **400-560K training samples** (measured, council round 5 — previous 1.2M estimate was pre-dedup)
Each sample: 200 events × 17 features = 3400 floats

This fits in memory on a single H100 (80GB). Can stream from disk if needed.

**Note:** All 25 symbols produce ~74K raw trades/day regardless of liquidity (suggests Pacifica API per-day collection cap). The grouping ratio varies: BTC/ETH/SOL ~4x, DOGE/AVAX ~8x. Illiquid symbols have higher grouping ratios.

### Training Loop

- Multi-task binary cross-entropy loss with **asymmetric horizon weighting**: `L = 0.10 * L_10 + 0.20 * L_50 + 0.35 * L_100 + 0.35 * L_500`. The 10-event horizon is dominated by bid-ask bounce (near-random); equal weighting lets noise gradients degrade the shared trunk. The 100-event and 500-event horizons carry the most theoretically meaningful signal (information half-life window). Treat weights as hyperparameters adjustable after linear baseline reveals per-horizon signal strength. (Kyle/Wyckoff/DL Researcher fix, round 3)
- **Horizon-specific label smoothing** (DL Researcher fix, round 4): `smoothed_target = target * (1 - ε) + ε/2`. Epsilon per horizon: 0.10 (10-event, noisiest), 0.08 (50-event), 0.05 (100-event), 0.05 (500-event). Prevents overconfidence on noisy labels without changing the loss function.
- AdamW(weight_decay=1e-4), **OneCycleLR** with max_lr=3e-4, pct_start=0.3 (30% warmup), cosine annealing. Warmup is critical while input BatchNorm statistics stabilize in early training. Sweep only max_lr ∈ {1e-4, 3e-4}; 1e-3 is too large for AdamW at this scale. (DL Researcher fix, round 4)
- Batch size: 256-1024 (GPU memory dependent)
- Epochs: 10-50 (monitor val loss for overfitting)
- **Training augmentation** (DL Researcher fix, round 4): (a) Additive Gaussian noise (σ = 0.05 × feature_std) on continuous features — not on is_open (bounded [0,1]). (b) Orderbook feature dropout: zero features 10-17 with p=0.15. Teaches the model to predict from trade features alone when book is stale.
- No class weighting initially (check base rate first)
- No recency weighting initially
- Input BatchNorm(17) handles feature scale differences; body uses LayerNorm for regime invariance. **Assert `model.eval()` for the entire test pass** — any call to `model.train()` mid-evaluation updates running stats with test data.

### Validation

- Walk-forward with **expanding training window** (Lopez de Prado fix, round 4): start with minimum 80 training days, add ~40 days per fold. **Realistically 2-3 truly independent test periods** over 160 days (the previous "minimum 4 folds" claim was arithmetically inconsistent with 120d train / 20d test — acknowledged). The last 20 days (Mar 5-25) overlap with the main branch's test period which has been used for 20+ experiments — treat that window with skepticism.
- **April hold-out (DESIGNATED 2026-04-02, before any April data viewed):** April 14+ is the untouched final evaluation set. April 1-13 may be used for development validation. This designation is irrevocable — no April 14+ data may be viewed, even informally, until the final model evaluation. (Lopez de Prado fix, round 4)
- **Embargo zones:** **600-event gap** between end of training data and start of test data at each fold boundary. The 500-event forward label for the last training sample peeks into the test period without this; the extra 100-event buffer absorbs off-by-one errors from random-offset sampling. (Lopez de Prado fix, rounds 3+4)
- **Pre-designated primary horizon: 100 events.** This is the go/no-go horizon. Report all four horizons, but power the statistical test on horizon-100 only. The other three are exploratory and must be labeled as such. This prevents cherry-picking the "best" horizon from 4 tests. (Lopez de Prado fix, round 3)
- Primary metric: **accuracy on ALL 25 symbols at the primary horizon** (universal, not cherry-picked)
- Secondary: per-symbol accuracy variance (low = universal, high = symbol-specific). Variance > 3% suggests the exchangeability assumption fails — evaluate per-liquidity-cluster (liquid: BTC/ETH/SOL; mid: AVAX/LINK/UNI; illiquid: FARTCOIN/PUMP/XPL).
- **Hold-out symbol test:** Exclude 1 symbol entirely from training (e.g., FARTCOIN), test exclusively on it. If the model fails on the held-out symbol, it learned symbol-specific patterns, not universal microstructure. (Kyle fix, round 3)
- **Multiple testing correction:** The declared 100 tests (4 horizons × 25 symbols) understate the true trial count by ~10-15x when hyperparameter sweeps are included (C sweep ≈ 400, lr sweep ≈ 300, architecture comparison ≈ 300, sequence length sweep ≈ 300, total ≈ 1,600). **Maintain `trial_log.csv`** (date, config, architecture, hyperparams, train_acc, test_acc per horizon) from experiment 1. Use total row count as T in DSR formula. Apply Holm-Bonferroni correction to the full trial set when reporting statistical significance. Designate primary architecture (CNN) before any data is viewed — do not evaluate alternative architectures on the same test set. (Lopez de Prado fix, rounds 3+4)
- Success: accuracy > 52% consistently across 20+ symbols **at the primary horizon (100 events)**

## Evaluation Protocol

### Phase 0: Label Validation (before anything else)
- Compute base rate and mean absolute return at each horizon
- If label is noise: stop or redesign labels

### Phase 0.5: Linear Baseline
- Logistic regression on flattened (200×17=3400) features, sweep C ∈ {0.001, 0.01, 0.1, 1.0}
- Report train vs test accuracy separately (watch for overfitting)
- If < 50.5% on all horizons: stop
- This is the floor the neural network must beat

### Phase 1: Prediction Accuracy (does it read the tape?)

| Metric | Target | Meaning |
|--------|--------|---------|
| Mean accuracy at **primary horizon (100 events)** across all 25 symbols | > 52% | Universal signal at the pre-designated horizon |
| Accuracy std across symbols | < 2% | Consistent reading (> 3% suggests non-exchangeability) |
| Symbols with accuracy > 51% at primary horizon | > 20/25 | Broad coverage |
| Held-out symbol accuracy (not in training) | > 51% | True universality, not memorization |
| All results | Holm-Bonferroni corrected | Accounts for 100 simultaneous tests |

### Phase 2: Trading Performance (can it trade?)

Only after Phase 1 targets are met:
- Add fee model and position sizing
- Evaluate Sortino, but across ALL symbols, not just winners
- Compare to current baseline (0.353 on 9/23)
- **Compute Deflated Sharpe Ratio (DSR)** — adjusts for number of trials, skewness, kurtosis. Without DSR, reported Sharpe is optimistically biased by an unknown amount. PSR < 0.95 means insufficient confidence in the Sharpe estimate. (Lopez de Prado fix, round 3)

## Implementation Plan

### Step 0: Label Validation + Data Validation (local, 15 min)
- Compute base rate, mean absolute return, and conditional return distribution (top decile/quartile) at 10/50/100/500 event horizons
- **Same-timestamp assumption validation:** compute distribution of (a) fill count per timestamp, (b) same-timestamp fills with mixed sides, (c) same-timestamp fills spanning > 3 price levels. **Expect 59% mixed-side events** (exchange mechanic, not grouping error — council round 5). The 5% threshold from round 3 is obsolete.
- **Dedup validation:** assert dedup rate 40-60% per symbol-day. Validate residual same-`(qty,price)` duplicates < 5% on illiquid symbols. Check post-dedup side distribution ~50/50.
- **OB cadence validation:** assert median snapshot interval in [15s, 35s]. Assert events before first snapshot are handled (zero-fill OB features).
- Deliverable: `scripts/test_label_signal.py`, `scripts/validate_event_grouping.py`

### Step 1: Data Pipeline (local CPU)

Build a PyTorch Dataset that:
1. Loads raw trade parquet files per symbol per day
2. **Deduplicates:** pre-April `drop_duplicates(subset=['ts_ms', 'qty', 'price'])`, April+ filter to `event_type == 'fulfill_taker'`
3. Groups same-timestamp trades into order events (validated in Step 0)
4. Aligns each event with nearest-prior orderbook snapshot — **vectorized `np.searchsorted` over all events at once**, not Python for-loop. **Guard `ob_idx >= 0`** — events before first snapshot get zero-filled OB features (56s gap measured on BTC day 1).
5. Computes the 17 per-event features (9 trade + 8 book-context)
6. Returns (sequence, label) tuples of shape (200, 17) and (4,)
6. Draws samples across all symbols and dates
7. Precomputes aligned features per symbol-day and caches to disk (.npz) to avoid reprocessing
8. **Pre-warms rolling windows** (median_event_qty, climax σ) from end of prior day. For the first calendar day per symbol (2025-10-16), mask the first 1000 events from training samples. **Do not mask other days** — masking loses 4M day-open events and systematically biases against day-open dynamics. Pre-warming is causally clean (past quantities, not future prices). All rolling windows (median, σ) must advance together in a single pass through the sorted event stream per symbol-day. (Practitioner fix, round 4)
9. **Handles day boundaries** for delta_imbalance_L1 (pre-warm from prior day or mask first event)

Deliverable: `tape_dataset.py`

### Step 1.5: Linear Baseline (local, 10 min)
- Flatten (200, 17) → 3400 features
- Logistic regression per horizon, per symbol, sweep C ∈ {0.001, 0.01, 0.1, 1.0}
- Report train vs test accuracy separately
- Deliverable: accuracy report, go/no-go decision

### Step 2: Prototype on CPU (local, 1 symbol, 1 day)

- Train 1D CNN on BTC, single day, to verify the pipeline works
- Check: loss decreases, no NaN, shapes are correct
- No performance expectations — just engineering validation

Deliverable: `tape_train.py` with working training loop

### Step 3: Full Training on RunPod H100

- Train on all 25 symbols, all 160 days
- lr sweep, architecture comparison (CNN vs GRU vs Transformer if warranted)
- Report accuracy per symbol per horizon

Deliverable: trained model + accuracy report

### Step 4: Analysis

- Which symbols does it predict best? Worst? Why?
- Which horizon has the most signal?
- What patterns has it learned? (gradient-based feature attribution)
- Does accuracy vary by time of day? By volatility regime?

## Compute Requirements

| Step | Hardware | Time |
|------|----------|------|
| Label + data validation | Local CPU | ~15 min |
| Data pipeline | Local CPU | ~3-5 hours (vectorized OB alignment) |
| Linear baseline | Local CPU | ~10 min |
| Prototype (1 symbol, 1 day) | Local CPU | ~10 min |
| Full training (CNN) | RunPod H100 | ~1-2 hours |
| Full training (Transformer) | RunPod H100 | ~2-4 hours |
| Analysis | Local CPU | ~30 min |

## Risks

1. **Label noise.** Binary direction over short horizons is inherently noisy. Multi-horizon prediction mitigates this — if 500-event horizon has signal but 10-event doesn't, we learn the right timescale.

2. **Order event grouping may lose information.** If two different market participants trade at the same millisecond, we group them as one event. This is rare but possible. Monitor the distribution of fills-per-event.

3. **Orderbook staleness.** Orderbook snapshots every ~24 seconds (measured) means ~10.6 events share the same book state. The book features will be step-wise constant, which is correct but means the model can't see within-snapshot book changes. The 24s cadence may actually filter HF quote noise (council-2, round 5).

4. **Overfitting to symbol-specific characteristics.** Even with relative features, different symbols have different volatility, spread, and depth distributions. The model might learn "this looks like a BTC event" rather than "this looks like aggressive buying." Monitor per-symbol accuracy variance.

5. **400-560K samples may not be enough.** After dedup + grouping, we have ~400-560K sequences (was 1.2M pre-dedup). Financial data is noisier than ImageNet. Overfitting is a real risk — keep models small (~91K params).

## What Success Looks Like

> A model that, given 200 order events from ANY symbol it has never seen (including symbols excluded from training), can predict the next-100-event direction at > 52% accuracy with statistical significance after multiple-testing correction. Not because it memorized BTC patterns, but because it learned what aggressive accumulation, liquidation cascades, and informed flow look like — universally.

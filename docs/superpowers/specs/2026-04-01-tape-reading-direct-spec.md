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

Script: compute next-100-trade direction for all symbols across all dates. Report:
- Up/down split (if 50.1%/49.9%, the label is pure noise)
- Mean absolute return over 100 trades (if < 1 bps, moves are too small to be meaningful)
- Compare horizons: 10, 50, 100, 500 trades forward

If the base rate is within 50 ± 0.5% AND mean absolute return is < 2 bps, the label is too noisy. Consider longer horizons or different labels (e.g., threshold-based: up > 5 bps, down < -5 bps, flat otherwise).

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

Per order event:
```
1. log_return:      log(vwap / prev_event_vwap)
2. log_total_qty:   log(total_qty / median_event_qty)
3. is_buy:          1 if buy side, 0 if sell
4. is_open:         fraction of fills that are opens (0 to 1)
5. time_delta:      log(ts - prev_event_ts + 1)
6. num_fills:       log(number of fills in this event)
7. price_impact:    (last_fill_price - first_fill_price) / mid — how much this order moved the price
```

This reduces noise (one order = one event, not multiple rows) and captures order-level signals (an order that fills across 5 price levels is more aggressive than one that fills at a single price).

**Sequence length adjustment:** With order events, 200 events covers more clock time than 200 raw trades. This is better — each event is a real decision, not a matching engine artifact.

## Input Representation

Per order event, 12 features from 2 data sources (trades + orderbook):

**From trade events (7 features):**
```
1. log_return:      log(vwap / prev_event_vwap)          — relative price change
2. log_total_qty:   log(total_qty / median_event_qty)     — relative order size
3. is_buy:          1 if buy, 0 if sell
4. is_open:         fraction of fills that are opens [0,1] — position flow signal
5. time_delta:      log(ts - prev_event_ts + 1)           — urgency
6. num_fills:       log(fill count)                        — order complexity
7. price_impact:    (last_fill - first_fill) / mid         — how much the order walked the book
```

**From orderbook (aligned by nearest prior snapshot, 5 features):**
```
8.  log_spread:      log(best_ask - best_bid + 1e-10)                   — current spread
9.  imbalance_L1:    (bid_qty_L1 - ask_qty_L1) / (bid_qty_L1 + ask_qty_L1)  — top of book imbalance
10. imbalance_L5:    (bid_qty_L1:5 - ask_qty_L1:5) / (bid_qty_L1:5 + ask_qty_L1:5)  — near book imbalance
11. depth_ratio:     log(total_bid_qty / total_ask_qty)                  — deep book asymmetry
12. trade_vs_mid:    (event_vwap - mid) / spread                        — where in spread this event executed
```

**Why level-separated imbalance:**
- `imbalance_L1` (best bid/ask only): most predictive, changes fastest, captures immediate liquidity
- `imbalance_L5` (top 5 levels): captures near-term supply/demand
- `depth_ratio` (all levels): captures structural positioning

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
- 10 events forward (very short term, ~0.5-1 sec)
- 50 events forward (short term, ~2-5 sec)
- 100 events forward (medium term, ~5-10 sec)
- 500 events forward (longer term, ~30-60 sec)

Binary per horizon: did price go up or down?

**Multi-task loss:** sum of binary cross-entropy across all 4 horizons. This forces the model to learn representations that work at multiple timescales — better features than any single horizon.

**Output:** 4 sigmoid heads, one per horizon.

## Mandatory Linear Baseline

**Before training ANY neural network, fit logistic regression on the same data.**

Flatten the (200, 12) input to 2400 features, fit logistic regression per horizon. Report accuracy.

If logistic regression achieves < 50.5% accuracy on all horizons across all symbols: **STOP.** The signal does not exist at this granularity, and a neural network will just overfit.

If logistic regression achieves 50.5-51% on some horizons: signal exists but is weak. Neural network may extract more, proceed cautiously.

If logistic regression achieves > 51%: clear signal, neural network should improve further.

## Architecture

### Option A: 1D CNN (simplest, try first)

```
Input: (batch, seq_len=200, 12)      — 200 order events, 12 features each

BatchNorm1d(12)                       — normalize across features
Conv1d(12 → 32, kernel=5, stride=1)   — local patterns (5-event motifs)
BatchNorm1d(32) + ReLU
Conv1d(32 → 64, kernel=5, stride=2)   — wider patterns, downsample
BatchNorm1d(64) + ReLU
Conv1d(64 → 64, kernel=5, stride=2)   — wider still
BatchNorm1d(64) + ReLU
GlobalAvgPool                          — compress to fixed length
Linear(64 → 4, sigmoid)               — 4 horizon predictions

~60K parameters
```

### Option B: Transformer (if CNN shows signal)

```
Input: (batch, seq_len=200, 12)

Linear(12 → 64)                       — project to model dim
PositionalEncoding(64)                 — sequence position
TransformerEncoder(64, 4 heads, 2 layers)
CLS token pooling → Linear(64 → 4, sigmoid)

~200K parameters
```

### Option C: LSTM (baseline sequential)

```
Input: (batch, seq_len=200, 12)

LSTM(12 → 64, 2 layers, bidirectional=False)
Last hidden state → Linear(64 → 4, sigmoid)

~80K parameters
```

## Training Strategy

### Data Loading

- Group raw trades into order events (same-timestamp trades = one event)
- Each training sample: 200 consecutive order events with aligned orderbook context
- Label: direction at 4 forward horizons (10, 50, 100, 500 events)
- Samples drawn from ALL symbols — no per-symbol distinction during training
- Random offset within each day to avoid alignment artifacts
- Shuffle across symbols and dates each epoch
- Orderbook alignment: precompute per-symbol per-day to avoid repeated joins

### Scale

Per symbol per day: ~140K trades → ~50-70K order events → ~300 non-overlapping samples of 200 events
Total: 25 symbols × 160 days × 300 samples ≈ **1.2M training samples**
Each sample: 200 events × 12 features = 2400 floats

This fits in memory on a single H100 (80GB). Can stream from disk if needed.

### Training Loop

- Multi-task binary cross-entropy loss (sum across 4 horizons)
- AdamW, lr sweep {1e-4, 3e-4, 1e-3}
- Batch size: 256-1024 (GPU memory dependent)
- Epochs: 10-50 (monitor val loss for overfitting)
- No class weighting initially (check base rate first)
- No recency weighting initially
- BatchNorm handles feature scale differences

### Validation

- Walk-forward: train on first 120 days, validate on next 20 days, test on last 20 days
- Primary metric: **accuracy on ALL 25 symbols** (universal, not cherry-picked)
- Secondary: per-symbol accuracy variance (low = universal, high = symbol-specific)
- Report accuracy per horizon (which timescale has most signal?)
- Success: accuracy > 52% consistently across 20+ symbols at any horizon

## Evaluation Protocol

### Phase 0: Label Validation (before anything else)
- Compute base rate and mean absolute return at each horizon
- If label is noise: stop or redesign labels

### Phase 0.5: Linear Baseline
- Logistic regression on flattened (200×12=2400) features
- If < 50.5% on all horizons: stop
- This is the floor the neural network must beat

### Phase 1: Prediction Accuracy (does it read the tape?)

| Metric | Target | Meaning |
|--------|--------|---------|
| Mean accuracy (all 25 symbols) | > 52% | Universal signal |
| Accuracy std across symbols | < 2% | Consistent reading |
| Symbols with accuracy > 51% | > 20/25 | Broad coverage |
| Best horizon accuracy | > 53% | At least one timescale has clear signal |

### Phase 2: Trading Performance (can it trade?)

Only after Phase 1 targets are met:
- Add fee model and position sizing
- Evaluate Sortino, but across ALL symbols, not just winners
- Compare to current baseline (0.353 on 9/23)

## Implementation Plan

### Step 0: Label Validation (local, 5 min)
- Compute base rate and mean absolute return at 10/50/100/500 trade horizons
- Deliverable: `scripts/test_label_signal.py`

### Step 1: Data Pipeline (local CPU)

Build a PyTorch Dataset that:
1. Loads raw trade parquet files per symbol per day
2. Groups same-timestamp trades into order events
3. Aligns each event with nearest-prior orderbook snapshot (np.searchsorted on timestamps)
4. Computes the 12 per-event features (7 trade + 5 book-context)
5. Returns (sequence, label) tuples of shape (200, 12) and (4,)
6. Draws samples across all symbols and dates
7. Precomputes aligned features per symbol-day and caches to disk (.npz) to avoid reprocessing

Deliverable: `tape_dataset.py`

### Step 1.5: Linear Baseline (local, 10 min)
- Flatten (200, 12) → 2400 features
- Logistic regression per horizon, per symbol
- Deliverable: accuracy report, go/no-go decision

### Step 2: Prototype on CPU (local, 1 symbol, 1 day)

- Train 1D CNN on BTC, single day, to verify the pipeline works
- Check: loss decreases, no NaN, shapes are correct
- No performance expectations — just engineering validation

Deliverable: `tape_train.py` with working training loop

### Step 3: Full Training on RunPod H100

- Train on all 25 symbols, all 160 days
- lr sweep, architecture comparison (CNN vs LSTM vs Transformer)
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
| Label validation | Local CPU | ~5 min |
| Data pipeline | Local CPU | ~2-3 hours |
| Linear baseline | Local CPU | ~10 min |
| Prototype (1 symbol, 1 day) | Local CPU | ~10 min |
| Full training (CNN) | RunPod H100 | ~1-2 hours |
| Full training (Transformer) | RunPod H100 | ~2-4 hours |
| Analysis | Local CPU | ~30 min |

## Risks

1. **Label noise.** Binary direction over short horizons is inherently noisy. Multi-horizon prediction mitigates this — if 500-event horizon has signal but 10-event doesn't, we learn the right timescale.

2. **Order event grouping may lose information.** If two different market participants trade at the same millisecond, we group them as one event. This is rare but possible. Monitor the distribution of fills-per-event.

3. **Orderbook staleness.** Orderbook snapshots every ~3 seconds means many events share the same book state. The book features will be step-wise constant, which is correct but means the model can't see within-snapshot book changes.

4. **Overfitting to symbol-specific characteristics.** Even with relative features, different symbols have different volatility, spread, and depth distributions. The model might learn "this looks like a BTC event" rather than "this looks like aggressive buying." Monitor per-symbol accuracy variance.

5. **1.2M samples may not be enough.** ImageNet has 1.2M images. We have 1.2M sequences. It's the same ballpark but financial data is noisier. Overfitting is a real risk — keep models small.

## What Success Looks Like

> A model that, given 200 order events from ANY symbol it has never seen, can predict the next-100-event direction at > 52% accuracy. Not because it memorized BTC patterns, but because it learned what aggressive accumulation, liquidation cascades, and informed flow look like — universally.

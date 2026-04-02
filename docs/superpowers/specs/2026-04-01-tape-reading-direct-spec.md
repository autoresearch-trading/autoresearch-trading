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
Raw trades (40GB) → sequential model → learned representations → direction
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

## Input Representation

Per trade, 10 features from 2 data sources (trades + orderbook):

**From trades (per trade):**
```
1. log_return:      log(price / prev_price)         — relative price change
2. log_qty:         log(qty / median_qty)            — relative size
3. is_buy:          1 if buy (open_long/close_short), 0 if sell
4. is_open:         1 if opening position, 0 if closing
5. time_delta:      log(ts - prev_ts + 1)            — time between trades (log-scaled)
```

**From orderbook (aligned to each trade by nearest prior snapshot):**
```
6. log_spread:      log(best_ask - best_bid + 1e-10) — spread at time of trade
7. book_imbalance:  (bid_depth - ask_depth) / (bid_depth + ask_depth)  — who has more size
8. bid_depth_norm:  log(total_bid_qty / median_depth) — how thick is the bid
9. ask_depth_norm:  log(total_ask_qty / median_depth) — how thick is the ask
10. trade_vs_mid:   (trade_price - mid) / spread      — where in the spread did this trade execute
```

**Why these 10:**
- Features 1-5 capture the trade itself (what happened)
- Features 6-10 capture the market context (what the book looked like when it happened)
- A buy hitting a thin ask (low ask_depth_norm, high trade_vs_mid) is aggressive — the model needs to see this
- All features are relative/normalized — no absolute prices or sizes that differ across symbols

**Orderbook alignment:** Each trade gets the most recent orderbook snapshot where `ob.ts_ms <= trade.ts_ms`. Orderbook is sampled every ~3 seconds, trades happen every ~10-20ms, so many consecutive trades share the same book state. This is correct — the book context doesn't change between snapshots.

**Not included (only available from Apr 1, not 160 days):**
- `cause` (liquidation flag) — would be very valuable but only 1 day of data
- `event_type` (taker/maker) — same
- `open_interest` — same
- `funding_rate` — available for 160 days but updates hourly, too slow to be per-trade context. Could be added later as a regime feature.

## Label

**Next-N-trade direction** (binary: did price go up or down over next N trades?)
- N=100 matches our current batch granularity
- Start simple with binary (up/down), not 3-class (flat/long/short)
- No fee-adjusted triple barrier — first learn to read the tape, then worry about profitability

## Architecture

### Option A: 1D CNN (simplest, try first)

```
Input: (batch, seq_len=1000, 10)     — 1000 consecutive trades, 10 features each

Conv1d(10 → 32, kernel=5, stride=1)  — local patterns (5-trade motifs)
Conv1d(32 → 64, kernel=5, stride=2)  — wider patterns, downsample
Conv1d(64 → 64, kernel=5, stride=2)  — wider still
GlobalAvgPool                         — compress to fixed length
Linear(64 → 1, sigmoid)              — binary direction

~55K parameters
```

Why 1D CNN first:
- Fast to train (parallelizable, unlike LSTM)
- Captures local patterns (sweeps, bursts, clusters)
- Proven on audio waveforms which have similar properties to trade tapes
- Can run on CPU for prototyping, GPU for full scale

### Option B: Transformer (if CNN shows signal)

```
Input: (batch, seq_len=1000, 10)

Linear(10 → 64)                       — project to model dim
PositionalEncoding(64)                 — sequence position
TransformerEncoder(64, 4 heads, 2 layers)
CLS token pooling → Linear(64 → 1, sigmoid)

~200K parameters
```

### Option C: LSTM (baseline sequential)

```
Input: (batch, seq_len=1000, 10)

LSTM(10 → 64, 2 layers, bidirectional=False)
Last hidden state → Linear(64 → 1, sigmoid)

~75K parameters
```

## Training Strategy

### Data Loading

- Each training sample: 1000 consecutive raw trades with aligned orderbook context + label
- For each sample, load trades and align nearest-prior orderbook snapshot per trade
- Samples drawn from ALL symbols — no per-symbol distinction during training
- Random offset within each day to avoid alignment artifacts
- Shuffle across symbols and dates each epoch
- Orderbook alignment: precompute per-symbol per-day to avoid repeated joins

### Scale

Per symbol per day: ~140K trades → ~139 non-overlapping samples of 1000 trades
Total: 25 symbols × 160 days × 139 samples ≈ **556K training samples**
Each sample: 1000 trades × 6 features = 6000 floats

This fits in memory on a single H100 (80GB). Can stream from disk if needed.

### Training Loop

- Binary cross-entropy loss (up/down prediction)
- AdamW, lr sweep {1e-4, 3e-4, 1e-3}
- Batch size: 256-1024 (GPU memory dependent)
- Epochs: 10-50 (monitor val loss for overfitting)
- No class weighting (up/down should be ~50/50)
- No recency weighting initially

### Validation

- Walk-forward: train on first 120 days, validate on next 20 days, test on last 20 days
- Primary metric: **accuracy on ALL 25 symbols** (universal, not cherry-picked)
- Secondary: per-symbol accuracy variance (low = universal, high = symbol-specific)
- Success: accuracy > 52% consistently across 20+ symbols

## Evaluation Protocol

### Phase 1: Prediction Accuracy (does it read the tape?)

| Metric | Target | Meaning |
|--------|--------|---------|
| Mean accuracy (all 25 symbols) | > 52% | Universal signal |
| Accuracy std across symbols | < 2% | Consistent reading |
| Symbols with accuracy > 51% | > 20/25 | Broad coverage |

### Phase 2: Trading Performance (can it trade?)

Only after Phase 1 targets are met:
- Add fee model and position sizing
- Evaluate Sortino, but across ALL symbols, not just winners
- Compare to current baseline (0.353 on 9/23)

## Implementation Plan

### Step 1: Data Pipeline (local CPU)

Build a PyTorch Dataset that:
1. Loads raw trade parquet files per symbol per day
2. Aligns each trade with nearest-prior orderbook snapshot (np.searchsorted on timestamps)
3. Computes the 10 per-trade features (5 trade + 5 book-context)
4. Returns (sequence, label) tuples of shape (1000, 10) and (1,)
5. Draws samples across all symbols and dates
6. Precomputes aligned features per symbol-day and caches to disk (.npz) to avoid reprocessing

Deliverable: `tape_dataset.py`

### Step 2: Prototype on CPU (local, 1 symbol, 1 day)

- Train 1D CNN on BTC, single day, to verify the pipeline works
- Check: loss decreases, no NaN, shapes are correct
- No performance expectations — just engineering validation

Deliverable: `tape_train.py` with working training loop

### Step 3: Full Training on RunPod H100

- Train on all 25 symbols, all 160 days
- lr sweep, architecture comparison (CNN vs LSTM vs Transformer)
- Report accuracy per symbol

Deliverable: trained model + accuracy report

### Step 4: Analysis

- Which symbols does it predict best? Worst? Why?
- What patterns has it learned? (gradient-based feature attribution)
- Does accuracy vary by time of day? By volatility regime?

## Compute Requirements

| Step | Hardware | Time |
|------|----------|------|
| Data pipeline | Local CPU | ~2 hours |
| Prototype (1 symbol, 1 day) | Local CPU | ~10 min |
| Full training (CNN) | RunPod H100 | ~1-2 hours |
| Full training (Transformer) | RunPod H100 | ~2-4 hours |
| Analysis | Local CPU | ~30 min |

## Risks

1. **The tape might be noise at the individual trade level.** Microstructure theory says individual trades are informative, but at 100-trade batches we're already at ~1-2 seconds. Individual trades at ~10-20ms might be below the information threshold.

2. **40GB seems like a lot but 556K samples isn't huge for deep learning.** ImageNet has 1.2M samples. We're in the right ballpark but not abundant.

3. **Label noise.** "Did price go up in the next 100 trades?" is noisy because 100 trades is ~1-2 seconds. Most of the time the answer is noise. May need longer horizons for cleaner labels.

4. **Overfitting to symbol-specific price scales.** Even with log returns, different symbols have different volatility distributions. The model might learn "BTC trades look like X" rather than "aggressive buying looks like X."

## What Success Looks Like

> A model that, given 1000 raw trades from ANY symbol it has never seen, can predict the next-100-trade direction at > 52% accuracy. Not because it memorized BTC patterns, but because it learned what aggressive accumulation, liquidation cascades, and informed flow look like — universally.

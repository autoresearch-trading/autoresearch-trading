# Tape Reading Pivot: Specification

## Goal

Determine whether the raw trade tape contains more predictive signal than our current 13-feature summary statistics, and if so, build a model that learns universal microstructure patterns across all 23 symbols — not per-symbol profit curves.

## Current State

- 13 handcrafted features per 100-trade batch (summary statistics: OFI, VPIN, etc.)
- Flat MLP, Sortino 0.353 on 9/23 symbols (39%)
- The model learns symbol-specific patterns, not universal tape reading
- 100-trade batching destroys sequential structure

## Core Question

**Does trade sequence order matter, or is the summary sufficient?**

If we shuffle trades within each 100-trade batch and the model performs identically, then the tape order contains no extra signal — our current summaries already capture everything. If performance drops, the sequence has information we're leaving on the table.

## Phase 1: Signal Existence Tests (no new model needed)

Simple statistical tests using existing data and infrastructure. Each test answers a specific question.

### Test 1.1: Shuffle Test

**Question:** Does trade ordering within a batch matter?

**Method:**
1. Take current v11b features computed normally → baseline Sortino
2. Shuffle trades randomly within each 100-trade batch before computing features → shuffled Sortino
3. If shuffled ≈ baseline: sequence order doesn't matter, current summaries are sufficient
4. If shuffled < baseline: sequence carries information our features partially capture

**Implementation:** Add a `shuffle_within_batch=True` flag to `compute_features_v9`. Shuffle trade rows within each batch before computing features. Run training, compare.

**Cost:** ~10 min (same model, same pipeline, one flag)

### Test 1.2: Granularity Test

**Question:** Do smaller batches (more steps, less aggregation) help?

**Method:**
1. Current: trade_batch=100 (~1-2 sec per step for BTC)
2. Test: trade_batch=25 (~0.3-0.5 sec per step, 4x more steps)
3. Test: trade_batch=10 (~0.1-0.2 sec per step, 10x more steps)
4. Same 13 features, same MLP, same training — only batch size changes

**What this tells us:**
- If smaller batches help: finer granularity has more signal, worth pursuing
- If smaller batches hurt: aggregation is actually a feature (noise reduction), and we should stay at 100
- If no difference: signal lives at the feature level, not the granularity level

**Implementation:** Change `TRADE_BATCH` constant. Requires cache rebuild (~30 min per batch size). Window size may need adjustment (window=50 at batch=100 covers ~100 sec; window=50 at batch=10 covers ~10 sec — need to keep the same time horizon).

**Cost:** ~40 min per batch size (cache rebuild + training)

**Important:** When changing trade_batch, adjust window_size to maintain the same time coverage:
- batch=100, window=50 → covers ~5000 trades (~100 sec)
- batch=25, window=200 → covers ~5000 trades (~100 sec)
- batch=10, window=500 → covers ~5000 trades (~100 sec)

### Test 1.3: Linear Predictability at Fine Granularity

**Question:** Can a simple linear model predict next-step direction from current features?

**Method:**
1. Compute features at batch=10 (fine granularity)
2. Label: sign of next-step return (binary: up/down)
3. Train logistic regression on the 13 features
4. Report: accuracy per symbol, mean accuracy across all 23 symbols

**What this tells us:**
- Accuracy > 51% on most symbols: linear signal exists at fine granularity
- Accuracy varies wildly across symbols: signal is symbol-specific, not universal
- Accuracy ≈ 50% everywhere: no linear signal at this granularity

**Implementation:** Standalone script, no changes to prepare.py or train.py.

**Cost:** ~5 min (logistic regression is fast)

### Test 1.4: Cross-Symbol Feature Universality

**Question:** Are the feature distributions similar across symbols?

**Method:**
1. For each of the 13 features, compute the distribution (mean, std, skew, kurtosis) per symbol
2. Compare distributions across symbols
3. High similarity → features capture universal microstructure
4. Low similarity → features are symbol-specific, tape reading will struggle

**Implementation:** Standalone analysis script.

**Cost:** ~2 min

## Phase 1 Decision Gate

| Result | Implication | Action |
|--------|-------------|--------|
| Shuffle hurts AND finer granularity helps | Sequence has signal, finer batches capture it | Proceed to Phase 2 |
| Shuffle neutral AND finer granularity helps | Aggregation loses signal, but not from sequence order | Try more features at finer granularity |
| Shuffle neutral AND finer granularity neutral | Current setup extracts all available signal | Stop — model is at ceiling, need more data |
| Shuffle hurts AND finer granularity neutral | Sequence matters but is already captured by features | Investigate which features carry the sequence signal |

## Phase 2: Representation Learning (only if Phase 1 passes gate)

### 2a: Volume Bars (Lopez de Prado)

Replace fixed 100-trade batches with volume-normalized bars. Each bar contains the same dollar volume rather than the same number of trades. This makes bars comparable across symbols (a BTC bar and a DOGE bar represent the same notional activity).

### 2b: Sequential Model on Raw Trades

If Phase 1 shows sequence matters, build a model that processes the raw trade sequence:

**Architecture options (in order of simplicity):**
1. 1D CNN over trades within each batch → flat MLP on top (HybridClassifier already exists)
2. LSTM over trade sequence → classification head
3. Transformer with positional encoding (needs RunPod)

**Input per trade:** (price, qty, is_buy, is_open, time_delta) — 5 features per raw trade
**Sequence length:** 100 trades (one batch)
**Output:** direction prediction (flat/long/short)

### 2c: Self-Supervised Pretraining

Train the sequential model to predict masked trade features (like BERT for trades) across ALL symbols. Then fine-tune the learned representations for direction prediction.

**Why this helps tape reading:**
- Forces the model to learn universal patterns (what does a buy sweep look like?)
- Not biased by profit labels during pretraining
- Should generalize across symbols because the pretraining task is symbol-agnostic

## Phase 2 Evaluation

**Primary metric:** Mean prediction accuracy across ALL 23 symbols (not Sortino of winners)

**Success criteria:**
- Accuracy > 52% on at least 18/23 symbols (universal, not cherry-picked)
- Low variance in accuracy across symbols (< 2% std)
- If this is met, THEN overlay profit evaluation (Sortino, fees, etc.)

## Non-Goals

- Don't optimize for Sortino in Phase 1 or Phase 2a/b — optimize for prediction accuracy
- Don't exclude symbols — the whole point is universal tape reading
- Don't use RunPod until Phase 1 confirms there's signal to capture
- Don't build a complex architecture before proving simple tests show signal exists

## Risk

The biggest risk is that 100-trade summaries already extract all the signal, and the raw tape is pure noise at the individual trade level. Phase 1 tests for this explicitly. If Phase 1 shows no signal, we save weeks of architecture work.

## Dependencies

- Phase 1: only needs current codebase + small scripts
- Phase 2a: modify prepare.py (volume bars)
- Phase 2b: modify train.py (new model architecture)
- Phase 2c: new pretraining script, RunPod for compute

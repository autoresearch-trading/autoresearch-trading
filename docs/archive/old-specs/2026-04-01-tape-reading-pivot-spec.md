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

Statistical tests on the RAW DATA (all 160 days, all 23 symbols). No train/val/test splits — these are data analysis questions, not model evaluations. Use time-series cross-validation where a model is involved.

**Data source:** Raw parquet files in `data/trades/`, `data/orderbook/`, `data/funding/` — all dates, all symbols.

### Test 1.1: Shuffle Test

**Question:** Does trade ordering within a batch matter?

**Method:**
1. Load raw trades for 8 representative symbols (mix of high/low liquidity)
2. Compute 13 features normally at batch=100 across ALL dates
3. Compute 13 features with trades shuffled randomly within each batch
4. Compare feature-return correlations (not Sortino — no model needed)
5. For each feature, compute `corr(feature[t], return[t+1])` in both cases
6. If shuffled correlations ≈ normal correlations: order doesn't matter
7. If shuffled correlations < normal correlations: sequence has signal

**Why correlations, not Sortino:** We want to measure whether the raw features carry more information when order is preserved. This doesn't require training a model — just statistics on the raw data.

**Implementation:** Standalone script `scripts/test_shuffle.py`.

**Cost:** ~5 min (no model training, no cache rebuild)

### Test 1.2: Granularity Test

**Question:** Do smaller batches (more steps, less aggregation) reveal more predictability?

**Method:**
1. Load raw trades for 8 representative symbols, ALL dates
2. Compute features at batch=100, batch=25, batch=10
3. For each granularity, compute feature-return cross-correlation at lag 1
4. Compare: does finer granularity increase or decrease predictability?

**What this tells us:**
- Finer = more predictable: aggregation loses signal, worth pursuing
- Finer = less predictable: aggregation reduces noise, stay at 100
- No difference: signal lives at feature level, not granularity level

**Important:** Compare correlations at the same time horizon. At batch=10, lag-1 covers 10 trades (~0.2 sec). At batch=100, lag-1 covers 100 trades (~2 sec). To compare fairly, also compute lag-10 at batch=10 (same time horizon as lag-1 at batch=100).

**Implementation:** Standalone script `scripts/test_granularity.py`.

**Cost:** ~10 min (feature computation at 3 granularities, no model training)

### Test 1.3: Linear Predictability Across All Symbols

**Question:** Can a simple linear model predict next-step direction, and is this universal?

**Method:**
1. Load raw trades for ALL 23 symbols, ALL dates
2. Compute features at batch=100 (current granularity)
3. Label: sign of next-step return (binary: up/down)
4. Fit logistic regression per symbol using time-series cross-validation (walk-forward: train on first 80%, predict last 20%)
5. Report accuracy per symbol and mean accuracy across all 23

**What this tells us:**
- Accuracy > 51% on 20+ symbols: universal linear signal exists
- Accuracy > 51% on 9 symbols only: signal is symbol-specific (confirms current state)
- Accuracy ≈ 50% everywhere: no linear signal, need nonlinear or sequential model
- Accuracy varies wildly: features are not universal

**Implementation:** Standalone script `scripts/test_linear_predictability.py`.

**Cost:** ~5 min (logistic regression is fast)

### Test 1.4: Cross-Symbol Feature Universality

**Question:** Are the feature distributions similar across symbols?

**Method:**
1. Load raw trades for ALL 23 symbols, ALL dates
2. Compute 13 features at batch=100 (using existing caches where available)
3. For each feature, compute per-symbol statistics: mean, std, skew, kurtosis
4. Compute pairwise Kolmogorov-Smirnov test between symbols for each feature
5. Report: which features are universal (similar distribution across symbols) and which are symbol-specific

**What this tells us:**
- Features with similar distributions across symbols → good for universal tape reading
- Features with different distributions → these are capturing symbol-specific characteristics, not microstructure

**Implementation:** Standalone script `scripts/test_universality.py`.

**Cost:** ~2 min (statistics on cached features)

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

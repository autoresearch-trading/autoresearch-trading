# Representation Learning Pivot — Council Synthesis

**Date:** 2026-04-10
**Reviewers:** Council-2 (Cont), Council-4 (Wyckoff), Council-5 (Practitioner), Council-6 (DL Researcher)
**Question:** How should the project pivot from "predict direction at 52%" to "learn meaningful tape representations"?

---

## Executive Summary

The council unanimously supports the pivot but demands falsifiability. The design:

1. **Self-supervised pretraining** on all 40GB using Masked Event Modeling (block masking) + contrastive regularization
2. **~400K param encoder** (up from 91K) producing 256-dim embeddings
3. **5 pre-registered go/no-go gates** to keep the research falsifiable
4. **Wyckoff-derived self-labels** from the 17 features for evaluation (no human annotation)
5. **Direction prediction becomes a downstream probing task**, not the primary objective

---

## Consensus Decisions

### Training Objective (Council-6, unanimous)

| Component | Weight | Details |
|-----------|--------|---------|
| **Masked Event Modeling** | 0.70 | Block masking (5-event blocks), 15% of events. Reconstruct in BatchNorm-normalized space |
| **SimCLR contrastive** | 0.30 | NT-Xent on global embeddings. Window-start jitter + noise + feature dropout as augmentations |
| **Direction prediction** | 0.00 during pretrain | Only used for probing evaluation and optional fine-tuning AFTER pretraining |

**Critical:** Exclude 3 carry-forward features (`delta_imbalance_L1`, `kyle_lambda`, `cum_ofi_5`) from MEM reconstruction targets — they are trivially predictable from adjacent events via copy.

**NOT recommended:**
- Next-event prediction (requires causal masking, conflicts with current bidirectional CNN)
- Temporal contrastive learning (circular: "same market state" is what we're trying to learn)

### Architecture (Council-6)

```
Input (batch, 200, 17)
BatchNorm(17)
Conv1d(17→64, k=5, d=1) + LayerNorm + ReLU + Dropout(0.1)
Conv1d(64→128, k=5, d=2) + LayerNorm + ReLU + Dropout(0.1)
Conv1d(128→128, k=5, d=4) + LayerNorm + ReLU + residual
Conv1d(128→128, k=5, d=8) + LayerNorm + ReLU + residual
Conv1d(128→128, k=5, d=16) + LayerNorm + ReLU + residual
Conv1d(128→128, k=5, d=32) + LayerNorm + ReLU + residual

MEM branch: Linear(128→17) at masked positions → MSE (14 of 17 features)
Global branch: concat[GAP(128), last_position(128)] → 256-dim embedding
  → Linear(256→256) + ReLU + Linear(256→128) → L2-norm → NT-Xent

~400K parameters total
After pretraining: discard MEM decoder + projection head, keep encoder + 256-dim embedding
```

### What the Model Should Learn (Council-4 Wyckoff + Council-2 Cont)

**10 tape states with observable signatures in the 17 features:**

| State | Key Features | Duration (events) | Fits 200-event window? |
|-------|-------------|-------------------|----------------------|
| Accumulation | High effort_vs_result sustained, is_open elevated, kyle_lambda low | 2,000-50,000 | No — learn local signature only |
| Distribution | Mirror of accumulation, is_open short-side | 2,000-50,000 | No — local signature only |
| Markup | Low effort_vs_result, positive log_return, expanding is_open | 50-500+ | Partial |
| Markdown | Mirror of markup | 50-500+ | Partial |
| Buying Climax | climax_score > 2.5, positive spike, high is_open | 1-20 | Yes |
| Selling Climax | climax_score > 2.5, negative spike, high effort_vs_result | 1-20 | Yes |
| Absorption | effort_vs_result > 1.5 sustained, flat log_return, high volume | 50-300 | Yes |
| Spring + Test | Negative spike + high effort_vs_result + is_open spike + recovery + declining volume retest | 30-200 | Yes |
| Upthrust | Mirror of spring | 30-200 | Yes |
| Shakeout | Rapid negative spike, moderate volume, immediate recovery | 5-30 | Yes |

**5 microstructure regimes (Cont):**
1. High information asymmetry (kyle_lambda elevated, persistent cum_ofi_5)
2. Noise flow (kyle_lambda ≈ 0, oscillating cum_ofi_5)
3. Liquidity provision vs. withdrawal (delta_imbalance_L1, spread dynamics)
4. Momentum vs. mean-reversion (log_return consistency, effort_vs_result)
5. Stressed vs. calm (extreme log_spread, depth_ratio, climax_score)

**Three load-bearing features (Wyckoff):**
1. `effort_vs_result` — the master signal (absorption vs ease-of-movement)
2. `climax_score` — phase transition markers
3. `is_open` — the DEX-specific Composite Operator footprint (no equivalent in traditional markets)

**200 events = ~10 minutes on BTC.** Sufficient for local patterns (springs, climaxes, short absorption). NOT sufficient for full Wyckoff cycles. A hierarchical architecture (sequence of embeddings) is the natural evolution for phase-level inference.

### Self-Labels from Features (Council-4 + Council-2, no human annotation)

Computable regime labels using only the 17 features (all causal, no lookahead):

| Label | Rule (simplified) |
|-------|-------------------|
| Absorption | mean(effort_vs_result) > 1.5 AND std(log_return) < 0.5σ AND volume elevated |
| Buying Climax | max(climax_score) > 2.5 AND positive spike AND prior uptrend |
| Selling Climax | max(climax_score) > 2.5 AND negative spike AND prior downtrend |
| Markup | Sustained positive log_return AND low effort_vs_result |
| Markdown | Mirror of markup |
| Spring | Negative spike + high effort_vs_result at low + is_open > 0.5 + recovery |
| Informed Flow | kyle_lambda > 75th pct AND persistent cum_ofi_5 |
| Stress | log_spread > 90th pct AND abs(depth_ratio) > 90th pct |

These labels serve for: contrastive pair construction, probing task evaluation, cluster validation.

### Falsifiability Framework (Council-5 — the critical contribution)

**5 pre-registered gates, evaluated in sequence. Failure at any gate = stop.**

| Gate | Test | Threshold | When |
|------|------|-----------|------|
| **0** | PCA + logistic regression baseline | CNN probe must exceed PCA by ≥ 0.5pp | Before pretraining |
| **1** | Linear probe on frozen embeddings, 100-event direction | > 51.4% on 15+/25 symbols (April 1-13) | After pretraining |
| **2** | Fine-tuned CNN vs logistic regression on flat features | Exceed by ≥ 0.5pp on 15+ symbols | After fine-tuning |
| **3** | Held-out symbol (pre-register: AVAX) | > 51.4% at primary horizon | After fine-tuning |
| **4** | Temporal stability (months 1-4 vs 5-6) | < 3pp accuracy drop on 10+ symbols | After fine-tuning |

**Additional diagnostics:**
- Symbol probe accuracy < 20% (embeddings must NOT encode symbol identity)
- CKA > 0.7 between two seed-varied runs (representation stability)
- Pretrained encoder must exceed random (untrained) encoder by ≥ 0.5pp
- Compute budget: **1 H100-day** before gates must be evaluated

### What the Representation Should NOT Capture (Council-2 + Council-5)

- Symbol identity (test: symbol classification probe < 20%)
- Time-of-day effects (test: hour-of-day probe should not be dominant)
- Absolute price level (already handled by feature normalization)
- Carry-forward artifacts in OB features (block structure is data pipeline artifact)

---

## Key Disagreements / Open Questions

### 1. Model size cap

- **Council-5:** Hard cap at 500K params given effective sample count (~100-250K after autocorrelation)
- **Council-6:** Recommends ~400K, with potential for 2M if representations show promise
- **Resolution:** Start at 400K, do not exceed 500K without clearing all gates first

### 2. Cross-symbol contrastive pairs

- **Council-6:** Use same-date, same-hour cross-symbol pairs as soft positives (liquid symbols only)
- **Council-2:** Memecoin microstructure may be fundamentally different; do not force invariance BTC↔FARTCOIN
- **Resolution:** Cross-symbol pairs only for top-6 liquid symbols initially (BTC, ETH, SOL, BNB, AVAX, LINK)

### 3. Direction prediction as evaluation

- **Council-5:** Direction probe > 51.4% is the hard gate — without it, representations are unfalsifiable
- **Council-4:** The value is in learning to SEE tape patterns, not in predicting direction
- **Resolution:** Gate 1 (direction probe) is non-negotiable for falsifiability. But representation quality is also assessed via Wyckoff label probes, cluster analysis, and cross-symbol consistency

### 4. Hierarchical architecture for full Wyckoff cycles

- **Council-4:** 200 events can only detect local patterns; full cycles need 10K+ events via a second-level model
- **Council-6:** Out of scope for initial spec; natural evolution after local representations prove useful
- **Resolution:** Note as future work. Current spec focuses on learning local tape patterns within 200 events.

---

## Implementation Roadmap

### Phase 0: Baselines (local CPU, 1 hour)
- Compute PCA + logistic regression baseline (Gate 0 reference)
- Compute random encoder + linear probe baseline
- Compute Wyckoff self-labels for all training data

### Phase 1: Pretraining (RunPod H100, ~12 hours)
- MEM (block masking) + contrastive on all pre-April data
- ~400K param encoder, 256-dim embeddings
- Monitor: MEM loss, contrastive loss, embedding collapse (batch std)
- Stop when loss plateaus (< 1% improvement in last 20% of epochs)

### Phase 2: Evaluation (local CPU, 2 hours)
- Gate 1: Linear probe on frozen embeddings → April 1-13
- Symbol probe (should fail: < 20% accuracy)
- Cluster analysis (k=8-16) colored by Wyckoff labels
- CKA between two seed-varied pretraining runs
- Wyckoff-state-specific accuracy breakdown

### Phase 3: Fine-tuning (conditional on Gate 1 pass)
- Add direction heads, freeze encoder 5 epochs, then unfreeze at lr=5e-5
- Gate 2: Compare to logistic regression baseline
- Gate 3: Held-out symbol (AVAX)
- Gate 4: Temporal stability

### Phase 4: Interpretation (conditional on Gates 1-4)
- What did it learn? Feature attribution per Wyckoff state
- Embedding trajectory analysis during known market events
- Cross-symbol regime correspondence
- Discrete codebook (k-means, k=128) → "market state vocabulary"

---

## Source Council Reviews

Detailed analyses available in:
- `docs/council-reviews/repr-learning-council2.md` (Cont: microstructure regimes)
- `docs/council-reviews/repr-learning-council4.md` (Wyckoff: tape state taxonomy)
- `docs/council-reviews/repr-learning-council5.md` (Practitioner: falsifiability gates)
- `docs/council-reviews/repr-learning-council6.md` (DL Researcher: self-supervised framework)

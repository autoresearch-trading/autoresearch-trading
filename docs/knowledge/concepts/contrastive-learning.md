---
title: SimCLR Contrastive Learning
topics: [pretraining, self-supervised, architecture]
sources:
  - docs/council-reviews/2026-04-10-round5-council-6-pretraining-mechanics.md
  - docs/council-reviews/repr-learning-synthesis.md
  - docs/experiments/step5-cluster-cohesion.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
last_updated: 2026-04-24
---

# SimCLR Contrastive Learning

## What It Is

The secondary pretraining objective (weight 0.10→0.40, annealed). NT-Xent loss
pushes augmented views of the same window together and different windows apart
in the 256-dim embedding space. This shapes the GLOBAL embedding geometry that
downstream probes and fine-tuning heads use.

## Our Implementation

- **Augmented views:** 2 per window → batch=256 produces 512 views, 256 positive pairs
- **Loss:** NT-Xent on L2-normalized 128-dim projections (via 256→256→128 projection head)
- **Temperature:** τ=0.5 anneal to τ=0.3 by epoch 10 (NOT ImageNet default of 0.1)

### Augmentation Parameters

| Augmentation | Value | Rationale |
|---|---|---|
| Window jitter | ±10 events | Best augmentation — forces position invariance |
| Gaussian noise (trade features) | σ=0.05 | 2% was near-identical views |
| Gaussian noise (OB features) | σ=0.15 | OB features naturally stale (~24s) |
| Gaussian noise (discrete features) | 0 | Noise changes meaning of is_open, num_fills |
| Feature dropout | p=0.10 | Per feature per event, zero to BN mean |
| Time dilation | [0.75, 1.25] | Speed invariance |

### Calibration Test

Before training, compute cosine similarity between augmented pairs using random
encoder. Target: 0.7-0.85. If > 0.95, augmentations too weak (trivial task).

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-10 | τ=0.5→0.3 not 0.1 | Low τ with financial data pushes apart genuinely similar market states using spurious features (symbol identity, time-of-day) | round5-council-6 |
| 2026-04-10 | Augmentation noise increased from σ=0.02 | Views were near-identical → trivial contrastive task | round5-council-6 |
| 2026-04-10 | Cross-symbol pairs deferred to run 2 | Implementation complexity; validate basic framework first. Add only if symbol probe > 30% | round5-council-6 |
| 2026-04-10 | Loss weight annealed 0.10→0.40 | Contrastive needs basic encoder structure before gradients are useful (convergence asymmetry with MEM) | round5-council-6 |

## Collapse Prevention

- **Projection head** (256→256→128 + L2-norm): most important anti-collapse mechanism
- **Effective rank monitoring:** flag if < 20 at epoch 5, < 30 at epoch 10
- **Gradient clipping:** max_norm=1.0 (prevents projection head instability)
- **If collapse detected:** increase τ by 2x, reduce lr by 2x. Do NOT restart.

## Measured Cross-Symbol Invariance (step3-r2, 2026-04-24)

The SimCLR objective targets cross-symbol invariance on 6 liquid anchors
(BTC/ETH/SOL/BNB/LINK/LTC). Measured on Feb held-out:

| Population | Mean cosine |
|---|---|
| within_symbol | 0.8948 |
| same_symbol_diff_hour | 0.8361 |
| cross_symbol_same_hour | 0.7339 |
| cross_symbol_diff_hour | 0.6967 |

**Cross-symbol same-hour delta = +0.037**, below council-5's +0.10
`some_invariance` threshold. Symbol-identity signal = +0.139 (4× stronger).
The current recipe (6-of-24 anchors, soft-positive weight 0.5) learned
per-symbol feature quality, not universal tape geometry. See
[cross-symbol invariance concept](cross-symbol-invariance.md).

This does not invalidate the Gate 1 pass (encoder > PCA on the pretrained
universe) — it qualifies the kind of signal the encoder learned. A future
universality-targeting run would need LIQUID_CONTRASTIVE_SYMBOLS widened to
12–15 and soft-positive weight annealed 0.5 → 1.0.

## Gotchas

1. τ=0.1 (ImageNet default) is too cold for financial data — causes spurious feature learning.
2. Window jitter at day boundaries must fall back to zero jitter.
3. Do NOT use time reversal (breaks causality) or event shuffling.
4. Cross-symbol soft positives are complex to implement in NT-Xent — defer.

## Related Concepts

- [MEM Pretraining](mem-pretraining.md) — the primary pretraining objective
- [Self-Labels](self-labels.md) — contrastive pair construction for labeled windows
- [Cross-Symbol Invariance](cross-symbol-invariance.md) — measured invariance outcome on step3-r2
- [Cluster Cohesion Experiment](../experiments/cluster-cohesion-diagnostic.md)

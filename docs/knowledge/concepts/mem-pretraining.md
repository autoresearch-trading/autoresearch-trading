---
title: Masked Event Modeling (MEM)
topics: [pretraining, self-supervised, architecture]
sources:
  - docs/council-reviews/2026-04-10-round5-council-6-pretraining-mechanics.md
  - docs/council-reviews/2026-04-10-round5-council-5-impl-risks.md
  - docs/council-reviews/repr-learning-synthesis.md
  - docs/council-reviews/council-6-step3-model-size.md
last_updated: 2026-04-24
---

# Masked Event Modeling (MEM)

## What It Is

The primary self-supervised pretraining objective (weight 0.70 initially, annealed).
Analogous to BERT's masked language modeling but for tape data: mask blocks of
order events, reconstruct their features from surrounding context. Forces the
encoder to learn the statistical structure of tape patterns.

## Our Implementation

- **Block masking:** 20-event blocks (NOT 5 — too small for RF=253 CNN)
- **Masking rate:** 20% (4 blocks per 200-event window)
- **Reconstruction targets:** 14 of 17 features (exclude delta_imbalance_L1,
  kyle_lambda, cum_ofi_5 — trivially copyable from neighbors via forward-fill)
- **Loss space:** MSE in BatchNorm-normalized space (not raw feature space)

### Critical: Mask Token Replacement Order

```python
# 1. BatchNorm on FULL, UNMASKED input (clean running statistics)
x_normalized = self.input_bn(x_raw)

# 2. Replace masked positions with 0 in BN-NORMALIZED space (= training mean)
x_masked = x_normalized.clone()
x_masked[:, mask, :] = 0.0

# 3. Forward pass with masked input
encoder_output = self.encoder(x_masked)

# 4. MEM loss against BN-normalized ground truth (14 features only)
mem_target = x_normalized[:, mask, :][:, :, non_carryforward_idx]
```

If BN runs AFTER masking, running statistics are contaminated by 15-20%
artificial zeros → systematic miscalibration at inference (no masking).

**Identity-task bug corrected in commit `29f23c0` (2026-04-23).** Step 3 run-0
passed the FULL, UNMASKED view to `enc(v1)`, then applied the mask only to
select which positions contributed to the MSE loss. The encoder saw ground
truth at masked positions → MEM became a trivial identity task. Council-6
identified this as bug #1 in the pretrain_step review. Fix: build `pos_mask`
before the forward pass; replace masked positions in `v1_normalized` with 0
(BN training mean); pass the masked tensor as encoder input. See
[step3 run-0 collapse diagnosis](../experiments/step3-run0-collapse-diagnosis.md).

Optional: learnable mask tokens (17 parameters, free from budget).

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-10 | Block size 20, not 5 | RF=253 covers entire 200-event window; 5-event gaps are trivially solvable by local interpolation from positions p-1 and p+5 | round5-council-6 |
| 2026-04-10 | Masking rate 20%, not 15% | Larger blocks need slightly higher rate to maintain similar total masked events | round5-council-6 |
| 2026-04-10 | BN before masking | BN after masking contaminates running stats with artificial zeros | round5-council-6 |
| 2026-04-10 | Loss annealing: 0.90→0.60 over 20 epochs | MEM converges faster than contrastive; static 0.70 leads to MEM over-specialization | round5-council-6 |

## Failure Modes

1. **Interpolation shortcut:** High-autocorrelation features (is_open, log_spread)
   are reconstructable by copying neighbors. Monitor per-feature reconstruction
   MSE. If log_spread MSE < 20% of baseline variance, model is interpolating.
2. **MEM over-specialization:** Encoder becomes good at local reconstruction but
   never develops globally-coherent embeddings. Mitigated by loss annealing
   (decreasing MEM weight over training).
3. **BN stat instability:** MEM targets shift as BN running stats converge.
   Consider freezing BN stats after epoch 5 if loss spikes.

## Gotchas

1. Block size 5 with RF=253 is a diagnostic test, not a learning task — increase to 20.
2. Diagnostic for too-easy masking: reconstruction MSE on EXCLUDED features
   (delta_imbalance_L1, kyle_lambda, cum_ofi_5). If MSE < 0.01 in BN space,
   the task is too easy.
3. MEM loss should be computed on the 14 non-carry-forward features only.

## Related Concepts

- [Contrastive Learning](contrastive-learning.md) — the other pretraining objective
- [Self-Labels](self-labels.md) — evaluation targets, not training targets
- [Step 3 Run-0 Collapse Diagnosis](../experiments/step3-run0-collapse-diagnosis.md) — the identity-task bug that invalidated run-0 MEM trajectory

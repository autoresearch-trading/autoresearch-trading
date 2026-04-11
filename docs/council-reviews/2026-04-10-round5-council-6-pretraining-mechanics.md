# Council-6 Round 5: Pretraining Mechanics Deep Dive

**Reviewer:** Council-6 (Deep Learning Researcher, Primary Architect)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

Four implementation-critical gaps must be resolved before the first H100 run: (1) mask replacement after BatchNorm, (2) block size 5→20 events, (3) NT-Xent τ=0.5 not 0.1, (4) augmentation noise σ=0.02 too weak.

## 1. Block Masking: RF=253 Makes 5-Event Blocks Trivial

With RF=253 (larger than the 200-event window), every position sees the entire input. A 5-event gap is filled by layer-1 convolutions from positions p-1 and p+5 — pure local interpolation.

**Fix:** Increase block size to 20 events, keep 20% masking rate. This produces 4 gaps of 20 events per window, requiring long-range context (layers 3+) to reconstruct.

**Diagnostic:** After 1 epoch, check reconstruction MSE on excluded features (delta_imbalance_L1, kyle_lambda, cum_ofi_5). If MSE < 0.01 in BN-normalized space, the task is too easy.

## 2. Mask Token Replacement — The BatchNorm Order

**Critical specification:**

```python
# Step 1: BatchNorm on FULL, UNMASKED input
x_normalized = self.input_bn(x_raw)

# Step 2: Replace masked positions with 0 IN BN-NORMALIZED SPACE
x_masked = x_normalized.clone()
x_masked[:, mask, :] = 0.0  # zero in BN space = training mean

# Step 3: Forward pass with masked input
encoder_output = self.encoder(x_masked)

# Step 4: MEM loss against BN-normalized ground truth (14 features)
mem_target = x_normalized[:, mask, :][:, :, non_carryforward_idx]
```

**Why:** If BN runs on masked input (zeros before BN), running statistics are contaminated by 15% artificial zeros. At inference (no masking), BN uses wrong statistics → systematic miscalibration.

Optional: add 17 learnable mask tokens instead of zeros (free from budget).

## 3. Augmentation Calibration

| Augmentation | Current Spec | Recommended | Rationale |
|---|---|---|---|
| Gaussian noise (trade features) | σ=0.02 | σ=0.05 | 2% of std is near-identical views |
| Gaussian noise (OB features) | σ=0.02 | σ=0.15 | OB features are naturally stale (~24s) |
| Gaussian noise (discrete features) | σ=0.02 | 0 | Noise changes meaning of is_open, num_fills |
| Feature dropout | p=0.05 | p=0.10 | 5% too mild |
| Window jitter | ±10 | ±10 (keep) | Best augmentation in the set |
| Time dilation | [0.8, 1.2] | [0.75, 1.25] | Slight widening |

**Calibration test:** Before training, compute cosine similarity between augmented pairs with random encoder. Target: 0.7-0.85. If > 0.95, augmentations too weak.

## 4. NT-Xent Temperature

**τ is unspecified in the spec.** Must be added.

- ImageNet default (τ=0.1): too cold for financial data. Sharp softmax pushes apart genuinely similar market states using spurious features (symbol identity, time-of-day).
- Financial data recommendation: **τ=0.5, anneal to τ=0.3 by epoch 10.**
- Too-high τ (>1.0): diffuse gradient, slow learning but safe.
- Too-low τ (<0.1): collapse risk, spurious feature learning.

## 5. Loss Weight Annealing

| Epoch | MEM Weight | Contrastive Weight | Rationale |
|---|---|---|---|
| 1-5 | 0.90 | 0.10 | Bootstrap encoder with reconstruction |
| 6-10 | 0.80 | 0.20 | Start contrastive as encoder develops structure |
| 11-15 | 0.70 | 0.30 | Steady state (spec's target) |
| 16-20 | 0.60 | 0.40 | Push contrastive to prevent MEM over-specialization |

Implementation: `mem_weight = max(0.50, 0.90 - epoch * 0.02)`

## 6. Embedding Collapse Prevention

**Monitor both:**
- Per-batch std (existing spec): flag if < 0.05 for 3 consecutive epochs
- **Effective rank** (NEW): flag if < 20 at epoch 5, < 30 at epoch 10

```python
cov = (embeddings.T @ embeddings) / batch_size
eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=0)
p = eigenvalues / eigenvalues.sum()
effective_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum())
```

**Prevention:** Add gradient clipping `clip_grad_norm_(params, max_norm=1.0)`. Not in spec.

**If collapse detected:** Increase τ by 2x, reduce lr by 2x. Do not restart.

## 7. Cross-Symbol Soft Positives

**Recommendation: Defer to run 2.** Standard NT-Xent is complex enough. Validate basic framework first. Add cross-symbol pairs only if symbol probe > 30%.

## 8. Checkpoint Strategy

Save three checkpoints:
1. **best_mem:** Lowest MEM reconstruction loss
2. **best_probe:** Highest direction probe accuracy (April 1-13, every 5 epochs)
3. **last:** Final epoch

**Primary checkpoint for fine-tuning: best_probe** (not best_mem).

**Stopping criterion:** Probe accuracy not improved for 10 consecutive epochs (not MEM loss < 1%).

# Council Review: DL Architecture — Tape Reading Direct Spec
**Reviewer:** Council-6 (Deep Learning Researcher)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Date:** 2026-04-01

---

## 1. Receptive Field Calculation (CNN Option A) — INCORRECT in Spec

The spec describes the receptive field (RF) as "10-event" for layer 2 and "20-event" for layer 3. These numbers are wrong, and the actual RF is much smaller than implied.

For a dilated 1D CNN, the RF per layer is:
  RF_layer = (kernel - 1) * dilation + 1

And the total RF stacks additively:
  Layer 1: kernel=5, dilation=1  → RF = (5-1)*1 + 1 = 5
  Layer 2: kernel=5, dilation=2  → RF = (5-1)*2 + 1 = 9; cumulative = 5 + 9 - 1 = 13
  Layer 3: kernel=5, dilation=4  → RF = (5-1)*4 + 1 = 17; cumulative = 13 + 17 - 1 = 29

**Total receptive field: 29 events.** Not 20 as implied.

This is still only 14.5% of the 200-event sequence length. The model with 3 dilated layers cannot see the full 200-event context — it sees a 29-event window at the final layer. This is a significant limitation.

To cover the full 200-event sequence, you need more layers. With the same kernel=5 pattern (doubling dilations), the required dilations are 1, 2, 4, 8, 16, 32 (6 layers total):
  RF = 5 + 9 + 17 + 33 + 65 + 129 - 5 = 253 events (exceeds 200, covers full sequence)

**Recommendation:** Add 3 more layers (dilations 8, 16, 32) to get RF >= 200. This keeps parameters manageable (~85K vs 65K) and ensures the model can theoretically see the full context window. Alternatively, acknowledge the 29-event window is intentional and label it as "local pattern detector" rather than full-sequence model.

The spec's comment "20-event receptive field" on layer 3 is misleading — it appears to be confusing the dilation multiplier (4) with the actual RF. Fix the comment regardless.

---

## 2. BatchNorm vs LayerNorm for Sequence Data

The spec uses BatchNorm1d throughout. This choice requires justification given the data characteristics.

**BatchNorm1d applied to the input (shape: batch x 16 x seq_len)** normalizes each of the 16 feature channels across the batch dimension AND the sequence dimension. This is appropriate here because:
- Features like log_return and log_total_qty have heavy tails that need stabilization
- The batch is large (256-1024 samples) so BN statistics are stable
- Features are heterogeneous: is_buy is binary [0,1], log_return is continuous with fat tails, time_delta can span orders of magnitude

**However**, BatchNorm has a known failure mode for time series: if train and test distributions diverge (e.g., market regimes), the running mean/variance accumulated during training will not match test statistics, causing silent degradation. This is especially acute for crypto, where volatility regimes can shift 3-5x between periods.

**LayerNorm** normalizes per-sample across the feature dimension, making it regime-invariant. For sequence models processing financial data, LayerNorm is generally preferred after the initial input normalization.

**Recommendation:** Keep BatchNorm1d(16) at the input (handles the feature scale heterogeneity problem). Replace the per-layer BatchNorm1d(32) and BatchNorm1d(64) with LayerNorm applied over the channel dimension. This combines the benefits: input normalization handles scale, per-layer LayerNorm handles distribution shift. The spec currently risks silent test set degradation from regime changes.

---

## 3. GlobalAvgPool — Information Loss Concerns

GlobalAvgPool (GAP) compresses (batch, 64, seq_len) → (batch, 64) by averaging across the temporal dimension. For a 200-position sequence, this discards all positional information — the model cannot distinguish "aggressive buying at position 5" from "aggressive buying at position 195."

For microstructure patterns, the temporal structure within the 200 events may matter:
- Climax events near the end of the sequence are more predictive than those at the start
- Acceleration patterns (increasing intensity over the last 20 events) require knowing position

**Alternatives:**
1. **Last-position pooling:** Take the final time step (position -1) from the CNN output. Preserves recency but discards early context.
2. **Concat [GAP, last_position]:** 128-dim representation capturing both global summary and recent state. +0 parameters in linear head if we keep output at 128→4.
3. **Attention pooling (learned):** Linear(64→1) applied per position → softmax → weighted sum. Adds ~200 parameters, allows the model to learn which positions matter. This is the recommended upgrade once the CNN baseline is validated.

**Recommendation:** Start with GAP as specified — it is simple and will work reasonably well for horizons like 100/500 events where recency matters less. After baseline validation, add concat [GAP, last_position] as a near-zero-cost improvement. Reserve attention pooling for iteration 2.

---

## 4. Multi-Task Loss Weighting

The spec uses an unweighted sum of 4 BCE losses across horizons (10, 50, 100, 500 events). This implicit equal weighting has problems:

**Problem 1: Horizon 10 is likely near-random.** At 0.5-1 second forward, price direction is dominated by bid-ask bounce and microstructure noise. Including it at equal weight may hurt the shared trunk — the gradient from a noise-dominated head degrades the shared representation.

**Problem 2: Horizon 500 covers ~30-60 seconds.** The label at t+500 requires a 500-event lookahead buffer at training time, meaning the last 500 positions of each day cannot be labeled. For a ~300 sample/day budget, this is a ~10% reduction in effective training data.

**Problem 3: Equal weighting assumes equal signal strength.** If the 100-event horizon has 3x the signal of the 10-event horizon, equal weighting means the noisy horizon contributes equal gradient to the shared trunk.

**Recommendation:** Use adaptive weighting. After the first epoch, compute per-horizon validation accuracy. Assign weights inversely proportional to loss (or directly proportional to accuracy - 0.5 baseline). This is a simple heuristic that won't hurt if all horizons have equal signal, and will help if they don't. Alternatively, ablate: train single-horizon models for each horizon and compare to multi-task — this is the correct way to verify multi-task actually helps here.

**On shared representations:** The spec has all 4 heads reading from the same GAP vector, which is correct for multi-task learning. The shared trunk (CNN layers) should be encouraged to learn universal features, not horizon-specific ones. If you observe that performance on short horizons improves at the expense of long horizons (or vice versa), consider gradient surgery or a two-level architecture with short- and long-range branches.

---

## 5. Transformer (Option B) at seq_len=200, d_model=64

The spec specifies TransformerEncoder(64, 4 heads, 2 layers). Each head has dimension 64/4 = 16. This is very small.

At seq_len=200, self-attention computes a 200×200 attention matrix per head — that is 40,000 attention scores computed from 16-dimensional keys and queries. With only 16 dimensions per head, the attention scores will be noisy and hard to train stably without careful initialization.

From v7 results (Sortino=0.061 on full Transformer with window=2000, H100), temporal architectures do not reliably outperform flat MLP at this data scale. The spec correctly positions Option B as "if CNN shows signal" — that ordering is right. However, if the Transformer is attempted, consider:

- d_model=128 with 4 heads (32 dim/head) — more stable attention
- Pre-norm (LayerNorm before attention, not after) — better gradient flow
- Rotary positional embeddings (RoPE) instead of sinusoidal — better for relative position encoding, which matters for financial sequences where absolute position is less meaningful than relative gaps

At 200 seq_len × 64 dim × 2 layers × 4 heads, self-attention is not the compute bottleneck — it is ~5% of forward pass time on a GPU. The cost concern the spec raises is valid for seq_len=2000+ but at 200, Transformer and CNN are comparable in speed.

**Verdict:** The Transformer option is worth trying only if the CNN validates signal at > 51% accuracy. Do not invest engineering effort in it before that milestone.

---

## 6. LSTM (Option C): GRU vs LSTM, Bidirectionality

The spec uses LSTM(16→64, 2 layers, bidirectional=False). This is correct on the bidirectionality choice — bidirectional LSTM is non-causal for online prediction.

**GRU vs LSTM:** GRU has ~30% fewer parameters than LSTM at equivalent hidden dimension (no separate cell state). At hidden_dim=64, GRU has ~2 gates vs LSTM's 3, which means:
- GRU: ~25K parameters per layer
- LSTM: ~33K parameters per layer

At this dataset scale (1.2M samples), the capacity difference is marginal. However, GRU trains faster, is less susceptible to gradient issues, and has been shown to match LSTM performance on most sequence tasks shorter than ~500 steps. At seq_len=200, GRU is the better default.

**Recommendation:** Replace LSTM with GRU. This is a strict improvement in implementation simplicity and comparable in modeling power.

**Depth concern:** 2-layer LSTM/GRU on seq_len=200 will have gradient flow issues for the early positions (vanishing over 200 steps × 2 layers). The CNN alternative avoids this by construction — each layer has bounded gradient paths. This is one concrete reason to prefer CNN as the first architecture.

---

## 7. Model Capacity: 65K Params vs 1.2M Samples

The spec puts ~65K parameters against ~1.2M training samples, giving a sample-to-parameter ratio of approximately 18:1.

For comparison:
- ImageNet-1M with ResNet-50 (25M params): ratio ~48:1
- This spec: 1.2M samples / 65K params: ratio ~18:1

The ratio is on the low end for a well-regularized model, but not dangerous given:
1. Financial sequences are noisier than images (label noise dampens effective gradient signal)
2. The model sees 200-step sequences, so effective "pixels" per sample = 3200 — more information per sample than a single label implies
3. Binary labels (not soft) reduce information per sample further

**The 65K parameter budget is appropriate for the first pass.** If the model underfits (train accuracy = val accuracy = 50.x%), scaling to 130K (double channels) is the right move. If it overfits (train >> val), the existing 65K budget with added regularization is the right move.

**Do not use 200K+ parameter models (Transformer, Option B) until the CNN validates signal.** The v7 Transformer result is a cautionary data point.

---

## 8. Dropout: Missing from All Three Architectures

The spec uses AdamW + BatchNorm but no dropout. The training section explicitly says "No class weighting initially" but does not mention dropout at all.

From the current MLP baseline (BEST_PARAMS), dropout=0.0 won because "model already regularized by small size." That finding does NOT transfer here because:
1. The tape-reading model operates on raw, noisier features (not 13 carefully engineered ones)
2. Binary direction labels have higher noise than triple-barrier labels with fee_mult=11
3. The multi-task setup (4 heads, shared trunk) can produce a different overfit profile

**Recommendation:** Include dropout in the architecture spec:
- Post-activation dropout at rate 0.1-0.2 in the CNN layers
- Apply it only to the first two Conv layers, not the final one before pooling
- Test with and without: do a 2-run ablation, not a full sweep

The principal-of-parsimony argument for no-dropout only applies when we have prior evidence from similar models on similar data. We do not have that here for the CNN on raw tape features.

---

## 9. Data Augmentation Opportunities

The spec does not mention data augmentation. For financial time series at this scale, several augmentations are low-risk and high-value:

**Additive Gaussian noise to log_return and log_total_qty:** Scale 0.05-0.1 (5-10% of typical feature std). Prevents memorization of exact price trajectories. Standard for financial seq2seq models.

**Feature dropout (zeroing random features per sample):** With probability 0.1, zero out 1-2 random features per training sample. Forces the model to not rely on any single feature, similar to how DropPath works in vision models. Especially useful here because orderbook features (11-16) may be stale (3-second snapshots) — the model should learn to work without them.

**Time reversal (with caution):** Reverse the 200-event sequence, flip the direction label. This doubles effective training data but the underlying physics of order flow is NOT time-symmetric (open_long and close_long are not equivalent under reversal), so this augmentation should only be applied to price-only features, not to is_buy or is_open flags. Use with extreme caution.

**Symbol mixing (not recommended):** Mixing sequences from two different symbols is common in NLP (SentenceMixup), but the microstructure distributions differ enough between BTC and FARTCOIN that mixed sequences would be non-physical. Skip.

---

## 10. Architecture Recommendation

Given this task (200-event sequences, 16 features, 1.2M samples, binary classification, 4 horizons), my recommendation:

**Build the following, in this order:**

1. **Linear baseline first** (as specified — this is correct).

2. **CNN Option A with fixes:**
   - Extend to 6 dilated layers (dilations 1,2,4,8,16,32) to achieve RF >= 200
   - Replace per-layer BatchNorm with LayerNorm (keep input BatchNorm)
   - Add dropout(0.1) after first two conv blocks
   - Replace GlobalAvgPool with concat[GAP, last_position] (128-dim → head)
   - Add weighted multi-task loss after first epoch based on per-horizon accuracy

3. **If CNN achieves > 51% accuracy:** Try GRU(64, 2 layers) as a comparison. The GRU will capture longer-range dependencies but at the cost of sequential computation.

4. **Do not build the Transformer** until steps 1-3 are complete and validated.

This ordering respects "simplest first," addresses the critical RF gap in the current spec, and gives incremental complexity steps that are each individually attributable.

---

## Summary of Issues by Priority

| Priority | Issue | Action |
|----------|-------|--------|
| HIGH | RF calculation wrong — 29 events, not 200 | Add 3 more dilated layers |
| HIGH | No dropout — will overfit on noisy binary labels | Add dropout(0.1) to conv layers |
| MEDIUM | BatchNorm in body — regime-shift risk | Replace body BN with LayerNorm |
| MEDIUM | Equal horizon weighting — noise head degrades trunk | Weight by per-horizon accuracy |
| MEDIUM | GlobalAvgPool discards recency | Use concat[GAP, last_pos] |
| LOW | LSTM over GRU — unnecessary complexity | Use GRU instead |
| LOW | No data augmentation specified | Add noise + feature dropout |
| INFO | Transformer too small (16 dim/head) | Increase d_model if Transformer attempted |

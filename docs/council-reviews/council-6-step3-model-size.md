---
title: Council-6 Review — Step 3 Model Size + Pretraining Configuration
date: 2026-04-23
author: council-6
status: accepted — triggers spec + plan amendment
sources:
  - docs/superpowers/plans/2026-04-15-step3-pretraining.md
  - docs/knowledge/decisions/mem-block-size-20.md
  - docs/knowledge/decisions/ntxent-temperature.md
  - docs/knowledge/concepts/mem-pretraining.md
  - docs/knowledge/concepts/contrastive-learning.md
---

# Council-6 Review — Step 3 Model Size + Pretraining Configuration

## Q1: Which `channel_mult`?

**Recommendation: `channel_mult = 1.0` (~400K params). Do not exceed the 500K spec cap.**

The framing of 1:1.6 data-to-params ratio (641K windows : 400K params) as "tight" is
misleading for a self-supervised setting. The ratio that matters for SSL is effective
supervision density, not raw window count. Each 200-event window with 20% block masking
(4 blocks of 20 events = 80 masked positions) and 14 reconstruction targets produces
80 × 14 = 1,120 MSE targets per window. Across 641K windows × 20 epochs, MEM alone
generates ~14.4 billion per-feature regression targets before contrastive pairs are
counted. The effective signal per parameter is ~36,000:1, which is comfortably above
any representation-learning overfitting threshold observed in the literature (MAE,
BEiT, data2vec all operate at comparable or lower effective ratios at similar parameter
scales). 625K at `channel_mult=1.25` does not buy meaningfully more capacity for this
task — the bottleneck is the 17-feature input vocabulary and 200-event context, not
encoder width.

The 500K hard cap should hold. It exists not because the SSL training signal is weak
but because (a) Gate 3 must generalize to AVAX zero-shot, and (b) a tighter model
forces the 256-dim embedding to be genuinely compressed rather than symbol-memorizing.
Exceeding the cap without clearing Gate 1 is premature optimization.

The 200K option (`channel_mult=0.7`) is weakly rejected. With RF=253 and
`hidden_channels` scaled to ~90, the model retains the full temporal coverage but
loses representational width at the point where the 14-feature reconstruction and
128-dim cross-symbol contrastive objectives compete for gradient signal. The
effective rank of the embedding is likely to saturate below the 20-dimensional floor
(the collapse diagnostic threshold in `contrastive-learning.md`). The `is_open`
autocorrelation signature (half-life 20 events) and `effort_vs_result` dynamics
require enough channel width to hold distinct representations simultaneously — 200K
is marginal for that co-representation.

## Q2: Epoch budget

**Keep 20–40 epochs with early stop. The chosen 400K size does not change this.**

With equal-symbol sampling producing ~18K windows per symbol per epoch, and 24
symbols, each epoch is ~432K gradient steps at batch 256 (roughly 1,690 steps/epoch).
Over 20 epochs that is ~33,800 steps; over 40 epochs ~67,600 steps. OneCycleLR with
20% warmup means the peak lr is not reached until step ~6,760 at 20 epochs or ~13,520
at 40 epochs — well within the H100 wall-clock budget.

The 1% MEM improvement stopping criterion applied over "the last 20% of epochs"
means: at 20 epochs, stop if MEM loss has not dropped 1% in the final 4 epochs. This
is appropriate. What "enough" looks like: the MEM loss curve should flatten into a
plateau reflecting genuine irreducible uncertainty (market noise), not underfitting.
At 400K params and this data scale, experience from audio/signal SSL (e.g., wav2vec
2.0, BYOL-A) suggests the plateau is reached well before epoch 30 for encoder sizes
below 10M params. 20 epochs with early stop is a safe default; increase to 40 only
if the probe accuracy at epoch 20 is still improving.

One important note from the knowledge base that the current `PretrainConfig` does
not implement: **loss annealing**. The accepted decision (`mem-pretraining.md`,
round-5) specifies MEM weight 0.90→0.60 and contrastive weight 0.10→0.40 over 20
epochs. The static 0.70/0.30 in `PretrainConfig` violates this decision and will
cause MEM over-specialization in early training. This must be added as
`mem_weight_schedule` and `contrastive_weight_schedule` fields or computed inline in
`pretrain_step`.

## Q3: Collapse / capacity diagnostics in first 2 epochs

**Too-small signal (underfitting at 200K):** MEM loss at epoch 2 should be < 0.6
(normalized space, ~40% reduction from random init at ~1.0). If MEM loss is still
above 0.80 at epoch 2, the encoder lacks width to represent the 14-feature
reconstruction jointly with the contrastive objective. Check: per-feature MSE on
`log_return` and `effort_vs_result` specifically (highest-variance features). If
these are still near-random (MSE > 0.9 × baseline variance), the model is
underfitting.

**Too-big / collapse signal (contrastive collapse):** embedding_std across the
batch should stay above 0.1 at all times. The `detect_embedding_collapse` threshold
of 1e-4 in `pretrain.py` is too conservative — at 256 dims, a std of 1e-3 is already
functionally collapsed for a 128-class probe. Raise the flag threshold to 0.05.
Additionally, monitor effective rank (count singular values of the 256×B embedding
matrix above 1% of max). Below rank 20 at epoch 5 is a collapse early warning (from
`contrastive-learning.md`). At 400K params this risk is modest but nonzero with the
SimCLR temperature.

---

## Critical Bugs and Spec Conflicts in the Plan

These are blocking issues that must be resolved before Task 12 (the GPU launch).
They are more urgent than the `channel_mult` question.

**Bug 1 — MEM objective is defeated in `pretrain_step`
(plan lines 1638–1656).** `enc(v1)` is called on the full, UNMASKED view. The mask
`pos_mask` is built afterward and only selects which positions contribute to the
loss. The encoder sees the ground-truth values at masked positions — MEM becomes a
trivial identity task. The correct order per the knowledge base
(`mem-pretraining.md`) is: BN on full input → zero-fill masked positions in
BN-normalized space → pass masked input to encoder → compute MSE against
BN-normalized targets. The fix requires building `pos_mask` before the forward
pass, replacing masked positions in `v1_normalized` with 0 (BN training mean), and
passing that as encoder input.

**Bug 2 — NT-Xent temperature `nt_xent_temperature: float = 0.10` in
`PretrainConfig`.** The accepted decision (`ntxent-temperature.md`) is τ=0.5
annealed to τ=0.3 by epoch 10. τ=0.1 is the ImageNet default explicitly rejected as
"too cold for financial data." This setting will drive collapse via the
spurious-feature-learning pathway described in the decision document.

**Bug 3 — Block masking parameters default to 5 events at 15%.** The
`block_mask(window_len=T, rng=rng)` call in the plan does not show explicit
`block_len` or `fraction` arguments. The accepted decision
(`mem-block-size-20.md`) is `block_len=20`, `fraction=0.20` (4 blocks per window).
With 5-event blocks, the RF=253 encoder can solve MEM by local interpolation from
p-1 and p+5, learning nothing about tape structure.

**Flag — Missing loss annealing schedule.** `PretrainConfig` has static
`mem_weight=0.70`, `contrastive_weight=0.30`. The knowledge base specifies dynamic
schedules (MEM 0.90→0.60, contrastive 0.10→0.40 over 20 epochs). Without annealing,
the contrastive objective receives no gradient signal in early training (encoder
has no structure yet → NT-Xent gradients are noise).

**Flag — No gradient clipping.** `contrastive-learning.md` specifies `max_norm=1.0`
grad clipping as the primary anti-collapse mechanism for the projection head.
`pretrain_step` has no `torch.nn.utils.clip_grad_norm_` call. On an H100 with
bf16/AMP, projection head gradients can spike by 10–100× during the early high-tau
phase.

**Flag — AMP/bf16 absent.** For a 24-hour H100 run, bf16 mixed precision
(`torch.amp.autocast` with bf16) doubles throughput with no accuracy cost for this
model size and task. Expected speedup: ~1.8×, cutting the 24h budget to ~13h
effective and giving more epoch headroom.

**Flag — `torch.compile`.** `torch.compile(enc, mode="reduce-overhead")` is
appropriate here. The dilated CNN with static shapes (B=256, T=200, F=17) will see
significant kernel fusion gains. Apply to encoder only, not to the MEM decoder
(shapes vary per masked-position count).

---

## Summary

Recommend `channel_mult=1.0` (~400K params, within the 500K hard cap); the
self-supervised signal density is ~36,000 targets per parameter and there is no
capacity shortfall at this scale. Three bugs in the `pretrain_step` stub are more
urgent than the size question and must be fixed before the GPU run:
(1) the encoder currently sees unmasked input, defeating the MEM objective entirely;
(2) NT-Xent temperature is set to 0.10 (should be 0.5→0.3 per accepted decision);
(3) block masking parameters need to be explicitly set to `block_len=20`,
`fraction=0.20` per the round-5 knowledge base decision.

Additional flags (loss annealing, grad clipping, bf16 AMP, `torch.compile`,
collapse-threshold bump, effective-rank monitor) should be applied in the same
amendment — all are cheap code changes with outsized impact on the 24h H100 run.

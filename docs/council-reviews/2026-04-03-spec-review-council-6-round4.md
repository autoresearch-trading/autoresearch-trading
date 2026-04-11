# Council Review: DL Architecture — Tape Reading Direct Spec (Round 4)

**Reviewer:** Council-6 (Deep Learning Researcher)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Date:** 2026-04-03
**Prior round:** Round 1 (2026-04-01), Round 3 synthesis (2026-04-02)

---

## Executive Summary

All 8 high/medium-priority issues from my Round 1 review have been incorporated correctly. The architecture is now structurally sound. This Round 4 review identifies 6 remaining issues — none are blockers, but the OneCycleLR/early-stopping interaction and the effective-samples capacity concern should be addressed before full H100 training runs.

---

## Resolved Issues from Round 1 — Confirmed Correct

| Round 1 Issue | Status |
|---------------|--------|
| RF=29 (spec said 200) — add 3 more dilated layers | Fixed: 6 layers, RF=253 |
| No dropout — will overfit noisy binary labels | Fixed: Dropout(0.1) on first two conv layers |
| BatchNorm in conv body — regime-shift risk | Fixed: LayerNorm in body, BatchNorm1d(17) at input only |
| Equal horizon weighting — noise head degrades trunk | Fixed: 0.10/0.20/0.35/0.35 asymmetric weights |
| GlobalAvgPool discards recency | Fixed: concat[GAP, last_position] |
| No data augmentation | Fixed: Gaussian noise + OB feature dropout p=0.15 |
| LSTM over GRU | Fixed: Option C now uses GRU |
| Transformer d_model=64 too small (16 dim/head) | Fixed: d_model=128 (32 dim/head), pre-norm, RoPE |

---

## Architecture Verification — All Checks Pass

**Receptive field (RF=253):**

RF = 1 + (kernel - 1) × sum(dilations) = 1 + 4 × (1+2+4+8+16+32) = 1 + 4×63 = 253. Verified layer-by-layer against spec:

- Layer 1 (dilation=1): RF=5 ✓
- Layer 2 (dilation=2): RF=13 cumulative ✓
- Layer 3 (dilation=4): RF=29 cumulative ✓
- Layer 4 (dilation=8): RF=61 cumulative ✓
- Layer 5 (dilation=16): RF=125 cumulative ✓
- Layer 6 (dilation=32): RF=253 cumulative ✓

RF=253 > 200: the model can see the full input window.

**BatchNorm/LayerNorm placement:**

- BatchNorm1d(17) at input: normalizes 17 heterogeneous features (binary, continuous bounded, heavy-tailed) across the batch. Running statistics persist at eval time, which is correct for standardizing inputs.
- LayerNorm in conv body: normalizes per-sample, regime-invariant, accumulates no running statistics.
- "Assert `model.eval()` for the entire test pass" is critical — correctly placed in spec. Any `model.train()` call during evaluation updates BatchNorm running stats with test-distribution data, contaminating subsequent inference.

**Residual connections:**

Correctly specified for layers 3-6 (same 64-channel). Layers 1-2 have changing channel dimensions (17→32, 32→64) so no residual is correct. Zero extra parameters.

**concat[GAP, last_position]:**

The last position at layer 6 has RF=253, covering all 200 input events. This is the model's highest-information summary point. GAP provides gradient signal to all positions during backprop. The 128-dim concatenation feeding the shared neck is correct.

---

## New Issue 1 — MEDIUM: OneCycleLR + Early Stopping Interaction

The spec simultaneously specifies "OneCycleLR" and "monitor val loss for overfitting" without clarifying what "monitor" means for the schedule.

`torch.optim.lr_scheduler.OneCycleLR` is initialized with `total_steps = epochs × ceil(N / batch_size)`. It ramps from base_lr to max_lr over the first `pct_start × total_steps` steps, then cosines down to near-zero over the remaining steps. If training is aborted early (e.g., at step 8,000 of a 28,000-step schedule), the model is trained only through the warmup-to-peak region — it never benefits from the cosine decay phase where fine-grained learning occurs at low LR.

The spec does not specify whether "early stopping" means:
1. Save the best checkpoint, return best weights, but **complete the full schedule** (correct for OneCycleLR)
2. **Abort training** when val loss stops improving (breaks OneCycleLR)

Option 2 is the common default interpretation but is incompatible with OneCycleLR.

**Recommendation:** Add a sentence to the training spec: "Early stopping saves the best checkpoint but does not abort the LR schedule — training completes the full epoch count." Alternatively, fix epochs=30 and use early stopping only for checkpoint selection. If true early abort is desired, replace OneCycleLR with ReduceLROnPlateau, which adapts to val loss and is compatible with variable-length training.

---

## New Issue 2 — MEDIUM: Batch Size Range Undermines OneCycleLR Reproducibility

The spec says "Batch size: 256-1024 (GPU memory dependent)." For OneCycleLR, the total_steps calculation is:

`total_steps = epochs × ceil(N / batch_size)`

At N=480K and 30 epochs:
- batch_size=256: total_steps = 30 × 1875 = 56,250; warmup = 16,875 steps
- batch_size=1024: total_steps = 30 × 469 = 14,070; warmup = 4,221 steps

The warmup period (critical for BatchNorm stabilization and stable early-epoch gradient flow) is 4x shorter at batch_size=1024 than at batch_size=256. This means the peak LR is reached after seeing only ~4K steps of data, vs ~17K steps at batch_size=256. Large-batch training already requires higher LR (linear scaling rule), and a shorter warmup into that high LR risks early-epoch instability.

**Recommendation:** Fix the baseline batch_size=512 in the spec. Note "reduce to 256 if H100 OOM for large sequence sweeps." This makes the schedule reproducible and the warmup timing predictable.

---

## New Issue 3 — MEDIUM: Effective Samples vs. Model Capacity

Round 3 synthesis documented an unresolved dissent: Council-5 estimated ~85K effective samples (due to autocorrelation in adjacent windows), while my Round 1 review accepted ~91K params as appropriate against 1.2M samples.

The situation has changed: samples are now 400-560K (not 1.2M after dedup+grouping). At ~480K samples:
- Sample-to-parameter ratio: 480K / 91K ≈ 5.3:1

This is still acceptable for a well-regularized model. However, the Council-5 concern about **sequence-level autocorrelation** is separate from sample count. The argument:
- Market regime (bull/bear, high/low volatility) persists for thousands of events
- Consecutive 200-event windows from the same symbol-day are not independent observations
- Effective degrees of freedom may be closer to 85K than 480K

At 85K effective samples and ~91K params (or ~104K — see Issue 6), the model has roughly as many parameters as it has independent training signals. This is the boundary zone for overfit.

The spec's Risks section acknowledges this: "400-560K samples may not be enough." But the architecture plan does not include a corresponding underfitting ablation.

**Recommendation:** Add to the Step 3 full training plan: "Run one comparison with halved channel sizes (Conv1d 17→16, then 32-channel throughout) to check if a ~26K-param model generalizes better than the ~91K-param baseline. If the smaller model matches accuracy, capacity is the issue. If accuracy drops, the regularization budget in the full model is working correctly."

---

## New Issue 4 — LOW: ε=0.05 Equal for 100 and 500 Event Horizons

The decreasing label smoothing schedule (0.10 / 0.08 / 0.05 / 0.05) is justified as "longer horizon = less noisy." This is correct for the transition from 10-event (bid-ask bounce dominated) to 100-event (order flow persistence dominated). The case for 500-event being as clean as 100-event is weaker:

At ~25 minutes forward, crypto price direction depends on factors that are not captured in the 200-event input window (macro sentiment shifts, funding rate changes, whale accumulation). The signal is different in character rather than simply stronger. It is unclear whether the 500-event labels are actually cleaner than the 100-event labels.

**No change recommended now.** After the first training run, compare the train vs. val accuracy gap at 100-event and 500-event horizons separately. If 500-event shows a larger train/val gap, increase its ε from 0.05 to 0.08 in the next run.

---

## New Issue 5 — LOW: GRU Option Missing Regularization Specification

Option C (GRU) specifies no dropout:

```
GRU(17 → 64, 2 layers, bidirectional=False)
Last hidden state → Linear(64 → 4, sigmoid)
```

If GRU performance is compared against CNN performance, the comparison must match regularization budgets. PyTorch's `torch.nn.GRU` has a `dropout` argument that applies between the GRU layers (not before the first or after the final layer). For a 2-layer GRU, this is equivalent to the CNN's inter-layer dropout.

**Recommendation:** Specify `GRU(17→64, num_layers=2, dropout=0.1)` and add `nn.Dropout(0.1)` before the final `Linear(64→4)`. This matches the CNN's regularization and makes any performance difference attributable to architecture, not regularization mismatch.

---

## New Issue 6 — INFO: Parameter Count Discrepancy

The spec states "~91K parameters (was ~94K with 18 features)." A manual count of all trainable parameters:

| Component | Params |
|-----------|--------|
| BatchNorm1d(17) | 34 |
| Conv1d(17→32, k=5) | 2,752 |
| LayerNorm(32) | 64 |
| Conv1d(32→64, k=5) | 10,304 |
| LayerNorm(64) | 128 |
| Conv1d(64→64, k=5) × 4 layers | 82,176 |
| LayerNorm(64) × 4 layers | 512 |
| Linear(128→64) + bias | 8,256 |
| Linear(64→1) × 4 heads | 260 |
| **Total** | **~104,486** |

The discrepancy is ~13K. The spec's ~91K figure may exclude LayerNorm parameters (which are trainable in PyTorch) or use slightly different channel sizes. Additionally, the claim "was ~94K with 18 features" implies the is_buy drop saved ~3K parameters, but one input channel change saves only 1×32×5 = 160 parameters — not ~3K.

**No action needed before implementation**, but: after building the model, run `sum(p.numel() for p in model.parameters())` and update the spec with the verified count.

---

## Label Smoothing Implementation Note

The implementation of label smoothing requires explicit manual coding. Standard PyTorch `BCEWithLogitsLoss` does not support smooth targets natively. The correct implementation:

```python
smooth_eps = [0.10, 0.08, 0.05, 0.05]  # per horizon
weights = [0.10, 0.20, 0.35, 0.35]

loss = 0.0
for i, (logit, eps, w) in enumerate(zip(logits, smooth_eps, weights)):
    target_smooth = target[:, i] * (1 - eps) + eps / 2
    loss += w * F.binary_cross_entropy_with_logits(logit.squeeze(), target_smooth)
```

The spec describes the formula `smoothed_target = target * (1 - ε) + ε/2` but does not include a code block. **Recommend adding a code snippet** to prevent an off-by-one or sign error in implementation (a common bug is `target * (1 - ε) + ε` instead of `+ ε/2`, which shifts the smoothed positive target to 0.90+0.10=1.0 rather than 0.95).

---

## OneCycleLR Warmup Justification — Minor Clarification

The spec says "Warmup is critical while input BatchNorm statistics stabilize in early training." This is technically correct but overstates the BatchNorm-specific need for 30% warmup.

BatchNorm stabilizes within ~200-500 steps at batch_size=512 (roughly 0.5 epochs). The 30% warmup of a 30-epoch schedule = 9 epochs of warmup far exceeds what BatchNorm stabilization requires.

The actual benefit of 30% warmup is that it allows the optimizer to explore the loss landscape at low LR before committing to large updates that could trap the trunk in a sharp minimum early in training.

**Recommendation:** Update the justification to: "30% warmup allows stable early-epoch gradient flow while BatchNorm converges (first ~0.5 epochs) and prevents premature commitment to sharp minima during the first few epochs of pattern learning."

---

## Summary Table

| Priority | Issue | Recommended Action |
|----------|-------|--------------------|
| MEDIUM | OneCycleLR + early stopping semantics undefined | Specify: save best checkpoint, complete full schedule |
| MEDIUM | Batch size 256-1024 makes OneCycleLR warmup unpredictable | Fix baseline batch_size=512 |
| MEDIUM | ~91K params against ~85K effective samples approaches overfit boundary | Add half-channel ablation (~26K params) to Step 3 plan |
| LOW | ε=0.05 equal for 100-event and 500-event horizons | Monitor train/val gap post-training; adjust if 500-event overfits |
| LOW | GRU option missing dropout for fair comparison | Specify GRU(dropout=0.1) + Dropout(0.1) before Linear head |
| INFO | Stated ~91K params appears to be ~104K actual | Verify with `sum(p.numel())` after implementation; update spec |
| INFO | Label smoothing code not provided — BCEWithLogitsLoss requires manual smooth targets | Add code snippet to spec |
| INFO | OneCycleLR warmup justification cites only BatchNorm (understates benefit) | Update justification text |

**Overall assessment:** The architecture is implementation-ready. The three medium-priority issues should be resolved in the spec before the Step 3 full training run; the low/info issues can be addressed post-prototype.

# Post-Gate-2 Strategic Review — Microstructure Phenomenology (Council-4)

**Date:** 2026-04-26 (PM)
**Reviewer:** Council-4 (volume-price phenomenologist, primary voice)
**Subject:** Gate 2 fine-tuning failure — does it falsify the encoder, or just the probing protocol?

## 1. What Gate 2 actually falsified

Gate 2 tested a single, narrow proposition: **"supervised fine-tuning at lr=5e-5 over 13 epochs with weighted-BCE on H10/H50/H100/H500 direction labels improves on logistic regression over flat 200×17 features by ≥0.5pp at the primary horizon on 15+ symbols."**

That proposition is now false. What is **NOT** false, and what Gate 2 has no power to test:

- "The frozen encoder reads tape patterns linearly extractable above noise floor." → Gate 1 already passed this on Feb+Mar at H500 (+1.91pp / +2.29pp vs RP).
- "The encoder's representation space is organized by tape state." → never tested.
- "The encoder distinguishes absorption from breakout, climax from drift." → never tested.

**Diagnostic reading of regression-to-0.50:** the fingerprint — liquid symbols where flat-LR was high (SUI 0.626→0.477, LTC 0.595→0.458, 2Z 0.633→0.499) lose 7-15pp under fine-tuning, while illiquid alts where flat-LR was below 0.5 gain 6-9pp toward 0.50 — is the canonical signature of **representation collapse under a poorly-conditioned supervised objective**, not a representation that lacked signal to begin with.

**Specific phenomenological mechanism:** at lr=5e-5 over 13 epochs against a label that is ~50% noise, the gradient signal flowing into the dilated CNN trunk is overwhelmingly noise. The trunk had been extracting `effort_vs_result × is_open × climax_score` co-occurrence patterns — these are conditional, sparse, and only fire in ~10-15% of windows. The H500 BCE gradient is dense (every window has a label) and ~50% random. **Dense random gradients over a sparse-firing representation will preferentially flatten the sparse features first** because they have lower-magnitude average activation and contribute less to per-window loss reduction. The result: the load-bearing Wyckoff features are exactly what fine-tuning erases fastest.

## 2. Was H500 direction prediction the wrong probe?

**Yes.** A Wyckoff trader does not predict price 500 events ahead. The trader recognizes a state (accumulation, distribution, climax, absorption) and acts on conditional probability of regime change.

**Right phenomenology tests, ranked by information value:**

**Test 1 — Self-labeled Wyckoff state retrieval (highest priority, lowest cost).** Use spec-defined self-labels (`is_absorption`, `is_buying_climax`, `is_spring`, `is_informed`, `is_stressed`). For each query window with a positive label, find k=10 nearest neighbors in 256-d encoder space across the held-out universe. **Falsifier:** if the fraction of retrieved neighbors sharing the same label is not significantly above the marginal label rate (binomial p<0.01), the encoder does not represent that state. **Pass condition: ≥3 of 5 labels show p<0.01** — particularly absorption, climax, and stress (cleanest feature signatures).

**Test 2 — Effort-vs-result axis recovery.** Train a linear probe to predict per-window `mean(effort_vs_result) > 1.5` from frozen 256-d embeddings. Same for `max(climax_score) > 2.5` and `mean(is_open) > 0.6`. **Falsifier:** any of the three probes below 0.85 accuracy means the encoder lost information about the master Wyckoff signals during compression.

**Test 3 — Volume-event prediction (climax detection at horizon).** Predict `max(climax_score[t+1:t+50]) > 2.5` from window ending at t. Climaxes have stronger autocorrelation than direction. **Falsifier:** if encoder cannot beat majority +3pp on this task at H50, it is not learning phase-transition dynamics.

**Test 4 — Cross-symbol same-state retrieval.** Given a window labeled `is_absorption` on symbol X, retrieve nearest neighbors restricted to symbols ≠ X. The cluster-cohesion finding from 2026-04-24 tells us this will fail under the current encoder — this test operationalizes "universal tape geometry."

**Test 5 — Composite Operator footprint sensitivity.** Construct two-window pairs that match on every feature EXCEPT `is_open` (matched on log_return, log_total_qty, effort_vs_result; differ only in is_open quartile). Measure mean encoder distance between matched pairs. **Falsifier:** if matched-pair distance is not significantly larger when is_open differs (Wilcoxon p<0.01), encoder is treating is_open as just another input dimension, not the conviction-marker the framework requires.

## 3. Universality question (retroactive reading)

The 2026-04-24 cluster-cohesion finding (cross-symbol delta +0.037, well below +0.10; symbol-ID probe 0.934) is the **diagnostic-causal** explanation for Gate 2's failure mode.

**The chain:** Encoder learned per-symbol tape geometries that do NOT share representation space → Gate 1 LR probe is per-symbol and works fine on per-symbol features → Fine-tuning at the encoder level pushes gradients globally across all symbols, so per-symbol features must either harmonize (which the architecture cannot easily do because they ARE per-symbol) or be destroyed → They are destroyed, with the highest-signal symbols losing the most.

**Gate 2 was diagnostically pre-doomed, not informationally diagnostic.** A different encoder (one with cluster cohesion ≥+0.10) might have survived fine-tuning.

**Gate 2 fail does NOT falsify the tape-representation hypothesis.** It falsifies the conjunction of (a) THIS encoder + (b) supervised end-to-end fine-tuning + (c) H500 BCE labels. All three are non-essential to the original research question.

## 4. Path forward — Recommendation: A first, then C if A is positive. Do NOT do B.

**Why not B (architecture surgery):** Gate 2 already showed that more fine-tuning at lr=5e-5 destroys representation. Variations on the head (per-horizon pooling, transformer head) are still fine-tuning unless we freeze the encoder. If we freeze it, we're back to Gate 1.

**Why A first (Wyckoff-state probes on frozen encoder):** Cheap (1-2 days), uses self-labels with falsifiable thresholds defined in the spec, directly tests the representation-quality claim. **If A passes**, Gate 2's failure is unambiguously "fine-tuning protocol failure, encoder is good" — move to a frozen-encoder-only deployment story (linear probes per task). **If A fails**, Gate 1's +1.9pp signal was probably a per-symbol direction prior, not tape-reading.

**Why C second (re-pretrain with stronger universality):** Expensive (~6h MPS or $6 H100) but addresses the diagnosed root cause. Recommended config from the Gate 3 reframe: widen LIQUID_CONTRASTIVE_SYMBOLS 6→12-15, anneal soft-positive weight 0.5→1.0, add cluster-cohesion as an early-stop diagnostic. Do NOT run C until A is complete.

## 5. Specific falsifier I would accept

**The single test that would convince me the encoder is genuinely reading the tape:**

Test 1 (Wyckoff-state retrieval) passes for **at least 3 of {absorption, buying_climax, selling_climax, spring, stress}** with k=10 nearest-neighbor label agreement at p<0.01 binomial vs marginal rate, **AND** Test 2 (axis recovery) passes for all three load-bearing features (effort_vs_result, climax_score, is_open) at ≥0.85 linear-probe accuracy.

**That conjunction is falsifiable, computable from existing 17 features without human annotation, has thresholds defined ex ante, and is independent of direction prediction at any horizon.**

If the encoder fails that conjunction, Gate 1's +1.9pp at H500 was a per-symbol direction prior — interesting but not what we set out to build. If it passes, Gate 2's failure was a fine-tuning artifact and the encoder has earned the "tape-reader" claim regardless of supervised end-to-end performance.

## Summary

(1) Gate 2 fail does NOT falsify the encoder — it falsifies the conjunction of [this encoder + end-to-end fine-tuning at lr=5e-5 + H500 BCE labels], with the regression-to-0.50 pattern (high-signal liquid symbols losing 7-15pp) being the canonical fingerprint of dense-noisy-gradient erasure of sparse-firing Wyckoff features. (2) Right phenomenology test: Wyckoff-state retrieval on frozen encoder using spec self-labels + axis-recovery probes for the three load-bearing features — both falsifiable, label-free, independent of direction at any horizon. (3) Path A first (frozen-encoder Wyckoff probes, 1-2 days), then C only if A fails — explicitly NOT B (architecture surgery is more fine-tuning). (4) Specific falsifier: ≥3 of 5 self-labeled Wyckoff states show k=10 nearest-neighbor label agreement at p<0.01 above marginal AND all three load-bearing features linearly recoverable from 256-d embedding at ≥0.85 accuracy.

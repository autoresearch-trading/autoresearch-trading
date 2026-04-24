# Council-5 Review: Step 4 Plan — Falsifiability + Multiple-Testing Hygiene

**Date:** 2026-04-24
**Reviewer:** council-5 (critical skeptic)
**Plan:** `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` (commit `ae7a970`)
**Evidence:** Gate 1 pass writeup, Gate 3 triage, cluster cohesion, surrogate sweep, spec amendment v2.

## Verdict

**SHIP WITH FOUR REQUIRED EDITS.** Plan is honest about three of four hidden DoFs (raises them as open questions). One is buried. The +0.5pp bar is not the right falsifier for "fine-tuning adds value" — it tests something already established.

## Q1 — Loss-weight rebalance: defensible IFF one schedule pre-committed before training

**Pass-chasing? No, but barely, and only if a specific piece is added.**

The argument "consistency-following with the amendment, not binding-gate amendment" survives scrutiny ONLY because H500-primary horizon was itself fixed by an ex-ante rule (council-5-imposed "shortest horizon at which PCA+LR ≥ 0.505"). Once H500 is fixed exogenously, weighting the loss to match the binding horizon is a defensible inductive choice.

**The buried DoF.** Open question 1 offers council-6 three loss-weight options. **Whichever is chosen MUST be committed before training.** A "let's try clean swap, fall back to annealed if it fails" path is a hidden grid-search over loss weights with held-out months as the selection criterion. Multiple testing on Feb+Mar.

**Required edit #1:** before launching, pre-commit ONE loss-weight schedule. Document choice + rationale. Second schedule may only be tried if first one's failure is debugged at architectural level (e.g., catastrophic forgetting), NOT by re-shooting the gate.

## Q2 — Per-symbol regression check: soft floor as written, real falsifier with right threshold

At Gate 1 measured per-symbol bootstrap CIs of ~0.09-0.13 (visible in `step5-gate3-triage.md` and `step5-surrogate-sweep.md`), a "1.0pp regression" criterion is INSIDE per-symbol sampling noise on every illiquid symbol and approximately at the noise edge on liquid pools. As written, criterion fires only on catastrophic destruction of per-symbol Gate 1 signal — useful as catastrophic-forgetting alarm, soft floor for the marginal-quality question.

**Right methodology:** compute per-symbol bootstrap CIs on BOTH the Gate 1 frozen-encoder LR and the fine-tuned CNN, on same Feb+Mar test pools, same 1000-resample protocol. Trip Criterion 2 if fine-tuned CNN's CI lower bound is below Gate 1 CI lower bound by **more than 1 CI half-width** (typically 0.045-0.065pp) on any single symbol AND point estimate regresses ≥1.0pp.

**Required edit #2:** rewrite Criterion 2 CI-aware: "fine-tuned CNN's per-symbol CI lower bound regresses below Gate 1 frozen-encoder LR's CI lower bound by more than the wider of (1.0pp, 1× CI half-width) on any single symbol." Without CI-aware language, criterion is theatre.

## Q3 — Multiple-testing on Feb+Mar reuse: real, but not Bonferroni-corrected

Plan correctly raises this as open question 4 but doesn't commit to a correction. Under deflated-Sharpe framework, repeated evaluation against same held-out window is exactly what deflation penalizes. After Gate 2, encoder will have been evaluated against Feb+Mar through:
1. Gate 1 H500 frozen-encoder LR (binding)
2. Gate 3 informational AVAX bootstrap (4 cells)
3. Cluster cohesion 6-anchor probe (Feb only)
4. Surrogate sweep 5 symbols × 2 months × 2 horizons = 20 cells (CI-aware)
5. Gate 2 fine-tuned CNN (this step)

≥5 distinct evaluations, not all binding-equivalent but all informationally selective on same encoder weights.

**Right correction is NOT Bonferroni on threshold.** Gate 2 tests fundamentally different model class (fine-tuned CNN vs flat-LR), not re-evaluation of SAME predictor on SAME hypothesis. Encoder representation reused but joint encoder+heads is new statistical object. Bonferroni would be over-correction toward "trust nothing."

**Right correction: Gate 2's threshold should be raised, but for a different reason** — see Q5.

**Required edit #3:** writeup must publish "trial count log" — every prior evaluation of `runs/step3-r2/encoder-best.pt` against Feb+Mar — and explicitly note that Gate 2 pass/fail must be interpreted in that light. No retroactive Bonferroni. Anti-amnesia hygiene, not threshold inflation.

## Q4 — Trial-count Bonferroni on Gate 2 threshold: NO

Combining Q3 logic with trial count: 6 trials × naive Bonferroni → ~1.3pp threshold. Most of the 6 trials were either informational (Gate 3, surrogate sweep) or non-overlapping in hypothesis space (cluster cohesion is geometric, not directional accuracy). Naive Bonferroni would be over-correction.

**However:** Q5's redesigned Criterion 3 implicitly addresses this by making Gate 2 a more discriminating test.

## Q5 — The "+0.5pp vs flat-LR" bar is a falsifier of the WRONG claim

**This is the single biggest issue.** Gate 1 already established encoder-LR beats flat-LR by +1.9-2.3pp on 17/24 (Feb), 14/24 (Mar). Gate 2's bar is +0.5pp on 15+/24. If fine-tuning preserves frozen-encoder representation roughly intact (expected at lr=5e-5, 5-epoch warmup), fine-tuned CNN inherits the +1.9-2.3pp Gate 1 margin "for free." Gate 2 as written falsifies "encoder still useful after we touched it" — already established by construction — NOT "fine-tuning adds incremental value."

**Plan needs Criterion 3 (NEW) directly testing fine-tuning's incremental value:**

> **Criterion 3 (NEW):** Fine-tuned CNN must beat **frozen-encoder LR** (Gate 1 winner) by ≥ 0.3pp balanced accuracy on **13+/24** symbols at H500, on BOTH Feb and Mar independently. This tests fine-tuning's marginal contribution; without it, Gate 2 is a guarantee, not a falsifier.

Numbers: 0.3pp / 13+/24 is softer than headline +0.5pp / 15+/24 because comparator is harder (frozen-encoder LR not flat-LR). 13/24 = 54.2% (slightly above majority). 0.3pp is one CI half-width — smallest detectable effect at our sample size.

**Required edit #4:** add Criterion 3 as binding pass condition. Frame Gate 2 as "fine-tuned CNN must beat BOTH flat-LR baseline AND frozen-encoder LR baseline." Without second comparator, Gate 2 cannot falsify "fine-tuning is a no-op."

## Open-question and escape-hatch audit

- **Open Q1 (loss weights):** addressed by required edit #1. Pre-commit one schedule.
- **Open Q2 (freeze duration):** not falsifiability concern — DL design call for council-6.
- **Open Q3 (per-symbol regression threshold):** addressed by required edit #2.
- **Open Q4 (eval data overlap):** addressed by required edit #3 (trial-count log).
- **600-event embargo:** spec line 278 still applies. Plan should explicitly state walk-forward construction uses 600-event gap between train tail (Jan 31) and test head (Feb 1). Currently silent. Sub-edit: add one line.
- **Test/train split:** spec uses calendar-month boundaries. Plan inherits but should state explicitly. Sub-edit: one line.
- **Hour-of-day probe re-emergence every 5 epochs:** real falsifier. Keep.

## Summary for orchestrator

(1) Loss-weight rebalance defensible Y/N: **YES**, conditional on pre-committing ONE schedule before training (open-question-1 menu is hidden grid-search if not closed). (2) Single biggest hidden DoF: **Gate 2's +0.5pp vs flat-LR is the wrong falsifier** — encoder-LR already beats flat-LR by +1.9-2.3pp at Gate 1; fine-tuning inherits margin. Gate 2 as written is guaranteed pass. (3) Tighten +0.5pp via Bonferroni? **NO** — add Criterion 3 (fine-tuned CNN beats frozen-encoder LR by ≥0.3pp on 13+/24, both Feb and Mar) instead. (4) **SHIP WITH FOUR REQUIRED EDITS.**

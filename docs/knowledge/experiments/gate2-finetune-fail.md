---
title: Gate 2 Fine-Tuning FAIL — Held-out CNN Underperforms Flat-LR
date: 2026-04-26
status: completed
result: failure
sources:
  - docs/experiments/step4-gate2-finetune.md
  - docs/experiments/step4-phase-b-third-strike-postmortem.md
  - docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md
  - docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md
  - docs/council-reviews/2026-04-26-post-gate2-strategic-c6.md
last_updated: 2026-04-27
---

# Experiment: Gate 2 Fine-Tuning FAIL

## Hypothesis

The +1.9-2.3pp Gate 1 margin (frozen-encoder linear probe at H500) carries
through to a fine-tuned CNN with weighted-BCE direction heads. Spec
threshold: ≥0.5pp over flat-LR on 15+/24 symbols at Feb+Mar held-out H500.

## Setup

- Checkpoint: `runs/step3-r2/encoder-best.pt` (E6, MEM=0.504, 376K params)
- Fine-tuning: freeze encoder 5 epochs (Phase A) → unfreeze at lr=5e-5 (Phase B)
- 4 direction heads (H10/H50/H100/H500), loss weights 0.10/0.20/0.50/0.20
- Walk-forward eval on Feb 2026 + Mar 2026 held-out (~21K + ~16K windows)
- Three binding criteria (all must hold on both months):
  - C1: ≥0.5pp over flat-LR on 15+/24 symbols
  - C2: no per-symbol regression > 1pp
  - C3: ≥0.3pp over frozen-encoder LR on 13+/24

## Result — FAIL on all three criteria, both months

| Criterion | Threshold | Feb 2026 | Mar 2026 |
|---|---|---|---|
| C1: vs flat-LR ≥0.5pp on 15+/24 | 15+/24 | **7/24 FAIL** | **10/24 FAIL** |
| C2: no per-symbol regression > 1pp | 0 violations | **16/24 FAIL** | **12/24 FAIL** |
| C3: vs frozen-encoder LR ≥0.3pp on 13+/24 | 13+/24 | **8/24 FAIL** | **12/24 FAIL** |

Aggregate H500 bal_acc (mean across 48 cells):
- flat-LR baseline: 0.5115
- frozen-encoder LR (Gate 1 protocol): 0.5061
- **fine-tuned CNN: 0.4947 (underperforms flat-LR by 1.7pp)**

## Diagnostic Pattern

**CNN regresses to 0.50.** Liquid symbols where flat-LR was high lost
7-15pp under fine-tuning:
- SUI 0.626 → 0.477 (-15pp)
- LTC 0.595 → 0.458 (-14pp)
- 2Z 0.633 → 0.499 (-13pp)
- ETH 0.543 → 0.484 (-6pp)
- LINK 0.605 → 0.530 (-7pp)

Illiquid alts where flat-LR was below 0.5 gained 6-9pp toward chance —
mean-reversion to noise floor, not signal extraction.

The +1.2pp val-fold gain (random 90/10 split, in-distribution) was
overfitting to training-period label imbalance. Walk-forward held-out
correctly falsified.

## Council-6 Diagnosis (2026-04-26 PM)

Shortcut learning at fine-tune time. Phase B fit day-conditional structure
visible in the in-distribution random val split, then collapsed to
per-symbol-and-hour priors on novel days. Evidence:
- CKA drift to Phase B = 0.061 (mild — NOT catastrophic forgetting)
- No train/val gap (NOT classical overfitting)
- Feb+Mar windows statistically similar to training (NOT distribution shift)

The per-symbol representation geometry has no shared trunk for fine-tuning
to specialize, so gradient signal collapses to per-symbol-and-hour
day-conditional shortcuts.

## Three Abort-Criterion Math Bugs On The Way

- **Bug #1 (Class A):** AM Phase A `BCE > 0.95×init` required β=0.632 from
  Gate 1 measured at 0.514. Patched (`8149aa8`).
- **Bug #2 (Class A):** PM Phase B `CKA > 0.95 after epoch 8` demanded ~3×
  faster encoder rotation than lr schedule supports. Patched (`322ab50`).
- **Bug #3 (Class B redundant guard):** Phase B `max(ΔCKA last 3) < 0.005`
  fired during scheduled OneCycleLR cosine cooldown. Adjudicated as Class B
  (subsumed by end-of-Phase-B CKA<0.95 upper bound, which the run passed).
  Pre-Gate-2 postmortem committed BEFORE eval (`f2f50dc`).

## Verdict — FAIL

Per pre-committed audit trail, no retry of Phase B with different
hyperparameters. The fine-tuning approach as configured is falsified.
Pretraining itself is NOT falsified — Gate 1 (frozen encoder) remains
valid.

## What We Learned

1. **Linear extractability ≠ end-to-end trainability.** A representation
   can carry a linearly-readable signal that supervised gradient descent
   destroys when applied jointly to encoder + head.
2. **Per-symbol-clustered geometry breaks fine-tuning.** Without a shared
   trunk across symbols, the gradient signal collapses to symbol-and-hour
   priors rather than amplifying the cross-symbol direction signal.
3. **Random 90/10 in-distribution val splits lie.** The +1.2pp val gain
   was completely artifactual — held-out walk-forward is the only trusted
   signal.
4. **Class A vs Class B abort taxonomy** was operationalized after this
   run produced three abort-criterion bugs (see Class A/B Abort Taxonomy
   decision).

## Related

- [Gate 1 Pass — Feb+Mar H500](../experiments/gate1-pass-feb-mar-h500.md)
- [Gate 4 Temporal Stability PASS](../experiments/gate4-temporal-stability-pass.md)
- [Class A/B Abort Taxonomy](../decisions/abort-criterion-taxonomy.md)
- [Cluster Cohesion Diagnostic](../experiments/cluster-cohesion-diagnostic.md) — explains the per-symbol-clustered geometry root cause

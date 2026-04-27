---
title: Random-Init Probe Protocol (Standard Cheap Arbiter)
date: 2026-04-27
status: accepted
decided_by: council-1 (methodology), council-5 (skeptic), council-6 (architect) — convergent recommendation
sources:
  - docs/council-reviews/2026-04-27-encoder-retrain-protocol.md
  - docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md
  - docs/experiments/goal-a-v2/2026-04-27-random-init-probe-plan.md
last_updated: 2026-04-27
---

# Decision: Adopt Random-Init Probe + 5b Adapter as Standard Cheap Arbiter

## What Was Decided

Before committing GPU compute to any encoder retrain (pretraining or
fine-tuning) on a new downstream task, run the **two-stage random-init
probe protocol**:

1. **Phase 0 — Linear probe:** forward-pass every cached window through a
   freshly-initialized encoder (no training); fit
   `LogisticRegression(class_weight='balanced')` on the 256-dim global
   embeddings. Compare to flat-LR baseline under unified day-blocked CV
   with paired bootstrap.
2. **Phase 1 — Non-linear adapter:** if Phase 0 fires ARCH_BOTTLENECK,
   train a small adapter head (`Linear(256→64) + ReLU + Dropout(0.2) +
   Linear(64→1)`, ~16K params) on the same frozen embeddings.

Cost: < 1 CPU-hour total for both phases. $0 cloud spend. Falsifiable.

## Why

1. **Unfalsifiability of the alternatives at small-n.** Pretrain-first
   (MEM+SimCLR) at n~100-200 positives can produce parity-with-flat numbers
   that we cannot distinguish from "successful autoencoder of the input
   features." End-to-end fine-tune at 376K params / 169 positives
   = 60× the v1 fine-tune ratio that already overfit.
2. **Manifold deficiency is empirically detectable.** The two-stage probe
   distinguishes linearity-artifact (closes most of the gap) from
   manifold-deficiency (closes <25% of the gap). See
   [manifold-deficiency](../concepts/manifold-deficiency.md).
3. **Compute discipline.** A $10-20 GPU run with no falsifier is worse than
   a $0 CPU run with one. The cheap probe is the right gating step.

## Alternatives Considered

| Approach | Why rejected as first move |
|---|---|
| Pretrain MEM+SimCLR first → re-probe | Unfalsifiable parity case (autoencoder artifact); MEM excludes cascade-relevant features anyway |
| End-to-end fine-tune with strong reg | Capacity ratio 60× the v1 overfit ratio; result wouldn't be diagnostic of architecture vs reg knobs |
| Skip probe, commit to GPU pretrain | Bypasses the falsifiability gate; v1 spent compute without a cheap arbiter |
| Symbolic / Bayesian feature analysis | Different question (do the features matter); doesn't probe encoder representation |

## Implementation Recipe

**Phase 0 — Linear probe:**
- 5-fold day-blocked CV partition, 600-event embargo at fold boundaries.
- 3 random encoder seeds {0, 1, 2}, BN1d `track_running_stats=False`.
- Pooled day-clustered bootstrap (1000 reps); paired bootstrap on
  `(AUC_encoder − AUC_flat)`.
- Decision tree: GREENLIGHT_FINETUNE if delta ≥ +0.02 AND delta_lo > 0;
  ARCH_BOTTLENECK if delta < −0.02; MATCHED_FLAT otherwise.

**Phase 1 — 5b adapter (only if Phase 0 fires ARCH_BOTTLENECK):**
- Same data assembly + folds as Phase 0 (re-use Phase 0 helpers).
- Adapter head: `Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)`.
- `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)`; AdamW(lr=1e-3, wd=1e-3).
- 50 epochs max, batch 256, no LR schedule, early stop on val AUC patience 5.
- Same paired bootstrap + decision tree.

**Reference implementations:**
- `scripts/random_init_probe.py` (Phase 0, 11 unit tests, reviewer-10 GREEN)
- `scripts/cascade_adapter_probe.py` (Phase 1, 7 unit tests)

## Impact

- Adopted as the standard pre-flight for any future encoder retrain
  proposal on this codebase.
- Council-1 + council-5 + council-6 ratification means subsequent reviews
  default to "did you run the probe?" as the first question.
- Combined with [pre-flight economics audit](pre-flight-economics-audit.md):
  any future Goal proposal must clear BOTH gates before architecture work.

## Related

- [Phase 0 Random-Init Probe (experiment)](../experiments/phase0-random-init-probe.md)
- [Phase 1 5b Adapter Test (experiment)](../experiments/phase1-adapter-test.md)
- [Manifold Deficiency (concept)](../concepts/manifold-deficiency.md)
- [Pre-Flight Economics Audit (decision)](pre-flight-economics-audit.md)

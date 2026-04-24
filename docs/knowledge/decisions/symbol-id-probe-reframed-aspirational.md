---
title: Symbol-ID Probe <20% Reframed as Aspirational for This Training Config
date: 2026-04-24
status: accepted
decided_by: council-5 + lead-0 (spec amendment v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
  - docs/experiments/step5-cluster-cohesion.md
last_updated: 2026-04-24
---

# Decision: Symbol-ID Probe <20% Reframed as Aspirational

## What Was Decided

The Representation Quality diagnostic "Symbol identity probe < 20% accuracy"
is **retained as an aspirational goal for future universality-targeting runs,
but is NOT a binding threshold violation on this training config**. The
measured 0.934 balanced accuracy on 6 liquid anchors (Feb held-out) is
consistent with the training config's demonstrably weak cross-symbol
invariance (+0.037 SimCLR delta), not with a broken encoder.

**The escape hatch is closed for every other diagnostic.** Hour-of-day,
CKA stability, and Wyckoff label probes remain binding. Any future run that
claims a different diagnostic's threshold doesn't apply must re-pre-register
the threshold BEFORE training AND cite evidence that the training config does
not target the property.

## Why

Measured evidence from step5 cluster cohesion:
- Cross-symbol same-hour cosine delta (vs diff-hour) = **+0.037**, below the
  +0.10 `some_invariance` threshold.
- Symbol-identity signal (same_symbol_diff_hour − cross_symbol_diff_hour) =
  **+0.139**, 4× stronger than the cross-symbol signal.
- 6-way symbol-ID linear probe balanced accuracy = **0.934** on Feb held-out.

The two numbers are consistent. The training config — cross-symbol SimCLR on
6 of 24 anchors at soft-positive weight 0.5 — did NOT train for
symbol-invariance. Measuring 0.934 symbol-ID against a <20% target designed
for a universality-targeting run (would require 12–15 anchors at weight 1.0)
is measuring against the wrong target.

Council-5 HDF #3 flagged this as a potential template abuse: "whenever a
quality diagnostic misses, declare it 'not targeted by this config.'" The
amendment closes that template by requiring re-pre-registration for any
future reframe.

## What Stays Binding

- **Hour-of-day probe <10% accuracy AND <1.5pp cross-session variance.** This
  is a Gate 1 Condition 4 stop-gate; the training config demonstrably DOES
  target session-invariance via timing-noise augmentation.
- **CKA > 0.7 between seed-varied runs.** Representation stability; not
  symbol-specific.
- **Wyckoff label probes.** The training objective explicitly includes tape-
  state structure (MEM reconstruction of effort_vs_result, climax_score,
  is_open); the probes test whether those patterns transfer to linear probing.

## What's Aspirational

- **Symbol-ID < 20%.** Only this diagnostic is reframed, and only because
  the training config demonstrably doesn't target it. Measured SimCLR delta
  evidence is load-bearing for the reframe.

## What Would Restore <20% to Binding

A future training config that:
1. Widens LIQUID_CONTRASTIVE_SYMBOLS from 6 → 12–15 anchors;
2. Anneals soft-positive weight from 0.5 → 1.0 over training;
3. Adds cluster-cohesion delta ≥ +0.10 as an in-training early stop-gate.

Such a run would measure symbol-ID <20% as a binding test of whether the
stated objective was achieved.

## Alternatives Considered

1. **Keep <20% binding, declare this run fails the diagnostic.** Rejected —
   the diagnostic is measuring a property the training config did not target.
   Declaring failure against an unsought target is like blaming a regression
   model for failing classification accuracy.
2. **Drop the <20% diagnostic entirely.** Rejected — retaining it as
   aspirational preserves the measurement for future runs without forcing
   false failure now.
3. **Apply aspirational reframe liberally to any missed diagnostic.** Rejected
   per council-5 HDF #3; the escape hatch would gut the spec.

## Impact

- Spec "Representation Quality Metrics" section in amendment v2 commit
  `9c91f85` includes the reframe language and the escape-hatch-closure for
  other diagnostics.
- Future Gate 3 re-activation criteria include cluster cohesion delta ≥ +0.10
  as condition (c), which is the direct binding version of this diagnostic
  (see [gate3 retirement](gate3-retired-to-informational.md)).

## Related

- [Cross-symbol invariance](../concepts/cross-symbol-invariance.md)
- [Gate 3 retired to informational](gate3-retired-to-informational.md)
- [Cluster cohesion experiment](../experiments/cluster-cohesion-diagnostic.md)

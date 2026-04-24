---
title: Horizon Selection Rule for Gate 1
date: 2026-04-24
status: accepted
decided_by: council-5 + lead-0 (spec amendment v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
last_updated: 2026-04-24
---

# Decision: Horizon Selection Rule for Gate 1

## What Was Decided

The primary binding horizon for Gate 1 is defined ex-ante as:

> **The shortest horizon in {10, 50, 100, 500} at which PCA+LR achieves
> balanced accuracy ≥ 0.505 on the held-out universe.**

On this encoder's data, only H500 meets that bar (PCA+LR balanced accuracy:
0.508 on Feb, 0.508 on Mar; 0.50x at H10/H50/H100). On this training run the
selection produces H500.

**Future training runs MUST apply this selection criterion before reading
encoder numbers.** No horizon-shopping on encoder-specific performance.

## Why

Council-5 Q4 identified the horizon drift from pre-registered H100 to
amendment-declared H500 as the single sharpest p-hacking risk in the Gate 1
amendment. The textbook definition of horizon-selection-post-hoc: the
pre-registration said H100, multiple horizons were computed, H100 came in at
noise, H500 showed signal, H500 was declared primary.

The defense in the amendment is that H100 is at noise floor for **every
predictor including PCA, RP, and shuffled** on this data — not "encoder
fails at H100, encoder passes at H500" but "H100 is not a testable horizon
at this sample size for any predictor." That is a power analysis, not
p-hacking. But without a pre-registered rule, future runs have an escape
hatch: compute all horizons, declare the best one primary.

This rule closes the escape hatch. The selection depends ONLY on flat
baselines (PCA+LR), not on encoder output. Any future run publishes the
PCA+LR balanced accuracy at each horizon on the held-out universe first; the
shortest horizon above 0.505 is primary; encoder numbers are then read against
that pre-selected horizon.

## Why 0.505 (not 0.51 or chance)

0.505 is 0.5pp above chance — below the Gate 1 Majority+1pp threshold
(encoder must beat 0.510) but above the null-hypothesis 0.500 band. It is the
smallest bar that forces PCA+LR to be doing *something* that a useful
predictor could improve on. Under 0.505 PCA+LR is indistinguishable from
coin-flipping; picking that horizon would make the Majority+1pp test trivial
(anything above chance wins).

## Alternatives Considered

1. **Horizon pre-fixed to H500.** Rejected — does not generalize across
   training runs; a future encoder with different training data may have its
   strongest signal at H100 or H50.
2. **Primary = longest horizon.** Rejected — adversarial to future runs that
   pretrain on more data and may have stronger short-horizon signal.
3. **Horizon = whichever the encoder does best on.** Rejected — this is
   exactly the p-hacking pattern the rule exists to prevent.

## Impact

- Spec Gate 1 section: horizon-selection rule added as binding language in
  amendment v2 (commit `9c91f85`).
- Every future Gate 1 report must publish the per-horizon PCA+LR balanced
  accuracy table before identifying the primary horizon.
- Horizon structure across all gates is now explicit (see
  [gate4 rewrite](gate4-rewrite-for-coherence.md) for the cross-gate map).

## Related

- [Gate 1 window amended](gate1-window-amended-feb-mar-h500.md)
- [Gate 4 rewrite](gate4-rewrite-for-coherence.md)

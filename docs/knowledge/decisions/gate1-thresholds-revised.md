---
title: Gate 1 — 4-Condition Dual-Control Threshold
date: 2026-04-15
status: accepted
decided_by: council-1 + council-5 + council-6 (round 6)
sources:
  - docs/council-reviews/council-1-gate0-rp-equivalence.md
  - docs/council-reviews/council-6-gate0-impact-on-pretraining.md
last_updated: 2026-04-15
---

# Decision: Gate 1 — 4-Condition Dual-Control Threshold

## What Was Decided

Gate 1 (linear probe on frozen pretrained embeddings, H100, April 1–13 held-out)
requires ALL four conditions to hold simultaneously — all stop-gates:

1. **Absolute floor:** balanced accuracy ≥ 51.4% on 15+/25 symbols.
2. **vs. Majority:** balanced accuracy > Majority-class baseline + **1.0pp** on 15+ symbols.
3. **vs. Random Projection:** balanced accuracy > RP-control + **1.0pp** on 15+ symbols.
4. **Session-decorrelation:** hour-of-day 24-class probe on the same frozen
   embeddings < **10%** accuracy AND stratified variance < **1.5pp** across UTC
   sessions (Asia 0–8 / Europe 8–16 / US 16–24).

## Why

The prior threshold — "exceed PCA by 0.5pp on 15+ symbols" — was invalidated by
the [Gate 0 4-baseline experiment](../experiments/gate0-4baseline-grid.md).
PCA+LR is statistically indistinguishable from both the majority-class predictor
and a frozen random projection on balanced accuracy. "Beat PCA by 0.5pp"
collapses to "beat chance by 0.5pp" — a 0.5pp edge over random can come from
session-of-day leakage or per-fold label imbalance, not learned microstructure.

Requiring dual controls (Majority AND RP) and tightening the margin to 1.0pp
forces the CNN probe to demonstrate adaptive, non-chance, non-linear structure.
Adding the hour-of-day probe catches the specific shortcut council-5 and council-6
identified: the encoder could satisfy (1–3) by learning session-of-day rather
than tape microstructure.

## Alternatives Considered

1. **Keep +0.5pp vs PCA.** Rejected — equivalent to "beat chance by 0.5pp"
   given PCA≈Majority.
2. **+2.0pp margin.** Rejected as premature pessimism — 1.0pp is
   approximately 1 SE at our fold sizes; tighter margins lack statistical power
   at the 25-symbol level.
3. **Skip the hour-of-day probe** and use only accuracy-based thresholds.
   Rejected — council-6 showed the CNN has denser access to session-of-day
   than the LR does (per-event `time_delta`, `prev_seq_time_span`) and
   MEM's reconstruction objective doesn't penalize learning session identity.

## Impact

- Spec amendment `1f86d52` replaces the Gate 1 section.
- Pretraining monitoring adds hour-of-day probe every 5 epochs (early warning).
- Pre-pretraining sanity check added: hour-of-day-only LR must NOT exceed
  PCA+LR on flat features by > 0.5pp on 5+ symbols (catches session leakage in
  the `_last` statistic block).
- CLAUDE.md gotchas #28, #29, #30 formalize the change.

## Related

- [Gate 0 Baseline Grid](../concepts/gate0-baseline.md)
- [Session-of-Day Leakage](../concepts/session-of-day-leakage.md)
- [Balanced Accuracy decision](balanced-accuracy-all-horizons.md)

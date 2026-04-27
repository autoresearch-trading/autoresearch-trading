---
title: Goal-A v2 Program Closure (Cascade-Onset Encoder Retrain)
date: 2026-04-27
status: completed
result: failure
sources:
  - docs/experiments/goal-a-v2-program-end-state.md
  - docs/experiments/goal-a-feasibility/
last_updated: 2026-04-27
---

# Experiment: Goal-A v2 Program Closure

## Hypothesis

Predict liquidation-cascade onset using Pacifica-unique `cause` /
`event_type` fields. Train v1's 376K-param dilated CNN end-to-end on a
cascade-onset binary label, target OOS AUC ≥ 0.85 to flip strategy
economics under maker fees.

## Setup

Two phases of cheap (CPU-only) falsifiable evidence:

1. **Feasibility chain (8 artifacts, `docs/experiments/goal-a-feasibility/`).**
   Per-symbol oracle headroom, maker-mode cost band, maker adverse-selection
   sim, open-flow imbalance novelty test, v1 encoder confidence-conditional,
   cascade synthetic-label, cascade real-label LR + robustness, cascade
   direction LR + marginal-long, cascade real-label OOS.

2. **Encoder retrain pre-flight.**
   - [Phase 0 random-init probe](../experiments/phase0-random-init-probe.md):
     ARCH_BOTTLENECK (paired delta −0.18 [−0.26, −0.11]).
   - [Phase 1 5b adapter test](../experiments/phase1-adapter-test.md):
     KILL_ARCH_BOTTLENECK_CONFIRMED (paired delta −0.13 [−0.20, −0.06]).

## Result

**Closed 2026-04-27 (PM-late).** Tag: `v2-program-closed`.

Positive findings:
- Cascade-onset prediction signal is real at flat-LR (unified-CV AUC=0.8373).
- Pacifica-unique fields work as labels.

Program-killing findings:
- Maker's Dilemma blocks economics independent of AUC
  (E[realized|filled] = −7.89bp; encoder cannot solve adverse selection).
- Taker-side does NOT reverse the math: `headroom_table.csv` enumerates
  300 cells under taker fees at 55% accuracy → 0/300 survivors.
- Encoder architecture cannot extract cascade signal beyond hand features
  (Phase 0 + Phase 1 both confirm).
- Pacifica-unique conditional axes exhausted (3 independent tests, 3
  negatives: `open_imbalance.md`, `cascade_direction.md`, `encoder_confidence.md`).
- April holdout permanently consumed → no untouched OOS remains.

## What We Learned

1. **Representation learning cannot create signal where venue economics
   already kill it.** Both v1 (direction prediction) and v2 (cascade-onset)
   found real signal at the +1pp range that fee-economics killed. The
   binding constraint was venue + execution, not architecture.
2. **The 5b adapter test is the right cheap arbiter** for "does encoder X
   add signal beyond flat features Y for task Z." Adopted as the standard.
3. **Pre-flight economics audit** before any architecture work is the v2
   methodology contribution. The next Goal-* program should ask: *is there
   a fee-clearing edge in the LABELS alone, evaluated at the highest
   plausible accuracy?* If no, encoder retrain is moot.
4. **The flat-LR unified-CV AUC=0.8373** is the published reference for
   cascade-onset prediction on this dataset. Single-shot OOS numbers like
   0.778 (commit `b0de994`) are downward-biased and should not be cited.

## Verdict

Goal-A v2 closed. ~50 CPU-min total compute spent across Phases 0 + 1; $0
cloud. Re-entry conditions documented in
`docs/experiments/goal-a-v2-program-end-state.md`:

1. ≥ 30 days fresh data accrual (rebuilds untouched holdout).
2. Pacifica-unique conditional bucket clears frac_positive ≥ 0.595 at
   n ≥ 400 on ≥ 5 symbols, Bonferroni-corrected, on FRESH data.
3. Venue fee/execution schedule changes.
4. New label class with mechanism + execution path that doesn't trigger
   Maker's Dilemma.

## Related

- [V2 Program Closure decision](../decisions/v2-program-closure.md)
- [Pre-Flight Economics Audit](../decisions/pre-flight-economics-audit.md)
- [Phase 0 Random-Init Probe](phase0-random-init-probe.md)
- [Phase 1 5b Adapter Test](phase1-adapter-test.md)

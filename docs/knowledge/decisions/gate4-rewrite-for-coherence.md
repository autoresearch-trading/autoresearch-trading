---
title: Gate 4 Rewritten for Coherence with Amended Gate 1
date: 2026-04-24
status: accepted
decided_by: council-1 + lead-0 (spec amendment v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-1-amendment-2026-04-24.md
last_updated: 2026-04-24
---

# Decision: Gate 4 Rewritten for Coherence with Amended Gate 1

## What Was Decided

Rewrite Gate 4 (temporal stability) for coherence after the amended Gate 1
changes "training period" to Oct 16 – Jan 31 and "held-out" to Feb + Mar. The
old language ("months 1-4 vs months 5-6") referenced nonexistent months and
would be silently drifted on the next run.

**New text:** within-training-period stability. Split training Oct 16 – Jan 31
into two halves (Oct–Nov vs Dec–Jan). Train a linear probe on each half
independently with the frozen encoder, evaluate both probes on the same
held-out Feb + Mar fold at H500. **STOP if balanced accuracy drops > 3pp
between the two within-period probes on > 10/24 symbols at H500.**

## Why

Council-1 Q5 flagged the original Gate 4 as silent drift. Under the amended
Gate 1 protocol (training Oct 16 – Jan 31, held-out Feb + Mar), there are no
"months 5-6 of training" — the original Gate 4 phrasing referenced months
that don't exist in the new protocol. Leaving it unchanged would force the
next run's analyst to either (a) invent a mapping, likely non-reproducibly,
or (b) silently skip Gate 4.

Additionally, the Feb-vs-Mar independent-pass requirement in amended Gate 1
already *is* a held-out-period temporal-stability test. Gate 4 needs to be a
distinct test of **within-training-period** stability to avoid double-dipping
on Gate 1's Feb-vs-Mar consistency check.

## Alternatives Considered

1. **Retire Gate 4 entirely and fold its function into Gate 1 Feb-vs-Mar.**
   Rejected — the two tests measure different properties. Gate 1 Feb-vs-Mar
   checks held-out-period robustness; Gate 4 checks whether the encoder
   learned training-period-specific noise.
2. **Re-map "months 1-4 vs 5-6" to "Oct-Dec vs Jan-Mar" naively.** Rejected —
   puts part of the held-out set into the "training" half and violates the
   held-out irrevocability.
3. **Cross-period stability: training-fold probe vs held-out-fold probe
   accuracy.** Partially redundant with the existing Gate 1 structure.

## Impact

- Spec section "Gate 4: Temporal Stability" rewritten in commit `9c91f85`
  (amendment v2).
- Horizon structure across gates now explicit:
  - **H500 primary binding** for Gate 1 and Gate 4.
  - **H100 informational only** after the H100-noise-floor finding.
  - **H10/H50 informational** at Gate 0 baseline context.
  - **Gate 3 reports H100 and H500** with bootstrap CIs (informational).
- The stop-gate threshold is preserved numerically: < 3pp drop, > 10/24 symbols.

## Related

- [Gate 1 window amended](gate1-window-amended-feb-mar-h500.md)
- [Horizon selection rule](horizon-selection-rule.md)

---
title: Goal-A v2 Program Closure
date: 2026-04-27
status: accepted
decided_by: council-5 STOP recommendation, validated by council-4 + council-6 alignment + Phase 0 + Phase 1 empirical results
sources:
  - docs/experiments/goal-a-v2-program-end-state.md
  - docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md
last_updated: 2026-04-27
---

# Decision: Close Goal-A v2 (Cascade-Onset Encoder Retrain)

## What Was Decided

Stop all compute on the cascade-onset encoder retrain. Tag the program as
`v2-program-closed` (parallel to `v1-program-closed`). Document re-entry
conditions. Until at least one re-entry condition holds, no further
Goal-A compute is warranted.

## Why

Three independent kill arguments converged in a single afternoon:

1. **Maker's Dilemma kills economics independent of AUC.** Universe median
   breakeven 0.311; E[realized|filled] = −7.89bp; 0/300 cells with breakeven
   in (0.50, 0.55). Even AUC=0.90 with top-1% precision 35% does not flip
   the post-fee envelope. **Encoder cannot solve adverse selection** — it's
   a fill-conditional venue property.
2. **Taker-side does NOT reverse the math.** `headroom_table.csv` already
   enumerates taker economics: 0/300 survivors at 55% accuracy. Cost-structure
   reversal is not viability reversal.
3. **Pacifica-unique conditional axes are exhausted.** Three independent
   conditional-subpopulation tests (`open_imbalance.md`,
   `cascade_direction.md`, `encoder_confidence.md`) are all negatives.
   With holdout consumed and a 400-cell search space, no v3 audit can
   produce a falsifiable GO signal on already-touched data.
4. **Encoder architecture cannot extract cascade signal beyond hand
   features.** Phase 0 and Phase 1 paired deltas firmly below zero (CIs do
   not cross). Manifold-deficiency, not linearity-artifact.

## Alternatives Considered

| Path | Why rejected |
|---|---|
| Pretrain MEM+SimCLR first | Council-5: unfalsifiable (autoencoder artifact risk). Council-4: MEM is hobbled (excludes cascade-relevant OFI features); SimCLR ±25 jitter destroys right-edge signal. |
| End-to-end fine-tune with strong reg | Council-6: 376K params / 169 positives = 2225 params/positive = 60× v1 ratio that already overfit. Not falsifiable at this n. |
| Cascade-aware MEM (re-include OFI features, drop SimCLR) | Conditional on 5b passing — 5b failed. |
| TAKER-side Pacifica-unique signal pivot (Goal-A v3 candidate) | Council-5 sanity check: economics already audited (0/300 survivors); Pacifica-unique axes already probed (3 negatives); search space too large to be falsifiable on consumed-holdout data. |
| Wait 30 days for fresh data | Re-classified as a re-entry CONDITION, not an alternative to closure. |

## Re-Entry Conditions

Any one suffices to reopen Goal-A:

1. **Fresh data accrual ≥ 30 days post-Apr-26.** Rebuilds untouched holdout.
2. **Pacifica-unique conditional bucket** clears `frac_positive ≥ 0.595`
   at `n ≥ 400` on `≥ 5 symbols`, Bonferroni-corrected for the 400-cell
   search space, on data not yet touched.
3. **Venue economics change** — fee schedule, taker-protected execution
   modes, or other structural change that reduces adverse selection.
4. **New label class** with BOTH (a) plausible mechanism for being learnable
   from raw tape (not flat features) AND (b) execution path that doesn't
   trigger Maker's Dilemma.

## Impact

- Frees compute budget for whatever the next Goal becomes.
- Reusable artifacts retained: `tape/` package (289+ tests), `data/cache/`
  (4453 shards), all probe scripts (`random_init_probe.py`,
  `cascade_adapter_probe.py`, `cascade_precursor_probe.py`).
- Methodology contribution: pre-flight economics audit before any
  architecture work. See [pre-flight-economics-audit](pre-flight-economics-audit.md).
- Pattern from v1+v2: representation learning cannot create signal where
  venue economics already kill it. Both programs found real +1pp signal that
  fee-economics blocked.

## Related

- [Pre-Flight Economics Audit](pre-flight-economics-audit.md)
- [Random-Init Probe Protocol](random-init-probe-protocol.md)
- [V1 Path A Program Closure](path-a-program-closure.md)
- [V2 program end-state document](../../experiments/goal-a-v2-program-end-state.md)

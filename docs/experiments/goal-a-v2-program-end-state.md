# Goal-A v2 Program End-State

**Date:** 2026-04-27 (PM-late, post-Phase 1)
**Closure tag** (recommended): `v2-program-closed`
**Predecessor:** v1 closed 2026-04-27 AM (`v1-program-closed`,
`docs/experiments/step4-program-end-state.md`).

## What v2 was

A pivot from v1's failed direction-prediction representation learning to
**cascade-onset prediction**, exploiting Pacifica-unique `cause` /
`event_type` fields. The hypothesis: predicting *liquidation-cascade onset*
(rather than direction) would lift AUC into a regime where top-k%
precision flips strategy economics under maker fees.

## What v2 produced

### Positive findings

1. **The cascade-onset prediction signal is real and OOS-generalizing**
   under flat-LR (83-dim hand features). In-sample AUC=0.815 [0.772, 0.848]
   on Apr 1-13 (commit `f55c9ac`); OOS AUC=0.778 [0.732, 0.833] on Apr 14-26
   (commit `b0de994`); both day-clustered, distinguishable from
   shuffled-OOS at H500 (n=96 cascades). Per-symbol concentration:
   SUI/AVAX/PENGU/XRP carry the signal at flat-LR.

2. **Unified-CV flat-LR baseline established at AUC=0.8373 [0.8087, 0.8652]**
   on the merged Apr 1-26 dataset (Phase 0, commit `3110abc`). The earlier
   single-shot OOS=0.778 was downward-biased by the flat-LR being a
   one-realization estimate — the unified 5-fold day-blocked CV gives a
   more accurate baseline.

3. **Pacifica-unique fields work as labels** but not as feature axes that
   beat already-published OFI signals. `open_imbalance.md` (council-4
   novelty test) confirmed this — extreme open-flow imbalance is captured
   by standard OFI; the Pacifica-unique `is_open` axis adds nothing.

### Negative findings (the program-killers)

1. **Maker's Dilemma blocks strategy economics independent of AUC.**
   `maker_adverse_selection.md` measured E[realized|filled] = -7.89bp; with
   universe median breakeven at 0.311, 0/300 cells with breakeven in
   (0.50, 0.55). Even AUC=0.90 with top-1% precision 35% does not flip
   economics because adverse selection consumes the post-fee envelope.
   **The encoder cannot solve this** — adverse selection is a fill-conditional
   venue property, not a label-prediction problem.

2. **Taker-side execution does not reverse the math.**
   `headroom_table.csv` enumerates 300 (symbol × size × horizon) cells under
   taker assumptions at v1's measured 0.55 accuracy; **survivors: 0/300**.
   The closest-to-survivor (PUMP@H500 $1k at 60% accuracy) clears at
   +0.51bp, well inside noise.

3. **The encoder architecture cannot extract cascade signal beyond hand
   features.** Two falsifiable experiments:
   - Phase 0 (random-init linear probe, commit `3110abc`): paired delta
     vs flat-LR = -0.1812 [-0.2594, -0.1063]. ARCH_BOTTLENECK fired.
   - Phase 1 (5b non-linear adapter, commit `7705319`): paired delta
     = -0.1307 [-0.2003, -0.0577]. KILL_ARCH_BOTTLENECK_CONFIRMED.
     Non-linear adapter closes only 4pp of the 18pp Phase 0 gap.

4. **Pacifica-unique conditional sub-populations are exhausted.** Three
   independent tests on the strongest available axes:
   - `open_imbalance.md` — conditioning on `is_open` extreme tail: failed
     at chance (0.500 frac_positive vs 0.55 needed).
   - `cascade_direction.md` — directional LR conditional on cascade-likely
     windows: AUC=0.441, marginal-long net-negative.
   - `encoder_confidence.md` — v1 encoder top-quintile confidence: at chance.

5. **Holdout permanently consumed.** April 14-26 was deliberately consumed
   on 2026-04-27 (gotcha #17, commit `b0de994`). No untouched
   cascade-labeled holdout remains. Any further evaluation runs on data
   already touched by the feasibility chain.

## What v2 cost

- Compute: ~50 CPU-minutes total across Phase 0 + Phase 1. **$0 cloud
  spend.** No GPU was launched.
- Wall-clock: ~3-4 hours of orchestration (this session) plus the prior
  feasibility chain (4ae3102 → b0de994).
- Code: `scripts/random_init_probe.py` (1786 LOC), `scripts/cascade_adapter_probe.py`
  (~1100 LOC), `scripts/cascade_precursor_probe.py` (5883 LOC), `scripts/goal_a_feasibility.py`,
  + 18 unit tests across the new probes. All retained for re-entry conditions.

## Why v2 closed (one paragraph)

Cascade-onset prediction is a real signal at the flat-LR level (AUC ~0.84 in
unified CV), but (a) the Maker's Dilemma blocks strategy economics under
maker fees independently of AUC, (b) taker-side economics are already audited
at 0/300 survivors at the v1 +1pp accuracy level, (c) the encoder
architecture cannot extract cascade signal beyond what hand features capture
(Phase 0 + Phase 1 paired deltas firmly below zero), and (d) all
Pacifica-unique conditional axes (`cause`, `event_type`, `is_open`) have
been probed and yielded nothing beyond standard microstructure. With the
April holdout consumed, any further work runs on already-touched data with
a 400-cell search space — falsifiable evaluation is not available. The
honest answer is: **this dataset on this venue does not have a tradeable
edge beyond already-priced microstructure for cascade-onset or directional
prediction at the horizons and execution regimes accessible to a
representation-learning approach.**

## Re-entry conditions

If any of these change, Goal-A becomes worth reopening:

1. **Fresh data accrual ≥ 30 days.** Rebuilds an untouched holdout; allows
   a CLEAN OOS test of any v3 hypothesis.

2. **A Pacifica-unique conditional bucket** clears the binding bar on the
   FRESH data (post-Apr-26): frac_positive ≥ 0.595 at n ≥ 400 events on
   ≥ 5 symbols, Bonferroni-corrected for the 400-cell search space, on
   data the team has not touched. (Council-5 PR3, `2026-04-27-pretrain-vs-endtoend-synthesis.md`
   §"Council-5 STOP recommendation".)

3. **Venue economics change.** Pacifica fee schedule moves; or a venue
   feature reduces adverse selection (e.g. taker-protected execution
   modes that didn't exist when this analysis ran).

4. **A new label class** beyond direction/cascade-onset is identified that
   has BOTH (a) a plausible mechanism for being learnable from raw tape
   structure (not flat features) AND (b) a plausible execution path that
   doesn't trigger the Maker's Dilemma.

Until at least one of these is true, no further Goal-A compute is
warranted.

## Audit trail

| Phase | Commits | Key artifact |
|---|---|---|
| v2 kickoff | `9bb29e3` | state.md pivot to cascade-onset |
| Feasibility chain (8 artifacts) | `4ae3102` → `b0de994` | `docs/experiments/goal-a-feasibility/` |
| OOS test (holdout consumed) | `b0de994` | `cascade_precursor_oos.md` |
| Phase 0 plan + impl + run | `aa5bdea`, `64e3587`, `694d14c`, `3110abc` | random-init linear probe |
| Phase 0 verdict: ARCH_BOTTLENECK | `40832d0` | -18.1pp gap |
| Council round 2 (3-voice synthesis) | `dc4e949` | `pretrain-vs-endtoend-synthesis.md` |
| Phase 1 impl + run | `ae821f7`, `bc4ef3b`, `7705319` | 5b adapter test |
| Phase 1 verdict: KILL_ARCH_BOTTLENECK_CONFIRMED | `ce55ed4` | -13.1pp paired delta |
| Council-5 sanity check on v3 pivot | (this session) | STOP per (3) recommendation |
| v2 closure | (next commit) | this document |

## What this means for future Goal-* programs

The v2 closure is a sharper version of v1's lesson: **representation
learning cannot create signal where none exists at the venue-economics level**.
Both v1 and v2 found real signal at the +1pp range that fee-economics killed.
The next Goal that proposes "learn a representation to lift trading
performance on this dataset" should fail-fast on the venue economics
question BEFORE any model code is written:

- Is there a fee-clearing edge in the *labels* alone, evaluated at the
  *highest plausible* prediction accuracy?
- If no, the encoder retrain question is moot. Pivot or stop.
- If yes, then ask whether flat features ALREADY capture that edge — if
  yes, no encoder needed; if no, encoder retrain becomes a real question.

This is the v2 contribution to the project's methodology — a pre-flight
economics audit that gates any future architecture work.

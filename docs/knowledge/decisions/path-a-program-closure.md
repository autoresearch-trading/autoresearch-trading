---
title: Path A — Close Program + Publish, Goal-A Abandoned for This Stack
date: 2026-04-27
status: accepted
decided_by: user (after lead-0 EV pushback on Path β)
sources:
  - .claude/skills/autoresearch/state.md (2026-04-27 state)
  - docs/experiments/step4-program-end-state.md
  - docs/council-reviews/2026-04-27-phase2-prereg-c1-methodology.md
  - docs/council-reviews/2026-04-27-phase2-prereg-c2-microstructure.md
  - docs/council-reviews/2026-04-27-phase2-prereg-c5-stops.md
last_updated: 2026-04-27
---

# Decision: Path A — Close Program + Publish

## What Was Decided

The tape representation learning program closes at end-state on
2026-04-27. The publishable artifact is `step4-program-end-state.md`
with the calibrated interpretation appendix and council-1 QA edits
applied. **Goal-A (profitable paper trading on this stack) is
abandoned.** Future Goal-A pursuit requires a different research
program with a different objective + likely different data + likely
different architecture.

## What's Discarded

- The specific encoder `runs/step3-r2/encoder-best.pt` as a profitable
  trading signal — fee-blocked at every horizon under DEX perp realism
- The Step 4 fine-tuning approach (Gate 2 falsified)
- Option β (cross-symbol re-pretrain on this same data + architecture)
  — EV doesn't pencil out

## What's Kept

- The pipeline (`tape/` package, 289 tests, 4003 cached shards, 641K
  windows) — reusable infrastructure
- The methodology (pre-registration discipline, abort-criterion
  taxonomy, Class A/B taxonomy, bootstrap CIs, multi-month held-out
  validation, 4-baseline Gate 0 grid)
- The diagnostic finding (flat aggregation destroys sequential signal;
  sequential CNN extracts +1pp at H500 but not enough to clear DEX perp
  fees)
- The load-bearing features (`effort_vs_result`, `climax_score`,
  `is_open` — Wyckoff-derived, reusable)
- The knowledge base (`docs/knowledge/INDEX.md`) — institutional memory

## Why Not Path β

Three Phase 2 pre-registration reviews (c-1 methodology, c-2
microstructure, c-5 stops) UNANIMOUSLY rejected Phase 2 as written.
The cross-symbol re-pretrain proposal had:

- Joint probability of laddering to Goal-A: ~10-25% under generous
  assumptions (3 conditions must all hit: cohesion delta improves to
  ≥+0.10, new edge size ≥+5pp at H100 over RP, fine-tuning works on
  new geometry)
- Cont-de Larrard tension: universalizing geometry could *reduce* per-
  symbol locally-readable signal, which is exactly what the LR probe
  extracts
- The interpretive question (encoder reads tape vs priors) cannot be
  cleanly answered by re-pretraining — only by a different
  measurement design

EV math on Path β: ~3 weeks focused work × ~15-25% conditional success
× (a profitable system) does not pencil against α's $0 close + publish.

## Alternatives Considered

- **Path α (close):** Accepted.
- **Path β (cross-symbol re-pretrain):** Rejected on EV.
- **Path C (different objective on same stack):** Deferred to a future
  research program if the user pursues Goal-A again. Genuinely different
  problem formulation (e.g., learn embeddings whose nearest-neighbor
  retrieval predicts realized slippage). Would be a new program with a
  fresh spec.
- **Path "skip publish, pivot now":** Rejected — saves ~1 session but
  loses the calibrated finding as institutional memory.

## What Future Goal-A Pursuit Should Change

The honest framing is: "self-supervised representation learning on raw
order events, with **this** objective + **this** architecture +
**this** data + **this** evaluation rubric, doesn't clear DEX perp
fees." Each of those is separable.

Most impactful changes per the program's own findings:

1. **Different objective** — drop direction prediction. Predict
   execution-aware quantities (signed mid-move conditional on
   tradeable size, time-to-fill, slippage cost, queue position).
2. **Different data** — orderbook events (L2 changes), not just
   trades + 24s OB context.
3. **Different evaluation** — backtested PnL net of fees + slippage as
   primary metric, not balanced accuracy. The current rubric was
   measurement-rigorous but trading-blind.
4. **Architecture is less load-bearing.** The CNN+SimCLR worked well
   enough for representation extraction; the bottleneck was
   signal-to-fee ratio, not architecture.

## Impact

- Program closes, knowledge base compiled, freeze tag applied.
- The reusable pipeline, methodology, and load-bearing features carry
  over to any future attempt.
- The +1pp Gate 1 / Gate 4 result is published as a calibrated
  representation-learning finding, not a trading claim.

## Related

- [Calibrated Interpretation](calibrated-interpretation-per-symbol-clustered.md)
- [Path D — Drop Multi-Probe Battery](path-d-drop-battery.md)
- [Class A/B Abort Taxonomy](abort-criterion-taxonomy.md)
- [Tape-State Diagnostic Off-Ramp](../experiments/tape-state-diagnostic-off-ramp.md)
- [Pivot to Representation Learning](pivot-to-representation-learning.md) — the original pivot from supervised; this program closes that arc

# Council Synthesis — Phase 0 Verdict, Pretrain vs End-to-End vs STOP

**Date:** 2026-04-27 (PM-late, post-Phase 0)
**Trigger:** Phase 0 random-init encoder linear probe LOSES to flat-LR by 18.1pp
(commit `3110abc`). ARCH_BOTTLENECK fired. The plan's decision tree leaves
"pretrain vs end-to-end-with-reg" as a council decision parameterized on gap
size — gap is large.

**Reviewers:** council-4 (phenomenologist), council-5 (skeptic), council-6
(architect). Ran in parallel.

## The three voices

### Council-5 — RECOMMENDATION: STOP

The 18pp gap and consumed-holdout combine to make BOTH proposed paths
unfalsifiable, AND the Maker's Dilemma kills strategy economics independently
of AUC:

1. **Pretrain-first is unfalsifiable** — if MEM+SimCLR lifts probe to
   ~flat-LR parity (0.84), we cannot distinguish "encoder learned
   microstructure" from "encoder learned a 17-feature autoencoder LR
   re-extracts the same flat signal from." Parity is the EXPECTED outcome of
   a successful autoencoder, not evidence of representation-beyond-features.
2. **End-to-end at n=169 is unfalsifiable** — CI half-width ~3.8pp, no
   untouched holdout, 376K params on 169 positives is the v1 overfit failure
   mode with no detector left.
3. **Maker's Dilemma is the actual blocker.** Universe median breakeven 0.311;
   0/300 cells with breakeven in (0.50, 0.55); E[realized|filled] = -7.89bp.
   Even AUC=0.90 with top-1% precision 35% does not flip economics when
   adverse selection consumes the post-fee envelope. **The encoder cannot
   solve adverse selection** — that's a fill-conditional venue property,
   not a label-prediction problem.

If overriding to pretrain anyway: MEM-only (no SimCLR), 0.5 H100-day cap,
pre-register kill threshold (probe LB > flat LB + 2pp), pre-commit Goal-A v3
framing for the failure case. Council-5's preferred Goal-A v3 frame: pivot to
TAKER-side execution where Maker's Dilemma math reverses.

### Council-6 — RECOMMENDATION: DIFFERENT EXPERIMENT (5b adapter)

Both pretrain-first and end-to-end-first are wrong moves. Run the cheapest
arbiter first: a non-linear adapter on the frozen random-init encoder.

**Why pretrain risks a tax:** MEM excludes `cum_ofi_5`, `delta_imbalance_L1`,
`kyle_lambda` from reconstruction targets — exactly the cascade-relevant
features. SimCLR's "same window ≈ jittered same window" prior is orthogonal
to cascade detection (cascades are anomalous within-day rare events; NT-Xent
treats them as noise). Default config is a tax not a destroyer; viable only
with cascade-aware adjustments.

**Why end-to-end is dead:** 376K params / 169 positives = 2225 params/positive
= 60× the v1 fine-tune ratio that already overfit. Council-6's regularization
recipe is good hygiene, not a falsifier.

**The right experiment (5b):** forward-pass every Apr 1-26 window through
the random-init `TapeEncoder` (already cached if Phase 0 wrote them), train
ONLY a small adapter head on the cascade label:

```
Linear(256 → 64) + ReLU + Dropout(0.2) + Linear(64 → 1)   ~16K params
BCEWithLogitsLoss(pos_weight=15.7), AdamW(lr=1e-3, wd=1e-3),
50 epochs, batch 256, no LR schedule, day-clustered 5-fold val,
early stop on pooled val AUC (patience 5).
```

Three pre-registered outcomes:
- **Adapter ≥ flat-LR + 2pp paired:** manifold contains cascade signal but
  Phase 0's linear probe was the bottleneck. End-to-end fine-tune justified
  next; risk of overfitting reduced because adapter scaling worked.
- **Adapter ≈ random-init linear probe (~0.65):** manifold is genuinely
  deficient; pretraining is the only path. Run cascade-aware MEM (no SimCLR).
- **Adapter beats random-init probe but loses to flat-LR by 5-15pp:** partial
  story. Run cascade-aware MEM-only → re-probe with same adapter as arbiter.

Cost: < 1 CPU-hour. Half a day end-to-end including writing the script.

### Council-4 — RECOMMENDATION: cascade-aware pretext, not generic MEM/SimCLR

**The cascade signature is a liquidity-depletion ramp punctuated by a
Composite Operator exit:** `kyle_lambda` climbing, `cum_ofi_5` one-sided,
`is_open` decaying from the smart-money side over ~200-300 events, then a
`climax_score` spike that marks the cascade ignition. SLOW summary-statistic
regime change with a FAST trigger.

**Why the random CNN loses:** Cascade-precursor is dominated by *level* and
*dispersion* of three features over the window, not their ordering:
- `mean(kyle_lambda)`, `mean(cum_ofi_5)`, `mean(is_open)`, `last(is_open)` →
  flat features capture this trivially.
- `max(climax_score)` → max-pool over the window beats any random
  positional-CNN encoding.

A random projection of (200, 17) → 256-dim preserves pairwise distances but
NOT the specific nonlinear summaries (rolling-z, max, last) that hand
features encode by construction. **Expected trained-CNN claw-back: 3-8pp,
not 18pp**, unless OB-feature sequence interactions matter more than
phenomenology suggests.

**MEM is genuinely hobbled here.** Excluding kyle_lambda, cum_ofi_5,
delta_imbalance_L1 from reconstruction removes the three most
cascade-discriminative axes from the pretext signal. Recommendation: re-include
kyle_lambda and cum_ofi_5 as MEM targets *for this Goal-A retrain*. The
"trivial copy" concern is weaker when the goal is cascade prediction, not
generic representation.

**SimCLR ±25 jitter is actively harmful** for right-edge cascade detection. A
window ending 5 events before cascade and the same anchor jittered +25 events
(ending 20 events AFTER cascade) are forced to embed identically — that
destroys the climax_ignition signature. Cap jitter at ±5 OR drop SimCLR.

**Custom pretext:** **distance-to-climax regression** — predict log(events
until next climax_score > 2.5) from each window. Self-labeled, directly
aligned with cascade-precursor structure, forces the encoder to learn the
slow-ramp-to-fast-trigger geometry that flat features cannot articulate. Pair
with a lighter MEM (kyle_lambda included, jitter-free) for local denoising.

## The synthesis

The three voices ANSWER DIFFERENT QUESTIONS:

- **Council-5** answers "should we trade cascades?" — NO, Maker's Dilemma.
- **Council-6** answers "what's the cheapest test to disambiguate the 18pp
  gap?" — 5b adapter test.
- **Council-4** answers "what should the pretrain objective look like IF we
  do pretrain?" — cascade-aware (re-include OFI features, drop SimCLR or
  cap jitter, custom distance-to-climax pretext).

**They don't conflict — they layer.**

The next move that respects all three: **run 5b** (council-6's adapter test)
as the cheapest possible falsifiable arbiter, but framed correctly:

1. 5b is NOT a step toward cascade-onset tradeable strategy (council-5 is
   right that path is dead per Maker's Dilemma).
2. 5b IS a test of whether the tape representation has signal beyond hand
   features — useful for ANY downstream task, including potential Goal-A v3
   pivot (council-5's TAKER-side framing, or other Pacifica-unique signals).
3. If 5b passes (adapter beats flat-LR by ≥ 2pp paired with delta CI > 0):
   we have a positive signal that justifies cascade-aware MEM as the next
   step (council-4's recipe), but ONLY if the program also has a TAKER-side
   downstream task in scope.
4. If 5b fails: architecture isn't the right hammer for THIS dataset; pivot
   away from the encoder retrain entirely. STOP per council-5.

This is the cheapest path that produces an actionable result.

## Pre-registered Phase 1 plan (5b adapter test)

| Aspect | Spec |
|---|---|
| Encoder | Random-init `TapeEncoder(EncoderConfig())`, 3 seeds {0, 1, 2}; embeddings cached from Phase 0 if available |
| Head | `Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)`, ~16K params |
| Loss | `BCEWithLogitsLoss(pos_weight=15.7)` |
| Optim | AdamW(lr=1e-3, wd=1e-3), no LR schedule |
| Schedule | 50 epochs, batch 256, early stop on pooled-val AUC, patience 5 |
| Val | Same 5-fold day-blocked CV with 600-event embargo as Phase 0 |
| Bootstrap | Paired day-clustered (1000 reps) on (adapter − flat-LR) delta |
| Cost | < 1 CPU-hour total across 3 seeds |

**Pre-registered outcomes (council-5 falsifiability):**
- **GREENLIGHT** (Tier A): adapter ≥ flat-LR + 0.02 AND paired-delta CI
  excludes 0 (delta_lo > 0). → Cascade-aware MEM justified, but ONLY if
  Goal-A v3 has a TAKER-side or non-Maker-fee downstream.
- **MATCHED** (Tier B): adapter ≈ flat-LR (delta CI overlaps 0) → manifold
  is neutral. STOP encoder retrain. Pivot per council-5.
- **KILL** (Tier C): adapter < flat-LR by > 0.02 → manifold is actively
  deficient. STOP and pivot. (Should not happen if 5b is doing what
  council-6 expects, but it's the falsifier.)

**Pre-flight: strategy-economics audit (parallel to 5b).** Independently
of 5b, validate council-5's "encoder cannot solve Maker's Dilemma" claim:
revisit `docs/experiments/goal-a-feasibility/maker_adverse_selection.md` and
ask whether ANY plausible (AUC, precision, signal-frequency) combination
flips economics. If yes — under what fill regime? If no — then 5b's pass
case routes to a Goal-A v3 framing (TAKER-side or non-economics deliverable),
and the user makes a research-direction call.

## Audit trail

- Phase 0 plan: `docs/experiments/goal-a-v2/2026-04-27-random-init-probe-plan.md`
- Phase 0 protocol: `docs/council-reviews/2026-04-27-encoder-retrain-protocol.md`
- Phase 0 result: `docs/experiments/goal-a-v2/random_init_probe_validator_report.md` (commit `3110abc`)
- Maker's Dilemma evidence: `docs/experiments/goal-a-feasibility/maker_adverse_selection.md`

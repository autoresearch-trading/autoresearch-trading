---
title: Phase 0 Random-Init Encoder Linear Probe (Goal-A v2)
date: 2026-04-27
status: completed
result: failure
sources:
  - docs/experiments/goal-a-v2/2026-04-27-random-init-probe-plan.md
  - docs/experiments/goal-a-v2/random_init_probe_validator_report.md
  - docs/council-reviews/2026-04-27-encoder-retrain-protocol.md
last_updated: 2026-04-27
---

# Experiment: Phase 0 Random-Init Encoder Linear Probe

## Hypothesis

A random-init `TapeEncoder` 256-dim global embedding either (a) carries
cascade-onset signal beyond what 83-dim hand features extract, or (b) does not.
The result arbitrates whether the cascade-onset encoder retrain program is
worth GPU compute. Council-1 + council-5 + council-6 ratified the protocol.

## Setup

- Cache: 4453 .npz shards on merged Apr 1-26 (April holdout consumed
  per gotcha #17).
- Cascade-H500 label: any `cause IN ('market_liquidation',
  'backstop_liquidation')` fill in `(anchor_ts, ts_at(anchor + 500)]`.
- 5-fold day-blocked CV on 26 days (~5 days/block), 600-event embargo at
  fold boundaries.
- Flat-LR baseline: `LogisticRegression(C=1.0, class_weight='balanced')` on
  83-dim flat features.
- Encoder probe: 3 random-init seeds {0, 1, 2}, `TapeEncoder` forward-pass
  with `track_running_stats=False` (gotcha #18); `LogisticRegression` head on
  256-dim global embeddings.
- Paired day-clustered bootstrap (1000 iters) on `(AUC_encoder − AUC_flat)`.
- Pre-registered decision tree: GREENLIGHT_FINETUNE if delta ≥ +0.02 AND
  delta_lo > 0; ARCH_BOTTLENECK if delta < −0.02; MATCHED_FLAT otherwise.
- Implementation: `scripts/random_init_probe.py` (1786 LOC, 11 unit tests,
  reviewer-10 GREEN).

## Result

| Model | Pooled AUC | Day-clustered CI |
|---|---|---|
| Flat-LR (83-dim) | 0.8373 | [0.8087, 0.8652] |
| Random encoder (median seed=1) | 0.6463 | [0.5802, 0.7246] |
| Per-seed range | 0.6330–0.6952 | — |
| Paired delta | **−0.1812** | **[−0.2594, −0.1063]** |

n_cascades_pooled = 169 (matches state.md ±20%); 27.2s CPU wall-clock; 5/5
sanity checks pass. Decision tier: **ARCH_BOTTLENECK**.

## What We Learned

1. Random-init dilated CNN's 256-dim global embedding does NOT linearly
   extract the cascade signal that 83-dim hand features carry trivially.
2. The flat-LR unified-CV AUC (0.8373) is materially HIGHER than the
   single-shot OOS=0.778 reported earlier (`b0de994`); council-1 was right
   that the OOS number was downward-biased. **0.8373 retires 0.778 as the
   reference baseline.**
3. The 18pp gap is large enough to fire the decision tree's
   ARCH_BOTTLENECK branch unambiguously.
4. CV partition matches reproducibly: Phase 1's flat-LR re-evaluation
   under the SAME folds returned EXACTLY 0.8373.

## Verdict

ARCH_BOTTLENECK. Routes to council debate (round 2) on whether to pretrain
or run a non-linear-adapter test (5b) before committing GPU compute. See
[Phase 1](phase1-adapter-test.md) for the follow-up.

## Related

- [Manifold Deficiency](../concepts/manifold-deficiency.md) — the
  pattern this experiment first observed.
- [Random-Init Probe Protocol](../decisions/random-init-probe-protocol.md)
  — the cheap-arbiter design adopted as a standard.

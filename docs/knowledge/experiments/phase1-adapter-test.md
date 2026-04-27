---
title: Phase 1 5b Cascade Adapter Test (Goal-A v2)
date: 2026-04-27
status: completed
result: failure
sources:
  - docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md
  - docs/experiments/goal-a-v2/cascade_adapter_validator_report.md
last_updated: 2026-04-27
---

# Experiment: Phase 1 — 5b Non-Linear Adapter Test

## Hypothesis

Phase 0's −18.1pp paired delta could reflect either (a) a linearity artifact
(the random-init manifold has cascade signal that linear LR cannot extract)
or (b) a manifold deficiency (the manifold itself has no cascade structure).
A small non-linear adapter on the SAME frozen embeddings disambiguates: a
linearity artifact closes the gap; a manifold deficiency does not.

Council-6's design from `2026-04-27-pretrain-vs-endtoend-synthesis.md`. $0
budget. Cheapest possible falsifier before any pretrain or end-to-end work.

## Setup

- Re-uses Phase 0's data assembly, CV partition, embargo, paired bootstrap,
  BH-FDR helpers (`scripts/random_init_probe.py`). Single source of truth.
- Adapter head: `Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)`,
  ~16K params.
- Loss: `BCEWithLogitsLoss(pos_weight=15.7)` (pos_weight = n_neg/n_pos at
  ~6% base rate).
- Optim: `AdamW(lr=1e-3, weight_decay=1e-3)`, no LR schedule, batch 256,
  max 50 epochs, early stop on pooled-val AUC patience 5.
- 3 random-init seeds {0, 1, 2} for the encoder; matching adapter inits.
- Same 5-fold day-blocked CV + 600-event embargo as Phase 0.
- Implementation: `scripts/cascade_adapter_probe.py`, 7 unit tests pass.

## Result

| Model | Pooled AUC | CI / range |
|---|---|---|
| Flat-LR (re-run, must match Phase 0) | 0.8373 | exact match — CV consistent |
| Random encoder + linear LR (Phase 0) | 0.6463 | [0.5802, 0.7246] |
| **Random encoder + non-linear adapter (median seed=2)** | **0.6941** | min 0.6571, max 0.7496 |
| Paired delta (adapter − flat) | **−0.1307** | **[−0.2003, −0.0577]** |

28.5s wall-clock; 7/7 sanity checks pass. Decision tier:
**KILL_ARCH_BOTTLENECK_CONFIRMED**.

## What We Learned

1. **The non-linear adapter closes only 4pp of the 18pp Phase 0 gap.** Going
   from linear to non-linear lifts pooled AUC from 0.6463 → 0.6941. Most of
   the gap is manifold deficiency, not linearity artifact.
2. **The phenomenology held.** Council-4 predicted 3-8pp claw-back; result
   landed at 4pp. Cascade signature is dominated by summary-statistic regime
   change (mean/max/last over the window) — a random projection of (200, 17)
   to 256-dim doesn't preserve those statistics.
3. **CV partition reproducibility validated.** Phase 1 flat-LR exactly
   matched Phase 0 (0.8373 = 0.8373). When the same 5-fold day-blocked CV is
   applied with the same random seed, results are bit-identical. This makes
   paired comparison defensible.
4. **5b adapter test design works as a cheap arbiter.** $0 compute, 28s,
   produces a pre-registered decision-tree verdict. Adopted as the standard
   "is the encoder doing anything beyond the hand features" test.

## Verdict

KILL_ARCH_BOTTLENECK_CONFIRMED. Council-5's STOP recommendation is now
empirically validated. Routes to v2 program closure — see
[v2 program closure](v2-program-closure.md).

## Related

- [Phase 0 Random-Init Probe](phase0-random-init-probe.md)
- [Manifold Deficiency](../concepts/manifold-deficiency.md)
- [V2 Program Closure](../decisions/v2-program-closure.md)

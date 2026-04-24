---
title: Cross-Symbol Invariance
topics: [contrastive, pretraining, representation-quality, evaluation]
sources:
  - docs/experiments/step5-cluster-cohesion.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
last_updated: 2026-04-24
---

# Cross-Symbol Invariance

## What It Is

Cross-symbol invariance is the property that the encoder places two windows
from different symbols observing the same market moment (same UTC hour, same
date) closer in embedding space than two windows from different symbols at
different hours. It is the geometric fingerprint of "universal tape features"
— a precondition for claiming a single encoder learned symbol-agnostic
microstructure (absorption, climax, informed flow) rather than per-symbol
lookup tables.

Operationally we quantify it through four cosine populations on the 256-dim
L2-normalized global embedding, measured on the 6 liquid SimCLR anchors
(BTC/ETH/SOL/BNB/LINK/LTC):

- `within_symbol` — same symbol, same (date, hour)
- `same_symbol_diff_hour` — same symbol, different hour
- `cross_symbol_same_hour` — different symbols, same (date, hour)
- `cross_symbol_diff_hour` — different symbols, different hour

The load-bearing statistic is the **delta** `cross_symbol_same_hour −
cross_symbol_diff_hour`: this isolates the cross-symbol shared-moment signal
from the narrow-cone geometry every population lives on.

## Measured on step3-r2 (2026-04-24)

| Population | Mean cosine |
|---|---|
| within_symbol | 0.8948 |
| same_symbol_diff_hour | 0.8361 |
| cross_symbol_same_hour | 0.7339 |
| cross_symbol_diff_hour | 0.6967 |

- **Cross-symbol same-hour delta: +0.037** — below council-5's pre-dispatched
  `some_invariance` threshold of +0.1.
- **Symbol-identity signal (same_symbol_diff_hour − cross_symbol_diff_hour):
  +0.139** — ~4× stronger than the cross-symbol-same-hour signal.
- **6-way symbol-ID linear probe balanced accuracy: 0.934** on Feb held-out
  (spec target <20%, measured 93.4%).

The absolute `cross_symbol_same_hour = 0.734 > 0.6` headline triggers the
spec-literal `strong_invariance` band, but that band does not control for
cone offset; every population lives above cosine 0.69. The delta — not the
absolute — is the honest reading.

## Why the SimCLR Delta Is Weak Here

The training recipe:
- Cross-symbol positive pairs drawn only from 6 of 24 pretraining symbols.
- Soft-positive weight = 0.5 (half of the same-window positive weight).
- No annealing of soft-positive weight across epochs.

Council-5's prior review (gate3 falsifiability) predicted that under these
parameters SimCLR never kicks in at sufficient gradient strength to force
cross-symbol feature sharing — the self-view positive dominates and the
encoder reward-hacks to "this is symbol X" rather than "this is absorption."

## Consequences

- **Gate 3 AVAX transfer failure was overdetermined.** See
  [Gate 3 retirement](../decisions/gate3-retired-to-informational.md).
- **Symbol-ID <20% target becomes aspirational for future universality-targeting
  runs** — see
  [Symbol-ID reframe](../decisions/symbol-id-probe-reframed-aspirational.md).
- **A universality-targeting successor run would need:** (a) LIQUID_CONTRASTIVE_SYMBOLS
  widened 6→12–15, (b) soft-positive weight 0.5→1.0 annealed, (c) cluster-cohesion
  delta ≥+0.10 as an in-training early stop-gate.

## Council-5 Thresholds (pre-dispatched)

| Band | Criterion | This run |
|------|-----------|----------|
| strong_invariance | cross_symbol_same_hour > 0.6 | TRIGGERED (0.734) |
| some_invariance | delta > +0.10 | NOT met (+0.037) |
| no_invariance | delta within 0.1 of 0 | TRIGGERED (+0.037) |

The two lower bands both fire — the `strong` band fires spuriously on the
absolute cosine alone. The amendment accepts the `some_invariance` failure as
the binding reading.

## Gotchas

1. **Do NOT read off the absolute cosine.** All populations live on a narrow
   cone (> 0.69). The delta is load-bearing.
2. **Measured on Feb only.** Council-5 flagged this as a minor tell —
   re-running on Mar or a held-out month costs ~5 seconds of inference.
3. **SimCLR was trained on the same anchors being measured.** This is an
   in-sample measurement of the invariance the training objective explicitly
   targeted; out-of-sample (AVAX) will be worse by design.

## Related Concepts

- [Contrastive Learning](contrastive-learning.md) — the training objective
- [Gate 3 retirement](../decisions/gate3-retired-to-informational.md)
- [Symbol-ID reframe](../decisions/symbol-id-probe-reframed-aspirational.md)
- [Cluster cohesion experiment](../experiments/cluster-cohesion-diagnostic.md)

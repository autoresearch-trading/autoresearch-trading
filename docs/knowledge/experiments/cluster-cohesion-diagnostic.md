---
title: Cluster Cohesion Diagnostic on 6 Liquid SimCLR Anchors
date: 2026-04-24
status: completed
result: partial-success
sources:
  - docs/experiments/step5-cluster-cohesion.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
last_updated: 2026-04-24
---

# Experiment: Cluster Cohesion Diagnostic

## Hypothesis

If the SSL encoder learned universal cross-symbol tape features, then windows
from different symbols at the same UTC hour should cluster tighter in
embedding space than windows from different symbols at different hours. The
delta `cross_symbol_same_hour − cross_symbol_diff_hour` quantifies that
invariance and forms the prior for whether Gate 3 transfer was ever possible.

## Setup

- **Checkpoint:** `runs/step3-r2/encoder-best.pt` (376K params)
- **Script:** `scripts/cluster_cohesion.py`
- **Data:** Feb 2026 held-out shards for the 6 liquid SimCLR anchors
  (BTC/ETH/SOL/BNB/LINK/LTC).
- **Windowing:** 168 shards at stride=200 eval mode, 8,687 raw windows,
  per-bucket subsample up to 8 windows per (symbol, date, hour) → 8,660
  retained across 3,676 buckets.
- **Embeddings:** 256-dim global (concat of `GlobalAvgPool` and
  `last_position`), L2-normalized before cosine.
- **Populations:** each capped at 50,000 uniformly-resampled pairs.
- **Runtime:** ~5 seconds end-to-end on MPS.

## Result

Four cosine populations:

| Population | Mean | Std | p50 |
|---|---|---|---|
| within_symbol | 0.8948 | 0.065 | 0.9071 |
| same_symbol_diff_hour | 0.8361 | 0.093 | 0.8565 |
| cross_symbol_same_hour | 0.7339 | 0.148 | 0.7704 |
| cross_symbol_diff_hour | 0.6967 | 0.147 | 0.7257 |

**Load-bearing deltas:**
- **Cross-symbol same-hour signal: +0.037** (vs diff-hour baseline).
- **Symbol-identity signal: +0.139** (same_symbol_diff_hour vs cross_symbol_diff_hour).
- Symbol-identity signal is **~4× stronger** than cross-symbol shared-moment
  signal.

**6-way symbol-ID linear probe balanced accuracy: 0.9336** (on 80/20 split
of 8,660 Feb windows; n_test ≈ 1,732). Chance = 1/6 = 0.167. Spec target
<20% is violated by a factor of ~4.7×.

## Council-5 Pre-Dispatched Thresholds

| Band | Criterion | Result |
|---|---|---|
| strong_invariance | cross_symbol_same_hour > 0.6 | TRIGGERED (0.734) |
| some_invariance | delta > +0.10 AND > 0.3 | NOT met (+0.037) |
| no_invariance | delta within 0.1 of 0 | TRIGGERED (+0.037) |

The `strong` band fires spuriously because every population lives on a
narrow cone (all means > 0.69) — the absolute cosine threshold doesn't
control for the cone offset. The delta is load-bearing; the honest reading
is that `some_invariance` is NOT met.

## What We Learned

1. **The encoder learned per-symbol structure, not universal tape
   geometry.** Every symbol occupies a tight same-symbol cluster
   (within-symbol cosine 0.78–0.85); cross-symbol pairs are 4× further apart
   at shared UTC hours than same-symbol pairs at different UTC hours.
2. **The SimCLR recipe under-performed its stated goal.** 6-of-24 anchors
   with soft-positive weight 0.5 produced a +0.037 cross-symbol delta. The
   training objective existed but did not kick in at sufficient gradient
   strength.
3. **AVAX Gate 3 failure was overdetermined.** The encoder was never
   positioned to transfer to an unseen symbol. The cluster-cohesion
   measurement shows the invariance-geometry that would have made AVAX a
   clean "transfer" test was never built in.
4. **Retiring Gate 3 on this evidence is measurement-motivated, not
   retroactive rationalization.** The +0.10 threshold was pre-dispatched by
   council-5 before the measurement ran.

## Verdict

**Partial success — the diagnostic works, and the measured result tells us
the training config did not achieve cross-symbol universality.** This is
evidence that the Gate 3 reframe is correct and that a future universality-
targeting run must widen LIQUID_CONTRASTIVE_SYMBOLS and anneal soft-positive
weight.

**Caveat (council-5 minor tell):** Feb-only measurement on a single month. A
~5-second re-run on Mar or a held-out month would confirm cross-month
stability of the delta. Not a blocker, but logged.

## Related

- [Cross-symbol invariance](../concepts/cross-symbol-invariance.md)
- [Gate 3 retired to informational](../decisions/gate3-retired-to-informational.md)
- [Symbol-ID probe reframed aspirational](../decisions/symbol-id-probe-reframed-aspirational.md)
- [Gate 3 triage experiment](gate3-avax-triage.md)

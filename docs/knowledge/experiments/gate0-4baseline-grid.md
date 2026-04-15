---
title: Gate 0 — 4-Baseline Grid (2026-04-15)
date: 2026-04-15
status: completed
result: partial-success
sources:
  - docs/experiments/gate0-summary.md
  - docs/experiments/gate0-baseline.md
  - docs/experiments/gate0-random-control.md
  - docs/experiments/gate0-majority-baseline.md
  - docs/experiments/gate0-shuffled-labels.md
last_updated: 2026-04-15
---

# Experiment: Gate 0 4-Baseline Grid

## Hypothesis

The spec's PCA+LR baseline at Gate 0 measures extractable signal in the 85-dim
flat features, establishing a reference for the CNN probe at Gate 1.

## Setup

- Cache: 4003 shards / 25 symbols / 161 days / 32M events / 641K windows@s=50.
- Walk-forward: 3-fold, 600-event embargo, min_train=2000, min_test=500.
- Four baselines: PCA+LR, Random Projection (frozen 85→20)+LR, Majority-class,
  Shuffled-labels PCA+LR.
- Metric: balanced accuracy at H10/H50/H100/H500.
- Commits: data pipeline `ec1ea5d` → OB NaN fix `95ca60c` → Gate 0 CLI fix
  `9de25c2` → RP control `c0bee9f` → majority + shuffled + balanced rendering
  `7ff1459`.

## Result

| Horizon | PCA | RP | Majority | Shuffled |
|---------|-----|----|---------|---------| 
| H10 | 0.5104 | 0.5091 | 0.5000 | 0.5030 |
| H50 | 0.5065 | 0.5013 | 0.5000 | 0.4997 |
| H100 | 0.5043 | 0.4983 | 0.5000 | 0.5010 |
| H500 | 0.5051 | 0.4993 | 0.5000 | 0.5033 |

Standard error ≈ 0.022 per symbol-fold. PCA's margin over Majority at H100 is
+0.004pp — within one SE. All four baselines statistically indistinguishable
from chance at medium/long horizons.

Shuffled-labels stays 0.500±0.003 across all horizons — pipeline leakage-free.

## What We Learned

1. **Flat aggregation destroys sequential signal.** 85-dim summary statistics
   carry no linearly-extractable direction signal beyond per-fold label imbalance.
2. **Raw accuracy was misleading.** 5 symbols (2Z, CRV, WLFI, XPL, PUMP) had
   3–10pp raw-vs-balanced inflation at H10 driven entirely by label skew.
3. **AVAX (Gate-3 held-out) balanced acc H10 = 0.4986 — below chance.** Gate 3
   will require genuine transferable representation learning.
4. **Pipeline is clean.** Shuffled-labels null test validates every stage:
   label construction, window alignment, fold assignment, embargo bookkeeping.

## Verdict

**Partial success.** Gate 0 passes the pipeline-cleanliness check (shuffled ≈ 0.500)
but its numeric results invalidate the original spec threshold. Reframed as a
noise-floor publisher, not a pass/fail gate. Gate 1 thresholds were rewritten to
require beating Majority AND RP by 1.0pp each, plus hour-of-day probe < 10%.

## Spec Consequences

- Gate 0 reframed (`1f86d52` commit).
- Gate 1 = 4-condition binding stop-gate.
- Balanced accuracy becomes universal metric.
- SimCLR augmentations strengthened for session-of-day decorrelation.
- Pre-pretraining session-of-day confound check added.

See [Gate 0 Baseline concept](../concepts/gate0-baseline.md) for current state.

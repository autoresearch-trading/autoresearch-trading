---
title: Gate 0 Baseline Grid
topics: [evaluation, baselines, methodology]
sources:
  - docs/council-reviews/2026-04-10-round5-council-1-gate0-methodology.md
  - docs/council-reviews/council-1-gate0-rp-equivalence.md
  - docs/council-reviews/council-5-gate0-falsifiability.md
  - docs/council-reviews/council-6-gate0-impact-on-pretraining.md
  - docs/experiments/gate0-summary.md
last_updated: 2026-04-15
---

# Gate 0 Baseline Grid

## What It Is

Gate 0 is **not a threshold-gate**. It publishes four baselines against which
Gate 1 is measured. The original spec had Gate 0 as "PCA + LR baseline" with a
51.4% threshold; council round 6 (2026-04-15) found that baseline
statistically indistinguishable from a majority-class predictor at every
horizon on balanced accuracy.

## Our Implementation

Four baselines over the same walk-forward folds (3-fold, 600-event embargo,
min_train=2000, min_test=500) at H10/H50/H100/H500:

| # | Baseline | Purpose |
|---|----------|---------|
| 1 | **PCA(n=20) + LogisticRegression** on 85-dim flat features | Headline baseline |
| 2 | **Random Projection (85→20, frozen) + LR** | Adaptive-structure control |
| 3 | **Majority-class predictor** (training-fold majority) | True noise floor |
| 4 | **Shuffled-labels PCA+LR** | Pipeline-clean null check |

Flat features = mean/std/skew/kurt/last per channel × 17 features = 85 dims.
See `tape/flat_features.py`.

**Metric: balanced accuracy at ALL horizons.** Raw accuracy is gameable via
per-fold label imbalance — see
[Balanced Accuracy decision](../decisions/balanced-accuracy-all-horizons.md).

Scripts: `scripts/run_gate0.py`, `scripts/run_random_baseline.py`,
`scripts/run_majority_baseline.py`, `scripts/run_shuffled_labels_baseline.py`.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-10 | StandardScaler before PCA | Without it, PCA captures timing variation not market state | round5-council-1 |
| 2026-04-10 | PCA on all pre-April, LR on training fold | PCA unsupervised (no label leakage); LR supervised | round5-council-1 |
| 2026-04-10 | 5 random encoder seeds | Single seed can vary ±0.5-1% — need mean for reliable comparison | round5-council-1 |
| 2026-04-15 | Gate 0 = publishing baseline grid, NOT threshold-gate | PCA ≈ Majority ≈ RP on balanced acc; 51.4% threshold was meaningless | council round 6 |
| 2026-04-15 | Balanced accuracy at ALL horizons | Raw acc inflated illiquid H10 by up to 9.9pp via label imbalance | council-1 / council-5 |
| 2026-04-15 | Shuffled-labels null check required | Proves pipeline is leakage-free before trusting any baseline | council-5 |
| 2026-04-15 | PCA n=20 fixed (not swept) | Sweep unnecessary when n=20 already matches RP performance | 4-baseline measurement |

## 2026-04-15 Results

Balanced accuracy, 25 symbols:

| Horizon | PCA | RP | Majority | Shuffled |
|---------|-----|----|---------|---------| 
| H10 | 0.5104 | 0.5091 | 0.5000 | 0.5030 |
| H50 | 0.5065 | 0.5013 | 0.5000 | 0.4997 |
| H100 | 0.5043 | 0.4983 | 0.5000 | 0.5010 |
| H500 | 0.5051 | 0.4993 | 0.5000 | 0.5033 |

Standard error ≈ 0.022 per symbol-fold. PCA's margin over Majority at H100 is
+0.004pp — within one SE. All four baselines statistically indistinguishable
from chance at medium/long horizons.

Shuffled-labels stays 0.500±0.003 across all horizons — pipeline is
leakage-free.

## Implications

1. **Flat summary statistics destroy the sequential signal.** Consistent with
   the CNN hypothesis but does NOT prove it — could also mean no signal.
2. **Raw accuracy was misleading.** 2Z H10 raw=0.621 → balanced=0.522 (9.86pp
   gap from label imbalance alone). Five outlier symbols collapsed on rebalance.
3. **The CNN must beat Majority+1pp AND RP+1pp** on 15+/25 symbols at H100 for
   Gate 1 to pass. See
   [Gate 1 Thresholds decision](../decisions/gate1-thresholds-revised.md).
4. **AVAX (Gate-3 held-out) H10 balanced = 0.4986 — below chance.** Gate 3 will
   require genuine transferable representation learning, not memorization.

## Gotchas

1. **Never tune on April 1-13** — every touch costs a trial for DSR. Keep
   April 14+ untouched.
2. **PCA destroys temporal ordering** — this is a feature (establishes floor
   for sequence-aware architectures).
3. **Low-volume symbols with N_test < 500 have no statistical power at 51.4%.**
4. **Raw-accuracy tables preserved for history** but balanced-accuracy is the
   primary reading. The `.md` renderer shows both.
5. **PCA fit per fold, not globally** — refit on train portion only, no leakage.
6. **RP matrix is frozen across all folds and symbols** — fair comparison to
   PCA (both use the same downstream LR).
7. **Shuffled labels per symbol, not global** — preserves per-symbol
   distribution; tests pipeline, not cross-symbol mixing.

## Related Concepts

- [Session-of-Day Leakage](session-of-day-leakage.md)
- [Balanced Accuracy decision](../decisions/balanced-accuracy-all-horizons.md)
- [Gate 1 Thresholds](../decisions/gate1-thresholds-revised.md)
- [Self-Labels](self-labels.md) — evaluation targets
- [April Hold-Out Window](../decisions/april-holdout-window.md) — test set design

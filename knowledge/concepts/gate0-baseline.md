---
title: Gate 0 PCA Baseline
topics: [evaluation, baselines, methodology]
sources:
  - docs/council-reviews/2026-04-10-round5-council-1-gate0-methodology.md
  - docs/council-reviews/data-sufficiency-synthesis.md
last_updated: 2026-04-10
---

# Gate 0 PCA Baseline

## What It Is

The reference baseline that all pretrained representations are measured against.
Flatten (200, 17) → 3400-dim, apply PCA, train logistic regression on PCA
components. If the pretrained encoder can't beat this, self-supervised learning
added nothing.

## Our Implementation

### Setup
1. **StandardScaler** on 17 features (fit on training data only) — CRITICAL, omitting this makes PCA dominated by time_delta scale
2. **PCA** on all pre-April data (unsupervised, no label leakage)
3. **PCA n:** sweep {20, 50, 100, 200}, use best-performing (make baseline strong)
4. **Logistic regression** C sweep {0.001, 0.01, 0.1} on training fold only
5. **C selection:** temporal inner split (last 20% of pre-April by date, NOT random)
6. **Evaluation:** April 1-13 only

### Random Encoder Baseline
- 5 seeds, report mean ± std
- `model.eval()` for BatchNorm running statistics
- Expected accuracy: 50.5%-51.0%

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-10 | StandardScaler before PCA | Without it, PCA captures timing variation not market state | round5-council-1 |
| 2026-04-10 | PCA on all pre-April, LR on training fold | PCA is unsupervised (no label leakage); LR is supervised | round5-council-1 |
| 2026-04-10 | Sweep n, not fixed 50 | n=50 may be too few (underpowered) or too many (noise) | round5-council-1 |
| 2026-04-10 | C selected on temporal inner split | Random split overestimates CV accuracy by ~1-2pp due to label autocorrelation | round5-council-1 |
| 2026-04-10 | 5 random encoder seeds | Single seed can vary ±0.5-1% — need mean for reliable comparison | round5-council-1 |

## Gotchas

1. Never tune C on April 1-13 — every touch costs a trial for DSR.
2. PCA destroys temporal ordering — this is a feature (establishes floor for
   sequence-aware architectures).
3. Low-volume symbols with N_test < 500 have no statistical power at 51.4%.
4. Base rate may differ between training (52-53% in bull market) and April —
   report per-symbol, per-period base rates.

## Related Concepts

- [Self-Labels](self-labels.md) — evaluation targets
- [April Hold-Out Window](../decisions/april-holdout-window.md) — test set design

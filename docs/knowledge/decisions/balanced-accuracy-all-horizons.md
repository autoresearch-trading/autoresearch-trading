---
title: Balanced Accuracy at ALL Horizons
date: 2026-04-15
status: accepted
decided_by: council-1 + council-5 + council-6 (round 6)
sources:
  - docs/council-reviews/council-1-gate0-rp-equivalence.md
  - docs/council-reviews/council-5-gate0-falsifiability.md
  - docs/experiments/gate0-summary.md
last_updated: 2026-04-15
---

# Decision: Balanced Accuracy at ALL Horizons

## What Was Decided

All evaluation gates (0, 1, 3, 4) use **balanced accuracy** as the primary metric
at every horizon (H10/H50/H100/H500). Raw accuracy is preserved only in
historical-reference tables.

## Why

The prior spec amendment (Step 0, 2026-04-14) required balanced accuracy only at
H500, arguing that H100 and H50 base rates stayed within ±2pp on liquid symbols.
That claim was **wrong for illiquid symbols**: Gate 0 measurements at 2026-04-15
show H10 raw-vs-balanced gaps up to 9.86pp on 2Z, 8.81pp on CRV, 8.12pp on WLFI.
These symbols inflated the "14/25 above 51.4%" H10 headline to a majority-class
artifact.

Council-5 found the smoking gun directly in the JSON output: the
`balanced_accuracy_mean` field was computed at every horizon but only displayed
in the summary for H500. Switching the display rule eliminates the inflation
without any code-level change beyond the renderer.

## Alternatives Considered

1. **Keep raw accuracy at H10/H50/H100, document the inflation risk.** Rejected
   — silently leaves a tripwire for future agents who trust the summary table.
2. **Use F1 instead of balanced accuracy.** Rejected — F1 is asymmetric
   (privileges one class); balanced accuracy is the honest average of true-positive
   and true-negative rates for binary classification.
3. **Drop illiquid symbols from the summary.** Rejected — some of the signal we
   care about may live in illiquid symbols; the fix belongs on the metric, not
   the universe.

## Impact

- Gate 0 summary .md rendering now shows both raw and balanced tables; balanced
  is labeled "council-preferred."
- Gate 1/2/3/4 thresholds implicitly shift, since a 51.4% balanced floor is
  harder to clear than a 51.4% raw floor on imbalanced symbols.
- Future experiments must report balanced accuracy as the primary number.
- Spec amendment in commit `1f86d52`; gotcha #28 added.

## Related

- [Gate 0 Baseline](../concepts/gate0-baseline.md)
- [Gate 1 Thresholds Revised](gate1-thresholds-revised.md)

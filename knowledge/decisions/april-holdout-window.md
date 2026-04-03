---
title: April Hold-Out Window
date: 2026-04-02
status: accepted
decided_by: Council-1 (Lopez de Prado methodology), round3-synthesis H2
sources:
  - docs/council-reviews/2026-04-01-spec-review-council-1.md
  - docs/council-reviews/round3-synthesis.md (H2)
last_updated: 2026-04-03
---

# Decision: April Hold-Out Window

## What Was Decided

Designate April 14+ as an untouched hold-out set. April 1-13 is available for
development validation. The April 14+ data must not be viewed, analyzed, or used
for any purpose -- including data quality checks -- until the final evaluation.

## Why

The March 5-25 test window was used for 20+ experiments on the main branch
(Council-1 flagged this as progressive contamination). Every accuracy report
against that window is biased upward by an unknown amount due to implicit
overfitting through researcher degrees of freedom.

Council-1 identified this as time-sensitive: "every day without designation risks
progressive contamination through informal data inspection." The April data from
Pacifica API provides a genuinely fresh test set, but only if it is protected
from any premature use.

The split at April 14 gives:
- **April 1-13 (dev validation):** 13 days with the new `event_type` field.
  Useful for validating the data pipeline (dedup via `fulfill_taker`), testing
  feature computation with clean direction data, and running quick sanity checks.
- **April 14+ (untouched hold-out):** The definitive, single-use evaluation set.
  Results reported against this window carry full statistical weight because no
  prior experiment has touched it.

## Alternatives Considered

- **Use all of April as hold-out:** Loses the only clean-direction data for
  pipeline development (April has `event_type` field that pre-April lacks).
- **Use March 25+ as hold-out:** Already contaminated by main-branch experiments.
- **No designated hold-out:** All results become subject to the Deflated Sharpe
  Ratio penalty from accumulated trials. Without a fresh test set, statistical
  significance is much harder to establish.

## Impact

1. CLAUDE.md gotcha 17: "April 14+ is untouched -- do not view, even for data
   quality checks."
2. All walk-forward validation during development uses Oct-March + April 1-13.
3. The final model evaluation on April 14+ is a one-shot test. If it fails,
   the model fails. No iteration on the hold-out set.
4. Trial count for DSR calculation resets to 1 for April 14+ results (fresh set),
   vs ~1,600 accumulated trials against March windows.

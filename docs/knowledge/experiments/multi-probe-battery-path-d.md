---
title: Multi-Probe Battery Calibration Failure — Path D Drop Without Measurement
date: 2026-04-26
status: abandoned
result: inconclusive
sources:
  - docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c1.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c4.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c5.md
last_updated: 2026-04-27
---

# Experiment: Multi-Probe Battery — Dropped Without Measurement (Path D)

## Hypothesis

Three pre-registered phenomenology probes on the frozen encoder would
test whether the representation captures Wyckoff tape-state structure
beyond directional signal:

- **C1:** Wyckoff absorption probe (window-level label: low realized
  variance after high effort)
- **C3:** ARI cluster–Wyckoff label alignment
- **C4:** Embedding trajectory at climax events

## Calibration Discovery

Before any encoder forward pass, `step0_validate` was run on Feb+Mar
held-out windows to confirm label firing rates. Result:

- `climax_score > 2.5` threshold: **0% of held-out windows fire** across
  all 24 non-AVAX symbols
- `climax_score > 3.0` threshold: 0% (unchanged)
- Empirical max `climax_score` across all symbols: **0.256** (PENGU)
- The pre-registered threshold of 3.0 is **~30× the empirical maximum**

Council-4 admitted the threshold was specified without checking the
empirical scale of `min(z_qty, z_ret)` after the rolling-1000 σ + MIN
operator. The MIN operator severely compresses the tail; even rare
joint-extreme events do not approach z=3.

step0's per-event Wyckoff labels (different operationalization) fire at
3-11% absorption per event — confirming the underlying phenomenology is
at least partially recoverable with proper window-level operationalization.
The c1/c3/c4 thresholds are incorrect, not the phenomenology.

## Path D Consensus (council-1, council-4, council-5)

All three converged: drop the battery without measurement.

- **c-1 (methodology):** running probes with miscalibrated thresholds
  produces uninformative measurement; reporting them as falsified would
  be a category error.
- **c-4 (substance):** admits over-specification error in original
  threshold spec; calibrated thresholds against Feb+Mar would be a new
  experiment requiring fresh pre-registration.
- **c-5 (skeptic):** the 0%-fire discovery IS itself a feasibility
  check; it does not require additional measurement to falsify.

## Verdict — DROPPED WITHOUT MEASUREMENT

The multi-probe battery cannot produce honest pass/fail outcomes given
the pre-registered thresholds. Per Path D, no measurement attempted.

**Anti-amnesia:** future writeups citing this program MUST disclose
"phenomenological richness untested due to operational label calibration
failure" rather than claiming the encoder failed phenomenology.

## What We Learned

1. **Pre-register label firing rates AND thresholds, not just thresholds.**
   The C1/C3/C4 spec defined thresholds in standardized-units space without
   confirming the underlying distribution had appropriate dynamic range
   in the eval window.
2. **Rolling-σ MIN-operator compresses tails severely.** `climax_score`
   max of 0.256 vs threshold 3.0 means the `min(z_qty, z_return)`
   construction makes the joint-extreme event rare to a degree the spec
   did not anticipate.
3. **The "tape vs priors" interpretation question stays open.** The
   probes that would have decided it never fired. The interpretation is
   adjudicated softly via cohesion + RankMe + symbol-ID instead (see
   [Calibrated Interpretation](../decisions/calibrated-interpretation-per-symbol-clustered.md)).

## Related

- [Climax Score](../concepts/climax-score.md)
- [Wyckoff Self-Labels](../concepts/self-labels.md)
- [Path D — Drop Battery Without Measurement](../decisions/path-d-drop-battery.md)
- [Tape-State Diagnostic Off-Ramp](../experiments/tape-state-diagnostic-off-ramp.md) — the follow-on diagnostic also declined on falsifiability grounds

---
title: Path D — Drop Multi-Probe Battery Without Measurement
date: 2026-04-26
status: accepted
decided_by: council-1 + council-4 + council-5 (unanimous)
sources:
  - docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c1.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c4.md
  - docs/council-reviews/2026-04-26-multi-probe-calibration-c5.md
last_updated: 2026-04-27
---

# Decision: Path D — Drop Multi-Probe Battery Without Measurement

## What Was Decided

The pre-registered multi-probe phenomenology battery (C1: Wyckoff
absorption probe; C3: ARI cluster–Wyckoff alignment; C4: embedding
trajectory at climax events) is **dropped without measurement** because
its pre-registered label thresholds fire on 0% of held-out windows.

Future writeups citing this program MUST disclose
"phenomenological richness untested due to operational label calibration
failure" — NOT "phenomenology probes failed."

## Why

`step0_validate` confirmed before any encoder forward pass:
- `climax_score > 2.5` threshold: 0% of held-out windows fire
- Empirical max `climax_score` across 24 non-AVAX symbols: 0.256 (PENGU)
- Pre-registered threshold of 3.0 is ~30× the empirical maximum

Council-4 admitted the threshold was specified without checking the
empirical scale of `min(z_qty, z_ret)` after the rolling-1000 σ + MIN
operator.

## Alternatives Considered

- **Path A — recalibrate thresholds and run:** Rejected. Recalibrating
  on Feb+Mar held-out and then evaluating on Feb+Mar would be data
  leakage. Recalibrating on training period and re-evaluating would
  require fresh pre-registration consuming amendment budget.
- **Path B — run with original thresholds and report 0%:** Rejected.
  The probes would not produce measurement; reporting them as falsified
  would be a category error.
- **Path C — substitute different label operationalizations:**
  Rejected. Step0's per-event labels are different probes; substituting
  them is not running C1/C3/C4 — it is running new probes that bypass
  pre-registration.
- **Path D — drop without measurement, anti-amnesia disclosure:**
  Accepted unanimously.

## Impact

- The program end-state writeup makes no phenomenological claim.
- The "encoder represents tape state" question stays soft-adjudicated
  via cluster cohesion + RankMe + symbol-ID (see
  [Calibrated Interpretation](calibrated-interpretation-per-symbol-clustered.md)).
- A subsequent Tape-State-Paired-Probe was proposed and also declined
  on falsifiability grounds (see
  [Tape-State Diagnostic Off-Ramp](../experiments/tape-state-diagnostic-off-ramp.md)).
- The Path D precedent (decline pre-commit, anti-amnesia disclose)
  becomes the program's pattern for under-specified diagnostics.

## Related

- [Multi-Probe Battery Path D](../experiments/multi-probe-battery-path-d.md)
- [Climax Score](../concepts/climax-score.md)
- [Wyckoff Self-Labels](../concepts/self-labels.md)

---
title: Gate 3 AVAX Triage — Bootstrap CIs + In-Sample Control
date: 2026-04-24
status: completed
result: inconclusive
sources:
  - docs/experiments/step5-gate3-avax-probe.md
  - docs/experiments/step5-gate3-triage.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
last_updated: 2026-04-24
---

# Experiment: Gate 3 AVAX Triage — Bootstrap CIs + In-Sample Control

## Hypothesis

The stride=50 AVAX Gate 3 point-estimate failure (encoder 0.514 H100 on Mar,
below the 51.4% threshold and below PCA) is either (H1) a clean falsifier of
"encoder transfers to unseen symbols" or (H2) an underpowered probe
artifact. Bootstrap CI overlap + in-sample control disambiguates.

## Setup

- **Checkpoint:** `runs/step3-r2/encoder-best.pt`
- **Script:** `scripts/avax_gate3_probe.py` (commit `ea07bda`)
- **Protocol:** time-ordered 80/20 per month, stride=50, balanced accuracy,
  **1000-resample percentile bootstrap 95% CI per cell**, **N=50 shuffled-null
  with μ±2σ reporting**, class_prior reported per cell.
- **Held-out:** AVAX Feb (2,360 windows) + Mar (1,898 windows), H100 + H500.
- **In-sample control:** LINK + LTC same protocol (4,498 Feb + 3,806 Mar
  windows — ~2× AVAX sample size).
- **Additional control:** AAVE alone (2,458 Feb + 1,958 Mar windows).

## Result

### AVAX (pre-registered Gate 3)
Encoder vs PCA CIs **overlap on 4/4 primary cells**:
- Feb H100: enc [0.488, 0.579] vs pca [0.502, 0.594]
- Feb H500: enc [0.442, 0.538] vs pca [0.474, 0.561]
- Mar H100: enc [0.467, 0.562] vs pca [0.452, 0.534]
- Mar H500: enc [0.410, 0.510] vs pca [0.507, 0.604] (narrowest overlap: 0.4pp, direction PCA > encoder)

Encoder CI lower bound never clears 51.4%. **Pre-registered Gate 3 threshold
not cleared at CI-aware rigor.**

### LINK+LTC in-sample control
**Encoder fails the same way on in-sample symbols.** Encoder below majority
on 3/4 cells (point estimates 0.495, 0.509, 0.494, 0.496). CI overlap vs
PCA on 4/4. None of 4 cells clears 51.4%, not even PCA. Effective n_test
~660–880, ~2× AVAX's sample, yet no encoder lift visible.

### AAVE control (optional)
Replicates AVAX pattern even more starkly — Feb H100: encoder +7.9pp over
PCA; Mar H100: PCA +7.0pp over encoder. Both inside each other's CIs — the
exact "lucky cell" pattern at single-symbol small-n.

### Shuffled null (all three arms)
N=50 shuffles give mean 0.4995–0.5040 with σ ≈ 0.02–0.03. Pipeline is
clean. The prior single-seed Apr AVAX shuffled=0.700 is retroactively
explained as one tail draw of this distribution.

## What We Learned

1. **AVAX is NOT anomalous.** Its failure pattern matches in-sample
   LINK+LTC + AAVE under the same protocol. The probe is underpowered for
   the encoder's measured Gate-1 signal (~1–2pp) at n_test ~400–900 per
   single-symbol single-month cell. See
   [underpowered single-symbol probe](../concepts/underpowered-single-symbol-probe.md).
2. **The stride=200 Feb H100 AVAX "pass" at 0.575 is confirmed spurious.**
   At stride=50 it becomes 0.531, inside [0.488, 0.579]. The previous point
   estimate was at the upper percentile of the new CI; stride=200 n=120 was
   always inside its own ±0.13 CI.
3. **Reporting point estimates without CIs is the same epistemology that
   produced both the spurious pass and would produce a spurious fail.**
   Council-5's Rank-1 ask (bootstrap CIs on every cell) was required
   methodological hygiene.
4. **Triage before amendment.** Council-5 gated the spec amendment on this
   triage; the results turned the Gate 3 reframe into a
   measurement-motivated retirement rather than a post-hoc
   "my model missed, move the bar" pattern.

## Verdict

**EXONERATED (inconclusive on the original pre-registered claim).** The
AVAX Gate 3 point-estimate failure is NOT a clean falsifier of
cross-symbol transfer. The probe cannot distinguish "encoder doesn't
transfer to AVAX" from "encoder's signal is invisible at this sample size
on ANY single-symbol 1-month pool." The combination of this experiment +
[cluster cohesion](cluster-cohesion-diagnostic.md) +
[surrogate sweep](per-symbol-surrogate-sweep.md) formed the three-legged
evidence base for
[retiring Gate 3 to informational](../decisions/gate3-retired-to-informational.md).

## Related

- [Gate 3 retired to informational](../decisions/gate3-retired-to-informational.md)
- [Bootstrap methodology](../concepts/bootstrap-methodology.md)
- [Underpowered single-symbol probe](../concepts/underpowered-single-symbol-probe.md)
- [Cluster cohesion experiment](cluster-cohesion-diagnostic.md)
- [Per-symbol surrogate sweep](per-symbol-surrogate-sweep.md)

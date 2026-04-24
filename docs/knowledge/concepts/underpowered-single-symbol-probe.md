---
title: Underpowered Single-Symbol Probe
topics: [evaluation, statistics, falsifiability, gate3]
sources:
  - docs/experiments/step5-gate3-triage.md
  - docs/experiments/step5-surrogate-sweep.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
last_updated: 2026-04-24
---

# Underpowered Single-Symbol Probe

## What It Is

The empirical finding that a 1-month single-symbol pretraining-density probe
(stride=50, n_test ≈ 300–900 per cell) has a 95% bootstrap CI width of
~0.09–0.12 on balanced accuracy, which **exceeds the encoder's measured Gate-1
signal amplitude** (+1.9–2.3pp mean lift; per-symbol lifts typically 1–3pp).
Any such probe is mathematically incapable of detecting the encoder's real
signal against chance.

This is a methodological property of the probe design, not a property of the
encoder or of any particular symbol.

## The Quantitative Statement

Binomial SE on balanced accuracy with n_test per class ≈ n/2:

- n_test = 120 (stride=200, 1 symbol, 1 month): SE ≈ 0.065 → CI ±0.13
- n_test = 400 (stride=50, 1 symbol, 1 month): SE ≈ 0.035 → CI ±0.07
- n_test = 900 (stride=50, 2 symbols, 1 month): SE ≈ 0.023 → CI ±0.045
- n_test = 16,000 (stride=200, 24 symbols, 1 month — Gate 1): SE ≈ 0.004 → CI ±0.008

The encoder's measured Gate-1 lift over baselines (~1-3pp) fits inside the
±0.045 CI at n=900 and inside the ±0.13 CI at n=120. Only at n≈16K does
the lift separate from noise.

Stride=50 overlapping windows give ~2× effective-independent samples vs
stride=200 at 4× raw density (75% window overlap with 200-event windows at
stride 50); the CI width narrows by ~√2, not √4.

## Evidence: the three triage measurements

1. **AVAX Gate 3 triage** — encoder CI overlaps PCA CI on 4/4 cells (narrowest
   overlap = 0.4pp on Mar H500). Point estimate never clears 51.4% CI lower
   bound. See [gate3 triage experiment](../experiments/gate3-avax-triage.md).
2. **LINK+LTC in-sample control** — same protocol at ~2× test sample size,
   encoder fails majority on 3/4 cells, encoder-vs-PCA CIs overlap on 4/4.
   **In-sample symbols fail the same way out-of-sample AVAX fails.**
3. **Per-symbol surrogate sweep** — 5 in-sample symbols (ASTER, LDO, DOGE,
   PENGU, UNI) × 2 months × 2 horizons = 20 cells, all with encoder CI
   strictly above PCA CI in only **1/20 cells** (ASTER Feb H500). At α=0.05,
   1/20 is exactly the chance rate for spurious separation — the surrogate
   sweep measures the protocol's null distribution. See
   [surrogate sweep experiment](../experiments/per-symbol-surrogate-sweep.md).

## The Implication for Gate 3

The pre-registered Gate 3 was "encoder > 51.4% at H100 on AVAX held-out."
Under the measurement conditions available — one month of AVAX at stride=50
gives n_test ≈ 400–460 — the bootstrap CI width dominates the signal
amplitude. The gate was underpowered by design.

- **Declaring Gate 3 a failure on point estimates inside their own CIs** would
  be retroactive false-negative inflation — no better than false-positive
  p-hacking in the opposite direction.
- **Declaring Gate 3 a pass on point estimates inside their own CIs** would be
  the classic p-hacking pattern.
- **The honest reading:** the protocol cannot falsify anything at this n_test,
  and the amendment retires Gate 3 to informational status with a re-activation
  clause requiring n_test ≥ 2000.

## When a Probe Is Underpowered

Heuristic: compare **probe n_test** against **expected signal amplitude** and
**95% CI width**:

- If CI width > signal amplitude → underpowered, probe is informational only.
- If CI width < signal amplitude / 2 → adequately powered.
- If signal amplitude unknown → use Gate-1 measured amplitude as prior (~1–3pp)
  and require CI width < 0.03.

At stride=50 a 24-symbol pool of ~16K test windows per month satisfies this;
a 1-symbol pool of ~400 test windows does not.

## Gotchas

1. **Point estimates inside their own CI are not a signal.** "Encoder 0.531,
   PCA 0.548, encoder 1.7pp below" is not "encoder fails" when the CIs are
   ±0.07.
2. **Larger n from stride shrinkage is not 1:1 n_effective.** Overlapping
   windows autocorrelate; effective sample size ≈ raw n / (1 + autocorrelation
   lag coefficient); at 75% overlap this is roughly raw n × √(1/overlap).
3. **Multi-cell rate of spurious separations.** 1/20 is chance at α=0.05. Do
   not promote a 1/20 surrogate-sweep hit to "real signal on one symbol" —
   that is the surrogate-sweep's sampling-variance tail draw.
4. **The in-sample-control principle generalizes.** For ANY single-symbol
   probe claim, run the same protocol on 2–3 pretraining symbols matched in
   size and check that encoder > PCA there. If not, the probe is too small;
   the claim cannot rest on n=1 symbol regardless of what the numbers show.

## Related Concepts

- [Bootstrap methodology](bootstrap-methodology.md)
- [Gate 3 retirement](../decisions/gate3-retired-to-informational.md)
- [Gate 3 triage experiment](../experiments/gate3-avax-triage.md)
- [Surrogate sweep experiment](../experiments/per-symbol-surrogate-sweep.md)

---
title: Calibrated Interpretation — Per-Symbol-Clustered Representation
date: 2026-04-27
status: accepted
decided_by: lead-0 + council-1 final QA
sources:
  - docs/experiments/step4-program-end-state.md (calibrated interpretation section)
  - docs/council-reviews/2026-04-27-tape-state-paired-probe-c4-design.md
  - docs/council-reviews/2026-04-27-tape-state-paired-probe-c5-falsifiability.md
last_updated: 2026-04-27
---

# Decision: Calibrated Interpretation — Per-Symbol-Clustered Representation

## What Was Decided

The program end-state writeup commits to a calibrated interpretation
rather than leaving the tape-vs-priors question "undetermined":

> The encoder produces a **per-symbol-clustered representation with
> linearly-extractable directional signal** (+1pp at H500, temporally
> stable). The +0.037 cross-symbol same-hour cosine delta indicates a
> weak shared geometry — possibly the universal sign-of-flow predicate
> (untested at the +0.037 magnitude against an appropriate null
> distribution; consistent with council-2's Cont-de Larrard
> symbol-specific OFI framing). The +0.139 same-symbol delta and 0.934
> symbol-ID probe indicate the dominant geometric structure is
> per-symbol clustering, not Wyckoff-phenomenological tape state.

## Why

Three independent pre-registered diagnostics converge on the same
geometric claim:

| Diagnostic | Value | Interpretation |
|---|---|---|
| Symbol-identity probe (6 liquid anchors) | 0.934 bal_acc | Encoder is nearly symbol-separable |
| Same-symbol-diff-hour vs cross-symbol-diff-hour cosine delta | +0.139 | Strong per-symbol clustering |
| Cross-symbol-same-hour vs cross-symbol-diff-hour cosine delta (SimCLR-trained-for axis) | +0.037 | SimCLR alignment is 3.8× weaker than per-symbol clustering |
| Per-symbol RankMe (median 41.4) vs pooled RankMe (64.2) | ratio 0.65 | Per-symbol uses fewer directions than the pool |

The 3.8× ratio is a magnitude statement, not a directional one;
magnitude statements need less evidence than causal ones. The "What
this rules out / does NOT rule out" structure preserves the residual
question (per-symbol tape-reading remains open).

## Alternatives Considered

- **Leave as "undetermined":** Rejected — the existing evidence is
  coherent enough to support the calibrated claim, and "undetermined"
  understates what the cohesion data actually shows.
- **Run the tape-state-paired-probe to adjudicate firmly:** Rejected
  on EV grounds; ~80% probability of confirming or aborting, ~15% of
  meaningful new info. See
  [Tape-State Diagnostic Off-Ramp](../experiments/tape-state-diagnostic-off-ramp.md).
- **Claim "encoder reads tape" outright:** Rejected — cluster cohesion
  delta +0.037 is below the +0.10 universality threshold that was
  pre-registered; the cross-symbol invariance claim is not supported.

## Council-1 QA Constraints (Binding)

- The phrase "most plausibly the universal sign-of-flow predicate" was
  softened to "possibly ... (untested at the +0.037 magnitude against
  an appropriate null distribution)" because the +0.037 has no measured
  null and "most plausibly" is a causal-mechanism claim resting on a
  single number.
- DSR effective N stays at **3** (Gates 1, 2, 4). The cohesion + RankMe
  + symbol-ID measurements are pre-registered representation-quality
  diagnostics whose pass/fail thresholds are reused in a new narrative
  frame; they are not new probes against the +1pp claim.
- Source-artifact pointers added so a future reader can reconstruct the
  evidence chain without grepping (cohesion in
  `docs/experiments/step5-cluster-cohesion.md`, RankMe in
  `runs/step4-r1-perhorizon/rankme-feb-mar.json`, symbol-ID in
  Gate 1 probe logs).

## Impact

- The program end-state writeup ships with a calibrated claim, not a
  hedged "undetermined."
- The mandated external headline reads:
  *"+1pp linearly-extractable direction signal at H500 within a
  per-symbol-clustered representation, stable across training-period
  halves but not amplifiable by supervised fine-tuning, with
  phenomenological richness untested due to operational label
  calibration failure."*
- Future programs revisiting this question on a different stack must
  start from this interpretation, not re-litigate it.

## Related

- [Cross-Symbol Invariance](../concepts/cross-symbol-invariance.md)
- [Cluster Cohesion Diagnostic](../experiments/cluster-cohesion-diagnostic.md)
- [Tape-State Diagnostic Off-Ramp](../experiments/tape-state-diagnostic-off-ramp.md)
- [Path A Program Closure](path-a-program-closure.md)

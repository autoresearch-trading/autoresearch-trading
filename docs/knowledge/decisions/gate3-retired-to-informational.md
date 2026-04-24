---
title: Gate 3 Retired to Informational Status
date: 2026-04-24
status: accepted
decided_by: lead-0 + council-1 + council-3 + council-5 (spec amendment v1+v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
  - docs/council-reviews/council-3-avax-microstructure.md
  - docs/council-reviews/council-1-amendment-2026-04-24.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
  - docs/experiments/step5-gate3-triage.md
  - docs/experiments/step5-cluster-cohesion.md
  - docs/experiments/step5-surrogate-sweep.md
last_updated: 2026-04-24
---

# Decision: Gate 3 Retired to Informational Status

## What Was Decided

Retire Gate 3 (held-out AVAX symbol > 51.4% at H100 held-out) from binding
pass/fail stop-gate status. AVAX probe numbers are still published per-month
per-horizon with bootstrap CIs, but they no longer have power to stop the
pipeline. AVAX remains irrevocably held-out from pretraining.

The retirement is contingent on three pre-dispatched evidence lines
converging; see "Why" below. A binding re-activation clause specifies the
numerical conditions for any future training run to restore Gate 3 to
pass/fail status.

## Why

Three independent lines of evidence, each measured against a pre-dispatched
threshold, converge on "the 1-month single-symbol probe cannot detect the
encoder's measured Gate-1 signal, regardless of symbol choice."

### Evidence 1 — Bootstrap CI overlap (council-5 Rank 1 pre-dispatch)

On AVAX Feb/Mar at stride=50 with 1000-resample bootstrap CIs, encoder-vs-PCA
CIs **overlap on 4/4 primary cells** (narrowest overlap 0.4pp on Mar H500, in
the direction PCA > encoder). Encoder CI lower bound never clears 51.4%. The
pre-registered threshold is not met at CI-aware rigor.
See [gate3 triage experiment](../experiments/gate3-avax-triage.md).

### Evidence 2 — In-sample control fails identically (council-3 pre-dispatch)

LINK+LTC on the SAME methodology at ~2× test sample size: encoder fails
majority on 3/4 cells, CI overlap on 4/4. AAVE replicates the same pattern.
**AVAX is not anomalous — the probe is underpowered for any mid-Tier-2 symbol
at n_test ~400–900.** See
[underpowered single-symbol probe concept](../concepts/underpowered-single-symbol-probe.md).

### Evidence 3 — Cluster cohesion delta below threshold (council-5 Rank 3 pre-dispatch)

On the 6 liquid SimCLR anchors (BTC/ETH/SOL/BNB/LINK/LTC), measured
cross-symbol same-hour cosine minus cross-symbol diff-hour cosine = **+0.037**,
below the pre-dispatched +0.10 `some_invariance` threshold. Symbol-identity
signal is +0.139 (4× stronger). 6-way symbol-ID probe balanced accuracy =
0.934. **The training config (6-of-24 SimCLR anchors, soft-positive weight
0.5) did not target cross-symbol universality.** AVAX transfer failure was
training-dynamics-overdetermined. See
[cross-symbol invariance concept](../concepts/cross-symbol-invariance.md).

## Honest Framing (per council-1 + council-5 amendment review)

This is a **post-hoc retirement of a pre-registered falsifier**. The
amendment's first-draft defense language was "Gate 1 passed before Gate 3
ran, therefore the reframe is measurement-motivated" — council-1 and
council-5 both flagged this as a non-sequitur. **Temporal order of passes
does not establish non-retroactive-rationalization.**

The real defense: the retirement criteria themselves (bootstrap CI overlap,
in-sample control failure, cluster delta <+0.1) were **pre-dispatched in
council reviews BEFORE the triage experiments ran**. All three measurements
are conservative point-estimate comparisons against thresholds set before the
data was collected. Under López de Prado's framework, retiring a gate
because IT cannot falsify the null is stronger evidence of a badly-designed
gate than of a bad model.

Council-5 explicitly flagged the "not retroactive rationalization" paragraph
as reading more defensive than it should. Dissent preserved: the paragraph's
causal argument **over-argues** the case; the honest defense is pre-dispatched
thresholds plus conservative readings, not temporal order.

## Re-activation Criteria (binding on any future training run)

Gate 3 may be re-activated as binding pass/fail IFF **ALL** of:

(a) `n_test ≥ 2,000` windows per held-out cell after stride ≤ 50 evaluation;
(b) 1000-resample bootstrap 95% CI on encoder balanced accuracy does NOT
include 0.500 on the control in-sample pool at matched n_test;
(c) cross-symbol SimCLR cluster-cohesion delta ≥ **+0.10** (measured as
`cross_symbol_same_hour − cross_symbol_diff_hour` on the liquid anchor set);
(d) these re-activation criteria are declared **BEFORE** the held-out AVAX
evaluation is run.

Absent all four, AVAX numbers remain informational. This closes the drift-back
loophole (council-1 Q4).

## AVAX Irrevocability Extension

AVAX MUST remain held-out from pretraining universes of any successor run that
wants to claim Gate 3 evaluation against this program's baseline. A future run
that includes AVAX in pretraining may not re-use this program's Gate 3
numbers as a baseline (council-5 medium-severity item).

## Alternatives Considered

1. **Option A: Drop Gate 3 entirely.** Rejected — the AVAX probe numbers are
   still useful informational evidence even if not binding.
2. **Option B: Broaden Gate 3 to a held-out set of 2–3 symbols** (council-3
   recommendation). Partially adopted as the per-symbol surrogate sweep
   diagnostic (Step-6), but not as a new binding Gate 3 — the surrogate sweep
   measured the protocol's null distribution (1/20 CI separations) rather than
   converting the transfer claim.
3. **Option D: Keep Gate 3 binding and declare failure on point estimates.**
   Rejected — it is retroactive false-negative inflation to fail a pre-
   registered gate on point estimates inside their own CIs.

## Impact

- Spec sections amended: "Gate 3: Cross-Symbol Transfer — INFORMATIONAL",
  re-activation criteria block, AVAX irrevocability clause.
- Gate 2 (fine-tuning) proceeds on the pretrained 24-symbol universe; Gate 3's
  informational status does not block Step 4.
- The "universal microstructure" framing is retired for this training config;
  the claim earned is "per-symbol feature quality on a 24-symbol pretraining
  universe," not "universal tape representations that transfer cross-symbol."
- Per-symbol surrogate sweep logged as Step-6 diagnostic (pre-committed in
  amendment v2, ran late same day — 1/20 CI separations, protocol-null-rate,
  validates the underpower interpretation).
- [Symbol-ID probe <20% target reframed aspirational](symbol-id-probe-reframed-aspirational.md).

## Related

- [Cross-symbol invariance](../concepts/cross-symbol-invariance.md)
- [Underpowered single-symbol probe](../concepts/underpowered-single-symbol-probe.md)
- [Bootstrap methodology](../concepts/bootstrap-methodology.md)
- [Gate 3 triage experiment](../experiments/gate3-avax-triage.md)
- [Cluster cohesion experiment](../experiments/cluster-cohesion-diagnostic.md)
- [Per-symbol surrogate sweep](../experiments/per-symbol-surrogate-sweep.md)
- [Symbol-ID reframe](symbol-id-probe-reframed-aspirational.md)

---
title: Tape-State-Paired-Probe Diagnostic — Off-Ramp on Falsifiability Grounds
date: 2026-04-27
status: abandoned
result: inconclusive
sources:
  - docs/council-reviews/2026-04-27-tape-state-paired-probe-c4-design.md
  - docs/council-reviews/2026-04-27-tape-state-paired-probe-c5-falsifiability.md
  - docs/experiments/step4-program-end-state.md (calibrated interpretation section)
last_updated: 2026-04-27
---

# Experiment: Tape-State-Paired-Probe Off-Ramp

## Hypothesis

A bucketed-cohesion paired probe could adjudicate the unresolved
interpretation question: does the encoder read tape volume-price
phenomenology, or does it read per-symbol direction priors?

**Proposed design (council-4):** Bucket eval windows by
(`effort_vs_result` × `is_open` tertiles). Compute mean cosine of
cross-symbol-same-hour pairs in encoder space and PCA-on-flat-features
space. If encoder cosine meaningfully exceeds PCA cosine within
tape-state buckets, encoder reads tape; if equivalent, encoder reads
per-symbol priors.

## Council-5 Falsifiability Rejection

c-5 rejected the design as proposed pre-commit on three grounds:

1. **Symbol-confound abort risk ~30-40%.** `is_open` and rolling-σ
   `effort_vs_result` distributions are symbol-shaped; the diagonal
   bucket cells likely fail a chi-square uniformity test on per-bucket
   symbol counts.
2. **Multiple-comparisons inflation.** 9 buckets but effective tests 4-6;
   without Bonferroni correction the diagnostic over-rejects null.
3. **Marginal power below +0.05 effect.** At realistic (symbol, date)
   block-effective sample size (100-500 per bucket), bootstrap CIs only
   resolve effects ≥+0.05. The diagnostic cannot honestly distinguish
   "encoder reads weak tape" from "encoder reads no tape."

c-5's binding pre-commit requirements before approval:
- Chi-square symbol-confound abort (3+ buckets fail uniformity → abort)
- 5/9 buckets clear threshold AND ≥2 in high-`effort_vs_result` row
- (symbol, date)-blocked bootstrap with 99.44% Bonferroni-corrected CI
  excluding zero
- Pre-published per-bucket effective block count
- Per-bucket effect size ≥+0.05 point estimate

## Council-4's Bucket Fix (Preserved for Future)

c-4 made one substantive correction during the design review:
**replace `climax_score` with `is_open` as the second bucket axis.**
`climax_score` has no empirical dynamic range to tertile (max 0.256 vs
the C1/C3/C4 threshold of 3.0 — same trap that killed the multi-probe
battery). `is_open` is bounded [0,1], pools cleanly across symbols, and
IS the DEX-specific Composite Operator footprint (the feature whose
20-trade autocorrelation half-life was the strongest persistent signal
in the dataset).

## Outcome Probability Distribution

| Outcome | Estimated probability | Information value |
|---|---|---|
| Aborted at symbol-confound test | 30-40% | Diagnostic non-result; interpretation stays unresolved (anti-amnesia adds artifact) |
| Threshold A (encoder reads symbol priors) | 30-40% | Confirms what cohesion + RankMe + symbol-ID already suggest |
| Threshold B (uniform tape signal, not Wyckoff) | 15-25% | Modest new info |
| Threshold C (Wyckoff-specific concentration) | 5-15% | Genuinely new |

~80% probability of confirming what existing evidence already shows or
aborting on confound grounds. Lead-0 took the off-ramp on EV grounds.

## Calibrated Interpretation Replaces Diagnostic

Instead of running a new probe, the program end-state writeup commits to
a calibrated interpretation grounded in existing evidence (cohesion
deltas +0.139 vs +0.037, ratio 3.8×; symbol-ID probe 0.934; per-symbol
RankMe 41.4 vs pooled 64.2): **"per-symbol-clustered representation
with linearly-extractable directional signal."**

This claim is anchored to council-2's Cont-de Larrard symbol-specific
OFI framing and is consistent with c-5's prior position that the
existing data adjudicates the question softly. The +0.037 cross-symbol
delta is described as "possibly the universal sign-of-flow predicate
(untested at the +0.037 magnitude against an appropriate null
distribution)" — c-1's binding QA edit on 2026-04-27.

## Verdict — DECLINED PRE-COMMIT

No measurement attempted. The interpretation question is adjudicated
softly via existing diagnostics. Future programs revisiting this
question on a different stack should start from c-4's bucket fix
(`effort_vs_result × is_open`, NOT × `climax_score`) and c-5's
falsifiability spine.

## What We Learned

1. **Existing evidence often adjudicates softly.** The cohesion +
   RankMe + symbol-ID triple already pointed at "per-symbol-clustered."
   Running a new diagnostic with high abort risk and high probability
   of confirming what's already known has low EV.
2. **Falsifiability gates can decline experiments pre-commit.** This is
   the second time on this program (after Path D on the multi-probe
   battery) that a pre-registered probe was retired before measurement.
3. **`is_open` is the right second-axis Wyckoff feature** for any
   future bucket-based tape-state diagnostic. `climax_score` empirically
   does not bucket usefully.

## Related

- [Multi-Probe Battery Path D](../experiments/multi-probe-battery-path-d.md)
- [Cluster Cohesion Diagnostic](../experiments/cluster-cohesion-diagnostic.md) — the existing soft adjudication
- [Calibrated Interpretation](../decisions/calibrated-interpretation-per-symbol-clustered.md)
- [Cross-Symbol Invariance](../concepts/cross-symbol-invariance.md)

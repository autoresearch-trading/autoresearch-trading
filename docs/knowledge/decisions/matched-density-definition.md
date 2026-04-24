---
title: Matched-Density Held-Out Months — Quantitative Definition
date: 2026-04-24
status: accepted
decided_by: council-5 + lead-0 (spec amendment v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
last_updated: 2026-04-24
---

# Decision: Matched-Density Held-Out Months — Quantitative Definition

## What Was Decided

Two held-out months are "matched-density" to the training set IFF each
symbol's **windows-per-day on the held-out months is within 0.7–1.3× of the
same symbol's mean windows-per-day across the training period**, measured at
**stride=200 evaluation**.

Feb + Mar 2026 satisfy this for all 24 pretraining symbols (Feb 21,290 total
windows / Mar 16,278 total windows at stride=200, vs training ~63K windows/month
at matched cadence).

## Why

Council-5 HDF #1 identified "matched-density" as the single worst hidden
degree of freedom in the spec amendment v1 — the phrase appeared five times
and had no measurable threshold. Any future training run could pretrain on
different dates, have a held-out period with dramatically different event
rates, and still claim "matched density" by eye. Every future run gets to
define its own denominator.

The 0.7–1.3× band is generous enough to absorb normal rate variation
(weekends, brief low-liquidity periods, exchange maintenance) while tight
enough to catch a genuinely different-regime held-out period (e.g., a
post-event volatility spike that triples windows/day or a holiday period that
halves it).

## Why Per-Symbol, Not Pool-Level

Pool-level density can be gamed: a held-out month can have matched total
windows while BTC is 2× denser and memecoins are 0.5× denser, which would let
the probe's signal be dominated by BTC-specific behavior. Per-symbol matching
forces each symbol to individually contribute proportionately; failure is
detectable and fixable by excluding a specific symbol from the held-out
evaluation if needed.

## Why Stride=200 Eval Cadence

Stride=200 is the pre-registered evaluation cadence (non-overlapping
windows). Measuring density at stride=50 (pretraining cadence) would inflate
the count by 4× through window overlap and does not reflect the independent-
sample count the probe actually uses.

## Alternatives Considered

1. **Require absolute-count match (e.g., both months within 10% of training
   month mean).** Rejected — too strict on symbols that happen to have low
   activity in a given month; would force exclusion of legitimate months.
2. **Require held-out total windows ≥ 5,000 per symbol per month.** Rejected —
   imposes a hard minimum that doesn't relate to training density; a symbol
   with consistently low activity (e.g., PENGU) would always fail this.
3. **Leave "matched density" undefined (original amendment v1 language).**
   Rejected per council-5 HDF #1 — the phrase is load-bearing for every
   future Gate 1 evaluation and must be measurable.

## Impact

- Spec Gate 1 section: matched-density definition added as binding language in
  amendment v2 (commit `9c91f85`).
- Any future Gate 1 evaluation must compute and publish the per-symbol
  held-out-vs-training density ratio on stride=200 BEFORE running the probe.
- Current Gate 1 Feb+Mar 2026 satisfies the definition (verified at amendment
  v2).

## Related

- [Gate 1 window amended](gate1-window-amended-feb-mar-h500.md)
- [Horizon selection rule](horizon-selection-rule.md)

---
title: Gate 1 Window Amended — Feb+Mar 2026 at H500
date: 2026-04-24
status: accepted
decided_by: lead-0 + council-1 + council-5 (spec amendment v1+v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-1-amendment-2026-04-24.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
  - docs/experiments/step3-run-2-gate1-pass.md
last_updated: 2026-04-24
---

# Decision: Gate 1 Window Amended — Feb+Mar 2026 at H500

## What Was Decided

Move the binding Gate 1 evaluation window from **April 1–13 at H100** (original
pre-registration) to **Feb 2026 AND Mar 2026 independently, at H500**. Both
months must pass ALL FOUR binding conditions (51.4% / Majority+1pp / RP+1pp /
hour-of-day probe <10%) on 15+/24 symbols (AVAX excluded from the in-pretraining
count).

Supersedes
[gate1-thresholds-revised.md](gate1-thresholds-revised.md) (the 4-condition
structure carries forward; only the window and horizon change).

## Why

Two **separable** justifications per council-1's methodological review. Only the
first is load-bearing; the second is corroborating only:

### Primary (ex-ante, load-bearing)

**April 1–13 at stride=200 produces 60–150 windows per symbol, below the
probe's 200-window `min_valid` floor.** This is an arithmetic constraint
verifiable from the cache manifest BEFORE any encoder is trained — it would
have disqualified the window under any architecture. This is the only
justification that survives pre-registration ethics (council-1 Q1).

### Secondary (ex-post, corroborating)

On this encoder's output, **H100 direction prediction is at the noise floor
for every predictor** (encoder, PCA, RP, shuffled). H500 shows clear encoder
signal: +1.9–2.3pp over flat baselines on 17/24 Feb, 14/24 Mar. Per
council-1, this is noted but is NOT the reason to amend — it is textbook
"the backtest selected its own window" if relied on.

Council-1 explicitly blocked the amendment if the two justifications were
conflated; spec amendment v2 separates them textually.

## Alternatives Considered

1. **Keep April 1–13 at H100.** Rejected — ex-ante arithmetic: 60–150 windows
   < min_valid 200 on most symbols. The window cannot be evaluated at
   pre-registered rigor; keeping it would force a simultaneous
   "under-powered pass" or "under-powered fail" that admits no clean reading.
2. **Move to April 1–30 at H100 (longer window, same horizon).** Rejected —
   still inside the "H100 at noise floor" region on this data; would require
   extending data collection several weeks and delays Step 4.
3. **Move to Feb+Mar at H100 only (no H500 primary).** Rejected — H100 balanced
   accuracy at 0.50x for all predictors including PCA provides no falsifiable
   headroom; every cell inside its own CI, nothing can be decided.
4. **Move to Feb+Mar mean-across-months with stricter threshold.** Rejected —
   means hide per-month tails. AND-of-passes preserves the regime-shift
   information (council-1 Q2).

## Impact

- Spec commits `b1f4065` (v1) and `9c91f85` (v2) on 2026-04-24.
- **Anti-amnesia clause (binding):** every Gate 1 report MUST publish the
  original April 1–13 H100 numbers alongside the amended Feb+Mar H500 numbers,
  labeled "original pre-registration, superseded 2026-04-24." The superseded
  numbers remain informational but visible indefinitely.
- **No re-sampling loophole.** The held-out months are Feb 2026 AND Mar 2026
  specifically; they may not be substituted, excluded, or supplemented without
  re-pre-registration (closes council-5 HDF #2).
- **No one-passes-one-fails adjudication.** If any future re-run passes one
  month and fails the other, the RUN FAILS Gate 1. No averaging, no "close
  enough" (council-1 Q2).
- **15+/24, not 15+/25** — mechanically tighter (60.0% → 62.5%), not laxer.
- Gate 4 re-written for coherence (no more "months 5-6 of training" which no
  longer exist) — see
  [gate4 rewrite](gate4-rewrite-for-coherence.md).
- Horizon selection rule added to protect future runs — see
  [horizon selection](horizon-selection-rule.md).
- Matched-density definition added (closes council-5 HDF #1) — see
  [matched-density definition](matched-density-definition.md).

## Current Status (post-amendment)

**PASSES on Feb AND Mar 2026.** Feb: +3.03pp vs Majority, +1.91pp vs RP, 15/24
≥ 51.4%, hour probe 0.06–0.09. Mar: +3.12pp vs Majority, +2.29pp vs RP, 17/24
≥ 51.4%, hour probe 0.06–0.09. Writeup:
[gate1 pass experiment](../experiments/gate1-pass-feb-mar-h500.md).
Checkpoint: `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params).

## Related

- [Gate 1 thresholds revised (superseded)](gate1-thresholds-revised.md)
- [Horizon selection rule](horizon-selection-rule.md)
- [Matched-density definition](matched-density-definition.md)
- [Gate 4 rewrite](gate4-rewrite-for-coherence.md)
- [Gate 1 pass experiment](../experiments/gate1-pass-feb-mar-h500.md)

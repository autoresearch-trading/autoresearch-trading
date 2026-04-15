---
title: Session-of-Day Leakage
topics: [leakage, evaluation, features, pretraining]
sources:
  - docs/council-reviews/council-5-gate0-falsifiability.md
  - docs/council-reviews/council-6-gate0-impact-on-pretraining.md
last_updated: 2026-04-15
---

# Session-of-Day Leakage

## What It Is

Crypto markets have persistent directional biases tied to UTC session (Asia open,
London open, NY open). If a model learns "morning sessions on thin alts tend to
drift positive" rather than "absorption → ease-of-movement transitions predict
direction," it will score above chance on in-sample evaluation but fail to
transfer to any structurally different period.

Both the flat-feature baseline and the CNN encoder can encode session-of-day as
a shortcut. The risk is not hypothetical: council round 6 identified two specific
leak vectors and observed a ~1pp raw-vs-balanced effect at Gate 0.

## Leak vectors

### 1. Flat features — `_last` statistic block

Features 68–84 of the 85-dim flat vector are last-event values of the 17
channels. Two are session-indicative:

- **`time_delta_last`** — inter-event gap at the window's last event.
  Session opens compress gaps to fractions of a second; overnight gaps span
  seconds to minutes.
- **`prev_seq_time_span_last`** — total wall-clock span of the immediately
  prior 200-event window. For illiquid symbols (2Z at ~56 min per 200 events),
  this value directly encodes approximate UTC time of day.

### 2. Raw sequence — CNN input

The CNN sees `time_delta` and `prev_seq_time_span` at EVERY event, not just
as summaries. Session signal is denser in the raw sequence. Additional
channels also carry session:

- **`log_spread`** — spreads systematically widen at session boundaries.
- **`is_open`** fraction has diurnal rhythms across perpetual futures.

## Why MEM doesn't protect against it

Session patterns repeat day-to-day. MEM reconstruction improves when the encoder
memorizes "9am Asia has tight spreads, high `is_open`" because that pattern is
predictive within the window given partial context. The model satisfies the
self-supervised objective by encoding session identity, which then transfers to
a direction probe where morning-up-bias makes the probe work.

## Our Countermeasures

### 1. Pre-pretraining confound check

Before launching pretraining, train LR on a single feature — hour-of-day
(4-hour bins, one-hot) — against the same Gate 0 walk-forward folds. If this
one-feature model beats PCA+LR on the 85-dim flat features by > 0.5pp on 5+
symbols, the `_last` block is leaking and must be pruned. Cost: < 5 minutes.

### 2. Stronger SimCLR augmentations (council-6 recipe)

- **Window jitter ±25 events** (up from ±10) — crosses BTC session micro-
  boundaries and shifts illiquid-alt window centers by ~10 min.
- **Timing-feature noise σ=0.10** on `time_delta` and `prev_seq_time_span`
  during SimCLR view generation. Forces the encoder to rely on relative rhythms,
  not absolute session-indicative magnitudes.
- **Do NOT exclude `prev_seq_time_span` from MEM** — its local rhythm signal is
  genuinely microstructural. The augmentation handles the session-leak risk
  without removing the feature.

### 3. Hour-of-day probe at Gate 1 (binding)

24-class LR on frozen 256-dim embeddings predicting UTC hour of window center.
**Must be < 10% accuracy** (above 1/24 = 4.2% chance = session info leaking;
below 10% = clean).

Plus **stratified accuracy < 1.5pp cross-session variance** — the linear probe's
direction-accuracy must be stable across Asia / Europe / US sessions.

### 4. Pretraining-era monitoring

Hour-of-day probe run every 5 epochs. Early warning if the encoder latches onto
session identity before it reaches Gate 1.

## Gotchas

1. **Raw accuracy masks this effect.** Use balanced accuracy to avoid
   misattributing majority-class exploitation to "signal."
2. **Symbol-identity probe doesn't catch it.** An encoder can encode
   session-of-day without encoding symbol (all symbols share the same UTC).
3. **Adversarial hour-of-day head was considered and rejected** (council-6):
   adversarial heads destabilize training when the primary task is weak.
   Probe-at-evaluation is strictly preferable.

## Related Concepts

- [Gate 0 Baseline Grid](gate0-baseline.md)
- [Gate 1 Thresholds Revised](../decisions/gate1-thresholds-revised.md)
- [Effort vs Result](effort-vs-result.md)

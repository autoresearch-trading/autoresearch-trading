---
title: SimCLR Augmentations — Strengthened for Session Decorrelation
date: 2026-04-15
status: accepted
decided_by: council-6 (round 6)
sources:
  - docs/council-reviews/council-6-gate0-impact-on-pretraining.md
last_updated: 2026-04-15
---

# Decision: SimCLR Augmentations — Strengthened for Session Decorrelation

## What Was Decided

Two changes to the SimCLR view-generation augmentation recipe:

1. **Window start jitter ±10 → ±25 events.** (Applied to both views.)
2. **New: timing-feature noise σ=0.10** Gaussian noise injected on
   `time_delta` and `prev_seq_time_span` during view generation.
   Five times the baseline `σ = 0.02 × feature_std`.

All other augmentations unchanged:

- Additive Gaussian noise σ = 0.02 × feature_std (continuous features only)
- Feature dropout p=0.05 per feature per event
- Time scale dilation: multiply time_delta by factor in [0.8, 1.2]

## Why

Council round 6 Gate 0 review found that the flat-feature baseline
(PCA+LR on 85-dim summary statistics) is indistinguishable from a
majority-class predictor on balanced accuracy. A significant chunk of the
residual signal in raw features is plausibly session-of-day leakage via
`time_delta` and `prev_seq_time_span`.

**Old jitter was too small.** ±10 events at BTC's median inter-event gap of
1.5s = ±15 seconds. Events inside the same trading minute. At BTC's 20K
events/day, ±10 is 0.05% of a day — far too small to decorrelate session identity.

**New jitter ±25 events** at BTC scale = ±37 seconds. Crosses minor
session-open micro-boundaries. On illiquid alts (56–68 min per 200-event
window), ±25 events shifts the window center by ~10 minutes, which is
meaningful.

**Timing-feature noise** forces the encoder to rely on *relative* rhythms
(gap ratios between events) rather than *absolute* session-indicative
magnitudes. σ=0.02 was too subtle; σ=0.10 is perceptible at the
scale of typical BTC gaps.

## Alternatives Considered

1. **Exclude `prev_seq_time_span` from MEM reconstruction.** Rejected —
   its local event-rate rhythm is a genuine microstructure signal (stress
   regimes have rapid bursts; thin markets have long gaps). Noise injection
   preserves utility while decorrelating absolute values.
2. **Adversarial hour-of-day head.** Rejected — adversarial heads
   destabilize training when the primary task is weak; compute budget is
   tight (1 H100-day).
3. **Larger jitter (±50).** Deferred — would start crossing day boundaries
   on BTC's fast event rate, violating gotcha #26 unless tied to day-id
   awareness. Current ±25 stays inside a single day for all symbols.

## Impact

- Spec amendment in commit `1f86d52` updates §Contrastive Augmentations.
- Builder-8 will implement in the Step 3 pretraining code.
- Hour-of-day probe every 5 epochs during pretraining provides the
  feedback loop that validates whether these changes worked.

## Related

- [Session-of-Day Leakage](../concepts/session-of-day-leakage.md)
- [Contrastive Learning](../concepts/contrastive-learning.md)
- [Gate 1 Thresholds Revised](gate1-thresholds-revised.md)

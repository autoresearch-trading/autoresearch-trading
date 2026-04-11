---
title: NT-Xent Temperature τ=0.5→0.3
date: 2026-04-10
status: accepted
decided_by: council-6
sources:
  - docs/council-reviews/2026-04-10-round5-council-6-pretraining-mechanics.md
last_updated: 2026-04-10
---

# Decision: NT-Xent Temperature τ=0.5→0.3

## What Was Decided

Set NT-Xent temperature τ=0.5, anneal linearly to τ=0.3 by epoch 10, then hold
constant. This was previously unspecified in the spec.

## Why

ImageNet default τ=0.1 is too cold for financial time series:
- Low τ → sharp softmax → only hardest negatives contribute → collapse risk
- Financial data has much higher intrinsic similarity between pairs than images
  (same market, same time period, similar features)
- Hard negatives in early training are often genuinely similar market states →
  pushing them apart teaches spurious features (symbol identity, time-of-day)

Higher τ → more uniform gradient contribution → slower but safer learning,
especially critical for the cross-symbol positive pairs planned for run 2.

## Alternatives Considered

- τ=0.1 (ImageNet standard): too cold, collapse risk
- τ=1.0: too warm, loss too uniform, slow convergence
- Fixed τ=0.3: viable but misses the benefit of warm start

## Impact

Must be added to the spec's hyperparameter table. Annealing schedule:
`tau = max(0.3, 0.5 - epoch * 0.02)` for epochs 1-10, then constant 0.3.

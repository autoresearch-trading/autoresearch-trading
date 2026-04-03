---
title: Drop is_buy Feature
date: 2026-04-02
status: accepted
decided_by: Council round 5 (council-4 primary, council-5 concurred)
sources:
  - docs/council-reviews/2026-04-02-council-4-tape-viability.md
last_updated: 2026-04-03
---

# Decision: Drop is_buy Feature

## What Was Decided

Remove `is_buy` from the 18-feature input representation, reducing to 17 features.

## Why

Three independent reasons, any one sufficient:

1. **59% ambiguity:** Pre-April data has mixed buy+sell fills at same timestamp
   (exchange reports both counterparties). Majority-vote direction is wrong for
   a significant fraction of events.

2. **No persistence:** Autocorrelation half-life of 1 event — essentially random.
   Compare to is_open (half-life 20 events). A feature with no autocorrelation
   adds noise, not signal.

3. **Distributional discontinuity:** April+ data has fulfill_taker field giving
   clean direction. Training on pre-April (ambiguous) + April (clean) creates a
   distribution shift the model must learn to ignore — wasted capacity.

## Alternatives Considered

- **Keep is_buy with masking:** Mark pre-April is_buy as missing. Rejected:
  model sees the feature sometimes, learns to depend on it, then fails on
  pre-April test windows.
- **Derive from log_return sign:** Redundant with log_return itself.
- **Keep only for April+ data:** Creates train/test distribution mismatch.

## What Replaced It

Directional information preserved in:
- `log_return` (feature 1) — signed price change
- `trade_vs_mid` (feature 14) — execution location vs midpoint, doubles as
  direction proxy

## Impact

Feature count: 18 → 17. Model params: ~91K (negligible change).

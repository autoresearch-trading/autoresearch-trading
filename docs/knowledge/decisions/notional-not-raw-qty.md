---
title: Use Notional Not Raw Qty for Cross-Symbol Features
date: 2026-04-02
status: accepted
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Decision: Use Notional Not Raw Qty for Cross-Symbol Features

## What Was Decided

Three features must use notional (qty x price) instead of raw qty for cross-
symbol comparability:
1. **depth_ratio** (feature 13): `log(max(bid_notional, 1e-6) / max(ask_notional, 1e-6))`
2. **kyle_lambda** (feature 16): uses `signed_notional` in the Cov/Var estimator
3. **cum_ofi_5** (feature 17): accumulates notional changes, normalized by rolling notional volume

## Why

The model trains on all 25 symbols jointly with no per-symbol indicator. For a
universal model, features must be comparable across symbols. Raw qty is not:

- 1 BTC lot at $68,000 = $68,000 notional
- 1 FARTCOIN lot at $0.50 = $0.50 notional
- 100 SOL lots at $150 = $15,000 notional

Under raw qty, a "1 lot bid vs 1 lot ask" depth_ratio is identical for BTC and
FARTCOIN. Under notional, BTC's 1-lot bid is $68K of intent while FARTCOIN's is
$0.50 -- fundamentally different levels of commitment. The model should see this
difference.

Council-2 identified that `imbalance_L1` (feature 12) in the existing
`prepare.py` already used notional. The spec inconsistently used raw qty for
depth_ratio and kyle_lambda. This was a gap, not a design choice.

## Alternatives Considered

- **Raw qty with per-symbol normalization:** Divide by per-symbol median qty.
  Rejected: changes the economic meaning. A 2x-median BTC order and a 2x-median
  FARTCOIN order are not equivalent in dollar commitment or information content.
- **Raw qty with symbol embedding:** Add a learned per-symbol embedding. Rejected
  for v1: adds parameters and the model should learn universal patterns, not
  symbol-specific adjustments.
- **Mixed approach (notional for some, raw for others):** Rejected: inconsistent
  and confusing. All cross-book features should use the same units.

## Impact

All orderbook-derived features that involve quantity now consistently use notional
(qty x price). This includes depth_ratio, kyle_lambda, cum_ofi_5, and
imbalance_L1/L5 (which already used notional). The epsilon guards for depth_ratio
(`1e-6`) are in notional units -- verified appropriate even for low-price symbols
since notional at L1 is always >> 1e-6 during normal trading.

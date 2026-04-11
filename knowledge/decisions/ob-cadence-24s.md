---
title: OB Cadence Is 24s Not 3s
date: 2026-04-02
status: accepted
decided_by: data-eng-13 (measured), Council-2 (impact analysis)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Decision: OB Cadence Is 24s Not 3s

## What Was Decided

The orderbook snapshot cadence is ~24 seconds (median), not ~3 seconds as the
original spec assumed. All feature designs and parameter choices were reviewed
and updated accordingly. The 24s cadence does not fundamentally undermine the
orderbook feature set but requires concrete adjustments to cum_ofi window size
and the OFI formula implementation.

## Why

data-eng-13 measured the actual OB cadence on BTC (2025-10-16): median ~24s,
mean ~24s, max 59s. This is 8x slower than the spec stated. The measurement was
consistent across BTC, FARTCOIN, and XPL -- it is a Pacifica API collection
parameter, not symbol-dependent.

Key consequences of the 8x cadence mismatch:
- `cum_ofi_20` covered ~480s (~8 min) instead of intended ~60s
- ~10.6 order events share the same aligned snapshot (not ~1-2)
- `delta_imbalance_L1` is zero ~90.6% of the time (not ~95% as spec estimated
  from 3s assumption -- actually slightly better)
- Price levels change between snapshots 60-80% of the time, making the naive
  OFI formula produce wrong-sign results during trends

## Alternatives Considered

- **Interpolate between snapshots:** Rejected. Interpolating book state between
  24s snapshots would be fabricating data. The step-function (carry-forward)
  approach is honest about what we know.
- **Drop OB features entirely:** Rejected. Council-2 showed that even at 24s,
  book features provide valid coarse context. The model already trains with OB
  feature dropout (p=0.15) for robustness.
- **Request higher-frequency OB data:** Not available from Pacifica API at this
  time.

## Impact

1. `cum_ofi` window reduced from 20 to 5 snapshots (separate decision)
2. Piecewise Cont 2014 OFI formula required (was optional at 3s, now blocking)
3. `kyle_lambda` recomputed per-snapshot instead of per-event (separate decision)
4. Static book features (log_spread, imbalance, depth_ratio) confirmed valid --
   they capture slow-moving market structure (minutes to hours scale)
5. 24s cadence may actually filter high-frequency quote flickering noise,
   improving signal quality of book features (Council-2 novel observation)

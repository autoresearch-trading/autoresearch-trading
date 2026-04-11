---
title: Orderbook Alignment
topics: [orderbook, data-pipeline, features]
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Orderbook Alignment

## What It Is

Orderbook snapshots and trade events arrive at fundamentally different cadences.
Order events fire every ~10-50ms during active trading; OB snapshots arrive every
~24 seconds (median, measured across BTC/FARTCOIN/XPL by data-eng-13). Every
trade event must be paired with its nearest prior OB snapshot to compute book
features (log_spread, imbalance, depth_ratio, trade_vs_mid, book_walk, etc.).

The alignment method is `np.searchsorted(ob_ts, event_ts, side='right') - 1`,
which returns the index of the last snapshot with timestamp <= the event
timestamp. This is causally clean: no future book state is ever used.

At 24s cadence with ~28K events/day on BTC, roughly 10.6 events share the same
aligned snapshot. This means 8 OB features are constant across blocks of ~10
consecutive events, creating a step-function pattern in the feature matrix.

## Our Implementation

1. Load OB timestamps into a sorted array `ob_ts`.
2. For all events at once: `ob_idx = np.searchsorted(ob_ts, event_ts, side='right') - 1`.
3. Events before the first snapshot (`ob_idx < 0`) get zero-filled OB features.
4. All OB features for event i are looked up from `ob_features[ob_idx[i]]`.

Vectorized over all events in a single symbol-day -- no Python for-loop.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-02 | Vectorized searchsorted | Python for-loop over 50K events/day is 6+ hours; vectorized is seconds | council-5, data-eng-13 |
| 2026-04-02 | Zero-fill pre-first-snapshot events | Honest missing data; rolling stats handle warm-up | data-eng-13 |
| 2026-04-02 | side='right' for equal timestamps | Snapshot known at that ms; causally valid | data-eng-13 |

## Gotchas

1. First trade of the day can precede first OB snapshot by ~56 seconds (measured
   on BTC 2025-10-16). All events in this gap get `ob_idx = -1` -- must guard.
2. OB has 10 levels per side, not 25. All symbols measured at exactly 10 bid + 10
   ask levels.
3. OB `bids`/`asks` columns are arrays of Python dicts, not structured arrays.
   Parse into flat numpy arrays first for vectorized feature computation.
4. During volatile periods, the aligned snapshot mid can be stale by up to 24s.
   `delta_imbalance_L1 == 0` implicitly signals staleness -- the CNN learns to
   discount book features when this is zero for many consecutive events.
5. Do NOT use a Python for-loop for alignment -- vectorize with searchsorted
   over all events at once (CLAUDE.md gotcha 2).

## Related Concepts

- [Kyle Lambda](kyle-lambda.md) -- per-snapshot computation, forward-filled to events
- [Cum OFI](cum-ofi.md) -- rolling sum over 5 snapshots, forward-filled
- [Book Walk](book-walk.md) -- uses mid/spread from aligned snapshot
- [Order Event Grouping](order-event-grouping.md) -- produces the event timestamps

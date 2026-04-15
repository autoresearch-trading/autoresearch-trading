---
title: OB Level Arrays — Zero-Fill, Not NaN-Fill
date: 2026-04-15
status: accepted
decided_by: analyst-9 (post-hoc analysis) + orchestrator
sources:
  - docs/experiments/cache-nan-investigation.md
last_updated: 2026-04-15
---

# Decision: OB Level Arrays — Zero-Fill, Not NaN-Fill

## What Was Decided

In `tape/io_parquet.py::expand_ob_levels()`, initialize the per-level flattened
arrays (bid_price_L1..L10, bid_qty_L1..L10, ask_price_L1..L10, ask_qty_L1..L10)
with `np.zeros(shape)` rather than `np.full(shape, np.nan)`.

## Why

On the first full-corpus cache build (2026-04-15, 4003 shards), the validator
flagged 117 shards with NaN contamination (879 cells total) across ~20
symbols. Clustering on 2026-02-02 through 2026-02-06 suggested a data incident.
Analyst-9 traced the bug: raw OB parquet occasionally delivers snapshots with
fewer than 10 levels on a side. `np.full(..., np.nan)` left the unfilled
slots as NaN, which propagated through:

- `depth_ratio` (722 cells) — 10-level sum of notional; any NaN contaminates.
- `imbalance_L5` (141 cells) — 5-level sum.
- `cum_ofi_5` (11 cells) — piecewise OFI touches L1 price/qty.
- `delta_imbalance_L1` (5 cells) — diff across snapshots.

The AAVE 2025-10-16 smoke test did not catch this because AAVE on that day
happened to have 10 levels throughout.

**Zero-fill is semantically correct.** A missing level means the book had no
visible liquidity at that depth — zero notional is the truthful description,
not "unknown." All downstream guards (`max(x, 1e-6)`, clipping) handle zero
cleanly.

## Alternatives Considered

1. **Drop affected shards.** Rejected — shards with 1-30 NaN cells out of
   thousands have mostly good data; dropping would lose months of signal.
2. **Interpolate between neighbors.** Rejected — invents liquidity the book
   didn't actually have. Zero is honest.
3. **Filter snapshots with <10 levels upstream.** Rejected — reduces effective
   cadence on thin-book symbols; many 3–5-level snapshots are legitimate
   market state, not errors.

## Impact

- Rebuild on 2026-04-15: 4003 shards, 0 critical validator failures (down from
  117). 15-minute rebuild.
- Four residual warnings are real extreme-dislocation events on illiquid symbols
  (KBONK/LDO/XPL specific dates) where `log_spread > 0` because ask > 3× bid.
  These are real market data, not code bugs — the feature correctly describes
  the dislocation.
- Commit `95ca60c` (`fix(tape): zero-fill missing OB levels instead of NaN`).
- CLAUDE.md gotcha #31 added.

## Gotchas

1. **Regression test added**: `tests/tape/test_io_parquet.py` constructs a
   3-bid-level × 10-ask-level synthetic snapshot and asserts `bid_price_L4..L10`
   are `0.0` (not NaN).
2. **Semantic consequence for `imbalance_L1`** on all-ask snapshots:
   `(0 - ask_not)/(0 + ask_not) = -1` — correctly flags a one-sided book.
3. **Never `np.full(np.nan)` for anything downstream of feature aggregation.**
   Add zero-fill or use sentinel-aware aggregations.

## Related

- [Orderbook Alignment](../concepts/orderbook-alignment.md)
- [Cumulative OFI](../concepts/cum-ofi.md)

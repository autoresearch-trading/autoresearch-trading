---
title: Order Event Grouping
topics: [data-pipeline, dedup, preprocessing]
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md
last_updated: 2026-04-10
---

# Order Event Grouping

## What It Is

Same-timestamp trades are fragments of one order being filled across multiple
price levels. Grouping them into single "order events" reduces noise (one order
= one data point, not multiple matching engine artifacts) and captures order-level
signals like book walk and aggressiveness.

## Our Implementation

### Dedup (MUST happen before grouping)

The exchange reports both counterparties of each fill as separate rows. 30-74%
of raw rows are duplicated buyer/seller pairs. `trade_id` is always empty.

**Pre-April data:** `df.drop_duplicates(subset=['ts_ms', 'qty', 'price'], keep='first')`
- Dedup on (ts_ms, qty, price) WITHOUT side
- Including side removes nothing — buyer/seller pairs differ only on side

**April+ data:** `df[df.event_type == 'fulfill_taker']`
- Filter to taker fills only — handles dedup and direction in one step
- Verify event_type is non-null for >99% of rows before using this filter

### Grouping

After dedup, group by timestamp. Per order event compute:
- vwap (quantity-weighted average price)
- total_qty (sum of quantities)
- num_fills (count)
- is_open (fraction of fills that are opens)
- book_walk (last fill price - first fill price)

### Key Numbers

- Raw trades → order events: ~5:1 reduction (140K raw → 28K events/day for BTC)
- 200 order events covers more clock time than 200 raw trades
- 59% of events have mixed buy+sell fills (exchange mechanic, not error)
- Training samples: ~400-560K total, not 1.2M

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-02 | Dedup key: (ts_ms, qty, price) without side | Including side removes nothing | council-5-dedup-direction.md |
| 2026-04-02 | April+ uses fulfill_taker filter | Cleaner than heuristic dedup | council-5-dedup-direction.md |
| 2026-04-02 | Drop is_buy feature | 59% ambiguous + half-life 1 + pre/post-April discontinuity | council-4-tape-viability.md |

## Gotchas

1. Dedup key must NOT include side — buyer/seller pairs differ on side
2. Pre-April and April+ use different dedup strategies
3. 59% mixed-direction events are expected, not a data quality issue
4. 100-trade batching (main branch approach) destroys tape signals

## Related Concepts

- [Kyle Lambda](kyle-lambda.md) — depends on signed flow from grouped events
- [Effort vs Result](effort-vs-result.md) — uses grouped event quantities

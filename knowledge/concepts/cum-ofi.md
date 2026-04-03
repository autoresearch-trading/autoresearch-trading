---
title: Cumulative Order Flow Imbalance (cum_ofi)
topics: [microstructure, orderbook, features, cont-2014]
sources:
  - docs/council-reviews/2026-04-02-council-2-ob-cadence.md
  - docs/council-reviews/2026-04-01-spec-review-council-2.md
  - docs/council-reviews/2026-04-02-spec-review-data-eng-13.md
  - docs/council-reviews/round3-synthesis.md (H3)
last_updated: 2026-04-03
---

# Cumulative Order Flow Imbalance (cum_ofi)

## What It Is

Order Flow Imbalance (OFI) measures the net directional pressure in the order
book by tracking changes in bid and ask depth at L1 between consecutive
snapshots. Cont et al. (2014) showed that cumulative OFI over a time window is
a strong short-horizon price predictor -- accumulated directional pressure forces
market makers to update prices to clear the imbalance.

The key insight from Cont 2014: delta-imbalance at a single snapshot is noisy,
but the cumulative sum over multiple snapshots captures sustained directional
pressure that the market must eventually price.

## Our Implementation (Feature 17)

```
cum_ofi_5 = sum(OFI_t for t in last 5 snapshots) / rolling_notional_volume
```

Uses the **piecewise Cont 2014 formula** (BLOCKING requirement from Council-2):

**Bid side:**
- If `best_bid_price_t > best_bid_price_{t-1}`: `delta_bid = +bid_notional_L1_t`
- If `best_bid_price_t == best_bid_price_{t-1}`: `delta_bid = bid_notional_L1_t - bid_notional_L1_{t-1}`
- If `best_bid_price_t < best_bid_price_{t-1}`: `delta_bid = -bid_notional_L1_{t-1}`

**Ask side (mirror):**
- If `best_ask_price_t < best_ask_price_{t-1}`: `delta_ask = +ask_notional_L1_t`
- If `best_ask_price_t == best_ask_price_{t-1}`: `delta_ask = ask_notional_L1_t - ask_notional_L1_{t-1}`
- If `best_ask_price_t > best_ask_price_{t-1}`: `delta_ask = -ask_notional_L1_{t-1}`

**OFI_t = delta_bid - delta_ask**

Computed per-snapshot, then summed over 5 snapshots (~120s at 24s cadence).
Normalized by rolling notional volume. Forward-filled to events.

## Why Piecewise, Not Naive

The naive formula (`bid_notional_t - bid_notional_{t-1}`) assumes the best bid
price stays constant between snapshots. At 24s cadence, price levels change in
60-80% of snapshot pairs during normal trading. When the best bid drops from
111235 to 111166, the naive formula can produce a positive value (spurious buying
signal) even though the bid fell (selling pressure). Council-2 called this
"anti-correlated with price direction during trends -- precisely when OFI has the
most predictive value." This is a correctness-level error, not a precision issue.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-02 | 5 snapshots, not 20 | 20 snapshots = ~480s at 24s cadence, longer than the primary 100-event prediction horizon (~300s). Cont 2014: OFI lookback should roughly match prediction horizon. | council-2-ob-cadence |
| 2026-04-02 | Piecewise Cont 2014, not naive | Price levels change 60-80% of the time at 24s; naive has wrong sign during trends | council-2-ob-cadence, data-eng-13 |
| 2026-04-02 | Notional, not raw qty | Cross-symbol comparability (BTC lots != FARTCOIN lots) | round3-synthesis H1 |
| 2026-04-02 | Normalize by rolling notional volume | Prevents magnitude from dominating cross-symbol | data-eng-13 |

## Gotchas

1. The naive delta-notional formula has the WRONG SIGN during trending markets
   at 24s cadence -- must use piecewise (CLAUDE.md gotcha 15).
2. Normalization denominator can be zero for very illiquid snapshots -- guard
   with `np.where(norm > 1e-10, cum_ofi / norm, 0.0)`.
3. Must use notional (qty x price) throughout, not raw qty (CLAUDE.md gotcha 14).
4. Forward-fill to events after computing per-snapshot. Same carry-forward
   pattern as delta_imbalance_L1.
5. First 5 snapshots of first calendar day have incomplete window -- use partial
   sum with available snapshots.

## Related Concepts

- [Orderbook Alignment](orderbook-alignment.md) -- snapshot cadence and alignment method
- [Kyle Lambda](kyle-lambda.md) -- complementary: cum_ofi measures net pressure, lambda measures price response to pressure
- [Effort vs Result](effort-vs-result.md) -- trade-side absorption, vs book-side pressure

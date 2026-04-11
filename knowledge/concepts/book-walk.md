---
title: Book Walk
topics: [microstructure, features, tape-reading]
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Book Walk

## What It Is

Book walk measures how far a single order walked through the order book, i.e.,
how many price levels it consumed to get fully filled. An order that fills at a
single level has zero book walk; an order that fills across 5 levels walked
aggressively through resting liquidity. This is a direct measure of order
aggressiveness and urgency.

In the Kyle (1985) framework, informed traders with strong and time-decaying
private information execute aggressively -- they cannot hide size when the
deadline is near. A large book walk signals an aggressive participant willing to
pay spread to get filled immediately. Council-2 recommended normalizing by spread
(not mid) so the feature measures "how many spreads did this order walk" -- this
is directly comparable across symbols and price regimes.

## Our Implementation (Feature 6)

```
book_walk = abs(last_fill_price - first_fill_price) / max(spread, 1e-8 * mid)
```

- **Unsigned:** abs() removes direction. Other features (log_return, trade_vs_mid)
  already carry direction. Unsigned measures pure aggressiveness.
- **Spread-normalized:** An order that walks 2 spreads on BTC and 2 spreads on
  DOGE has consumed comparable resting liquidity regardless of dollar prices.
- **Zero-spread guard:** `max(spread, 1e-8 * mid)` prevents division by zero
  when snapshot shows equal bid/ask (can happen in snapshot rounding).
- **Single-fill events:** `last_fill == first_fill`, so book_walk = 0. This is
  the majority of events -- most orders fill at a single level.

Originally named `price_impact` in the spec. Renamed to `book_walk` (round3-
synthesis L1) to avoid terminological collision with the theoretical concept of
price impact (permanent vs transitory market-wide impact), which is measured by
kyle_lambda.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-01 | Spread-normalized, not mid-normalized | Mid-normalization conflates price level with liquidity; spread measures book consumption | council-2 |
| 2026-04-01 | Unsigned (abs) | Direction already in log_return and trade_vs_mid; unsigned measures pure aggressiveness | council-2 |
| 2026-04-02 | Renamed from price_impact | Avoids confusion with theoretical price impact (Kyle/Hasbrouck) | round3-synthesis L1 |
| 2026-04-02 | Staleness bounded, no fix needed | During 24s stale snapshot, spread underestimate biases book_walk upward ~2x worst case; signal ordering preserved | council-2-ob-cadence |

## Gotchas

1. Uses spread from the aligned OB snapshot, which can be up to 24s stale. During
   trending markets, stale spread underestimates actual spread, causing upward
   bias in book_walk. The model can learn to condition on `delta_imbalance_L1 == 0`
   (stale book indicator).
2. Zero-spread guard is essential -- snapshot can show bid == ask due to rounding
   (CLAUDE.md gotcha 10).
3. Most events (single-fill) have book_walk = 0. The informative tail (nonzero
   values) is where the signal lives.

## Related Concepts

- [Kyle Lambda](kyle-lambda.md) -- measures market-wide price impact per unit flow; book_walk measures single-order aggressiveness
- [Effort vs Result](effort-vs-result.md) -- complementary: effort_vs_result detects absorption (high volume, no walk); book_walk detects aggression (walking through levels)
- [Orderbook Alignment](orderbook-alignment.md) -- source of spread and mid for normalization

---
title: Kyle Lambda
topics: [microstructure, price-impact, features, orderbook]
sources:
  - docs/council-reviews/2026-04-02-council-3-kyle-lambda.md
  - docs/council-reviews/round3-synthesis.md (C1, H1)
last_updated: 2026-04-03
---

# Kyle Lambda

## What It Is

Kyle's lambda measures price impact per unit of signed order flow: how much does
the midpoint move per dollar of net buying/selling pressure. Higher lambda =
market makers repricing aggressively (informed flow regime). Lower lambda =
noise trading regime.

Estimator: `lambda = Cov(delta_mid, cum_signed_notional) / Var(cum_signed_notional)`

## Our Implementation (Feature 16)

- **Per-SNAPSHOT, not per-event.** Event-level had ~2 effective observations per
  50-event window at 24s OB cadence (Council-3 analysis). Per-snapshot over 50
  snapshots (~20 min) gives 50 real observations.
- **Uses delta_mid, not delta_vwap.** delta_vwap conflates intra-order book walk
  (already captured by book_walk feature 6) with the market maker's information-
  driven price update. Biases lambda upward for large orders under noise trading.
  (Round 3, C1)
- **Uses notional** (qty x price), not raw qty, for cross-symbol comparability.
  (Round 3, H1)
- **Forward-filled** from snapshot timestamps to event timestamps.
- **Variance guard:** When Var(cum_signed_notional) < epsilon, output 0.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-02 | Use delta_mid not delta_vwap | Avoids conflating book walk with information | round3-synthesis.md C1 |
| 2026-04-02 | Per-snapshot not per-event | 24s OB cadence leaves ~5 non-zero delta_mid per 50-event window | council-3-kyle-lambda.md |
| 2026-04-02 | Use notional not raw qty | BTC lots != FARTCOIN lots | round3-synthesis.md H1 |
| 2026-04-02 | Retain despite sparse updates | Works as regime indicator per snapshot | council-3-kyle-lambda.md |

## Gotchas

1. Per-event kyle_lambda had ~2 effective observations per window — inflates
   variance ~10x, contributes noise not signal (Council-3 OLS analysis)
2. 59% of pre-April events have ambiguous direction — signed_notional is
   corrupted for those events (Council-4: lambda is the "one exception")
3. Must use notional for signed_qty, not raw qty
4. Variance guard required: zero-flow windows produce div-by-zero

## Related Concepts

- [Order Event Grouping](order-event-grouping.md) — source of signed flow data
- [Effort vs Result](effort-vs-result.md) — Wyckoff pattern that works without direction
- Orderbook Alignment — 24s cadence determines effective observations
- Depth Ratio — also uses notional, also per-snapshot

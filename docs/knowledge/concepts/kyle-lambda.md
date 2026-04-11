---
title: Kyle Lambda
topics: [microstructure, price-impact, features, orderbook]
sources:
  - docs/council-reviews/2026-04-10-round5-council-3-information-regimes.md
last_updated: 2026-04-10
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
| 2026-04-02 | Per-snapshot not per-event | 24s OB cadence leaves ~5 non-zero delta_mid per 50-event window | council-3-kyle-lambda.md |
| 2026-04-02 | Retain despite sparse updates | Works as regime indicator per snapshot | council-3-kyle-lambda.md |

## Four-Quadrant Interaction Map (with effort_vs_result)

| Lambda | Effort vs Result | Market State | Signal |
|---|---|---|---|
| **High** | **Low** | **Informed flow** — efficient directional move | Momentum continuation |
| **Low** | **High** | **Absorption** — volume absorbed, price stable | Trend reversal prep |
| **High** | **High** | **Stress/Liquidation** — forced selling into thin book | NOT informed — confound |
| **Low** | **Low** | **Drift** — thin volume, fragile momentum | High reversal risk |

Critical: High-lambda + high-effort is stress, not information. Informed flow
probes must condition on spread to distinguish (Round 5, Council-3).

## Signed Flow Convention

```
cum_signed_notional_j = sum over events in [snapshot_{j-1}, snapshot_j):
    notional_open_long + notional_close_short - notional_open_short - notional_close_long
```

For mixed-side events (59% of pre-April): aggregate signed components per
snapshot period. Do NOT assign a single sign per event then aggregate.

## Lambda Confounds Information + Illiquidity

On liquid symbols (BTC, ETH, SOL): information component dominates.
On memecoins (FARTCOIN, PUMP): liquidity component dominates. The encoder
can condition on log_spread + depth_ratio to disentangle (Round 5, Council-3).

## Gotchas

1. Per-event kyle_lambda had ~2 effective observations per window — inflates
   variance ~10x, contributes noise not signal (Council-3 OLS analysis)
2. 59% of pre-April events have ambiguous direction — signed_notional is
   corrupted for those events (Council-4: lambda is the "one exception")
3. Must use notional for signed_qty, not raw qty
4. Variance guard required: zero-flow windows produce div-by-zero
5. Winsorize at rolling 99th percentile per symbol to prevent Cov/Var explosion (Round 5, Council-2)
6. Lambda must remain an INPUT feature — 20-min timescale exceeds 10-min RF; model cannot recover it (Round 5, Council-3)
7. Pre-warm from prior day's last 50 snapshots; use min_periods=10 for first calendar day (Round 5, Council-2)

## Related Concepts

- [Order Event Grouping](order-event-grouping.md) — source of signed flow data
- [Effort vs Result](effort-vs-result.md) — Wyckoff pattern that works without direction
- [Orderbook Alignment](orderbook-alignment.md) — 24s cadence determines effective observations
- [Cum OFI](cum-ofi.md) — complementary: cum_ofi measures net pressure, lambda measures price response
- Depth Ratio — also uses notional, also per-snapshot

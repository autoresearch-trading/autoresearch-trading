# Council-2 Round 5: Orderbook Feature Implementation Review

**Reviewer:** Council-2 (Rama Cont — Order Flow Microstructure)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

The piecewise OFI formula is BLOCKING — must be empirically validated via sign-correlation with subsequent price movement. The depth_ratio epsilon at 1e-6 may produce spurious extremes for KBONK-tier symbols.

## 1. Piecewise Cont 2014 OFI — Exact Formula

**Bid side:**
- bid_price_t > bid_price_{t-1}: `delta_bid = +bid_notional_L1_t`
- bid_price_t == bid_price_{t-1}: `delta_bid = bid_notional_L1_t - bid_notional_L1_{t-1}`
- bid_price_t < bid_price_{t-1}: `delta_bid = -bid_notional_L1_{t-1}`

**Ask side (mirror):**
- ask_price_t < ask_price_{t-1}: `delta_ask = +ask_notional_L1_t`
- ask_price_t == ask_price_{t-1}: `delta_ask = ask_notional_L1_t - ask_notional_L1_{t-1}`
- ask_price_t > ask_price_{t-1}: `delta_ask = -ask_notional_L1_{t-1}`

**OFI_t = delta_bid - delta_ask**

**At 24s cadence, price changes in 60-80% of BTC snapshot pairs.** The naive formula gets the sign wrong during exactly the regimes when OFI is most predictive.

**Boundary: levels disappear (one-sided book).** If prior_notional=0, treat as new level appearing: `delta = +current_notional` regardless of price direction.

## 2. Kyle Lambda Edge Cases

- **Var ≈ 0 (quiet period):** Guard `where(Var > 1e-20, cov/var, 0.0)` is correct. Threshold safe for all symbol scales.
- **Stale market (Δmid=0 for all 50 snapshots):** Cov=0 naturally, returns 0. Correct.
- **Start of day (<50 snapshots):** Pre-warm from prior day. Use min_periods=10 for first calendar day.
- **Winsorization:** Clip at rolling 99th percentile per symbol, causal. Prevents Cov/Var ratio explosion.

## 3. Imbalance L5 — Use 1/k Harmonic Weights

**Weights: 1, 1/2, 1/3, 1/4, 1/5** (not exponential 1, 0.5, 0.25, 0.125, 0.0625).

Exponential places too much emphasis on L1 — L5 contributes only 6.25%, making imbalance_L5 near-equivalent to imbalance_L1. Harmonic 1/k preserves meaningful L2-L5 contributions.

```
imbalance_L5 = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
```

## 4. Staleness

No explicit staleness feature needed. `time_delta` (feature 4) implicitly captures staleness — large time_delta co-occurring with unchanged OB features signals stale book.

## 5. delta_imbalance_L1 — Carry-Forward, NOT Spike-and-Zero

The value at each event is the delta from the most recent snapshot, REPEATED. Not zero between snapshots with a spike at snapshot arrival. Verify in implementation.

## 6. Cross-Symbol Scale Leakage

| Feature | Scale Issue | Risk |
|---|---|---|
| log_spread, imbalance_L1/L5, trade_vs_mid, delta_imbalance_L1 | Dimensionless by construction | None |
| depth_ratio | Absolute epsilon 1e-6 may fire on thin books | LOW — validate in Step 0 |
| kyle_lambda | 5000x raw scale difference BTC vs KBONK | LOW — BatchNorm + equal-symbol sampling handles |
| cum_ofi_5 | Normalized by rolling volume | LOW — guard for zero denominator |

**Action:** Step 0 check — fraction of KBONK snapshots where min(bid_notional, ask_notional) < 1e-6. If >1%, use dynamic epsilon floor.

## Implementation Checklist (Before Caching)

1. OFI is piecewise (validate sign-correlation with subsequent mid-price movement)
2. kyle_lambda is per-SNAPSHOT (50 snapshot observations, not 50 events)
3. cum_signed_notional aggregates all events per snapshot period
4. All OB features use notional (qty × price)
5. depth_ratio epsilon validated for thin-book symbols
6. imbalance_L5 uses 1/k weights
7. delta_imbalance_L1 is carry-forward (repeated), not spike
8. First snapshot of day: OFI=0, lambda warm-up from prior day

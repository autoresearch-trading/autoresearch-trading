# Council-3 Round 5: Information Regime Detection Review

**Reviewer:** Council-3 (Albert Kyle — Price Impact Theory)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

Kyle_lambda is correctly specified (per-snapshot, 50 snapshots). The signed-flow convention is confirmed as (open_long + close_short) - (open_short + close_long) per snapshot period. The most actionable finding is the four-quadrant lambda × effort_vs_result interaction map that predicts the stress/liquidation regime will confound the informed flow probe.

## 1. Signed Flow Convention

```
cum_signed_notional_j = sum over events in [snapshot_{j-1}, snapshot_j):
    notional_open_long + notional_close_short - notional_open_short - notional_close_long
```

For mixed-side events (59% of pre-April data): aggregate the signed components per snapshot period. Do NOT assign a single sign per event then aggregate.

## 2. Lambda Interpretation in DEX

Lambda measures aggregate price sensitivity of the book to net order flow. Two confounded causes:
- **Information asymmetry** (genuine Kyle lambda): liquidity providers reprice after directional sweep
- **Thin liquidity** (illiquidity): even uninformed orders sweep levels mechanically

On liquid symbols (BTC, ETH, SOL): information component dominates.
On memecoins (FARTCOIN, PUMP): liquidity component dominates.

**Partial remedy:** Encoder sees log_spread + depth_ratio alongside lambda → can learn to condition interpretation.

**Informed flow label fix:** Add `log_spread < rolling_50th_pct` to prevent thin-book illiquidity from firing the label.

## 3. 20-Minute Window

50 snapshots at ~24s = ~20 minutes. At 50 observations, OLS has df≈48 — adequate for ordinal regime classification. No change recommended.

## 4. Lambda vs. Amihud

Lambda superior specifically for absorption detection:
- Absorption: high unsigned volume, near-zero net flow → Amihud high, lambda near-zero
- Directional sweep: high volume in one direction → Amihud high, lambda high
- Lambda distinguishes these; Amihud cannot.

## 5. Four-Quadrant Interaction Map

| Lambda | Effort vs Result | Market State | Signal |
|---|---|---|---|
| **High** | **Low** | **Informed flow** — efficient directional move, thin opposing book | Momentum continuation |
| **Low** | **High** | **Absorption** — large volume absorbed, price stable | Trend reversal preparation |
| **High** | **High** | **Stress/Liquidation** — forced selling into thin book | NOT informed flow — confound risk |
| **Low** | **Low** | **Drift** — thin volume, fragile momentum | High reversal risk |

**Critical:** High-lambda + high-effort is stress, NOT information. The informed flow probe must distinguish this via spread conditioning.

## 6. Lambda as Input vs. Reconstruction Target

**Lambda must remain an input:** 20-minute timescale exceeds the encoder's 10-minute receptive field. The model cannot recover lambda from within-window features.

**Lambda as validation probe:** After pretraining, train linear probe on frozen embeddings to predict high-lambda regimes. If it succeeds, the encoder learned information regime structure from the other 16 features.

Lambda is correctly excluded from MEM reconstruction (forward-filled → trivially copyable). The ~50% of masked blocks at snapshot boundaries where lambda transitions are non-trivial to reconstruct, but the blanket exclusion is the safe conservative choice.

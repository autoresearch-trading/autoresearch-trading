# Phase 2 Pre-Registration — Microstructure Realism (Council-2 / Cont)

**Date:** 2026-04-27
**Binding for:** April 14+ untouched-data Sortino evaluation of the frozen SSL encoder.

## 1. Pacifica fee structure (binding assumptions)

Pacifica perp public docs disclose taker 5bps / maker 2bps; on-chain settlement adds ~0.5–1bp gas-equivalent at typical congestion; MEV/sandwich risk on a public-mempool DEX adds 0.5–1bp adversely on takers. **Binding round-trip cost assumption: 8bps taker-taker, 4bps maker-maker, 6bps mixed.** The Phase 2 backtest MUST default to **6bps round-trip** unless a maker-fill model is explicitly justified per §5. Funding cost: H500 ≈ 30min on BTC but several hours on illiquid alts; if any horizon spans a funding tick (8h interval), deduct |funding_rate| × position_notional. Default funding cost: 1bp/8h average on liquid symbols, accrued per held interval.

## 2. Per-symbol minimum-fillable size and slippage (the whale-fill loophole)

The encoder's prediction is computed on a window whose **last event is the trade that moved L1 mid**. A real trader cannot enter at that mid. **Binding entry model:** trader enters at `t+1` order event, paying full top-of-book slippage as a function of intended size vs `imbalance_L1`/`imbalance_L5`/`depth_ratio`/`log_spread` from the same input feature set the encoder consumed. Per-symbol minimum fill size = **median of 200-event window total notional × 0.5** (guarantees the trade is at least half the size of what the encoder saw); per-symbol max fill = **L5 ask-side notional × 0.25**, slippage = `0.5 × spread + 0.5 × (size / L5_notional) × spread` on takers. On 2Z/KBONK/PUMP/KPEPE this typically forces 5–15bps one-way slippage — adding 10–30bps round-trip on top of fees. **Illiquid alts are effectively untradeable under this model**; that is the correct conclusion, not a parameter to tune away.

## 3. Best horizon to trade — fee-vs-edge arithmetic

H500: +1pp × ~30bps |return| = 0.3bps gross edge vs 6bps fees → **fee-blocked by ~20×**. April-1-13 inflation (PCA 0.596, RP 0.634) confirms much of H500 is directional drift any predictor catches; encoder margin over RP is +1.4pp, not the headline +14.8pp.
H100: +6.2pp over shuffled, but only **+3.7pp over RP** at ~10bps |return| → 0.37bps gross vs 6bps fees + 5× turnover → **fee-blocked**.
H50/H10: encoder margin over RP is ≤1pp; horizons are noise-dominated per Cont decay profile. **Fee-blocked.**

**Council-2 finding: NO horizon clears fees at the current edge size.** The minimum encoder-vs-RP margin to make H100 tradeable would be **≥7pp on liquid symbols** (gross edge ~0.7bps × 100 trades = 70bps absorbing 60bps fees + 10bps slippage). The current +3.7pp is half what is needed.

## 4. Per-symbol momentum-baseline comparator — MANDATORY

The encoder beating majority-class is **insufficient** for a tradeable claim. Phase 2 binding criterion: encoder Sortino must exceed the 4-feature momentum-baseline (council-2 §4 of the prepublication review) Sortino on **≥12/24 non-AVAX symbols** at the chosen horizon. Beating majority-class proves only that the encoder has direction; beating momentum proves the encoder adds value over what any practitioner would build first. Without this gate, Phase 2 PASS reduces to "the encoder learned per-symbol drift," which is not a representation-learning claim.

## 5. Execution model for Sortino

**Binding:** taker-taker fills at L1 ± half-spread + size-dependent slippage (§2), 6bps round-trip fees, funding accrued per 8h tick held. Maker-rebate model is **rejected for Phase 2**: claiming maker fills at H100 with adverse-selection-aware sizing requires a separate queue-position model that this program has not built; allowing it would let the backtest rescue itself with unmodeled assumptions. If maker fills are desired in a future phase, that requires its own pre-registration.

## Recommendation

**Phase 2 cannot pass at the current edge size. Recommend Path 2: re-pretrain before evaluating Phase 2.** The H500 +1pp result, even taken at face value, sits ~20× below the fee floor; the H100 +6.2pp-over-shuffled shrinks to +3.7pp over RP and remains fee-blocked at 5× turnover. Running Phase 2 on the current encoder will produce a negative Sortino across most symbols and consume DSR budget (effective N → 4) on a foregone conclusion. The honest move is to acknowledge the program produces a +1pp linearly-extractable signal that is real but sub-fee, and either (a) re-pretrain with execution-cost-aware objectives (e.g., predict signed-and-thresholded mid-move conditional on size > tradeable-min), or (b) close the program at its current publishable end-state without a tradeable claim.

## Summary

(1) Pacifica round-trip cost binding at **6bps mixed taker/maker**, plus per-symbol size-dependent slippage modeled from `imbalance_L1`/`imbalance_L5`/`depth_ratio`/`log_spread`; illiquid alts (2Z/KBONK/PUMP/KPEPE) are effectively untradeable under this model. (2) **No horizon clears fees** at the current edge size — H500 fee-blocked ~20×, H100 encoder-vs-RP margin (+3.7pp) is half the +7pp needed. (3) Per-symbol momentum-baseline comparator is **MANDATORY** — encoder Sortino > momentum Sortino on ≥12/24 symbols, otherwise Phase 2 PASS reduces to "encoder learned per-symbol drift" which is not a representation-learning claim. (4) Maker-rebate model rejected for Phase 2 (no queue-position model built). (5) **Recommendation: Path 2 (re-pretrain with execution-cost-aware objective) before any Phase 2 evaluation; running Phase 2 on current encoder is foregone-conclusion negative result that consumes DSR budget for no information gain.**

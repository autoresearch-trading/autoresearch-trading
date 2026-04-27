# Council-2 Review (Cont / Order Flow Microstructure) — Pre-Publication Strengthening

**Date:** 2026-04-26 (PM, post Gate 4 PASS, post multi-probe drop)
**Subject:** Step 4 program end-state writeup; pre-publication strengthening request
**Verdict:** Publish as-is, with one cheap optional probe noted in §2.

## 1. Is +1pp at H500 a meaningful signal in the OFI literature's frame of reference?

No, and the writeup should not pretend otherwise — but the writeup already doesn't. Cont-Kukanov 2014 reports OFI R² ≈ 0.65–0.75 against contemporaneous price changes at 10s–1min on equity LOBs with continuous quoting. That regime has (a) sub-second OB updates, (b) two-sided market making against an exogenous fundamental, (c) direction explicitly observable in signed trade flow. None of that translates to DEX perp at H500. H500 ≈ 500 trade events ≈ 30 min on BTC, ~hours on KBONK; at that horizon Cont's OFI signal has long since decayed (their own Fig. 4 shows half-life at lag ~5–10 events on equities, with predictability essentially gone by lag 50). Asking H500 balanced accuracy to compare against Cont's R² is a category error.

The right reference frame for +1pp at H500 is **the noise floor above majority-class on illiquid crypto perp**, not the OFI predictability ceiling. Gate 0 established that floor at ≈0.500±0.003. +1pp above that floor on 15+/25 symbols, temporally stable across two training-period halves, and resistant to fine-tuning collapse, is **a real but small informational structure** — consistent with the encoder having captured something past lag-50 OFI dynamics, well below what Cont-Kukanov measures at their native lag. It is not "tape reading" in the human-trader sense; it is "linearly-extractable directional residual after the symbol prior is accounted for." The writeup's headline already says exactly this.

## 2. Pre-publication probes that don't require multi-probe calibration

I recommend **one optional, single-probe addition** that is cheap, falsifiable, and uses only existing checkpoint + features, with thresholds calibrated on Oct-Jan training-period (not Feb+Mar):

**OFI-axis recovery probe.** Linear regression on frozen 256-d embeddings → `mean(cum_ofi_5 over last 100 events)` per window. Report Spearman ρ on Feb+Mar held-out, per symbol. Threshold: median ρ ≥ (Oct-Jan training-period median ρ − 0.05) on ≥14/24 symbols. Continuous target, no positive-rate calibration, sidesteps the entire C1/C3/C4 disaster.

**Why this one:** `cum_ofi_5` is the canonical Cont-flavored microstructure variable in the input set. If the encoder cannot linearly recover the recent-window mean of its own input OFI on held-out months, the +1pp signal is a label-side artifact, not order-flow representation. If it can, the signal carries microstructure content. Either outcome is informative.

**DSR cost:** This is one new probe against the same Feb+Mar slice. Per council-1's accounting that lifts effective N from 3 to 4. Council-1 should sign off before running.

Spread-regime and notional-imbalance recovery (the other examples you suggested) are redundant with OFI-axis recovery: they probe the same encoder-input-recovery hypothesis on correlated features. One probe is enough.

**My honest recommendation: skip even this one.** The writeup is already coherent without it, and adding any post-Gate-4 probe spends amendment budget on a question that doesn't change the headline.

## 3. Per-symbol cluster cohesion (delta +0.037, symbol-ID 0.934) — supports or undermines tape-reading?

**Expected from an OFI perspective; not a red flag.** Cont's own follow-up work (Cont, Kukanov, Stoikov 2014; Cont-de Larrard 2013) emphasizes that price-impact coefficients (λ, γ) are symbol-specific and scale with average daily volume, tick size, and queue depth. There is no "universal OFI signature" across instruments — the *functional form* (signed flow predicts impact) is universal, but the *coefficient regime* is not. KBONK and BTC live in different microstructure universes: KBONK has ~30 events/day vs BTC's ~50K, queue depths differ by 3 orders of magnitude, tick-to-spread ratios differ by 1–2 orders. A representation that collapsed them into shared geometry would be encoding *less* OFI structure, not more.

The cross-symbol delta of +0.037 is consistent with "symbol-specific OFI coefficients learned per symbol, with weak shared geometry from the universal sign-of-flow predicate." Symbol-ID 0.934 is consistent with the encoder having learned the per-symbol OB regime (depth, tick, cadence). Neither falsifies tape-reading; both are consistent with it. The writeup's framing — "consistent with both (a) per-symbol direction priors and (b) per-symbol tape geometries" — correctly notes the test is undecidable here.

## 4. Falsifiable test that the +1pp is not a per-symbol momentum/mean-reversion artifact

**Cheap test, runnable on existing checkpoint:** compare frozen-encoder LR vs a **per-symbol-conditional momentum baseline** at H500. Specifically: a logistic regression with 4 hand-built features per window — (sign of last 50-event return, sign of last 200-event return, abs(return) recent/long ratio, sign(cum_ofi_5 last 100)). Train per symbol on Oct-Jan; evaluate on Feb+Mar. If the encoder LR beats this momentum-baseline by ≥0.5pp on ≥12/24 symbols at H500, the +1pp is not exhausted by per-symbol momentum/mean-reversion. If the encoder ties or loses, the +1pp is "what a per-symbol momentum predictor already captures, in slightly compressed form."

This is **one** additional probe against Feb+Mar; same DSR cost as §2's OFI-axis recovery. It is sharper for the falsifiability question you asked: it directly tests whether the encoder's signal is incremental over the obvious symbol-conditional baseline that any practitioner would build first. Threshold derived from training-period: encoder LR vs momentum-LR on Oct-Jan, set Feb+Mar pass at training-period delta − 0.3pp on ≥12/24 symbols.

If you run only one strengthening probe, run this one rather than OFI-axis recovery — it answers a question that materially changes the headline, where OFI-axis recovery only adds color.

## Recommendation

**Publish as-is.** The writeup's headline ("+1pp linearly-extractable direction signal at H500, stable across training-period halves, not amplifiable by supervised fine-tuning") is well-calibrated to what the data shows. The disclosures around DSR effective N=3, Gate 2 FAIL, Gate 4 PASS, and multi-probe drop are appropriate and honest. From an OFI/microstructure perspective there is nothing in the existing writeup that overstates the result.

If the user wants strengthening, the **per-symbol momentum baseline at H500** (§4) is the single highest-value cheap probe — it sharpens the falsifiability claim. The OFI-axis recovery probe (§2) is a weaker second choice. Both incur DSR effective N=4. Council-1 should sign off on either before execution. Neither is required for the publishable end-state.

**Council-2 final position:** publish as written. Both proposed probes are *optional strengthenings*, not blockers.

## Summary

(1) Publish the program end-state writeup as-is — the +1pp at H500 is correctly framed against Gate 0's noise floor (not against Cont-Kukanov's lag-1-10 OFI ceiling, which is a different regime); the writeup is well-calibrated. (2) Per-symbol cluster cohesion (cross-symbol delta +0.037, symbol-ID 0.934) is **expected** under symbol-specific OFI coefficients (Cont-de Larrard 2013) and is NOT a red flag for tape-reading; the writeup's framing of the per-symbol vs universal question as "undecidable here" is correct. (3) If any strengthening is desired, the single highest-value optional probe is a **per-symbol momentum-baseline comparator at H500** with training-period-calibrated threshold (encoder LR vs 4-feature momentum LR; training delta − 0.3pp on ≥12/24 symbols); OFI-axis recovery is a weaker second choice. Both incur DSR effective N=4, council-1 sign-off required, neither is necessary for publication.

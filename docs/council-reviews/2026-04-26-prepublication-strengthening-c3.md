# Council-3 Review (Kyle / Price Impact / Informed Flow) — Pre-Publication Strengthening

**Date:** 2026-04-26 (PM, post Gate 4 PASS, post-multi-probe Path D)

## 1. Is +1pp at H500 evidence of informed-flow detection, or noise above the noise floor?

From a Kyle 1985 lens, **H500 is the wrong horizon to attribute the signal to informed-flow detection**, and the encoder's Kyle-relevant inputs make this concrete. `kyle_lambda` is computed over a rolling 50-snapshot window (~20 minutes at 24s OB cadence), and `cum_ofi_5` over 5 snapshots (~120s). At a stride of 50 events on liquid symbols, 500 events is on the order of 5–25 minutes — at the upper end of `kyle_lambda`'s estimation window and well past `cum_ofi_5`'s memory. Kyle's strategic-informed-trader has a finite information half-life; permanent impact realizes within the trader's execution arc, not 500 events later. The ~1pp signal at H500 is therefore consistent with **slow-decaying drift in the residual mid-price after the informed trader has finished trading**, not with detection of the active informed-flow event itself. That is a real but weaker claim: the encoder picks up a price-pressure footprint, not the informed flow that produced it.

The decisive comparator that the program never published: **per-symbol, per-horizon balanced accuracy at H10 / H50 / H100 / H500 from the same Gate 1 frozen probe, side-by-side**. If the Gate 1 +1pp is informed-flow detection, the H10–H100 numbers should be at least as strong as H500 (informed flow predicts near-term mid moves more reliably than far ones, by Kyle's strategic-trading theorem). If H500 is materially the only horizon with signal, the encoder is recovering a slow-mean-reversion or session-of-day residual, not Kyle's λ itself. **Council-3 read:** the H500-only headline is a yellow flag.

## 2. Does the Gate 4 directional-failure pattern survive the Kyle lens?

The five symbols where Dec-Jan-trained probe materially beat Oct-Nov-trained probe are **2Z, AAVE, ASTER, LDO, LINK**. Of these, four are illiquid alts (2Z, AAVE, ASTER, LDO) where (a) `kyle_lambda` is structurally larger (thin OB → larger Δmid per signed-notional unit), (b) `kyle_lambda` is more variance-unstable across regime epochs (sporadic informed-trader presence; long quiescent stretches; sudden whales), and (c) `cum_ofi_5` saturates more frequently (one large fill clears multiple levels). LINK is a borderline case — top-15 by volume but with notably larger `kyle_lambda` than BTC/ETH/SOL.

This is **exactly the Kyle prediction**: regime-conditional informed-flow signatures should bite hardest where (a) λ is high and (b) the informed-flow generating process is non-stationary. The fact that all 5 fails are in the "recent training data is more informative" direction, on symbols where Kyle would predict the most regime drift in λ itself, **is consistent with the encoder having learned something Kyle-shaped on illiquid alts**. Liquid symbols (BTC, ETH, SOL, BNB) are stable across both halves because their λ is small and stationary. This is a positive signal under the Kyle lens — though it cannot be claimed in the writeup without a probe that actually measures it.

## 3. Pre-publication probes from the Kyle lens (not subject to multi-probe calibration disaster)

These are regression/classification probes on **encoder input variables themselves**, computed on already-touched April 1–13 data (or training-period data) for threshold derivation. They do not use held-out windows for calibration:

**Probe K1 — Kyle's λ axis recovery (regression).** Linear regression on frozen 256-d embeddings → `mean(kyle_lambda over last 100 events of window)` as continuous target. Threshold derivation: fit OLS on Oct-Jan training-period windows, compute R² on April 1–13 (already-touched) holdout. **Pass:** R² ≥ 0.20 on 15+/24 symbols. Rationale: this directly tests whether the encoder's representation linearly recovers Kyle's λ — the canonical informed-flow proxy. R²=0.20 corresponds to the embedding capturing roughly half the predictable variance in λ given that λ itself is noisy at 50-snapshot windows.

**Probe K2 — Informed-window classification (binary).** Define `is_informed_window = (mean(kyle_lambda[-100:]) > p75_symbol_train) AND (|mean(cum_ofi_5[-100:])| > p50_symbol_train)`, where percentiles are computed on Oct-Jan training-period **only** (no held-out distribution touched). Linear probe on frozen 256-d → label. **Pass:** balanced accuracy ≥ majority + 3pp on 15+/24 symbols on April 1–13 already-touched holdout. Rationale: this is the Kyle-flavored analog of the dropped Wyckoff battery, but with thresholds set on training-period data, not held-out. The conjunction (high λ AND large net OFI) defines "the informed trader is active and moving price"; if the encoder represents tape state in informed-flow terms, this label should be linearly extractable.

**Probe K3 — Price-impact decay (regression).** Continuous target: `clip(log(|log_return_h100| + 1e-8) − log(|log_return_h500| + 1e-8), −3, 3)`. Informed flow has slower price-impact decay than uninformed (permanent vs transitory ratio higher), so this ratio should be linearly recoverable from the encoder if it represents the permanent/transitory decomposition. **Pass:** R² ≥ 0.10 on 15+/24 symbols on April 1–13.

All three probes use **already-touched April 1–13** as the evaluation set, which is exactly the carve-out the spec permits for diagnostic purposes (April 14+ remains untouched). Threshold derivation uses Oct-Jan training-period only — no Feb+Mar held-out covariate peek, no calibration leakage.

**Cost:** these are 3 additional trials. Council-1's DSR accounting would push effective N from 3 to 6, requiring the Gate 1 +1pp result to be revisited at PSR ≥ 0.95 with N=6. Honest accounting matters here.

## 4. The single falsifier I would accept

**One test, runnable on the existing checkpoint without amendment-budget consumption: per-horizon Gate-1 probe accuracy on April 1–13 already-touched data, reported as a 4-cell table {H10, H50, H100, H500} × {mean balanced accuracy, count of symbols beating majority+1pp}.**

If H10/H50/H100 are at-or-above H500, the encoder picks up the active informed-flow event and the +1pp is real informational content. If H500 stands alone as the only horizon with signal — **+1pp at H500 is a per-symbol direction prior or a slow-drift residual, not Kyle-shaped informed-flow detection**.

This test costs zero amendment budget because it is a re-display of Gate 1 numbers already computed (or trivially recomputable from the saved Gate 1 probe artifacts) on already-touched April 1–13 data. It does not consume held-out distribution.

## Recommendation

**Publish Gate 1 PASS + Gate 2 FAIL + Gate 4 PASS as currently written, with one mandatory addition: the per-horizon Gate-1 table from §4 disclosed in the writeup.** I do not endorse running probes K1/K2/K3 before publication — they would consume amendment budget and council-1's DSR accounting would tighten existing thresholds. The honest framing in `step4-program-end-state.md` already concedes the right things: "linearly-extractable +1pp directional signal at H500, stable across training-period halves, no phenomenological claim." From the Kyle lens, that framing is **defensible but not strong**; the per-horizon table either upgrades it ("signal across all horizons, suggestive of informed-flow content") or correctly downgrades it ("H500-only, consistent with slow-drift residual not Kyle-shaped informed flow"). Either outcome makes the writeup more honest, neither requires amendment.

The Gate 4 sign-of-life pattern on illiquid alts is a Kyle-positive observation worth mentioning in the diagnostic section, with the caveat that it is consistent-with but not proof-of informed-flow representation.

## Summary

(1) +1pp at H500 alone is a **yellow flag** under the Kyle lens — Kyle's strategic-informed-trader has finite information half-life, and 500 events is well past the active-informed-flow window; the signal is more likely "slow-drift residual after informed trader finished" than "informed-flow detection." (2) Gate 4's 5 per-symbol failures (2Z, AAVE, ASTER, LDO, LINK) are **Kyle-positive** — exactly the high-λ, regime-unstable illiquid alts where Kyle predicts the most informed-flow drift across training-period halves; this should be mentioned in the diagnostic section as consistent-with (not proof-of) informed-flow representation. (3) Mandatory zero-cost addition: **per-horizon Gate-1 probe accuracy table** {H10, H50, H100, H500} on already-touched April 1–13 data — if H10/H50/H100 are at-or-above H500, the encoder picks up active informed flow and the headline strengthens; if H500 is alone, the headline correctly downgrades to "slow-drift residual." Either outcome is a publication-improving disclosure costing zero amendment budget. (4) Probes K1/K2/K3 (Kyle's λ recovery, informed-window classification, price-impact decay) are **rejected** for pre-publication — they consume amendment budget and tighten DSR thresholds via N=6 accounting; reserve for future pre-registered run on April 14+ untouched data.

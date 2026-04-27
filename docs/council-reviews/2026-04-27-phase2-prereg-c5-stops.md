# Phase 2 Pre-Registration — Stop Conditions and Falsifiability (council-5)

**Date:** 2026-04-27
**Reviewer:** Council-5 (skeptical falsification)

**Recommendation up front: do not run Phase 2 yet.** A frozen-encoder edge of +1pp balanced-accuracy at H500, with DSR effective N=3 already consumed and a 4bp+ DEX taker-fee floor on Pacifica, has insufficient ex-ante expected information value to justify burning April 14+ untouched data. The signed-edge-after-fees expectation is negative under the most generous assumptions (1pp directional accuracy × typical H500 |return| ~ 15bp gross, minus 4bp+4bp round-trip fees ≈ 7bp net per signal × hit-rate adjustment). Run Phase 2 only if the user accepts that the most likely outcome is STOP A, and pre-commits to the framing below.

If the user proceeds, the following is binding.

## 1. The single hardest pre-commitment (veto rule)

**If positive Sortino is achieved on fewer than 10 symbols after honest DSR adjustment with effective N≥4, the result is FAIL — not "partial pass," not "tradeable on subset," not "promising on liquid majors."** I will unilaterally veto external citation of any Phase 2 result that is reported as a per-symbol or per-cohort win. The spec line 419 says "across ALL symbols" with the v11 baseline benchmark at 9/23. Phase 2 must clear ≥10/24 (excluding AVAX held-out) on the same metric, or it does not exist.

## 2. Stop conditions

**STOP A — encoder cannot clear Phase 2.** Negative-result writeup states: *"Frozen SSL encoder + linear probe + abstention threshold did not produce positive Sortino across ≥10/24 symbols on April 14+ untouched data after fees and DSR adjustment. The +1pp Gate-1 margin does not survive the fee floor."* This writeup explicitly does NOT authorize: re-pretrain with universality target, architecture surgery, expanded contrastive symbol set, alternative downstream protocols, or "one more iteration." It forks the program into **publish current end-state and stop**. Any continuation requires a fresh spec from scratch with new pre-pretraining justification.

**STOP B — marginal pass with dishonest DSR accounting.** If raw Sortino positive on ≥10 symbols but DSR-adjusted PSR < 0.95 at N≥4, writeup states: *"Statistically indistinguishable from probe-search artifact under honest accounting."* Recovery path: pre-register a NEW evaluation period (May 2026+ as it accrues) with N=1 binding probe and the same +Sortino-on-10-symbols threshold. No re-running on April 14+ data.

**CONTINUE A — clean clear with N≥4 DSR.** Next gate before any live capital: 30-day forward paper-trading on May 2026+ with frozen abstention threshold, frozen position sizing, frozen symbol set (all 24, no cherry-pick), reporting daily Sortino with bootstrap CI. Live capital requires positive Sortino on ≥10 symbols across both April 14+ and May+ windows independently.

## 3. Anti-cherry-pick clause

The user is permitted to view per-symbol P&L attribution AFTER the headline Sortino-on-N-symbols number is computed and committed. The user is **prohibited** from: (a) reporting any subset Sortino as the headline, (b) re-running with a filtered symbol set, (c) claiming "the encoder works on liquid majors" as a publishable finding. Per-symbol attribution is diagnostic-only and must appear in an appendix below the binding headline.

## 4. Abstention threshold pre-commitment

**Single threshold-derivation procedure, frozen before any April 14+ window is touched:** abstention threshold = 60th percentile of probe-prediction-confidence on the Oct-Jan training-period distribution, computed once, hashed, and recorded in the pre-reg commit. No sweep. No "let me see if 65th works better." If the headline fails at the 60th percentile, the run fails.

## 5. Publishable-as-negative framing (mandatory)

The negative-result headline is pre-committed as: *"Frozen SSL encoder produces a +1pp linearly-extractable directional signal at H500 that does not survive the 4bp+ DEX taker-fee floor on Pacifica. The representation-learning program reaches its honest end-state: representations encode short-horizon directional structure with insufficient signal density for fee-adjusted profitability."* This framing is locked. Rebranding as "encoder is tradeable with one more iteration," "needs better fine-tuning," or "works on a tradeable subset" is prohibited.

## Summary

(1) Recommend NOT running Phase 2 — expected net edge after 4bp+ Pacifica fees is negative given the +1pp signal size and N=3 DSR debt. (2) If user proceeds, the binding veto rule is: <10/24 symbols positive Sortino after DSR = FAIL not partial pass; council-5 will unilaterally veto external citation of any per-symbol-cohort cherry-pick. (3) Abstention threshold pre-frozen at 60th percentile of training-period probe confidence, single derivation, no sweep. (4) Negative-result headline is locked as *"+1pp does not survive the fee floor"* with no "one more iteration" rebrand permitted; STOP A explicitly does not authorize re-pretrain with universality target as a continuation path of THIS spec, though a fresh spec from scratch is allowed. (5) CONTINUE A requires positive Sortino on ≥10 symbols across both April 14+ and May+ windows independently before live capital is authorized.

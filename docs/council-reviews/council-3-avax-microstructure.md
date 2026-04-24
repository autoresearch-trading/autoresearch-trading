# Council-3 Review: Is AVAX a Structurally Appropriate Held-Out Symbol?

**Date:** 2026-04-24
**Reviewer:** council-3 (market microstructure theory)
**Inputs:**
- `docs/experiments/step5-gate3-avax-probe.md`
- `docs/experiments/step3-run-2-gate1-pass.md`
- CLAUDE.md (especially gotcha #25 on AVAX exclusion)

## Verdict

**AVAX is structurally ambiguous — probably the wrong single-symbol held-out choice, but the Gate 3 failure is NOT cleanly explainable by microstructure anomaly alone. Recommend Option B (broaden to a held-out SET).**

## Microstructure Analysis

### 1. Liquidity tier — AVAX sits in a middle zone the encoder saw little of in this exact form

The 25-symbol universe splits empirically into three tiers by window count (gotcha #21):
- **Tier 1 (majors / anchors):** BTC (67K), ETH (63K), HYPE (44K), SOL (41K) — these dominate SimCLR positive pairs and equal-sampling still gives them massive absolute event diversity.
- **Tier 2 (mid-liquid L1/infra/DeFi):** BNB, LINK, LTC, AAVE, AVAX, XRP, DOGE — roughly 20-35K windows each. AVAX sits here.
- **Tier 3 (illiquid alts and memecoins):** 2Z, CRV, LDO, UNI, FARTCOIN, KBONK, KPEPE, PUMP, PENGU — 18-22K windows, much slower inter-event rhythm.

Looking at the probe data: AVAX Feb stride=50 gave **2,360 windows for one month**, implying an annualized pace that puts AVAX firmly in the Tier 2 mid-liquid zone — comparable to LINK and LTC (its two closest structural analogues in the anchor set). This is the good news for the transfer hypothesis: LTC specifically was chosen as the AVAX substitute in cross-symbol SimCLR precisely because its liquidity profile is a close match.

### 2. Kyle's λ regime — AVAX is likely indistinguishable from LINK/LTC, distinct from BTC/ETH

Kyle's λ scales inversely with depth at L1-L5 and directly with informed-trade arrival rate. For a mid-liquid L1 token on a DEX perp venue, the λ distribution should look like:
- **BTC/ETH:** very low λ, tight Cov(Δmid, signed flow), heavy maker-side absorption → the "deep" regime.
- **LINK/LTC/AVAX:** moderate λ, visible price impact per $100K notional, intermittent informed flow episodes → the "medium" regime.
- **Memecoins (FARTCOIN, KBONK, KPEPE):** highly variable λ, extreme climax_score tails, sudden liquidity vacuums → the "thin and gappy" regime.

AVAX should live in the same λ quintile as LINK and LTC. The encoder was trained with LTC explicitly in the liquid-anchor SimCLR set — so AVAX's λ regime was seen during pretraining through LTC. **This argues against "AVAX is structurally unrepresented."**

### 3. Composite Operator footprint (`is_open` dynamics) — the informed-flow signature is preserved across Tier 2

`is_open` autocorrelation half-life (≈20 trades, per CLAUDE.md findings) is a venue-level artifact of Pacifica's trade-event semantics, not a per-symbol phenomenon. The *level* of `is_open` can differ by symbol (memecoins skew long-open in rallies; BTC is more balanced), but the *persistence structure* — which is what the dilated CNN's receptive field actually captures — is a DEX-wide property. AVAX's Composite Operator footprint should be readable by any encoder that learned is_open autocorrelation at all.

### 4. Funding/oracle — no known AVAX-specific quirk on Pacifica

AVAX on Pacifica uses the standard ~1h funding interval and oracle mark; no Jupiter-style auction artifact, no ASTER-style launchpad dynamics, no PUMP-style listing event concentration. The funding feature is not in the 17-feature input anyway — this is a dead lead.

### 5. Event-class anomaly: AVAX is the ONLY pure L1-smart-contract token among the mid-liquid held-out candidates

Here's the genuinely concerning piece. Among Tier 2 symbols:
- **LINK:** oracle/infrastructure token — highly correlated with DeFi narrative cycles
- **LTC:** payments L1, old-guard, correlated with BTC
- **AAVE:** DeFi protocol token
- **AVAX:** L1 smart-contract platform, correlated with ETH/SOL beta but with distinct funding regimes during AVAX-specific ecosystem news (subnet launches, C-Chain events)
- **BNB:** exchange token, venue-specific flow
- **XRP/DOGE:** retail-flow dominated, legal-catalyst driven

**AVAX's closest structural twin — SOL — is in the anchor set as a pretraining symbol.** The encoder saw SOL's tape extensively. If the encoder learned *any* L1-platform tape structure, it should transfer from SOL to AVAX. The fact that it doesn't (0/6 cells) is actually informative: it suggests the encoder's representations are more symbol-conditional than regime-conditional.

## The Falsifier Interpretation

Reading the step5 report carefully: stride=200 Feb H100 showed encoder +7.9pp over PCA (n=120, inside ±0.13 CI), stride=50 showed encoder −1.7pp vs PCA (n=472, outside noise). This is not "AVAX is weird" — this is "higher-density evaluation collapses the apparent signal to noise." The encoder does not encode AVAX's tape usefully even though SOL and LTC (AVAX's structural near-twins) were both extensively in the training set.

**Microstructure theory's prediction:** if the encoder had learned universal tape features (absorption, climax, informed flow signatures that depend on relative rhythms and book shape, not absolute symbol identity), it should transfer to AVAX at Tier-2 quality — roughly what we see on LINK and LTC in Gate 1 (where both were in +1.9-2.3pp winners). The fact that it does not transfer is evidence the encoder learned a symbol-conditional manifold, with SOL-region and LTC-region being distinct from the "AVAX region" the encoder never visited.

## Caveats to the falsifier claim

1. **n=1 held-out symbol.** We cannot distinguish "encoder fails to transfer to AVAX specifically" from "encoder fails to transfer to novel mid-Tier-2 symbols generally." Two symbols with plausibly similar microstructure (LINK in-sample, AVAX out-of-sample) disagreeing on transferability is a single data point.
2. **Sampling/class-balance confound not ruled out.** AVAX's H500 label balance in Feb/Mar may genuinely differ from LINK/LTC in a way that handicaps LR fit independent of representation quality. H3 in the step5 report is not yet distinguished from H1.
3. **LTC substitution was a concession.** The original spec had AVAX as anchor; pulling it for held-out status and backfilling with LTC creates an asymmetry: AVAX is the only symbol in the universe that was *designed* to be structurally similar to a SimCLR anchor without *being* one. This is exactly the configuration most likely to expose symbol-conditional representations.

## Recommendation: Option B (broaden Gate 3 to a held-out SET)

**Do NOT accept the stride=50 AVAX failure as a clean falsifier of universal-tape representations.** Too much is loaded on one symbol whose closest structural analogue (SOL) is in-sample. Do:

1. **Amend the spec to redesignate Gate 3 as a held-out set**: pick 3 symbols spanning the Tier 2 and Tier 3 regimes — e.g., one mid-liquid L1 (AVAX), one DeFi token (AAVE or CRV), one memecoin (PENGU or KBONK). Pass criterion: encoder > flat on ≥ 2/3 symbols at H100 balanced accuracy.
2. **Retrain with these 3 symbols excluded** (the AAVE/CRV/PENGU exclusion is a non-trivial retraining cost — ~10% of pretraining data). If retraining is cost-prohibitive, run Gate 3 on the existing checkpoint against AAVE/CRV/PENGU as well as AVAX, knowing AAVE/CRV/PENGU were *in-sample* — this gives an "in-sample transfer to held-out labels" floor to compare the AVAX out-of-sample number against. If in-sample AAVE on this checkpoint also fails to beat PCA on Feb/Mar stride=50, then the AVAX failure is a probe/data issue, not a transfer issue.
3. **Only if AVAX fails AND in-sample Tier 2 symbols pass on the same held-out months** is the representation confirmed symbol-conditional. That's the clean experimental design.

## Final answer to the question posed

**No, AVAX is not structurally anomalous enough to pre-excuse the Gate 3 failure** — its closest microstructural twin (SOL) was a pretraining anchor and its liquidity peer (LTC) substituted in cross-symbol SimCLR specifically to cover AVAX's regime. The failure is therefore *compatible* with a clean falsifier for universal-tape representations. **However, n=1 held-out symbol is insufficient to make that falsifier stick** — the spec should broaden Gate 3 to a held-out set of 2-3 symbols spanning Tier 2 and Tier 3 before declaring the universality hypothesis falsified.

## Summary

AVAX is structurally close enough to in-sample SOL (L1 platform) and LTC (matched liquidity tier, substituted into cross-symbol SimCLR precisely to cover AVAX's regime) that good universal-tape representations *should* transfer — the Gate 3 failure is therefore a real but underpowered falsifier, not a spec-design mistake. Recommend **Option B**: broaden Gate 3 to a 2-3 symbol held-out set (e.g., AVAX + AAVE + one memecoin) spanning Tier 2 and Tier 3 microstructure regimes, and run an in-sample-control probe (LINK/LTC on same Feb/Mar stride=50 protocol) to distinguish "symbol-conditional representations" from "stride=50 small-n noise on any mid-liquid symbol."

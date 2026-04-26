# Post-Gate-2 Strategic Review — Falsification (Council-5)

**Date:** 2026-04-26 (PM)
**Reviewer:** Council-5 (skeptical falsification)
**Trigger:** Gate 2 failed (-1.7pp at H500 on Feb+Mar; all 3 binding criteria failed both months)

## 1. What was actually falsified

**Falsified (binding):** the claim that **fine-tuning the pretrained CNN with the spec-prescribed protocol** (freeze 5 epochs → unfreeze at lr=5e-5, 0.10/0.20/0.20/0.50 horizon weights, walk-forward 600-event embargo) produces a direction predictor that beats flat-LR by ≥0.5pp at H500 on 15+/24 symbols on Feb+Mar held-out.

**Specifically NOT falsified:**
- "The encoder learns useful representations" — Gate 1 still passes on the same Feb+Mar window with frozen LR.
- "SSL on tape data is fundamentally broken" — no evidence; only one config tested.
- "The encoder is fragile to fine-tuning" — possible, but only one fine-tune protocol was tried.

**The danger of over-falsifying.** -1.7pp at H500 on a fine-tuning protocol with ONE seed, ONE configuration, on a held-out period that is ONE realization of the data-generating process, falsifies exactly that protocol on that data. It does not falsify SSL on tape.

## 2. Three interpretations, ranked

### Interpretation A: "Encoder fine; fine-tuning blew it up." — WEAKEST (~10-20%)

**For:** Gate 1 passes (frozen probe), Gate 2 fails (fine-tuned), liquid symbols regressed specifically.

**Against:** lr=5e-5 with frozen-warmup is the standard "gentle fine-tune" recipe. Symbol-ID probe = 0.934 → embeddings are 4× more discriminative for symbol than for cross-symbol microstructure state. Mean-reversion of illiquid alts to 0.50 is what you'd see if **the encoder's signal on those symbols was always near noise** and fine-tuning regularized them toward the prior — not what you'd see if fine-tuning destroyed real signal.

### Interpretation B: "Encoder at natural ceiling." — STRONGEST (~50-60%)

**For:**
- Gate 1 mean +1pp absolute, not +5pp — small signal.
- Phase A (frozen probe within fine-tune scaffold) matched Gate 1 (+0.6pp over majority). **Smoking gun: the heads themselves cannot extract more signal than logistic regression already does.**
- Per-symbol H500 PCA+LR gets +0.8pp; encoder gets +1.9pp — encoder margin over PCA is 1.1pp. Fine-tuning would need to amplify this and didn't.
- SSL literature on financial microstructure plateaus near linear-probe ceiling — fine-tuning gains typically <1pp.
- 627K windows / 400K params is tight (1:1.6 data-to-params). The fact that linear probe only extracts +1pp is consistent with there being only +1pp of linearly-extractable directional signal.

**Most honest characterization:** *the encoder achieved per-symbol feature quality at a +1pp linear-probe ceiling above the noise floor; both the ceiling and the per-symbol nature are properties of the training config, not bugs to fix.*

### Interpretation C: "Encoder learned per-symbol artifacts." — STRONG SECONDARY (~25-35%)

**For:** Cluster cohesion delta = +0.037, symbol-ID probe = 0.934, Phase B's +1.2pp val-fold gain disappearing to -1.7pp on held-out is the canonical "model learned period-specific features" failure mode.

**Against:** Gate 1 PASSES on Feb+Mar — by Gate 1's binding criteria. If the encoder were purely period-fragile, Gate 1 would have failed.

**Why the proposed remedy is dangerous:** re-pretrain with widened LIQUID_CONTRASTIVE_SYMBOLS, soft-positive weight 0.5→1.0, more epochs is **expensive, consumes amendment-budget quota** (per the 2026-04-24 amendment-budget clause), and is a direct manifestation of "unearned universality." There is no evidence widening would fix the symbol-ID issue rather than just make symbol-ID also read 0.934 on the wider set.

## 3. Skeptical questions answered

**Q1: Is B the null we should defend against, or accept?** B is the null. Accept it as the operating hypothesis until falsified by NEW evidence.

**The framework error:** the spec's editorial gloss "Pretraining added nothing" if Gate 2 fails is too strong. Gate 1 PASSED → encoder produces +1pp linearly-extractable direction signal. That's not nothing. It's just not amplifiable by direction-supervised fine-tuning.

**Q2: Coherent multi-probe path or post-hoc?** There IS a coherent path, BUT only if pre-committed before any new probes. The spec already lists multi-probe diagnostics; what is post-hoc is **converting them from "diagnostic" to "binding" after Gate 2 failed.** To make this honest: pre-commit IN WRITING before running any new probe.

**Q3: Should you run Gate 4 on frozen encoder before further work?** YES. Cheapest, most diagnostic, lowest-risk next step. ~2 hours CPU. Reuses existing checkpoint. **Decisive in both directions:**
- Gate 4 passes → B leads; encoder at ceiling but stable; multi-probe battery is next.
- Gate 4 fails → C leads; encoder learned period-specific features; the +1pp Gate 1 margin is partly memorization; **even more reason to stop**.

There is no version of Gate 4 outcome that supports a re-pretrain.

**Q4: What stop conditions for the program?** Adopt these. Pre-commit them. They are binding now.

## 4. Pre-Registered Multi-Probe Battery (proposed binding, 2026-04-26 PM)

**To be ratified by user before any new measurement runs.**

> **Multi-Probe Representation Quality Battery (binding):**
>
> Encoder produces a positive program result iff Gate 4 PASSES (<3pp drop on >14/24 symbols at H500 between Oct-Nov-trained and Dec-Jan-trained probes evaluated on Feb+Mar) **AND** at least 2 of the following 4 conditions hold on Feb+Mar held-out:
>
> 1. **Wyckoff absorption probe** balanced-acc > majority+2pp on 12+/24 symbols, where absorption labels are computed from spec-defined formula `mean(effort_vs_result[-100:]) > 1.5 AND std(log_return[-100:]) < 0.5*rolling_std AND mean(log_total_qty[-100:]) > 0.5` on held-out windows.
> 2. **CKA seed-stability ≥ 0.75** between the Run-2 epoch-6 checkpoint and a fresh seed-1 run trained with identical config on a 50% subsample of training data.
> 3. **Cluster purity for at least 2 Wyckoff states** at k=16 k-means: ≥40% purity in at least one cluster vs. ≤10% null expectation.
> 4. **Embedding trajectory test:** for ≥10 manually-identified climax events on held-out data, embedding distance jump at the event >2σ above the within-symbol-day distance distribution, on ≥7/10 events.
>
> If Gate 4 fails OR fewer than 2 of conditions 1-4 hold, the program result is negative. Stop. Write up.

## 5. Program Stop Conditions (proposed binding)

1. **Gate 4 fails** (>3pp drop on >10/24 symbols at H500): STOP. Write negative result. The frozen encoder is non-stationary; no amount of fine-tuning or re-pretraining at this scale fixes data-level non-stationarity.

2. **Gate 4 passes BUT multi-probe battery fails** (fewer than 2 of 4 conditions): STOP. Write negative result. The encoder is at ceiling AND extracts only direction-flavored signal at +1pp; this is the publishable negative finding.

3. **Gate 4 passes AND multi-probe battery passes (≥2 of 4)**: NOT a green light to ship; a green light to write up the diagnostics, then design ONE follow-up experiment with explicit pre-registered thresholds. No re-pretrain without council-1 + council-5 sign-off and amendment-budget consumption.

4. **Any tradeable claim** requires: positive Sortino across ≥10 symbols on April 14+ untouched data, AFTER fees, AFTER Deflated Sharpe Ratio adjustment for number of probes run.

**Most probable program outcome:** writeup titled "*Self-supervised pretraining on DEX perpetual tape data achieves +1pp linearly-extractable direction signal at the H500 horizon, stable across two held-out months but not amplifiable by supervised fine-tuning. Negative result with respect to a tradeable edge after costs.*" Most research programs do not produce tradeable systems. This is the modal outcome and it is fine.

## 6. What would convince me program is worth shipping?

In strict order of decisiveness:

1. **Gate 4 passes AND ≥3 of 4 multi-probe conditions hold.** Strongest evidence the encoder learned genuine microstructure.
2. **Wyckoff absorption probe ≥ majority+3pp on 12+/24 symbols.** Absorption is the spec's master signal.
3. **Embedding trajectory shows reproducible >2σ jumps at 7/10 manually-identified Wyckoff events.** Qualitative-but-falsifiable.
4. **CKA ≥ 0.85 between two seed-varied runs.**

**What would NOT convince me:**
- "Re-pretrain with wider contrastive symbols passes Gate 2 by 0.3pp." Too small, no pre-commit.
- "Fine-tuning with different hyperparameters passes Gate 2." Already pre-committed against.
- "We found a different held-out window where Gate 2 passes." Re-sampling loophole, closed by 2026-04-24 amendment.

## Closing skeptical note

You wrote: *"the temptation is to find a path forward that doesn't involve admitting the program may not produce a positive result."* You are correct that this is the temptation. The pre-registered stop conditions above are designed to remove your discretion at the moment when you'll most want to exercise it. **Sign them now. Once Gate 4 runs, signing them will be post-hoc.**

The +1pp Gate 1 margin is small enough that confirmation bias on the next probes will be very strong. When the Wyckoff absorption probe comes back at +1.8pp (just under the +2pp pre-committed threshold), the temptation will be enormous to argue "+1.8pp is statistically significant given the bootstrap CI, surely this counts." It does not count. **Pre-commit the bootstrap-CI rule too: point estimate must clear the threshold; CI lower bound is published but is NOT the test.**

## Summary

(1) Interpretation B leads (~50-60% weight, encoder at +1pp natural ceiling) with strong secondary contribution from C; A is weakest. (2) Cheapest next test is Gate 4 on the frozen encoder (~2h CPU, no amendment-budget cost, decisive in both directions). (3) Adopt this binding stop condition NOW: program writes negative result and stops if Gate 4 fails OR if Gate 4 passes but fewer than 2 of {Wyckoff absorption probe > majority+2pp on 12+/24, CKA seed-stability ≥0.75, cluster purity ≥40% on a Wyckoff state, embedding trajectory >2σ jumps on 7/10 held-out events} hold. (4) Shipping requires Gate 4 + ≥3 of 4 multi-probe + Phase 2 positive Sortino on April 14+ after fees and DSR adjustment — anything less is a publishable +1pp negative result, the modal honest outcome.

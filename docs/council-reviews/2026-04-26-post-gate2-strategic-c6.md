# Post-Gate-2 Strategic Interpretation — Council-6 (DL Architect)

**Date:** 2026-04-26 (PM)
**Reviewer:** Council-6 (deep learning researcher)

## Q1: What did the encoder LEARN during Phase B that doesn't transfer?

**Verdict: Hypothesis C is dominant, with a smaller B contribution. A is unlikely.**

The decisive evidence is the train/val curves combined with the held-out reversal:

- Train BCE 0.6907, val BCE 0.6907 (no gap) — rules out classical overfitting (memorizing training windows).
- But val fold = same symbols, same days, different windows → the model can fit **stride-level day-conditional structure** (the same trading session's micro-regime, the same hour-of-day OB cadence pattern, the same per-symbol vol-of-vol on that day) without memorizing individual windows.
- A 600-event embargo within a day does NOT eliminate this — it only embargoes label leakage at fold boundaries, not the daily-regime covariate.
- On Feb+Mar held-out, days are entirely new. The day-conditional structure that Phase B learned to lean on doesn't exist for those days. The encoder has nothing to fall back on except the per-symbol class prior, hence regression-to-0.50.

**A (per-symbol calibration corrections) is unlikely** because the worst regressions are concentrated on symbols where flat-LR was FAR ABOVE 0.500 (SUI 0.626, LTC 0.595, 2Z 0.633). If the encoder were learning a per-symbol calibration delta, we would expect symmetric pull toward the encoder's "per-symbol mean prediction" — not preferentially demolishing symbols where the linear-on-summaries model had real signal.

**The diagnostic signature of C is exactly what was reported:** monotone val descent + held-out reversal with sign reversal concentrated on liquid symbols (where day-conditional structure is richest because event density per day is highest, so the encoder has the most per-day patterns to memorize). KPEPE/AAVE/KBONK gains are an artifact: flat-LR was *below* 0.5 on those symbols, so any push toward 0.5 looks like a "gain" — it isn't; it's the same regression-to-0.50 pattern, scored kindly because the baseline was on the wrong side of the line.

## Q2: Is the linear-trunk-then-per-horizon-head architecture wrong?

**Partially — but it is not the load-bearing failure.**

The 256→64 bottleneck IS aggressive — we throw away 75% of the representation in one matmul before any per-horizon specialization. For a 256-dim embedding with measured eff_rank ~215, projecting to 64 dims is a hard information bound: we're capping the head at 64 directions in embedding space.

**But the regression-to-0.50 pattern is NOT explained by head bottleneck alone.** A bottleneck-limited head should produce **biased-but-confident** predictions, not predictions clustered at 0.5. A head trained to predict the per-symbol prior on shortcut features, then evaluated on a distribution where those shortcuts don't fire, produces **uncertain-near-0.5** predictions — which is what we observe.

**Architectural verdict:** the bottleneck is suboptimal but secondary. The primary issue is what the encoder+head jointly learned to lean on, not how much capacity the head has.

## Q3: Is the loss-weight schedule wrong?

**Yes, but not in the direction your H500-de-weighting concern points to.**

H10 has up to 9.9pp class imbalance inflation (gotcha #28). On illiquid symbols H10 BCE on the imbalanced label has a larger absolute value than H500 BCE because H10's label distribution is further from 0.5. **Even at weight 0.10, H10 contributes outsized gradient when the model can find shortcut features for the majority class.**

The fine-tuning loop then trains the encoder to **encode the per-horizon class prior at low loss cost on H10**, which tilts the shared trunk toward representations that preserve symbol-conditional class prior info. H100/H500 heads share that trunk and inherit the bias.

**Concrete recommendation if a future Phase B is run:** use **balanced BCE** (per-symbol per-horizon class-weighted) at every horizon, OR drop H10 from the multi-task loss entirely.

## Q4: Is the encoder representation INSUFFICIENT for direction prediction at H500?

**Probably yes for this architecture, but the encoder is salvageable for non-direction tasks.**

The arithmetic is brutal:
- Gate 1 frozen-encoder LR: +1.0pp over majority on average.
- Phase A frozen-encoder DirectionHead: +0.6pp.
- Phase B unfrozen: +1.2pp on val fold but −1.7pp on held-out.

The +1.2pp val-fold gain is consistent with **0.4-0.6pp of preserved signal + 0.6-0.8pp of in-period overfitting**. On held-out months, the overfitting flips sign (it has nowhere to retrieve the in-period signal from), which exactly produces the −1.7pp deficit (1.0pp Gate-1 ceiling minus ~2.7pp of mean-reverted noise = -1.7pp).

This isn't catastrophic forgetting (CKA drift is mild), it isn't classical overfitting (no train/val gap), and it isn't pure distribution shift. It's the encoder operating at its **true generalization ceiling**, with fine-tuning adding noise that doesn't transfer.

## Q5: Highest-information-value test (A/B/C/D)?

**(D) Walk-forward temporal validation during Phase B is the highest-IV test, but it is BANNED by the no-retry pre-commitment.** Acknowledged. Recommended sequence given that constraint:

- **Gate 4 (frozen encoder, Oct-Nov vs Dec-Jan probes evaluated on Feb+Mar)** — analogous to (D) but on the frozen encoder; tests whether the +1pp Gate 1 signal is itself stationary across training period halves.
- **(A) Frozen encoder + larger head with C-search regularization** — only relevant if Gate 4 passes. Tests the head bottleneck.
- **(B) Re-pretrain with stronger cross-symbol invariance** — the highest-cost / lowest-information test in this batch. Save for after the cheaper diagnostics.

## Q6: Failure mode in standard DL terms

This is **shortcut learning at fine-tune time** in the Geirhos / Beery / Lapuschkin sense:

- NOT catastrophic forgetting: 0.061 CKA drift is well within acceptable elastic adaptation.
- NOT classical overfitting: no train/val gap (because val is in-distribution).
- NOT pure distribution shift: Feb+Mar windows are statistically similar; the issue is the encoder learned to lean on day-conditional cues that are absent on novel days.
- IS at the encoder's true generalization ceiling, COMBINED with shortcut-learning during fine-tuning that consumed the signal margin.

The standard pattern: model finds a representation that minimizes the available training signal by exploiting any correlation that holds within the training distribution. In our case, that correlation is "windows from this symbol on this day in this hour have these labels." On novel days, the symbol-and-hour cues remain, but the day-specific class skew doesn't, so predictions collapse toward the symbol-and-hour-conditioned prior. On a balanced metric, that prior averages 0.5 → regression-to-0.50.

## Summary

(1) **Encoder ceiling combined with fine-tune-time shortcut learning** best explains regression-to-0.50: Phase B fit day-conditional structure visible in the in-distribution random val split (Hypothesis C) and routed H10's class-prior gradient through the shared trunk (Hypothesis B residual), consuming the encoder's measured ~1pp Gate-1 signal margin and flipping it to −1.7pp on held-out months where day-specific cues are absent.

(2) **Highest-IV next test (given no-retry pre-commitment) is Gate 4 on the frozen encoder** (Oct-Nov vs Dec-Jan probes evaluated on Feb+Mar) — analogous to walk-forward temporal validation but without re-running Phase B; (A) frozen-encoder larger head is the appropriate follow-up only if Gate 4 passes.

(3) **The encoder representation IS salvageable for non-direction downstream tasks** (Wyckoff label probes, regime classification, cluster-based market-state vocabulary) — Gate 1 demonstrated decodable directional signal exists, the embedding has eff_rank ~215 with embed_std 0.769, and CKA stability is good; what is NOT salvageable under this fine-tuning protocol is direction prediction at H500 above flat-LR.

(4) **Gate 4 prediction (frozen encoder, Oct-Nov vs Dec-Jan, evaluated on Feb+Mar at H500): I expect Gate 4 to PASS with mild stress** — the frozen Gate-1 encoder has no fine-tune-induced shortcut overfitting, so within-period drift will look like the Gate-1 ~1pp ceiling on each half rather than a sign-flipped reversal; the worry is that Oct-Nov-trained probe may show 1.5-2.5pp lower balanced accuracy than Dec-Jan-trained probe on 6-10 symbols (regime drift in the training period), but I do not predict >3pp drop on >10/24 symbols, so the binding stop-gate should hold. If Gate 4 DOES fail, that would be evidence the encoder's ~1pp signal is itself unstable across the training period, which would retroactively reframe Gate 1 as having passed on a marginal regime-conditional artifact.

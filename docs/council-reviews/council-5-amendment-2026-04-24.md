# Council-5 Review: Gate 1 + Gate 3 Spec Amendment 2026-04-24

**Date:** 2026-04-24
**Reviewer:** council-5 (critical skeptic)
**Reviewed artifact:** Commit `b1f4065` on `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`.
**Evidence reviewed:**
- `docs/experiments/step3-run-2-gate1-pass.md`
- `docs/experiments/step5-gate3-avax-probe.md`
- `docs/experiments/step5-gate3-triage.md`
- `docs/experiments/step5-cluster-cohesion.md`
- `docs/council-reviews/council-5-gate3-avax-falsifiability.md` (my prior review)
- `docs/council-reviews/council-3-avax-microstructure.md`

## Verdict

**SIGN OFF WITH TWO PRE-COMMITTED FOLLOW-UPS AND THREE REQUIRED EDITS.** The amendment is defensible. It is not retroactive rationalization in the strict sense — the triage and cluster-cohesion experiments I demanded were actually executed, the numbers are conservative readings of them, and the Gate 1 pass is genuinely anterior to the Gate 3 reframe in the commit log. But there are three hidden degrees of freedom that would let a future researcher re-shape the verdict, and one paragraph of language that reads more defensive than it should. I would sign off conditional on: (a) a precise definition of "matched-density", (b) a pre-committed re-run policy if Feb-OR-Mar ever splits, and (c) pre-committing the per-symbol surrogate sweep as the single follow-up diagnostic.

## Q1 — Is Gate 3's retirement "earned by measurements"?

**Yes, barely.** Bootstrap CIs show encoder-vs-PCA overlap on 4/4 AVAX cells AND on 4/4 LINK+LTC in-sample control cells. That pair of measurements is load-bearing — it turns "AVAX failed" into "the protocol is underpowered." The cluster-cohesion delta (+0.037 vs the +0.1 "some_invariance" threshold I pre-dispatched, with symbol-ID at 0.934) independently shows the training config never targeted universality. Three evidence lines, each with a pre-specified pass/fail criterion, converging on the same conclusion, is the right standard.

**What's missing.** The cluster-cohesion run is Feb-only on 6 anchors. It's a single measurement on a single month. If the amendment is retiring a pre-registered binding gate on the strength of that measurement, I'd want it repeated on Mar or a held-out month to confirm cross-month stability of the delta. Cost: ~5 seconds of inference. Not a blocker, but cheap enough that omitting it is a minor tell. The per-symbol surrogate sweep (n=1 → n=5 on the transfer claim) was NOT run, and it was my Rank-2 ask. I'd pre-commit it as Step-6 diagnostic even if the reframe ships now.

## Q2 — "This amendment is not retroactive rationalization"

The paragraph's causal argument is weaker than it presents. "Gate 1 passed before Gate 3 ran, therefore the reframe is measurement-motivated" is a non-sequitur — the *order* of passes and failures does not establish that the reframe is justified. What establishes that is the triple evidence line. The PhD committee answer would be: "Gate 1 passing first is irrelevant; what matters is that the Gate 3 reframe criteria were specified in a prior council review before the triage measurements were taken, the triage results were conservative readings of pre-specified statistical thresholds, and the cluster-cohesion threshold (+0.1 delta) was my pre-dispatched line in the sand in `council-5-gate3-avax-falsifiability.md`."

**Cleaner defense language for the spec paragraph:**
> "The Gate 3 reframe is motivated by measurement: (1) bootstrap CI overlap between encoder and PCA was the pre-dispatched falsifier (council-5 Rank 1); (2) in-sample LINK+LTC control was the pre-dispatched disambiguator (council-3 recommendation); (3) cluster cohesion delta +0.1 was the pre-dispatched 'some_invariance' threshold (council-5 Rank 3). All three readings are conservative point-estimate comparisons against thresholds set before the experiments ran. The Gate 1 pass is noted but is not the justification."

The current text ("Gate 1 passed before Gate 3 ran") implicitly defends against a p-hacking accusation with temporal order — which is the wrong defense.

## Q3 — 15+/24 vs 15+/25

**Mechanical correction, defensible, but under-documented.** AVAX was in the 25-count; excluding AVAX from the Gate 1 probe universe (because it's held-out) arithmetically implies 24. The amendment writes "AVAX excluded from the in-pretraining-universe count" which is correct. BUT: the original pre-registration said 15+/25. Going from 15/25 → 15/24 lowers the fractional bar from 60% → 62.5%, which *tightens* the gate slightly — so this is moving the goalposts in the harder direction, which is fine. However, it should be explicitly noted in the amendment: **"15+/24 is NOT a loosening; it is mechanically tighter by 2.5 percentage points on the fraction."** Without that note, a casual reader sees "15/25 → 15/24" and assumes laxer.

## Q4 — H500 primary vs H100 informational: is this horizon p-hacking?

**This is the single sharpest risk in the amendment.** The pre-registration said H100. The Gate 1 run produced results at multiple horizons, H100 came in at noise floor, H500 showed signal, and the amendment now declares H500 primary. That IS horizon-selection-post-hoc — it meets the textbook definition.

**Defense:** the writeup's argument is that H100 is at noise floor *for every predictor including PCA, RP, and shuffled*. If H100 balanced accuracy is indistinguishable from chance for the flat baselines too, then H100 is not a horizon where any predictor can demonstrate edge on this data at this window size — it is not "encoder fails at H100, encoder passes at H500," it is "H100 is not a testable horizon at this sample size." That is a genuinely different claim from p-hacking; it is a power analysis.

**What I need to sign off on this:** the spec should PRE-REGISTER a horizon-selection criterion for future runs, e.g., **"the primary horizon is the shortest horizon at which PCA+LR achieves balanced accuracy ≥ 0.505 on the held-out universe."** That makes the H500 choice reproducible and prevents future runs from horizon-shopping.

**Missing from the amendment:** the horizon-selection rule. Add it, or future runs have an escape hatch.

## Q5 — Further measurements before sign-off?

Two pre-commitments, neither blocking the amendment ship:

1. **(c) per-symbol surrogate sweep — PRE-COMMIT as Step-6 diagnostic.** Council-3 explicitly recommended this in `council-3-avax-microstructure.md` Option B, and I asked for it as Rank-2 in my prior review. Cost ~3 hours. Without it, the "universality would require a different training config" claim rests on one held-out symbol + one in-sample-control pair. Five surrogate held-outs would convert n=1 to n=5 on the transfer claim and retire the question properly. If the amendment ships without pre-committing this, I want it logged as Future Work item.

2. **(d) final-epoch checkpoint probe — DO NOT pre-commit.** Low expected value as I said in the prior review. If it changes the verdict it creates a new problem (MEM-minimum is wrong). Skip.

## Q6 — Hidden degrees of freedom in the amended spec

Three. Ranked by how much they could bite:

**#1 — "matched-density held-out months" is undefined.** The phrase appears five times in the amended spec and is never given a measurable threshold. What ratio of windows-per-day on the training set vs held-out set counts as "matched"? If a future run pretrains on different dates and the held-out event-rate is, say, 0.85× training, does that still count? **Fix:** define it: "matched-density requires held-out-month windows-per-symbol-per-day within 0.7–1.3× of training-window density per symbol, on stride=200 eval." Without this, every future run gets to define its own denominator.

**#2 — "BOTH Feb AND Mar must hold, independently" — implicit re-sampling hazard.** The spec says both months must pass. What happens if a future retrained checkpoint passes Feb 0.531 and fails Mar 0.511 on Condition 1 (15+/24)? Can the researcher re-sample by adding April 1-13 as a third month and dropping whichever month failed? Currently the language is silent. **Fix:** add: "The held-out months are Feb 2026 AND Mar 2026 specifically; they may not be substituted, excluded, or supplemented without re-pre-registration." Without this, a future researcher has a free pass to drop the bad month.

**#3 — "This training config" as a shield for the symbol-ID <20% miss.** The amendment reframes the <20% target as aspirational for "future universality-targeting runs," and cites the measured 0.934 as explicable by "the current 6-of-24 soft-positive-0.5 recipe." This creates a template: whenever a quality diagnostic misses, declare it "not targeted by this config." **Fix:** require that any future run which claims to target a quality diagnostic must re-pre-register the threshold BEFORE training, AND any diagnostic present in the spec (symbol-ID, hour-of-day, CKA) that is not explicitly reframed to aspirational must remain binding.

Medium-severity items (worth noting but not fixing now):
- The Gate 3 reframe says "AVAX cache stays; AVAX stays excluded from pretraining." Good. But: does AVAX get to become a pretraining anchor in a future universality-targeting run? If yes, the Gate 3 pre-designation is not actually irrevocable. I'd make this explicit: AVAX is irrevocably held-out from pretraining for any future run that wants to cite this program's pre-registration.
- The amendment doesn't cap the number of future "this training config didn't target X" reframes. Worth a general amendment-budget clause: the spec has been amended twice (council round 6 on 2026-04-15; this amendment on 2026-04-24); a third binding-gate amendment without a new pre-registered experiment should require out-of-band review.

## Summary for orchestrator

Sign off WITH three required edits and one pre-committed follow-up. (1) The single worst hidden degree of freedom is the undefined "matched-density" phrase — five uses, zero measurable threshold, and it is the denominator every future run will tune against. (2) Pre-commit the per-symbol surrogate sweep (c) as a Step-6 diagnostic; skip (d) final-epoch probe. (3) The paragraph "This amendment is not retroactive rationalization" — specifically its causal argument that "Gate 1 passed before Gate 3 ran" justifies the reframe — reads as retroactive rationalization to me; the honest defense is that the triage thresholds were pre-dispatched in my prior review before the experiments ran, and THAT is what makes this measurement-motivated rather than pass/fail-chasing.

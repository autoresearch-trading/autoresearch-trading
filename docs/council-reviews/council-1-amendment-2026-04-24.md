# Council-1 Review: Spec Amendment 2026-04-24 (Gate 1 window / Gate 3 retirement)

**Date:** 2026-04-24
**Reviewer:** council-1 (financial-ML methodology; López de Prado voice)
**Reviewed artifact:** Commit `b1f4065` on `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`.
**Evidence reviewed:**
- `docs/experiments/step3-run-2-gate1-pass.md`
- `docs/experiments/step5-gate3-triage.md`
- `docs/experiments/step5-cluster-cohesion.md`
- `docs/council-reviews/council-5-gate3-avax-falsifiability.md`
- `docs/council-reviews/council-3-avax-microstructure.md`

## Verdict

**Approve-with-amendments on pre-registration ethics.** The Gate 1 window change is defensible as a sample-size constraint but needs textual separation between the pre-registrable (stride × min_valid) justification and the post-hoc (H100 noise-floor) corroborator, plus an explicit binding commitment to always-report-the-original-April-window. The single most important fix is to add a concrete, numerical pre-commitment for Gate 3 re-activation. The dissent-worthy call is that the "not retroactive rationalization" framing in the Gate 3 section is methodologically weaker than the more honest framing "we are retiring a gate that neither model nor in-sample control can falsify — that is evidence of underpowered-gate design, not of a bad model." **I would BLOCK the amendment on the Gate 4 coherence issue (it still references nonexistent "months 5-6 of training") and on the Gate 3 re-activation criteria gap; the Gate 1 textual tightening and horizon-drift fix can ship as follow-ups in the same edit pass.**

## Q1 — Hindsight bias / pre-registration hygiene on the April 1–13 → Feb+Mar move

**Position: defensible on technical grounds, but the amendment language is not yet bulletproof — one sentence is load-bearing and wrong.**

The move is defensible IFF the under-power argument is a genuine sample-size constraint rather than a window-shopping ex post. The spec claims "April 1–13 at stride=200 produces 60–150 windows/symbol, below the probe's 200-window `min_valid` floor" — that is a real sample-size constraint, verifiable from cache statistics WITHOUT reading the Feb/Mar pass numbers. Good. However, "H100 direction prediction is at noise floor for every predictor tested" is NOT an ex-ante sample-size argument; it is an ex-post measurement on THIS data with THIS encoder. The amendment conflates these two justifications. Only the first survives pre-registration ethics; the second is exactly what López de Prado calls "the backtest selected its own window."

**Required edits:**
- Separate the two justifications textually. The stride=200 × 200-window floor is an arithmetic constraint on ANY encoder (discoverable pre-run from the cache manifest). The H100 noise-floor argument should be flagged as "post-hoc corroborating evidence, not the primary reason for the amendment."
- Commit to reporting the original pre-registered window (April 1–13 at H100) alongside the amended one in every Gate 1 reference forever. The current text says April is "still produced as informational output" — make this binding: every Gate 1 report MUST publish the April 1–13 H100 number with its CI, labeled as the original pre-registration, even after it is superseded. This is the anti-amnesia clause.
- Add the trial count to the deflated-Sharpe discount: "Gate 1 has now been evaluated against 2 windows (April 1–13 H100 pre-registered; Feb+Mar H500 amended). Future gates on the same encoder must account for this in multiple-testing corrections."

## Q2 — Multi-month AND-of-passes vs OR vs mean

The **AND** formulation (BOTH Feb AND Mar must pass independently) is the right call for falsifiability and the amendment should keep it, but two issues remain. First, "AND on two months" is not a deeply-multi-split test — it has only one more "nail" than a single-month test, and two correlated months (Feb and Mar on the same data regime) are not two independent hypotheses. A passing run that clears Feb+Mar by tight margins should still be treated as "one signal, replicated once," not "two independent positives."

Second, what the spec DOESN'T specify is the "one-passes-one-fails" adjudication rule. Make it binding: **"If any future re-run passes one month and fails the other, the RUN FAILS Gate 1 — no adjudication, no averaging, no 'close enough.' That is the entire point of independent passes."** Without this explicit language, the first time this happens (and it will happen — two-month pass rates don't have 100% replication on noisy probes) there will be pressure to call it a "mostly pass" and move forward. The amendment should slam that door shut now.

**Do not switch to mean-across-months with a stricter threshold.** Means hide per-month tails; the AND structure preserves the information that regime shifts produce.

## Q3 — Gate 3 retirement defensibility (the single biggest hole)

**This is where the amendment has its single biggest methodological hole.** The current justification is "this amendment is not retroactive rationalization because Gate 1 passed before Gate 3 ran." That is the wrong test. The relevant question is not "did Gate 3 run after Gate 1?" — it is "was the criterion for retiring Gate 3 pre-committed, or is it being articulated for the first time now?"

The honest answer is: the retirement criterion (three-legged evidence: bootstrap-CI overlap + in-sample-control failure + cluster-cohesion delta < +0.1) is being articulated for the first time in this amendment. Council-5's prior review dated 2026-04-24 specified the pre-dispatched thresholds, so they ARE pre-registered in a weaker sense. But there was no pre-committed retirement rule in the original 2026-04-10 spec.

**What would make this defensible to a hostile PhD committee:**
- Reframe Gate 3's status change as "retired informationally because the retirement criterion ITSELF is defined post-hoc." Do not claim non-retroactive-rationalization — claim "this retirement is methodologically conservative: (a) the original binding threshold cannot be cleared at CI-aware rigor, (b) the in-sample control also cannot clear it, (c) therefore the original criterion was underpowered to falsify anything, including the null." Under López de Prado's framework, retiring a gate because IT cannot falsify the null is *stronger* evidence of a badly-designed gate than of a bad model.
- Add a pre-commitment for re-activation (see Q4). Without it, the amendment reads as "the gate was dropped because the model couldn't clear it."
- Explicit language: **"We acknowledge this is a post-hoc retirement of a pre-registered falsifier. The methodological alternative — ignoring the underpower evidence and declaring Gate 3 a failure on point estimates inside their own CIs — would be retroactive false-negative inflation, which is no better than the false-positive it was designed to prevent. We choose to be explicit about the retirement rather than silent about the underpower."**

## Q4 — Measurement-unit specification for re-activation

**Yes, add a concrete re-activation criterion.** Without it, Gate 3 drifts back into binding status through informal accumulation. Concrete language I'd ship:

> **Gate 3 re-activation criteria (any FUTURE training run).** Gate 3 may be re-activated as binding pass/fail IFF all of: (a) n_test ≥ 2000 windows per held-out cell after stride ≤ 50 evaluation, (b) 1000-resample bootstrap 95% CI on encoder balanced accuracy does not include 0.500 on the control in-sample pool measured at matched n_test, (c) cross-symbol SimCLR cluster-cohesion delta ≥ +0.10 (measured as cross_symbol_same_hour − cross_symbol_diff_hour on the liquid anchor set), (d) the re-activation criteria are declared BEFORE the held-out AVAX evaluation is run. Absent all four, AVAX numbers remain informational.

Without this, the loophole is obvious: future runs will "look at" AVAX numbers, discover a good one, and pressure-lift them into binding status. Close the door now.

## Q5 — Gate 4 coherence under the new Gate 1 protocol

**Gate 4 is no longer coherent and the amendment has a silent drift.** Gate 4 says "training months 1-4 vs months 5-6" — but under the amended Gate 1, "training" is Oct 16 – Jan 31 (4 months: Oct/Nov/Dec/Jan) and held-out is Feb+Mar. There are no "months 5-6" of training anymore. The temporal-stability test needs to be redefined as a property of either (a) within-training-period stability (split Oct-Jan into two halves), (b) held-out-period stability (Feb vs Mar already reported in Gate 1), or (c) cross-period stability (training-period probe accuracy vs held-out-period probe accuracy).

Additionally, the Feb-vs-Mar independent-pass requirement in Gate 1 is ALREADY a temporal-stability test in disguise. If Gate 1 passes on both Feb and Mar, then by definition cross-month stability on the held-out set is within whatever the Gate 1 margins are. Gate 4 as written now double-dips or becomes a training-period-only test.

**Recommendation:** Rewrite Gate 4 as "within-training-period stability: Oct-Nov vs Dec-Jan, measured via balanced-accuracy drop on held-out Feb+Mar fold, <3pp on >10/24 symbols" — OR retire Gate 4 explicitly and fold its function into the Gate 1 Feb-vs-Mar consistency check. Leaving it as "months 1-4 vs 5-6" is silent drift.

## Q6 — Horizon drift across gates

**Yes, this is drift and it should be fixed in the same commit as the Gate 4 rewrite.** Gate 0 reports all horizons. Gate 1 is now H500-primary. Gate 4 reports all horizons. Gate 3 (informational) reports H100 and H500. This is not wrong per se — you CAN have gates at different horizons — but the spec should name the horizon structure explicitly:

- H500 is the PRIMARY binding horizon (Gate 1 pass/fail hinges on it).
- H100 is INFORMATIONAL and reported-but-not-binding across all gates after the council-5 noise-floor finding.
- H10/H50 are published for Gate 0 baseline context but are not binding anywhere post-amendment.
- Gate 4's "all horizons" language should explicitly say "Gate 4 binding threshold on H500 only; H10/H50/H100 reported informationally."

Without this, a future analyst will read the spec and see conflicting horizon mandates. Fix both (Q5 and Q6) in one edit.

## Summary to orchestrator

Approve-with-amendments; do not ship as-is. Fix in this order: (1) Gate 3 re-activation criteria, (2) Gate 4 coherence + horizon structure, (3) Gate 1 textual tightening (separate ex-ante / ex-post justifications, anti-amnesia clause, no-adjudication-on-one-month-fail), (4) rewrite the "not retroactive rationalization" paragraph to the more honest underpower framing. These are all surgical language edits with clear evidence backing and should fit in a single follow-up commit.

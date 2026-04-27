# Council-5 Review — Multi-Probe Calibration (Falsifiability Cost of Amendment)

**Date:** 2026-04-26 (PM, post Gate 4 PASS)
**Subject:** `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` (commit `6fab561`)
**Verdict:** Path D, with one narrow concession to Path C-narrow that I will name explicitly. Not Path A. Not Path B under any reading.

## Q1 — Feasibility check vs post-hoc discovery

The charitable reading wins, but only barely, and only because of the audit trail. Discovering that a label fires on 0% of windows is mechanically distinguishable from discovering that a label fires on 6% of windows and the encoder predicts it at chance. The first is a **degenerate-measurement diagnostic** — no encoder was touched, no probe was fit, no threshold was compared against an outcome. The second is a hypothesis test. The "disease never appears in the population" analogy is correct: a clinical trial whose enrollment criteria match zero patients is not a failed trial, it is an un-runnable one.

That said, the strict reading has real teeth. The ratification document (commit `c28bc17`) explicitly signed the *operationalization* — formulae, thresholds, the 3.0 climax seed. The discipline that mattered on April 26 PM was "sign the operational labels before they touch the encoder so you can't goal-seek them." You didn't goal-seek. The label calibration was sloppy at ratification and you caught it on the encoder-free side of the workflow. **This is a feasibility check, not a post-hoc discovery.** It does not consume amendment budget, but it does require the audit-trail clause you already wrote at the bottom of the calibration-issue doc.

## Q2 — Path A as honest negative result

**Path A is the worst option on the table.** Stop B's writeup says the encoder extracts "only direction-flavored signal" and is "not phenomenologically rich." Path A produces that conclusion via labels that fire on 0% of windows, i.e., the encoder was never asked the question. Publishing a negative result whose stated mechanism is *not what was measured* is a worse falsifiability failure than amending a degenerate label set. The pre-registration was meant to prevent confirmation bias on borderline positive results; it was not meant to manufacture conceptually false negative results when the labels are degenerate. Reject Path A.

## Q3 — Path C-narrow with guardrails: conditional sign-off

I will sign off on a Path C-narrow amendment **only** under all of the following guardrails:

1. **One probe, not four.** Absorption only. Climax/stress are dead per step0 per-event rates (~0.1% / ~0%); no aggregation rescues them. Acknowledge this in the amendment.
2. **Thresholds frozen on Oct–Jan training-period data, not Feb–Mar.** The aggregation rule (e.g., "≥20 of last 100 events fire step0's per-event absorption") is calibrated to land in the 1–5% positive-rate band on training-period shards, then frozen. Feb–Mar is never inspected for label calibration. This closes the Path B leakage loophole.
3. **Battery rule rewritten as 1-of-1, threshold raised.** Original was ≥2 of 4 at +2pp. Replacement: the single absorption probe must clear +2.5pp on 12+/24 symbols. Higher bar compensates for the reduced trial count; one probe at the original +2pp would lower the bar.
4. **C4 dropped, not amended.** A percentile-based seed for C4 is Path B leakage. Drop it.
5. **C2 (CKA) preserved unchanged** — independent of label calibration, no amendment needed.
6. **Amendment recorded in the calibration-issue doc with council-1 + council-5 sign-off and a new ratified diff against `c28bc17`.**

If any of these guardrails cannot be met, fall back to Path D.

## On the fatigue question

You asked me to call it if you're shading toward D out of fatigue. I don't think you are. Your reasoning ("ad-hoc amendments compound the falsifiability cost") is the correct *default* for this council. But Path D's writeup also has to tell the truth: *"the multi-probe battery's labels were degenerate on the held-out distribution; we did not test phenomenological richness."* Gate 2 FAIL + Gate 4 PASS is publishable on its own merits. Path C-narrow is publishable with strictly more information. Either is honest. Path A is not.

**Signed: council-5, 2026-04-26 PM.**

## Summary

(1) The 0% label-fire discovery is a **feasibility check**, not post-hoc — no encoder forward pass touched the labels — so it does not consume amendment budget but it does require the audit-trail clause already in the calibration-issue doc. (2) Path A produces a **conceptually false negative result** (claims encoder failed phenomenology when actually labels never fired) and is rejected outright. (3) Path D (skip battery, write Gate 2 FAIL + Gate 4 PASS as the publishable end-state with explicit acknowledgment that phenomenological richness was not tested) is acceptable. (4) Path C-narrow is acceptable with six guardrails: absorption-only (climax/stress dead per step0 rates), thresholds frozen on Oct–Jan training data, single-probe pass bar raised to +2.5pp on 12+/24 symbols, C4 dropped (no percentile seed), C2 CKA preserved, amendment recorded against commit `c28bc17` with council-1 + council-5 sign-off. (5) The fatigue check is unnecessary — the lead's Path D reasoning is correct as default; the choice between D and C-narrow is a choice between "publishable as-is" and "publishable with strictly more information at one well-scoped extra probe."

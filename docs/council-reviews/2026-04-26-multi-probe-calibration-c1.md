# Council-1 Review (Lopez de Prado / Methodology) — Multi-Probe Calibration

**Date:** 2026-04-26 (PM, post Gate 4 PASS)
**Subject:** `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` (commit `6fab561`)
**Status:** Advisory — endorses Path D with a narrow Path C-narrow fallback.

## A. Multiple-comparisons / DSR accounting

The trial-count question is the cleanest of the three. AFML (Ch. 8, Backtest Statistics) is unambiguous: the denominator for the Deflated Sharpe Ratio counts **all configurations evaluated against the held-out distribution that informed any selection decision**, regardless of whether they "ran" against the model. The label-feasibility check IS a measurement on the held-out distribution. You looked at Feb+Mar feature data to determine that thresholds were uncalibrated. That is a peek — at the *covariate side*, not the *outcome side*, but a peek nonetheless against the same temporal slice you will probe with the encoder.

Concretely:

- **Original ratified trial count:** N=5 (Gate 4, C1, C2, C3, C4).
- **Path C-narrow (single re-calibrated probe):** trials drop to N=2 in *intent*, but the amendment itself adds one "selection trial" because the calibration check used Feb+Mar to choose which labels survive. Effective N for DSR = **3** (Gate 4 + amended probe + the calibration selection itself).
- **Bonferroni denominator for the amended probe:** 3, not 2. Per-probe α = 0.05/3 ≈ 0.0167 (two-sided).
- **DSR adjustment:** apply Bailey & López de Prado (2014) Eq. 9 with N=3 and the empirical skew/kurtosis of the bootstrap null. The threshold +2pp on the amended probe should be raised to whatever point estimate satisfies PSR ≥ 0.95 against N=3, which on typical bootstrap-balanced-accuracy null distributions adds roughly +0.4 to +0.7 standard errors to the required margin. **Practical effect: the amended C1's threshold should rise from "+2pp on 12+/24 symbols" to approximately "+2.5pp on 12+/24 symbols" or "+2pp on 14+/24 symbols"** — pick one ex ante, not both.

## B. Sequential-decision integrity

This is where Path C-narrow becomes most expensive. Gate 4 has resolved positively. The amendment-budget clause exists precisely because amendments are cheaper to justify when they sit in front of an unfavorable result and harder to justify when they sit behind a favorable one. We are in the **post-Gate-4 amendment regime**. The asymmetry is real: a battery whose first component passed and whose second component is now being re-spec'd creates the canonical "garden of forking paths" hazard López de Prado warns about (AFML §11.6, "The Backtest is a Research Tool, Not a Validation Tool"). The fact that no encoder forward pass touched C1/C3/C4 mitigates but does not eliminate the cost: you know the favorable Gate 4 outcome, and any amendment is now made under that knowledge.

**Cost surcharge:** treat the amendment as carrying one additional "implicit trial" beyond the calibration-selection trial in (A). DSR effective N = **4**, not 3, for any probe amended after Gate 4 resolved.

## C. The "publishable null" question

Falsifiability requires the negative result to accurately describe what was tested. Path A produces a Stop B writeup whose causal claim ("encoder fails phenomenology") is **not what the data shows** — what the data shows is "labels were uncalibrated and never had a chance to fire." Publishing Path A as Stop B is a Type-III error (right answer to the wrong question) and is methodologically worse than amending. **Path A is rejected.**

This leaves Path C-narrow vs Path D. **Path D is methodologically superior** for three reasons:

1. **Honest mechanism description.** Gate 2 FAIL + Gate 4 PASS already constitutes a coherent, publishable finding ("stable +1pp ceiling, not amplifiable by fine-tuning"). The phenomenological battery was intended to *strengthen* that claim, not enable it.
2. **No DSR surcharge.** Path D consumes zero amendment budget; the writeup explicitly notes the battery was undefinable on operational data, which is itself a finding about spec phenomenology.
3. **Preserves amendment budget for a future, properly pre-registered phenomenology run** with labels calibrated on April-1-13 (already-touched) data, then frozen before April-14+ evaluation.

## Recommendation

**Endorse Path D.** Write up Gate 2 FAIL + Gate 4 PASS as the publishable end-state. Document the calibration discovery (commit `6fab561`) as a finding about spec-label operationalization, not as a failed encoder test. Do not amend the ratified pre-registration.

If the user insists on Path C-narrow against this advice, the binding accounting rules are:

- **Trials counted for DSR:** N=4 (Gate 4 + amended probe + calibration selection + post-Gate-4 amendment surcharge).
- **Threshold for the amended probe:** raise from +2pp/12+ to **+2.5pp on 12+/24 symbols** OR **+2pp on 14+/24 symbols** — chosen and committed BEFORE the encoder forward pass.
- **Required disclosure in any writeup citing the amended probe:**
  1. Reference to `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` commit `6fab561`.
  2. The original C1/C3/C4 spec, the amended single-probe spec, and the Feb+Mar empirical positive rate that motivated the amendment.
  3. The DSR-adjusted threshold and the rationale for N=4.
  4. Explicit acknowledgment that C2/C3/C4 were dropped without measurement and cannot be retroactively reinstated.
- **What Path C-narrow does NOT authorize:** any quantile-based label fit on Feb+Mar (Path B is rejected outright as test-set leakage, AFML §7.4); any re-introduction of dropped conditions if the amended probe fails; any bootstrap CI lower-bound substitution for the point-estimate test.

The amendment-budget clause exists for exactly this moment. Path D spends nothing and produces an honest writeup; Path C-narrow spends one budget unit and tightens the threshold; Path A spends nothing but corrupts the falsifiability frame. The methodologically cheapest *and* most honest option is Path D.

## Summary

(1) **Path D endorsed** on falsifiability + DSR-accounting grounds; the calibration discovery used Feb+Mar covariate data so any probe amended now incurs an "implicit trial" against that slice and we are in the post-Gate-4-resolved regime where amendment cost is highest. (2) Path A is **methodologically rejected** as a Type-III error (right answer to the wrong question — the writeup's stated mechanism is not what the data showed). (3) Path C-narrow is acceptable only with binding DSR effective N=4, threshold raised to +2.5pp/12+ or +2pp/14+ chosen ex ante, and full amendment-history disclosure including the original spec, the amended spec, the empirical positive rate that motivated it, and the explicit non-reinstatement clause for C2/C3/C4. Path D preserves amendment budget for a future properly pre-registered phenomenology run with calibration on April-1-13 (already-touched) data and evaluation on April-14+ untouched data — that is the methodologically superior way to answer the phenomenological-richness question if it remains worth answering.

# Council Review: Financial ML Methodology (Lopez de Prado) — Round 4

**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Reviewer:** Council-1 (Financial ML Methodology, AFML 2018)
**Date:** 2026-04-03
**Prior reviews:** Round 1 (2026-04-01)
**Context:** Re-review after Rounds 2-3 council changes and Round 3 synthesis

---

## Executive Summary

The spec has improved substantially since Round 1. Five of the six critical and high-priority issues I raised have been correctly addressed: the primary horizon pre-designation, the 600-event embargo (upgraded from my Round 1 recommendation of 500), the Bonferroni-corrected 51.4% linear baseline threshold, the trial_log.csv requirement, and the DSR/PSR requirement in Phase 2. The April hold-out designation and hold-out symbol test are now concrete and irrevocable.

Three issues remain requiring action before implementation. One is a critical internal contradiction that will produce a live defect. Two are high-priority statistical validity concerns that do not block experiments but will undermine the interpretability of final results.

---

## 1. Round 1 Issue Resolution Status

| Issue | Round 1 Severity | Round 4 Status |
|-------|-----------------|----------------|
| No multiple testing correction | Critical | Partially resolved — trial_log.csv added; Phase 1 table still hardcodes T=100 |
| No primary horizon pre-designation | Critical | Resolved — 100-event horizon designated, others exploratory |
| Missing embargo zones | Critical | Resolved — 600-event embargo specified |
| 52% threshold not statistically justified | High | Partially resolved — 51.4% added to baseline section; residual in Evaluation Protocol |
| No DSR/PSR plan | High | Resolved — DSR required in Phase 2, PSR < 0.95 criterion stated |
| Logistic regression overfitting | High | Resolved — C sweep {0.001, 0.01, 0.1, 1.0} required, train/test separation required |
| Walk-forward fold count insufficient | High | Partially resolved — 2-3 folds honestly acknowledged; PSR implications not addressed |
| Label overlap between adjacent samples | Medium | Acknowledged as not affecting walk-forward evaluation — correct for evaluation but nominal N is inflated |
| Hold-out symbol test | Medium | Resolved — FARTCOIN explicitly held out |
| Go/no-go gate AND→OR logic | Critical (Round 3) | Resolved — OR logic specified |
| Linear baseline threshold 50.5%→51.4% | Critical (Round 3) | Resolved in main body; CRITICAL RESIDUAL in Evaluation Protocol section |
| 2 bps return threshold ambiguity | Medium | Resolved — replaced with 90th-percentile vs fee threshold |

---

## 2. CRITICAL DEFECT — Internal Contradiction on Linear Baseline Threshold

**Severity: Critical. Will cause a live defect if the Evaluation Protocol section is used as the implementation reference.**

The spec contains two contradictory stop thresholds for the linear baseline at the primary horizon.

**Location 1 — Mandatory Linear Baseline section (correct):** The text beginning at line 179 states: "If fewer than 15/25 symbols achieve logistic regression accuracy above 51.4% at the primary horizon: STOP." This is the Bonferroni-corrected significance boundary correctly derived from N=6,000 and alpha/4=0.0125: critical accuracy = 0.50 + 2.24 × sqrt(0.25/6000) = 51.45%.

**Location 2 — Evaluation Protocol section (incorrect):** The Phase 0.5 summary at line 313-316 states: "If < 50.5% on all horizons: stop." This is the old threshold that was explicitly identified as defective in the Round 3 synthesis (item C3: "50.5% is inside the null band and would pass pure noise 69% of the time"). The fix propagated to the Mandatory Linear Baseline section but not to the Evaluation Protocol section.

**Why this matters at implementation time:** Engineers building the pipeline will consult the Evaluation Protocol section as the compressed implementation checklist. The phase table is what gets turned into conditional branching in code. An implementer following the phase table will gate on 50.5%, which passes pure noise models 69% of the time and will never trigger the stop condition.

**Fix required:** Update line 315 to read: "If fewer than 15/25 symbols achieve accuracy above 51.4% at the primary horizon (100 events): stop. See Mandatory Linear Baseline section for full criteria." Remove "on all horizons" — the gate is per-symbol at the primary horizon only, not an aggregate across all four horizons.

---

## 3. Multiple Testing Framework — Phase 1 Table Hardcodes Wrong T

**Severity: High. Does not block experiments but will make the final evaluation report misleading.**

The Multiple Testing paragraph at line 303 correctly states: use total trial_log.csv row count as T in the DSR formula and Holm-Bonferroni correction. This is the right framework.

The problem is the Phase 1 evaluation table at line 326:

> "All results | Holm-Bonferroni corrected | Accounts for 100 simultaneous tests"

When the final model is evaluated, trial_log.csv will contain approximately 1,600 rows. Applying Holm-Bonferroni at T=1,600 instead of T=100 changes the required accuracy threshold as follows.

**Quantitative analysis at N=6,000 samples per symbol per fold:**

| Holm rank | T total | Per-test alpha | Critical z | Required accuracy |
|-----------|---------|---------------|-----------|------------------|
| 1 (most significant) | 1,600 | 3.13e-5 | 4.01 | 52.6% |
| 25 | 1,600 | 3.17e-5 | 4.00 | 52.6% |
| 100 | 1,600 | 3.33e-5 | 3.99 | 52.5% |
| 1,500 | 1,600 | 4.95e-4 | 3.29 | 52.1% |
| 1,600 (least significant) | 1,600 | 0.05 | 1.64 | 50.9% |

The stated Phase 1 target of "accuracy > 52% across 20+ symbols" requires that each of the 20+ symbols independently clear 52%. Under Holm-Bonferroni at T=1,600, the 20 strongest results are at ranks 1-20, requiring approximately 52.6%. The stated 52% target would fail the correction for these top-ranked tests by 0.6 percentage points.

**Practical implication:** A CNN achieving 52.2% mean accuracy across 20 symbols after 1,600 trials does not pass Holm-Bonferroni at T=1,600. The Phase 1 table says it does. The spec body says it does not. When the final evaluation report is written, this contradiction will be unresolved unless the table is fixed first.

**Recommendation:** Replace "Accounts for 100 simultaneous tests" with "Use T = total rows in trial_log.csv at time of final evaluation. At the planned T=1,600, the Holm-corrected required accuracy for the primary test is approximately 52.6% (see Multiple Testing section for formula)." Update the success target from "> 52%" to "> 52.6% (at T=1,600, Holm-corrected)" or state it as a function of the trial count at evaluation time.

---

## 4. Walk-Forward Evaluation — Embargo Implementation Specifics and PSR

**Severity: High for PSR. Medium for embargo implementation detail.**

### 4.1 Embargo Measurement Needs Explicit Anchor Point

The 600-event embargo is correctly specified, but the implementation requires a precise anchor: is the embargo measured from the last window START in training data, or from the last EVENT in training data?

For stride=200 windows, the last training window starts at position P_last and extends to P_last+199. The 500-event label for this window covers events [P_last+200, P_last+699]. Without a precise anchor, an implementation that measures embargo from P_last (window start) rather than P_last+199 (last event) would place the first test event at P_last+600, while the last training label extends to P_last+699. Test events [P_last+600, P_last+699] appear in the last training window's label — lookahead bias survives the nominal embargo.

**The correct anchor:** Embargo measured from the index of the FINAL EVENT in the last training sample (P_last+199). Test data begins at index P_last+199+601 = P_last+800. This ensures zero overlap between any training label horizon and any test input window.

**Fix:** Add to the Validation section: "The 600-event embargo is measured from the index of the final event in the last training sample, not from the start of the last training window. Test data begins at final_train_event_index + 601."

### 4.2 Two to Three Folds Cannot Support PSR From Fold-Level Means

With 2-3 truly independent test periods, the variance on the mean accuracy estimate is large. From the Round 3 synthesis: v11b showed std=0.220 on Sortino across 4 folds (mean=0.261). With 2 folds, the standard error of the mean is sigma / sqrt(2) ≈ 0.22/1.41 ≈ ±0.16 Sortino — more than half the baseline value.

The Probabilistic Sharpe Ratio (AFML equation 8.1) must be computed from the daily P&L return series within test periods, not from fold-level accuracy averages. A fold-level mean from 2 observations has no meaningful PSR. The daily P&L series within the test periods provides N=20 or N=40 daily observations, which is the correct input to PSR. N<30 is still marginal for reliable gamma_1 and gamma_2 estimates, but it is orders of magnitude better than N=2.

**Recommendation:** Add to Phase 2: "PSR is computed from the daily P&L return series within test periods (N = number of trading days in test windows). Report N alongside PSR. If N < 30, supplement PSR with 95% bootstrap confidence interval on the Sharpe ratio (10,000 resamples of daily P&L). Fold-level accuracy averages have no meaningful PSR — do not compute PSR from 2-3 fold-level means."

---

## 5. Label Smoothing Values — Statistical Analysis of ε = 0.10 / 0.08 / 0.05 / 0.05

**Severity: Medium. Values are defensible but the effective gradient analysis reveals a potential design issue.**

### 5.1 Combined Effect of Smoothing and Loss Weights

Label smoothing modifies the effective gradient contribution from each horizon. At initialization (sigmoid output ≈ 0.5), the expected squared gradient magnitude is proportional to (0.5 - ε/2)². Combined with the loss weight w_h, the effective gradient contribution is:

| Horizon | Weight (w_h) | Smoothing (ε) | Effective factor: w × (0.5 - ε/2)² | Share of total |
|---------|-------------|--------------|-------------------------------------|---------------|
| 10-event | 0.10 | 0.10 | 0.10 × 0.2025 = 0.02025 | 9.2% |
| 50-event | 0.20 | 0.08 | 0.20 × 0.2116 = 0.04232 | 19.2% |
| 100-event | 0.35 | 0.05 | 0.35 × 0.2256 = 0.07897 | 35.8% |
| 500-event | 0.35 | 0.05 | 0.35 × 0.2256 = 0.07897 | 35.8% |
| **Total** | | | **0.22051** | **100%** |

The label smoothing adds modest additional suppression to the 10-event head (9.2% vs raw weight share of 10.0%) and equally modest additional suppression to the 50-event head (19.2% vs 20.0%). The effect is small — loss weights are the dominant driver of gradient balance.

### 5.2 The 10-Event Head May Degenerate

With only 9.2% of gradient share and a near-random label at the 10-event horizon, the 10-event head's Linear(64→1) weights will receive gradients that are dominated by random noise rather than signal. AdamW weight_decay=1e-4 will push these weights toward zero over training. The likely result: the 10-event head learns to predict 0 logit (50% probability) for all inputs, which is the maximum-entropy prediction for a near-random label and is technically correct.

A degenerate 10-event head has two consequences:
1. It does not contribute the intended short-timescale regularization to the shared trunk features
2. It will report approximately 50% accuracy during evaluation, not providing meaningful information

**Detection:** After prototype training, compute variance of 10-event head logits across the test set. If variance < 0.05 (all outputs near 50%), the head has degenerated.

**If degeneration is confirmed:** Either remove the 10-event head from the shared loss entirely (keep as output head with zero gradient flow, or drop) and redistribute its 10% weight to the remaining heads, or raise w_10 to 0.20 and lower ε_10 to 0.05 to match the 100/500-event configuration.

### 5.3 ε Values Are Not Calibrated to Empirical Label Noise

The ε values are design choices without calibration to the actual label noise. The correct calibration under the Symmetric Label Noise model: ε = 2 × (1 - r_h), where r_h is the probability that a label at horizon h is informative (not random).

At the 10-event horizon: if the base rate is 50.3% (nearly random), the effective signal fraction is low and ε=0.10 understates the noise — the correct ε would be 0.20-0.40.

**Recommendation:** After Step 0 base rate validation, recalibrate ε for each horizon. If a horizon's base rate is within 50 ± 0.2%, raise ε to 0.20. If within 50 ± 0.5%, ε=0.10 is reasonable. If within 50 ± 1%, ε=0.05 is appropriate.

---

## 6. Go/No-Go Gate Analysis — Remaining Issues

### 6.1 Step 0: "10/25 Symbols" Stop Threshold Is Unexplained

The spec states: "Stop if base rate is within 50 ± 0.5% for more than 10/25 symbols at that horizon."

The 10/25 threshold is not derived from a statistical criterion. However, it is internally consistent with the 15/25 linear baseline gate: if exactly 15 symbols have signal, exactly 10 are noisy — the base rate gate fires at the same configuration as the linear baseline gate. This consistency is correct but accidental. An implementer reading only the base rate gate in Step 0 cannot know it is calibrated to match the Step 1.5 gate.

**Fix:** Add one sentence: "This threshold is set to match the 15/25 minimum required for the linear baseline gate — if fewer than 15 symbols have non-noise base rates, the linear baseline gate will almost certainly not be reached."

### 6.2 Step 0: 90th Percentile Return Gate — Pooling Method Unspecified

"90th-percentile absolute return is below fee_mult (11 bps) at the primary horizon" does not specify whether this is per-symbol or pooled.

If pooled across all symbols, BTC's high-volatility returns will dominate. XPL's low absolute volatility may always produce p90 returns below 11 bps even when its returns are economically meaningful relative to XPL's own fee structure. Pooling will mask this.

**Recommended specification:** "Per-symbol: stop if fewer than 15/25 symbols individually have 90th-percentile absolute return above 11 bps at the primary horizon. Consistent with majority-rule logic used in the linear baseline gate."

### 6.3 The "2 bps" Mean Return Report Is Now Orphaned

The Pre-Step section at line 39 still mentions "mean absolute return per horizon (if < 2 bps over the full horizon, moves are too small)" as a report item. The actual stop condition was changed to the 90th-percentile ≥ 11 bps criterion, making the 2 bps mention a potentially confusing vestigial criterion. Clarify it as "report only — not a stop condition."

---

## 7. New Concerns from Spec Changes

### 7.1 The Mar 5-25 Window: "Treat with Skepticism" Is Not a Protocol

The spec states the Mar 5-25 window "has been used for 20+ experiments on main branch — treat with skepticism." In the 2-fold expanding window structure (fold 1 test: Jan 5 - Feb 13; fold 2 test: Feb 14 - Mar 25), the contaminated Mar 5-25 window falls entirely within fold 2 — the only major historical test fold. Fold 2 test results cannot be presented as clean.

**The practical consequence:** The spec effectively has no clean historical test fold. The April 1-13 development window and April 14+ final hold-out are the only statistically clean test data. The historical walk-forward results (Oct 2025 - Mar 2026) should be explicitly labeled as "exploratory, not hypothesis-testing" due to main-branch contamination.

**Recommendation:** Add one sentence to the Validation section: "Due to 20+ experiments run against the Mar 5-25 window on the main branch, all historical walk-forward test results (Oct 2025 - Mar 2026) are treated as exploratory and not used for hypothesis testing. The April 1-13 development validation and April 14+ final hold-out are the only statistically clean test sets for this research."

### 7.2 Effective Sample Size at the 500-Event Horizon

The spec states 400-560K training samples at stride=200. At the 500-event horizon, adjacent samples share 300 events of label overlap (events [P+400, P+699] appear in both window i's label and window i+1's label, representing 60% overlap). Serial label correlation inflates nominal sample count.

Under a first-order autocorrelation model with rho=0.60, the effective sample size for the 500-event horizon is approximately N_eff = N_nominal / (1 + 2rho) = N_nominal / 2.2 ≈ 45% of nominal. The 400-560K nominal samples become approximately 180-250K effective samples at this horizon. Per-symbol N_eff ≈ 8,000 (still adequate), but confidence intervals on training accuracy that use nominal N=16,000 per symbol will be too narrow by a factor of sqrt(2.2) ≈ 1.5.

**This does not affect walk-forward test accuracy** (correctly acknowledged in the spec). It affects reported training accuracy confidence intervals. Add: "Confidence intervals on training accuracy at the 500-event horizon use N_eff ≈ N_nominal/2.2 due to 60% label autocorrelation between adjacent stride-200 samples. Walk-forward test accuracy uses non-overlapping test periods with 600-event embargo and is unaffected."

### 7.3 Phase 2 DSR Needs a Position-Sizing Rule

DSR cannot be computed from accuracy alone. It requires a daily P&L series, which requires a position-sizing rule. The spec mentions "add fee model and position sizing" but specifies no rule. Without a rule, Phase 2 is underspecified.

**Minimum viable specification:** Use fixed-fraction Kelly sizing: f* = 2p - 1 where p is the realized accuracy at the primary horizon, capped at 1% of notional per trade. Daily P&L = sum of (f* × signed_return_100_event - fee_rate) per symbol per trading day. This gives the daily P&L series from which to estimate gamma_1, gamma_2, and ultimately DSR.

State the SR_0 benchmark values explicitly: SR_0 = 0 (strategy beats nothing) and SR_0 = current_baseline_sortino ≈ 0.353 (strategy beats the flat MLP). Both are meaningful comparisons.

### 7.4 Augmentation + Dropout + Weight Decay + Label Smoothing: Risk of Over-Regularization

The model applies four simultaneous regularization mechanisms: Gaussian input noise (σ=0.05), OB feature dropout (p=0.15), Dropout(0.1) in first two layers, weight_decay=1e-4, plus label smoothing. With 91K parameters and 200K effective samples, the regularization-to-capacity ratio is high.

Over-regularization manifests as train accuracy ≈ val accuracy ≈ just above random. The model cannot fit the training data. The correct diagnostic is the training accuracy trajectory across epochs — if it plateaus at 51% after 5 epochs, the model is under-learning.

**Recommendation:** Add to Step 2 prototype: run one comparison — full regularization vs augmentation disabled (keep Dropout + weight_decay only). If training accuracy improves substantially without augmentation while validation accuracy holds, the augmentation is providing diminishing returns and can be tuned down.

---

## 8. Summary and Priority Table

| Finding | Severity | Timing |
|---------|---------|--------|
| Evaluation Protocol line 315 says 50.5%, contradicts 51.4% in main body | CRITICAL | Fix before any code runs |
| Phase 1 table hardcodes T=100; at T=1,600, required accuracy ≈ 52.6% | HIGH | Fix before final evaluation report |
| Embargo must be measured from last event index, not last window start | HIGH | Fix in tape_dataset.py implementation |
| PSR requires daily P&L series, not fold-level means — no position sizing rule for Phase 2 | HIGH | Add position-sizing rule before Phase 2 |
| 10-event head: only 9.2% gradient share, may degenerate — check logit variance | MEDIUM | Monitor during Step 2 prototype |
| ε values not calibrated to empirical label noise from Step 0 | MEDIUM | Recalibrate after Step 0 |
| "10/25 symbols" base rate stop threshold is unexplained | MEDIUM | Add one sentence linking to 15/25 linear baseline gate |
| 90th-percentile return gate pooling method unspecified | MEDIUM | Add per-symbol majority-rule specification |
| Mar 5-25 contamination: "skepticism" is not a protocol | MEDIUM | Explicitly label walk-forward results as exploratory |
| N_eff ≈ 45% of nominal at 500-event horizon — confidence intervals inflated | MEDIUM | Add N_eff note to Validation section |
| "2 bps" mean return in Pre-Step text is orphaned criterion | LOW | Clarify as report-only |
| Over-regularization risk from four simultaneous regularizers | LOW | Add calibration comparison in Step 2 |
| cum_ofi_5 brackets only 40% of primary horizon window | LOW | Acknowledged; sweep {3,5,10} will resolve |

**Net assessment:** The spec is ready for implementation with one blocking fix (the 50.5% contradiction in the Evaluation Protocol section). Three high-priority issues (embargo anchor, PSR computation plan, Phase 1 table T correction) should be resolved before the final evaluation report is written. The label smoothing values are defensible for the prototype but require post-Step-0 recalibration if the 10-event horizon proves near-random. All Round 1 critical issues have been addressed or substantially mitigated.

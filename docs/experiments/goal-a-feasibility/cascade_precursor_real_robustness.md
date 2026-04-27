# Goal-A cascade-precursor (stage 2) — day-clustered bootstrap robustness

**Question.** The stage-2 real-cause-flag probe reported pooled AUC = 0.817 [0.771, 0.858] at H500 with per-window bootstrap.  Cascade events cluster intra-day (contagion), so the leave-one-day-out folds may not be independent.  This document re-tests the AUC and precision-at-top-1% under a **day-clustered bootstrap** (resample the 7 cascade days WITH REPLACEMENT, take ALL windows from each sampled day, repeat 1000×).  The binding distinguishability test compares the day-clustered real-AUC lower bound against the day-clustered shuffled-AUC upper bound.

**Inputs.** Reuses `cascade_precursor_real_per_window.parquet` (held-out predictions from the stage-2 LR run).  No re-training; this is a bootstrap analysis on existing predictions.  Shuffled-baseline AUC uses the saved `pred_proba_shuffled` column — same predictions as the prior run, NOT a fresh shuffle.

## 1. Day-clustered AUC at H100 and H500

| H | AUC real (mean over boot) | 95% CI (day-clustered) | Per-window CI (prior run) |
|---|---|---|---|
| H100 | 0.783 | [0.700, 0.852] | [0.689, 0.870] |
| H500 | 0.815 | [0.772, 0.848] | [0.771, 0.858] |

## 2. Day-clustered shuffled-baseline AUC + distinguishability

| H | AUC shuffled (mean over boot) | 95% CI (day-clustered) | real lo > shuffled hi? |
|---|---|---|---|
| H100 | 0.303 | [0.154, 0.428] | YES |
| H500 | 0.400 | [0.285, 0.505] | YES |

## 3. Per-day attribution (H500)

| date | n_cascades | n_windows | day AUC | precision@top-1% | leave-this-day-out pooled AUC drop |
|---|---|---|---|---|---|
| 2026-04-03 | 0 | 59 | n/a (single-class) | 0.000 | +0.001 |
| 2026-04-04 | 3 | 407 | 0.629 | 0.000 | +0.010 |
| 2026-04-06 | 13 | 203 | 0.743 | 0.000 | -0.014 |
| 2026-04-07 | 9 | 291 | 0.890 | 0.667 | +0.008 |
| 2026-04-09 | 13 | 283 | 0.818 | 0.000 | +0.001 |
| 2026-04-10 | 8 | 212 | 0.744 | 0.500 | -0.010 |
| 2026-04-13 | 27 | 348 | 0.843 | 0.667 | +0.008 |

**Top 2 days driving the pooled AUC: 2026-04-04, 2026-04-07 (leave-out drops +0.010, +0.008).**  Pooled H500 AUC = 0.817.

## 4. Day-clustered precision-at-top-1% at H500

Real precision@top-1% (day-clustered mean): 0.276 [0.067, 0.471]

Shuffled precision@top-1% (day-clustered mean): 0.069 [0.000, 0.214]

Above the 5% (10× lift over base rate ~0.5%) tradeable threshold? **YES** (threshold from cascade_precursor_real.md §5).

## 5. Verdict

Day-clustered H500 real AUC = 0.815 [0.772, 0.848].  Day-clustered H500 shuffled AUC = 0.400 [0.285, 0.505].  Real lo > shuffled hi: **YES**.  Top 2 days driving signal: 2026-04-04, 2026-04-07 (combined leave-out drop = +0.018).  Implied pooled AUC after removing both: 0.797.

**Result kind: robust to day-level clustering.**

## 6. Methodological notes

* **Day-clustered bootstrap.** Each iteration samples 7 days WITH replacement from the 7 available cascade days (Apr 3, 4, 6, 7, 9, 10, 13), takes ALL windows from each sampled day, computes AUC on the concatenated fold.  This treats each day as the unit of independence, consistent with the cascade-contagion concern flagged in the stage-2 writeup.

* **Shuffled baseline reuses saved predictions.** The shuffled-label AUC uses the `pred_proba_shuffled` column already in the per-window parquet — no fresh shuffle, no re-training.

* **Per-day AUC may be NaN** when a day has zero or all positive labels (single-class within day → AUC undefined).  These rows are dropped from the leave-one-day-out attribution but remain in the CSV.

* **April 14+ untouched.** No raw or cached April 14+ data was loaded.

_Robustness analysis ran in 2.4 s on existing per-window predictions (1000 bootstrap iterations).  No LR re-training, no April 14+ data touched._

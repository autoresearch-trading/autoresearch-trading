# Goal-A cascade-precursor (OOS test) — does AUC=0.815 generalize to Apr 14-26?

**Question.** The Apr 1-13 in-sample LR (83-dim flat features, balanced class weight, leave-one-day-out CV) reported pooled cross-symbol AUC = **0.815 [0.772, 0.848]** at H500 with day-clustered bootstrap, and top-1% precision = 27.6% (lift 6.86×).  The marginal-long strategy at top-1% / 0.5% / 0.1% was net-negative — not directly tradeable.  **This run tests whether the AUC=0.815 signal generalizes to Apr 14-26 (genuinely held-out at the time of the in-sample fit).**

**Protocol.** ONE universe-wide LR (`LogisticRegression(class_weight='balanced', C=1.0)`) fit on ALL Apr 1-13 data — no leave-one-day-out, since the test set is genuinely held-out.  Stride=200 evaluation windows on Apr 14-26 cache shards.  Real cascade label = any `cause IN ('market_liquidation', 'backstop_liquidation')` fill in (anchor_ts, ts_at(anchor + H)].  Day-clustered bootstrap CI (resample the 11 OOS days with replacement, 1000 iters).  Shuffled-OOS baseline: labels permuted within day on the same OOS predictions (label permutation test).  Per-symbol cells reported for symbols with ≥ 3 real cascades on Apr 14-26 at H500.

**Hard constraint (anti-amnesia).** The Apr 14+ holdout has been DELIBERATELY consumed for this test.  After this run, no untouched cascade-labeled holdout remains; future OOS evaluations require either (a) waiting for new data accrual, or (b) splitting the merged Apr 1-26 dataset.  This is the binding generalization test for the cascade-precursor program.

## 1. Sample size on OOS (Apr 14-26)

| H | n_total (windows) | n_cascades | base rate | n_days |
|---|---|---|---|---|
| H100 | 2082 | 24 | 0.0115 | 11 |
| H500 | 1532 | 96 | 0.0627 | 11 |


## 2. Pooled cross-symbol AUC OOS vs in-sample

| H | AUC OOS (day-clustered) | AUC in-sample | Δ_AUC | AUC OOS shuffled (day-clustered) |
|---|---|---|---|---|
| H100 | 0.752 [0.638, 0.859] | 0.784 [0.689, 0.870] | -0.032 | 0.501 [0.410, 0.650] |
| H500 | 0.778 [0.732, 0.833] | 0.817 [0.771, 0.858] | -0.039 | 0.596 [0.542, 0.663] |

## 3. Distinguishable from shuffled-OOS baseline?

Binding statistical test: real-OOS AUC CI lower bound must strictly exceed shuffled-OOS AUC CI upper bound (day-clustered bootstrap).

* H100: real OOS CI [0.638, 0.859] vs shuffled OOS CI [0.410, 0.650] → **NOT distinguishable** (lo - shuffled_hi = -0.012).
* H500: real OOS CI [0.732, 0.833] vs shuffled OOS CI [0.542, 0.663] → **DISTINGUISHABLE** (lo - shuffled_hi = +0.068).

## 4. OOS precision-at-top-1% (lift over base rate)

In-sample held precision-at-top-1% = 27.6% at H500 (lift 6.86×).  Does this hold OOS?

| H | base rate | precision@top-1% OOS (day-clustered) | lift |
|---|---|---|---|
| H100 | 0.0115 | 0.0000 [0.0000, 0.0000] | 0.00 |
| H500 | 0.0627 | 0.2543 [0.0000, 0.5556] | 4.06 |

## 5. Per-symbol OOS distribution (≥ 3 cascades, AUC > 0.65 OOS @ H500)

Symbols with ≥ 3 OOS cascades at the horizon of interest (descriptive only; multiple-comparisons not adjusted).

| symbol | H | n_cascades | AUC OOS (day-clustered) | AUC > 0.65 OOS? |
|---|---|---|---|---|
| BTC | H100 | 4 | 0.674 [0.327, 0.921] | YES |
| SOL | H100 | 3 | 0.619 [0.428, 0.843] | NO |
| ETH | H100 | 8 | 0.550 [0.257, 0.739] | NO |
| HYPE | H100 | 3 | 0.486 [0.000, 0.820] | NO |
| SUI | H500 | 4 | 0.971 [0.948, 0.985] | YES |
| AVAX | H500 | 3 | 0.949 [0.917, 0.972] | YES |
| PENGU | H500 | 3 | 0.924 [0.854, 0.977] | YES |
| XRP | H500 | 8 | 0.770 [0.716, 0.852] | YES |
| BNB | H500 | 3 | 0.726 [0.600, 0.836] | YES |
| ENA | H500 | 3 | 0.673 [0.558, 0.784] | YES |
| SOL | H500 | 9 | 0.612 [0.515, 0.769] | NO |
| ETH | H500 | 23 | 0.590 [0.434, 0.745] | NO |
| AAVE | H500 | 4 | 0.585 [0.066, 0.942] | NO |
| HYPE | H500 | 14 | 0.531 [0.300, 0.728] | NO |
| BTC | H500 | 20 | 0.529 [0.382, 0.729] | NO |

**6 / 11 per-symbol cells clear AUC > 0.65 OOS at H500.**

## 6. AVAX OOS (held out from v1 contrastive training)

AVAX OOS at H500: AUC = 0.949 [0.917, 0.972]  (n_cascades = 3).  AVAX was excluded from v1 contrastive training; the cascade LR did not use that encoder, so AVAX is just another per-symbol cell here.

## 7. Verdict (per decision matrix)

**GENERALIZES.**  H500 OOS AUC > 0.75, CI lower bound > 0.65, distinguishable from shuffled-OOS baseline.  The Apr 1-13 in-sample signal extends to Apr 14-26.  Precision-at-top-1% OOS = 0.2543.  Encoder retrain on the merged Apr 1-26 dataset (~150 cascades) is worth committing GPU compute to.

## 8. Methodological flags

* **Day-clustered bootstrap is the binding test.**  Per-window bootstrap on tightly clustered cascade data understates uncertainty (prior commit `e2715ec` proved this).  All AUC and precision CIs in this writeup resample the OOS days with replacement.

* **Single LR fit, no fold-CV.**  Apr 14-26 is genuinely held-out, so the protocol is fit-once-on-train, predict-once-on-test.  No model selection, no hyperparameter search.

* **Apples-to-apples with in-sample.**  Same 83-dim flat features, same `LogisticRegression(class_weight='balanced', C=1.0)`, same cascade label definition (`cause IN ('market_liquidation', 'backstop_liquidation')`).  Per-symbol minimum is 3 cascades on OOS (vs 5 in-sample) because the OOS window is shorter.

* **Holdout permanently consumed.**  The Apr 14+ data was loaded by this script via the unsafe-loader code path.  No untouched cascade-labeled holdout remains; future OOS evaluation requires new data accrual or merged-dataset splitting.

* **Distribution-shift caveat.**  If Apr 14-26 has a structurally different cascade frequency or volatility regime than Apr 1-13, the OOS gap can reflect domain shift rather than overfit.  Compare base rate and n_cascades across the two folds before drawing strong conclusions.

* **Shuffled-OOS AUC > 0.50 is expected under cascade contagion.**  The shuffled-OOS baseline permutes labels WITHIN each day (preserving per-day cascade count).  Day-clustered bootstrap resamples days with replacement: if the LR's day-mean prediction correlates with the day's cascade rate (volatility regime), the shuffled AUC drifts above 0.5 even though the within-day rank carries no signal.  The distinguishability test (real-lo > shuffled-hi) correctly accounts for this — the real AUC must exceed the contagion floor, not the 0.5 chance line.

_OOS pipeline ran in 16.2 s.  CPU-only.  Apr 14+ holdout permanently consumed by this run._

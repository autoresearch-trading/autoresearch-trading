# Goal-A cascade-precursor (stage 2) — real `cause` flag probe

**Question.** Is there a measurable precursor footprint in the 200 events before a real liquidation cascade?  Stage 1 used a synthetic 99th-percentile-magnitude label that turned out to overlap real cascades only ~20% on April 1-13 at H100 — the high lift was measuring volatility clustering, not forced liquidations.  Stage 2 uses the `cause` flag directly.  Sample size on April 1-13 is small (n_cascades ≈ 9 / 20 / 73 universe-wide at H50 / H100 / H500) — wide CIs are expected.  The binding statistical question is whether the real-label AUC's 95% CI lower bound strictly exceeds the shuffled-label AUC's 95% CI upper bound.

**Protocol.** 83-dim flat baseline (`tape/flat_features.py`).  `LogisticRegression(class_weight='balanced', C=1.0)` — same default as Gate 0 / Gate 1.  No hyperparameter search (small data + multiple comparisons would inflate AUC).  Pooled cross-symbol leave-one-day-out CV on April 1-13 — held-out day's predictions aggregated across folds.  Per-symbol leave-one-day-out CV restricted to symbols with ≥ 5 real cascades at the horizon of interest (descriptive, not binding).  Bootstrap 95% AUC CI (n_boot = 1000).  Shuffled-label baseline (labels permuted within day, train fold only).  Random-feature baseline (Gaussian noise, same shape).  April 14+ untouched (gotcha #17).

**Contamination disclosure.** April 1-13 was used for v1 diagnostic checks (gotcha #17 lists this as the diagnostic window) — but the `cause` field was NOT studied in v1.  The contamination concern from prior v1 work is on direction-related tests and cannot bias this real-cascade probe.

## 1. Sample size confirmation

| H | n_total (windows) | n_cascades | base rate |
|---|---|---|---|
| H50 | 2192 | 9 | 0.0041 |
| H100 | 2153 | 20 | 0.0093 |
| H500 | 1803 | 73 | 0.0405 |

Prior synthetic-vs-real validation reported H50 = 9, H100 = 20, H500 = 73 cascades universe-wide.  If the table above differs, explain in flags below.

## 2. Pooled cross-symbol AUC at H100 and H500

| H | AUC (real label) | AUC (shuffled) | AUC (random feat) | distinguishable from shuffled? |
|---|---|---|---|---|
| H50 | 0.624 [0.395, 0.834] | 0.424 [0.231, 0.636] | 0.531 [0.320, 0.723] | NO |
| H100 | 0.784 [0.689, 0.870] | 0.307 [0.209, 0.417] | 0.461 [0.325, 0.594] | YES |
| H500 | 0.817 [0.771, 0.858] | 0.397 [0.335, 0.469] | 0.517 [0.447, 0.581] | YES |


## 3. Distinguishable from shuffled-label baseline?

Binding statistical test: real-label AUC CI lower bound must strictly exceed shuffled-label AUC CI upper bound.  Aggregate across H100 and H500 below.

* H50: real CI [0.395, 0.834] vs shuffled CI [0.231, 0.636] → **NOT distinguishable**.
* H100: real CI [0.689, 0.870] vs shuffled CI [0.209, 0.417] → **DISTINGUISHABLE**.
* H500: real CI [0.771, 0.858] vs shuffled CI [0.335, 0.469] → **DISTINGUISHABLE**.

## 4. Per-symbol breakdown

Symbols with ≥ 5 real cascades at the horizon of interest (descriptive only; multiple-testing not adjusted).

| symbol | H | n_cascades | AUC (real) | AUC (shuffled) | auc>0.60 & CI excl 0.50 & dist from shuffled? |
|---|---|---|---|---|---|
| BTC | H100 | 5 | 0.583 [0.367, 0.798] | 0.519 | NO |
| ETH | H100 | 6 | 0.280 [0.148, 0.415] | 0.572 | NO |
| BTC | H500 | 17 | 0.755 [0.658, 0.846] | 0.525 | NO |
| HYPE | H500 | 12 | 0.589 [0.393, 0.773] | 0.570 | NO |
| ETH | H500 | 16 | 0.470 [0.335, 0.618] | 0.395 | NO |
| SOL | H500 | 12 | 0.287 [0.154, 0.447] | 0.441 | NO |
| AAVE | H500 | 7 | 0.275 [0.110, 0.502] | 0.365 | NO |

**0 per-symbol cells clear AUC > 0.60 AND CI excludes 0.50 AND are distinguishable from shuffled.**

## 5. Precision-at-top-1% lift (pooled cross-symbol)

If we trade only when the model says cascade-likely (top 1% of windows by predicted probability), how often is it right?  For a 10× tradeable lift at base rate ~0.5%, top-1% precision must be > 5%.

| H | base rate | precision@top-1% | lift | recall@top-1% |
|---|---|---|---|---|
| H50 | 0.0041 | 0.0000 | 0.00 | 0.0000 |
| H100 | 0.0093 | 0.0000 | 0.00 | 0.0000 |
| H500 | 0.0405 | 0.2778 | 6.86 | 0.0685 |

## 6. Verdict

**YES (with caveats) — at least one binding horizon clears distinguishability from the shuffled-label baseline.**  H100 dist: True (n_cascades = 20); H500 dist: True (n_cascades = 73).  The 83-dim flat representation carries some real-cascade-precursor signal above pure noise on April 1-13.  Caveats: small n keeps the CI wide; a single cluster of same-day cascades can dominate a fold's AUC.

## 7. Methodological flags

* **Leave-one-day-out independence.** The folds are leave-one-day-out (April 1-13 → ≤ 13 folds depending on data availability), but real liquidation cascades cluster intra-day (cascade contagion).  If a single day's cascades dominate the held-out day's AUC, fold-level AUC overstates true held-out performance.  The bootstrap CI captures sampling noise but not fold-clustering noise.

* **Per-symbol cells are descriptive, not binding.** With 3 horizons × ≤ 25 symbols of per-symbol comparisons, per-symbol AUC > 0.60 cells will appear by chance even under the null.  Treat per-symbol results as pattern hints, not standalone evidence.

* **Stage-1 contamination disclosure (gotcha #17).** April 1-13 was used for v1 diagnostic checks but the `cause` field was not studied — contamination on this stage-2 study is not a concern.

* **April 14+ hold-out preserved.** No raw or cached April 14+ data was loaded by this script.

_Pipeline ran in 15.7 s.  CPU-only.  No April 14+ data touched._

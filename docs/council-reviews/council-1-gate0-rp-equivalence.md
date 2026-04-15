# Council-1 (Lopez de Prado methodology) — Gate 0 / RP Equivalence Review

**Date:** 2026-04-15
**Reviewer:** council-1
**Inputs:** `scripts/run_gate0.py` (SHA `9de25c2`), `scripts/run_random_baseline.py` (SHA `c0bee9f`), `docs/experiments/gate0-baseline.md`, `docs/experiments/gate0-random-control.md`

## 1. What the finding says

PCA(20) fitted on training data and a frozen random Gaussian projection (seed=42, 85×20, column-normalized) produce statistically indistinguishable mean accuracy at every horizon: H10 Δ +0.0003, H50 +0.0065, H100 +0.0058. The two outliers (2Z H10=0.6208, CRV H10=0.6159) are matched or exceeded by RP (0.6203, 0.6299).

**Mechanistic interpretation:** by Johnson–Lindenstrauss, random projections approximately preserve pairwise distances when reducing 85→20. The LR's solution space is nearly identical under PCA vs RP. PCA adds no adaptive value; the accuracy comes from the LR's ability to find a decision boundary in a generic 20-dim subspace, not from data-adaptive dimensionality reduction.

## 2. Does Gate 0 hold up?

No. Gate 0 as defined measures "can LR find signal in these flat features" rather than "does adaptive dimensionality reduction help." Any model that projects the flat features reasonably will match. The 51.4% threshold is unsupported: at H100, only 7/25 symbols clear it under PCA, 5/25 under RP — majority are within noise of chance.

**PSR / statistical check missing.** With 3 folds × ~500 test windows per fold, standard error ≈ √(0.25/500) = 0.022. A 0.510 vs 0.500 difference is within one standard error. No paired t-test or Wilcoxon signed-rank reported for PCA vs RP — the stated deltas lack statistical support.

## 3. Missing control — majority-class predictor

The most important gap. For each test fold, predict the training-fold majority class. If this matches PCA, Gate 0 measures class imbalance, not signal. The spec mentioned a "random untrained encoder + linear probe" variant but did not include majority class.

**Required Gate 0 control set, priority order:**
1. Majority-class predictor (per fold, training-fold majority).
2. Stratified random predictor (predict with training-fold class probabilities).
3. Temporal-stratified base rate (majority class of test period — most honest variant).
4. Random untrained CNN encoder + linear probe.

## 4. Implications for Gate 1/2

The current framing — "exceed Gate 0 by 0.5pp on 15+ symbols" — is insufficient given PCA ≈ RP. The true bar:

1. **CNN probe > majority-class** per symbol, signed-rank p < 0.05 on fold accuracies.
2. **Gate 1 must evaluate on April 1–13 held-out period**, not the same Oct–Mar data that defined Gate 0.
3. **Representation diagnostics gate alongside accuracy**: symbol identity <20%, CKA >0.7, Wyckoff label probes. A CNN that matches PCA on accuracy but fails these is just re-implementing PCA in higher dimensions.

## 5. Threshold recommendation

- Replace absolute 51.4% at Gate 0 with "must exceed majority-class by >0 on 15+/25 symbols."
- Keep 51.4% at Gate 1 but ONLY on held-out April data, AND pair it with "CNN beats RP-control by ≥1.0pp."
- Apply **balanced accuracy symmetrically** to H10/H50/H100 (the H500 argument applies with reduced severity at shorter horizons too).

## 6. Immediate required actions

1. Run majority-class predictor against cache (per symbol × per horizon, same walk-forward).
2. Report per-fold accuracy standard deviation in the JSON output.
3. Re-render the Gate 0 summary in balanced accuracy terms (data is already in JSON).
4. Commit a spec amendment formalizing the above.

## Summary

The RP equivalence is real and methodologically critical. Gate 0 currently measures signal in flat features, not adaptive structure. Without a majority-class baseline, there is no noise floor and no guarantee that >50% accuracy reflects information rather than class imbalance. The spec needs Gate 0 revision; the 51.4% threshold belongs at Gate 1 only, and only on truly held-out April data.

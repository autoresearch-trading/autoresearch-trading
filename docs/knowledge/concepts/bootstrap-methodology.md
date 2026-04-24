---
title: Bootstrap CI + Shuffled-Null Methodology
topics: [evaluation, statistics, falsifiability, methodology]
sources:
  - docs/experiments/step5-gate3-triage.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
last_updated: 2026-04-24
---

# Bootstrap CI + Shuffled-Null Methodology

## What It Is

The statistical protocol for turning a small-n point-estimate balanced
accuracy into a falsifiable reading. Three components, all required on any
single-symbol or single-month probe:

1. **Per-cell 1000-resample percentile bootstrap 95% CI.** For each test fold,
   resample predictions-with-labels 1000× with replacement, recompute balanced
   accuracy, report [2.5%, 97.5%] percentile interval.
2. **N=50 shuffled-labels null.** 50 independent permutations of test labels,
   evaluate predictor, report null mean and σ (±2σ upper bound). Replaces the
   single-seed shuffled control which was too noisy at small n.
3. **Per-cell class prior reported.** Test-fold fraction of the positive class.
   Balanced accuracy on a 50/50 split is statistically different from a 75/25
   split at fixed n; without the prior the reader cannot interpret the CI.

## Why Required

Council-5's gate3 falsifiability review (2026-04-24) established that without
CIs the Gate 3 AVAX writeup was making eyeball claims about significance. At
stride=200 n=120, binomial SE on balanced accuracy is ~0.065 per class → 95%
CI ~±0.13 on balanced accuracy. Any single-cell point estimate inside ±0.13 of
0.500 is indistinguishable from chance. Reporting point estimates alone is the
same epistemology that produced the original stride=200 "Feb H100 pass" at
0.575 (encoder +7.9pp over PCA) that **did not replicate** under stride=50
higher-density evaluation.

Shuffled-null at single seed was equivalently brittle: the gate3 writeup's
Apr H500 shuffled=0.700 was one tail draw of a distribution with σ≈0.09 at n=60.
N=50 shuffles brought the null back to μ=0.4995, σ=0.029, retroactively
explaining the anomaly as "one bad draw."

## Our Implementation

`scripts/avax_gate3_probe.py` (commit `ea07bda`) implements all three:

```
for cell in cells:
    for predictor in [encoder_lr, pca_lr, rp_lr, majority, shuffled_pca_lr]:
        y_pred = predictor.predict(X_test)
        boot_accs = [balanced_accuracy_score(y_true[idx], y_pred[idx])
                     for idx in bootstrap_resample(n_test, n=1000)]
        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])

    # N=50 shuffle (replaces single-seed)
    null_accs = [balanced_accuracy_score(shuffle(y_true), y_pred_pca)
                 for _ in range(50)]
    null_mean, null_sigma = np.mean(null_accs), np.std(null_accs)

    # Always report
    class_prior = y_true.mean()
```

## Gotchas

1. **Bootstrap at stride=50 overlapping windows is not fully independent.** 75%
   overlap between consecutive windows means effective n < raw n. The CI width
   is conservative (slightly narrow); treat as an upper bound on confidence.
2. **Class prior is not the same as fold balance.** At short horizons (H100)
   drift-dominated periods can skew train/test priors independently. Report
   both train and test priors if they differ by > 5pp.
3. **Shuffled-null σ scales with 1/sqrt(n_test).** Apr AVAX n=60 had σ=0.09; Feb
   AVAX n=460 had σ=0.021. Budget shuffle count accordingly — N=50 is
   sufficient above n=200, rise to N=200 below n=100.
4. **Multiple testing.** Running 5 predictors × 2 horizons × 3 months × 5
   symbols is 150 cells; at α=0.05 expect ~7.5 spurious "separations" by chance.
   The 1/20 surrogate sweep CI-separation rate
   ([surrogate sweep experiment](../experiments/per-symbol-surrogate-sweep.md))
   is the empirical null baseline.
5. **Narrow CI does not make a non-separated result a pass.** Encoder CI
   [0.488, 0.579] and PCA CI [0.502, 0.594] overlap → pre-registered threshold
   NOT cleared at CI-aware rigor, even if both point estimates look "around the
   mark."

## When Binding

- Any single-symbol single-month probe (Gate 3, surrogate sweep, AAVE control).
- Any claim of "encoder > baseline" on n_test < 2000.
- Any pre-registered threshold crossing where the raw margin is < 3pp.
- Re-activation of retired Gate 3 (see
  [re-activation criteria](../decisions/gate3-retired-to-informational.md)
  condition (b)).

Large-pool Gate 1 evaluations (n_test ≈ 16K across 24 symbols per month) do
not require per-cell bootstrap at the same rigor because the per-symbol CI is
already within the Gate-1 sub-condition structure — but per-cell CIs remain
recommended and cheap.

## Related Concepts

- [Underpowered Single-Symbol Probe](underpowered-single-symbol-probe.md)
- [Gate 3 retirement](../decisions/gate3-retired-to-informational.md)
- [Gate 3 triage experiment](../experiments/gate3-avax-triage.md)

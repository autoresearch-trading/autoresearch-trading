# Summary of changes
## Formalization Complete: Hawkes Branching Ratio from Fixed-Size Batch Arrival Rates

### File: `RequestProject/HawkesBatchEstimator.lean`

All 11 theorems are formally proved in Lean 4 with no `sorry` statements and only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

### Key Findings — Two Claims are FALSE

**Claim 4 (Equivalence condition) is FALSE.** The original claim stated:
> Var(R)/E[R] = 1/(1-n)² iff Var(T_B)/E[T_B]² = 1/(B · E[λ])

A formal counterexample is provided (`claim4_original_is_false`) with B=2, μ=3, n=1/2. The **correct** equivalence condition is:
> Var(R)/E[R] = 1/(1-n)² iff Var(T_B)/E[T_B]² = 1/((1-n)² · E[λ])

This is proved as `equivalence_condition_corrected`.

**Claim 2 (Rate formula) needed correction.** The original formula omitted a factor of E[λ]:
- ❌ Original: `el * varT / meanT² = (1 + 2n/(1-n)²) / B`  
- ✅ Correct: `el * varT / meanT² = el * (1 + 2n/(1-n)²) / B`

This means the variance/mean ratio of batch rates depends on μ through E[λ] = μ/(1-n), not just on n alone. The batch-rate estimator thus requires knowledge of the mean rate to extract n, unlike the fixed-time-window case.

### What Was Proved

| Theorem | Claim | Status |
|---------|-------|--------|
| `expected_duration_eq` | Claim 1(a) | ✅ Proved |
| `equivalence_condition_corrected` | Claim 4 (corrected) | ✅ Proved |
| `claim4_original_is_false` | Claim 4 (original) | ✅ Disproved |
| `rate_variance_mean_formula` | Claim 2 (corrected) | ✅ Proved |
| `estimator_at_poisson` | Claim 5 (r=1 → n̂=0) | ✅ Proved |
| `estimator_in_unit_interval` | Claim 5 (r>1 → n̂∈(0,1)) | ✅ Proved |
| `estimator_monotone` | Claim 5 (monotonicity) | ✅ Proved |
| `estimator_continuous_at_ratio` | Claim 3 (consistency) | ✅ Proved |
| `adjusted_estimator_at_baseline` | Claim 5 (r=C → n̂=0) | ✅ Proved |
| `adjusted_estimator_in_interval` | Claim 5 (r>C → n̂∈(0,1)) | ✅ Proved |
| `branching_ratio_recovery` | Main theorem (n̂ recovers n) | ✅ Proved |

### Approach

The probabilistic foundations (Hawkes processes, Wald's identity, delta method, SLLN) are not in Mathlib, so they are abstracted as hypotheses on the moments. The formalization captures and proves all the algebraic/analytic content: the moment identities, the estimator properties, the continuity for consistency, and the branching ratio recovery theorem. The two errors in the original claims were discovered and formally demonstrated.
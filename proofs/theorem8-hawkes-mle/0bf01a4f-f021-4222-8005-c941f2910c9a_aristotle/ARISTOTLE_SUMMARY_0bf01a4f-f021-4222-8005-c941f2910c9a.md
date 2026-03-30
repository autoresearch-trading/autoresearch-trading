# Summary of changes
Successfully formalized and proved all claims about Hawkes Process MLE Consistency and Branching Ratio Estimation in Lean 4, in the file `RequestProject/HawkesProcess.lean`.

## Definitions formalized
- `hawkesIntensity`: The intensity λ(t) = μ + α · Σ exp(-β·(t - t_i))
- `hawkesIntegral`: The integral ∫₀ᵀ λ(s) ds
- `hawkesLogLikelihood`: The log-likelihood ℓ(μ,α,β)
- `branchingRatio`: n = α/β
- `branchingRatioEstimator`: n̂ = 1 - 1/√r (method-of-moments estimator)
- `recursiveR`: The O(N) recursive computation R_i
- `intensitySum`: The O(N²) direct sum for comparison
- `expectedIntensity`: μ/(1 - α/β)

## Theorems proved (16 total, all sorry-free)

**Claim 1(a) — Intensity positivity:**
- `sum_exp_nonneg`: The exponential sum term is non-negative
- `hawkes_intensity_pos`: λ(t) > 0 for all t
- `hawkes_intensity_ge_mu`: λ(t) ≥ μ

**Claim 1(b) — Integral well-definedness:**
- `one_minus_exp_nonneg`: Each 1 - exp(-β(T-tᵢ)) ≥ 0
- `one_minus_exp_le_one`: Each 1 - exp(-β(T-tᵢ)) ≤ 1
- `hawkes_integral_nonneg`: The integral term is non-negative

**Claim 2 — Recursive computation:**
- `recursive_computation_exact`: R_i equals the exact sum Σ_{j<i} exp(-β(tᵢ-tⱼ)) (proved by induction)

**Claim 3 — Score equations:**
- `harmonic_mean_from_score`: The first score equation Σ 1/λ(tᵢ) = T implies the harmonic mean equals N/T

**Claim 4 — Branching ratio estimator:**
- `branching_ratio_estimator_in_unit_interval`: n̂ ∈ (0,1) when r > 1
- `branching_ratio_estimator_poisson`: n̂ = 0 when r = 1
- `branching_ratio_estimator_tendsto_one`: n̂ → 1 as r → ∞
- `branching_ratio_estimator_consistent`: n̂ = n when r = 1/(1-n)²

**Claim 5 — Stationarity boundary:**
- `expected_intensity_pos`: E[λ] > 0 when α < β (stationary)
- `expected_intensity_not_pos_of_supercritical`: E[λ] ≤ 0 when α ≥ β (supercritical)
- `branching_ratio_stationary_iff`: (0 < α/β < 1) ↔ (α < β)
- `variance_mean_ratio_gt_one`: 1/(1-n)² > 1 when 0 < n < 1

All proofs use only standard axioms (propext, Classical.choice, Quot.sound).
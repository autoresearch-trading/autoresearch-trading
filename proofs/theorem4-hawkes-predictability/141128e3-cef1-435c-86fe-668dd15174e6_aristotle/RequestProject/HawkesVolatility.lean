/-
# Hawkes Self-Excitation: Formalization of Deterministic Mathematical Content

## Summary of Analysis

The original "theorem" contains three claims about Hawkes processes, volatility clustering,
and trading strategies. Upon rigorous mathematical analysis, we find:

### What IS formalizable and provable:
- The geometric decay formula from Claim 1 (as a purely analytic statement about sequences)
- The convergence properties of the predicted variance formula
- The monotonicity and limiting behavior of the correlation bound from Claim 3
- The numerical evaluation at hawkes_ratio = 0.8

### What is NOT formalizable as stated:
- **Claim 1**: Calls α/β > 0.5 "supercritical" — this is incorrect terminology.
  In Hawkes process theory, subcritical is α/β < 1, critical is α/β = 1,
  supercritical is α/β > 1. The formula for E[σ²(t+k) | λ(t)] conflates
  intensity prediction with realized variance prediction, which requires
  additional modeling assumptions not stated.

- **Claim 2**: Entirely circular. "Strictly positive when τ is chosen such that
  the Hawkes clustering signal is not yet priced into implied vol" assumes
  the conclusion (market inefficiency). This is an empirical claim about
  markets, not a mathematical theorem.

- **Claim 3**: "price_level_absorption" is not a standard mathematical object.
  The correlation bound 1 - exp(-r/(1-r)) is presented without derivation
  and the relationship between an undefined "absorption feature" and returns
  cannot be formally stated.

### What we prove below:
We formalize and prove the deterministic analytic facts that form the
mathematical backbone of the claims:

1. `predicted_variance_decay`: The predicted variance formula converges to
   the base variance as the horizon k → ∞ (Claim 1 core).

2. `predicted_variance_predictable`: When λ(t) ≠ μ and hawkes_ratio ∈ (0,1),
   the predicted variance differs from base variance for all finite k (Claim 1 predictability).

3. `correlation_bound_mono`: The bound function 1 - exp(-r/(1-r)) is strictly
   increasing on (0,1) (Claim 3 structure).

4. `correlation_bound_limit`: The bound approaches 1 as r → 1⁻ (Claim 3 limiting behavior).

5. `correlation_bound_at_08`: At r = 0.8, the bound exceeds 0.98 (Claim 3 numerical).
-/

import Mathlib

open Real Filter Topology

noncomputable section

/-! ## Predicted Variance Formula (Claim 1 Core)

We define the predicted variance formula from Claim 1 as a function of
the branching ratio r = α/β, the intensity ratio λ₀/μ, the base variance,
and the prediction horizon k.
-/

/-- The predicted variance formula from Claim 1:
  E[σ²(t+k) | λ(t)] = σ²_base · (1 + r^k · (λ₀/μ - 1))
where r = hawkes_ratio = α/β, and λ₀/μ is the current intensity ratio. -/
def predictedVariance (σ2_base : ℝ) (r : ℝ) (intensity_ratio : ℝ) (k : ℕ) : ℝ :=
  σ2_base * (1 + r ^ k * (intensity_ratio - 1))

/-
PROBLEM
When 0 < r < 1, the predicted variance converges to σ²_base as k → ∞.
This is the core mathematical content of Claim 1: predictability decays geometrically.

PROVIDED SOLUTION
Unfold predictedVariance. We need to show σ2_base * (1 + r^k * (intensity_ratio - 1)) → σ2_base. Since 0 < r < 1, r^k → 0, so r^k * (intensity_ratio - 1) → 0, so 1 + r^k * (intensity_ratio - 1) → 1, so the product → σ2_base * 1 = σ2_base. Use tendsto_pow_atTop_nhds_zero_of_lt_one and arithmetic limit lemmas.
-/
theorem predicted_variance_decay (σ2_base : ℝ) (r : ℝ) (intensity_ratio : ℝ)
    (hr0 : 0 < r) (hr1 : r < 1) :
    Tendsto (fun k => predictedVariance σ2_base r intensity_ratio k)
      atTop (𝓝 σ2_base) := by
  convert Tendsto.const_mul σ2_base ( tendsto_const_nhds.add ( tendsto_pow_atTop_nhds_zero_of_lt_one hr0.le hr1 |> Filter.Tendsto.mul_const _ ) ) using 1 ; ring!;

/-
PROBLEM
When the intensity ratio differs from 1 (i.e., λ(t) ≠ μ) and 0 < r < 1,
the predicted variance is strictly different from σ²_base for every finite k.
This formalizes "STRICTLY predictable for k steps ahead whenever λ(t) ≠ μ".

PROVIDED SOLUTION
Unfold predictedVariance. We need predictedVariance ≠ σ2_base, i.e. σ2_base * (1 + r^k * (intensity_ratio - 1)) ≠ σ2_base. Since σ2_base ≠ 0, it suffices to show 1 + r^k * (intensity_ratio - 1) ≠ 1, i.e. r^k * (intensity_ratio - 1) ≠ 0. Since 0 < r, r^k > 0, so r^k ≠ 0. And intensity_ratio ≠ 1 means intensity_ratio - 1 ≠ 0. Product of two nonzero reals is nonzero.
-/
theorem predicted_variance_predictable (σ2_base : ℝ) (r : ℝ) (intensity_ratio : ℝ) (k : ℕ)
    (hσ : σ2_base ≠ 0) (hr0 : 0 < r) (hr1 : r < 1) (hint : intensity_ratio ≠ 1) :
    predictedVariance σ2_base r intensity_ratio k ≠ σ2_base := by
  unfold predictedVariance;
  simp_all +decide [ sub_eq_iff_eq_add ];
  aesop

/-! ## Correlation Bound Function (Claim 3 Core)

We define the correlation bound from Claim 3:
  f(r) = 1 - exp(-r / (1 - r))

This is a purely analytic function whose properties we can rigorously prove.
Note: The original claim that this bounds the correlation between an
undefined "absorption feature" and future returns is NOT formalizable,
but the properties of the bound function itself ARE provable.
-/

/-- The correlation bound function from Claim 3:
  f(r) = 1 - exp(-r / (1 - r)) -/
def correlationBound (r : ℝ) : ℝ :=
  1 - Real.exp (-(r / (1 - r)))

/-
PROBLEM
The correlation bound is strictly between 0 and 1 for r ∈ (0, 1).

PROVIDED SOLUTION
Unfold correlationBound. We need 0 < 1 - exp(-r/(1-r)) and 1 - exp(-r/(1-r)) < 1. For the upper bound: exp(-r/(1-r)) > 0, so 1 - exp(...) < 1. For the lower bound: since r > 0 and 1 - r > 0, we have r/(1-r) > 0, so -r/(1-r) < 0, so exp(-r/(1-r)) < 1, so 1 - exp(-r/(1-r)) > 0.
-/
theorem correlation_bound_range (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) :
    0 < correlationBound r ∧ correlationBound r < 1 := by
  exact ⟨ sub_pos.2 <| Real.exp_lt_one_iff.2 <| by exact neg_lt_zero.2 <| div_pos hr0 <| sub_pos.2 hr1, sub_lt_self _ <| Real.exp_pos _ ⟩

/-
PROBLEM
The argument r/(1-r) is strictly increasing on (0,1), which combined
with exp being increasing shows the bound is strictly increasing.

PROVIDED SOLUTION
correlationBound r = 1 - exp(-r/(1-r)). The function g(r) = r/(1-r) is strictly increasing on (0,1) (derivative is 1/(1-r)^2 > 0). The function h(x) = 1 - exp(-x) is strictly increasing. So the composition h ∘ g is strictly increasing on (0,1). Use StrictMonoOn and compose monotonicity.
-/
theorem correlation_bound_strictly_increasing :
    StrictMonoOn correlationBound (Set.Ioo 0 1) := by
  exact fun x hx y hy hxy => sub_lt_sub_left ( Real.exp_lt_exp.mpr <| neg_lt_neg <| by rw [ div_lt_div_iff₀ ] <;> nlinarith [ hx.1, hx.2, hy.1, hy.2 ] ) _

/-
PROBLEM
As r → 1⁻, the correlation bound approaches 1.

PROVIDED SOLUTION
As r → 1⁻, r/(1-r) → +∞, so -r/(1-r) → -∞, so exp(-r/(1-r)) → 0, so 1 - exp(-r/(1-r)) → 1. Use Tendsto composition: r/(1-r) tends to atTop as r → 1 from below, then exp(-x) → 0 as x → +∞, then 1 - 0 = 1.
-/
theorem correlation_bound_tendsto_one :
    Tendsto correlationBound (𝓝[<] 1) (𝓝 1) := by
  -- We'll use the fact that as $r$ approaches $1$ from the left, $r / (1 - r)$ approaches infinity.
  have h_lim : Filter.Tendsto (fun r : ℝ => r / (1 - r)) (nhdsWithin 1 (Set.Iio 1)) Filter.atTop := by
    norm_num [ Filter.tendsto_atTop, Filter.eventually_inf_principal, nhdsWithin ];
    exact fun b => Metric.eventually_nhds_iff.2 ⟨ ( |b| + 1 ) ⁻¹, by positivity, fun x hx hx' => by rw [ le_div_iff₀ ] <;> cases abs_cases b <;> nlinarith [ mul_inv_cancel₀ ( by linarith : ( |b| + 1 : ℝ ) ≠ 0 ), abs_lt.1 hx ] ⟩;
  convert tendsto_const_nhds.sub ( Real.tendsto_exp_atBot.comp <| Filter.tendsto_neg_atTop_atBot.comp h_lim ) using 2 ; norm_num [ correlationBound ]

/-
PROBLEM
At r = 0.8, the bound is at least 0.98.
This verifies the numerical claim in the original theorem.
Proof: r/(1-r) = 0.8/0.2 = 4, so bound = 1 - exp(-4) ≈ 1 - 0.0183 = 0.9817 > 0.98.

PROVIDED SOLUTION
correlationBound 0.8 = 1 - exp(-0.8/0.2) = 1 - exp(-4). We need 1 - exp(-4) ≥ 0.98, i.e. exp(-4) ≤ 0.02. Since exp(-4) = 1/exp(4) and exp(4) ≥ 50 (because exp(4) ≈ 54.598), we have exp(-4) ≤ 1/50 = 0.02. To bound exp(4): use exp(x) ≥ 1 + x + x²/2 + x³/6 + x⁴/24, which at x=4 gives 1 + 4 + 8 + 32/3 + 256/24 = 1 + 4 + 8 + 10.667 + 10.667 ≈ 34.33. Actually we need exp(4) ≥ 50. Use more terms or use that exp(1) ≥ 2.718 so exp(4) ≥ 2.718^4. Or simply: the Taylor polynomial of degree 6 at x=4 gives 1 + 4 + 8 + 32/3 + 32/3 + 128/15 + 256/45 which exceeds 50. Try using native_decide or norm_num extensions if available. Alternative: show exp 4 ≥ 50 using exp_ge_one_add_of_nonneg repeatedly or the power series lower bound.
-/
theorem correlation_bound_at_08 :
    correlationBound 0.8 ≥ 0.98 := by
  unfold correlationBound ; norm_num [ Real.exp_neg ] ; ring_nf ; norm_num; (
  field_simp;
  have := Real.exp_one_gt_d9.le ; norm_num at * ; rw [ show Real.exp 4 = ( Real.exp 1 ) ^ 4 by rw [ ← Real.exp_nat_mul ] ; norm_num ] ; nlinarith [ pow_le_pow_left₀ ( by positivity ) this 4 ] ;)

/-! ## Hawkes Process: Branching Ratio Properties

Additional facts about the branching ratio that are purely algebraic. -/

/-
PROBLEM
The critical threshold for Hawkes processes is α/β = 1, not 0.5 as
incorrectly stated in the original claim. A Hawkes process is stationary
(subcritical) iff α/β < 1.

We prove: the expected number of descendants per event in a Hawkes process
with kernel α·exp(-β·t) is exactly α/β (the branching ratio).
This is ∫₀^∞ α·exp(-β·t) dt = α/β.

PROVIDED SOLUTION
∫ t in Ioi 0, α * exp(-(β*t)) = α * ∫ t in Ioi 0, exp(-(β*t)). The integral ∫₀^∞ exp(-β*t) dt = 1/β for β > 0. So the result is α * (1/β) = α/β. Use integral_exp_neg_mul or similar Mathlib lemma for the exponential integral on (0, ∞). The key fact is that ∫ t in Ioi 0, exp(-(β*t)) dt = 1/β.
-/
theorem hawkes_branching_ratio_integral (α β : ℝ) (_hα : 0 < α) (hβ : 0 < β) :
    ∫ t in Set.Ioi 0, α * Real.exp (-(β * t)) = α / β := by
  rw [ MeasureTheory.integral_const_mul, div_eq_mul_inv ];
  have := integral_exp_neg_mul_rpow zero_lt_one hβ ; norm_num [ Real.rpow_neg_one ] at this ; aesop

/-
PROBLEM
The unconditional intensity of a stationary Hawkes process is μ/(1 - α/β)
when α/β < 1. We verify that this is well-defined and greater than μ.

PROVIDED SOLUTION
We need μ / (1 - α/β) > μ. Since α/β < 1 and α > 0, β > 0, we have 0 < α/β < 1, so 0 < 1 - α/β < 1. Since μ > 0 and 0 < 1 - α/β < 1, dividing μ by a positive number less than 1 gives a result greater than μ. Use div_lt_iff or field_simp + positivity/linarith.
-/
theorem hawkes_unconditional_intensity (μ α β : ℝ) (hμ : 0 < μ) (hα : 0 < α)
    (hβ : 0 < β) (hstable : α / β < 1) :
    μ / (1 - α / β) > μ := by
  rw [ gt_iff_lt, lt_div_iff₀ ] <;> nlinarith [ mul_div_cancel₀ α hβ.ne' ]

end
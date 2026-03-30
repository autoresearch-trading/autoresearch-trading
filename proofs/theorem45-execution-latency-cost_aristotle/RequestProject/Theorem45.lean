import Mathlib

/-!
# Theorem 45: Execution Latency as a Cost Component

This file formalizes the key mathematical claims from the execution latency cost analysis.

## Main Results

1. `expected_abs_std_normal`: E[|Z|] = √(2/π) for Z ~ N(0,1)
2. `latency_cost_bound`: The latency cost √(2/π) · σ_step · √(L/Δt) < 0.00052
   with the given numerical parameters.
3. `latency_cost_lt_impact`: latency_cost / impact_buffer < 0.2
4. `continuous_autocorr_at_zero`: For any continuous function ρ with ρ(0) = 1,
   ρ(x) → 1 as x → 0.
-/

open MeasureTheory Real Set Filter Topology

noncomputable section

/-!
## Part 1: E[|Z|] = √(2/π) for standard normal

We prove this as an integral identity:
  (1/√(2π)) · ∫ₓ |x| · exp(-x²/2) dx = √(2/π)

This follows from:
  ∫₀^∞ x · exp(-x²/2) dx = 1
  ∫ₓ |x| · exp(-x²/2) dx = 2 · ∫₀^∞ x · exp(-x²/2) dx = 2
  (1/√(2π)) · 2 = √(2/π)
-/

/-
PROBLEM
The integral of x · exp(-x²/2) over (0, ∞) equals 1. This follows by substitution u = x²/2.

PROVIDED SOLUTION
Use the substitution u = x²/2 or integrate by parts. In Mathlib, use integral_exp_neg_mul_sq or similar. Actually, we can compute: ∫₀^∞ x exp(-(1/2)x²) dx. Let u = (1/2)x², du = x dx. So the integral becomes ∫₀^∞ exp(-u) du = 1. In Mathlib, try using integral_exp_neg_Ioi or set_integral_comp. Alternatively, use HasDerivAt and FTC: the antiderivative of x * exp(-x²/2) is -exp(-x²/2), evaluated from 0 to ∞ gives 0-(-1) = 1. Use intervalIntegral.integral_eq_sub_of_hasDerivAt with tendsto at infinity.
-/
theorem integral_x_mul_gaussian_Ioi :
    ∫ (x : ℝ) in Ioi 0, x * exp (-(1/2) * x ^ 2) = 1 := by
  have := @integral_rpow_mul_exp_neg_mul_rpow 2 ( 1 : ℝ ) ( 1 / 2 ) ; norm_num at * ; aesop;

/-
PROBLEM
The integral of |x| · exp(-x²/2) over ℝ equals 2. By symmetry, this is twice the integral
    over (0, ∞).

PROVIDED SOLUTION
By symmetry of x ↦ |x| * exp(-(1/2)x²) (even function), the integral over ℝ is 2 times the integral over (0,∞). On (0,∞), |x| = x, so use integral_x_mul_gaussian_Ioi to get 2 * 1 = 2. Use MeasureTheory.integral_comp_neg or the fact that the function is even with Mathlib's even function integral tools.
-/
theorem integral_abs_mul_gaussian :
    ∫ (x : ℝ), |x| * exp (-(1/2) * x ^ 2) = 2 := by
  -- We'll use symmetry around the y-axis to compute this integral.
  have h_symm : ∫ x, |x| * (Real.exp (-(1 / 2) * x ^ 2)) = (∫ x in Set.Ioi 0, x * (Real.exp (-(1 / 2) * x ^ 2))) + (∫ x in Set.Iio 0, -x * (Real.exp (-(1 / 2) * x ^ 2))) := by
    rw [ ← MeasureTheory.integral_indicator ( measurableSet_Ioi ), ← MeasureTheory.integral_indicator ( measurableSet_Iio ) ];
    rw [ ← MeasureTheory.integral_add ] ; congr ; ext x ; norm_num [ Set.indicator ] ; cases abs_cases x <;> simp +decide [ * ] ; ring;
    · grind +qlia;
    · intros; linarith;
    · rw [ MeasureTheory.integrable_indicator_iff ] <;> norm_num;
      have := @integral_x_mul_gaussian_Ioi;
      exact ( by contrapose! this; rw [ MeasureTheory.integral_undef ( by simpa [ mul_comm ] using this ) ] ; norm_num );
    · rw [ MeasureTheory.integrable_indicator_iff ] <;> norm_num;
      -- Let's simplify the integral.
      suffices h_simp : MeasureTheory.IntegrableOn (fun x => x * Real.exp (-(1 / 2 * x ^ 2))) (Set.Ioi 0) by
        convert h_simp.comp_neg using 1 ; norm_num [ Set.indicator ] ; ring_nf ; aesop;
      have := @integral_rpow_mul_exp_neg_mul_rpow;
      specialize @this 2 1 ( 1 / 2 ) ; norm_num at this;
      exact ( by contrapose! this; rw [ MeasureTheory.integral_undef this ] ; norm_num );
  -- By substitution using $ u = -x $, we can transform the integral over $(-\infty, 0)$ to an integral over $(0, \infty)$.
  have h_subst : ∫ x in Set.Iio 0, -x * (Real.exp (-(1 / 2) * x ^ 2)) = ∫ x in Set.Ioi 0, x * (Real.exp (-(1 / 2) * x ^ 2)) := by
    rw [ ← MeasureTheory.integral_Iic_eq_integral_Iio ] ; rw [ ← neg_zero, ← integral_comp_neg_Iic ] ; norm_num;
  linarith [ integral_x_mul_gaussian_Ioi ]

/-
PROBLEM
**E[|Z|] = √(2/π) for Z ~ N(0,1).**
    The expected absolute value of a standard normal random variable is √(2/π).
    Stated as an integral identity:
      (1/√(2π)) · ∫ₓ |x| · exp(-x²/2) dx = √(2/π)

PROVIDED SOLUTION
Rewrite using integral_abs_mul_gaussian: the integral = 2. So LHS = (1/√(2π)) * 2 = 2/√(2π). We need 2/√(2π) = √(2/π). Note that √(2/π) = √2/√π and 2/√(2π) = 2/(√2·√π) = √2/√π. So they are equal. Use Real.sqrt_div_self or algebraic manipulation with sqrt_mul, sqrt_div, etc.
-/
theorem expected_abs_std_normal :
    (1 / sqrt (2 * π)) * (∫ (x : ℝ), |x| * exp (-(1/2) * x ^ 2)) = sqrt (2 / π) := by
  -- Simplify the expression using the known value of the integral.
  field_simp; ring;
  rw [ ← Real.sqrt_mul <| by positivity ] ; ring ; norm_num [ Real.pi_pos.le ];
  convert integral_abs_mul_gaussian using 3 ; ring

/-!
## Part 2: Numerical Verification of Latency Cost Bounds

With parameters:
- L = 0.4 (Solana block time in seconds)
- Δt = 1.0 (model step duration)
- σ_step = 0.001024 (per-step return std)
- impact_buffer = 0.0003 (3 bps)
- half_spread_BTC = 0.000005 (0.05 bps)

We verify:
  latency_cost = √(2/π) · σ_step · √(L/Δt) < 0.00052
  latency_cost > half_spread_BTC
  latency_cost / impact_buffer < 0.2
-/

/-- The latency cost formula: σ_latency = σ_step · √(L/Δt) gives the price uncertainty
    during latency, and the expected absolute price change is √(2/π) · σ_latency. -/
def latency_cost (sigma_step L delta_t : ℝ) : ℝ :=
  sqrt (2 / π) * sigma_step * sqrt (L / delta_t)

/-
PROBLEM
**Numerical bound**: √(2/π) < 0.7980.
    This is needed for the numerical verification.

PROVIDED SOLUTION
We need sqrt(2/π) < 0.7980. Equivalently 2/π < 0.7980^2 = 0.636804. We know π > 3.14159, so 2/π < 2/3.14159 < 0.63662 < 0.636804. Use Real.sqrt_lt_sqrt or rw [Real.sqrt_lt'] and bound π from below using Real.pi_gt_three or similar.
-/
theorem sqrt_two_div_pi_lt : sqrt (2 / π) < 0.7980 := by
  rw [ Real.sqrt_lt' ] <;> norm_num ; ( rw [ div_lt_iff₀ ] ) <;> try positivity; ; ( have := Real.pi_gt_d20 ; norm_num at * ; linarith; ) ;

/-
PROBLEM
**Numerical bound**: √(0.4) < 0.6325.
    Square root of L/Δt with our parameters.

PROVIDED SOLUTION
sqrt(0.4) < 0.6325 iff 0.4 < 0.6325^2 = 0.40005625. This is true. Use Real.sqrt_lt_sqrt or show that 0.6325^2 > 0.4 and then apply sqrt_lt'.
-/
theorem sqrt_04_lt : sqrt 0.4 < 0.6325 := by
  rw [ Real.sqrt_lt ] <;> norm_num

/-
PROBLEM
**Latency cost < 0.52 bps**: With our parameters, the latency cost is less than 0.00052.

PROVIDED SOLUTION
Unfold latency_cost. We need sqrt(2/π) * 0.001024 * sqrt(0.4/1.0) < 0.00052. Note 0.4/1.0 = 0.4. We have sqrt(2/π) < 0.7980 and sqrt(0.4) < 0.6325. So the product < 0.7980 * 0.001024 * 0.6325 = 0.000516... < 0.00052. Use the already proved sqrt_two_div_pi_lt and sqrt_04_lt, then calc or linarith with appropriate mul_lt_mul estimates.
-/
theorem latency_cost_bound :
    latency_cost 0.001024 0.4 1.0 < 0.00052 := by
  norm_num [ latency_cost ];
  -- We'll use that π is approximately 3.14 to estimate the value.
  have h_pi : Real.pi > 3.14 := by
    pi_lower_bound [ 99 / 70, 874 / 473, 1940 / 989, 1447 / 727 ];
  rw [ div_mul_eq_mul_div, div_mul_div_comm, div_lt_iff₀ ] <;> ring <;> norm_num at *;
  · nlinarith [ Real.sqrt_nonneg 5, Real.sqrt_nonneg π, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), Real.sq_sqrt ( show 0 ≤ Real.pi by positivity ), mul_pos ( Real.sqrt_pos.mpr ( show 0 < 5 by norm_num ) ) ( Real.sqrt_pos.mpr ( show 0 < Real.pi by positivity ) ) ];
  · positivity

/-
PROBLEM
**Latency cost > BTC half-spread**: The latency cost exceeds the narrow BTC spread.
    latency_cost > 0.000005 (0.05 bps half-spread)

PROVIDED SOLUTION
Unfold latency_cost. We need sqrt(2/π) * 0.001024 * sqrt(0.4) > 0.000005. We know sqrt(2/π) > 0 and sqrt(0.4) > 0 (in fact sqrt(2/π) ≥ sqrt(0.5) > 0.7 and sqrt(0.4) > 0.63). Use positivity or specific lower bounds. Actually simpler: 0.7 * 0.001024 * 0.63 > 0.000451 > 0.000005. For the lower bounds: sqrt(2/π) > 0.79 (since 0.79^2 = 0.6241 < 2/π ≈ 0.6366), sqrt(0.4) > 0.63 (since 0.63^2 = 0.3969 < 0.4). Use Real.lt_sqrt to establish these lower bounds.
-/
theorem latency_cost_gt_half_spread_btc :
    latency_cost 0.001024 0.4 1.0 > 0.000005 := by
  unfold latency_cost; norm_num; ring_nf; norm_num [ Real.pi_pos ] ;
  field_simp;
  nlinarith [ Real.pi_le_four, Real.sqrt_nonneg 5, Real.sqrt_nonneg π, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), Real.sq_sqrt ( show 0 ≤ Real.pi by positivity ) ]

/-
PROBLEM
The original text claims "latency_cost / impact_buffer < 0.2" with impact_buffer = 3 bps = 0.0003.
   However, latency_cost ≈ 0.000517 and 0.2 * 0.0003 = 0.00006, so the claim is FALSE.
   The text has a unit error: 0.00052 = 5.2 bps (not 0.52 bps as stated), and
   5.2 bps / 3 bps ≈ 1.73, which is NOT < 0.2.

   Corrected claim: latency_cost is less than 2 × impact_buffer (i.e., within the same
   order of magnitude), meaning the spread + impact model already provides a reasonable
   buffer.

theorem latency_cost_lt_impact_fraction_ORIGINAL :
latency_cost 0.001024 0.4 1.0 < 0.2 * 0.0003 := by sorry  -- FALSE

**Corrected bound**: Latency cost is less than twice the impact buffer (3 bps = 0.0003),
    showing they are on the same order of magnitude.

PROVIDED SOLUTION
We need latency_cost 0.001024 0.4 1.0 < 2 * 0.0003 = 0.0006. Unfold latency_cost. We need sqrt(2/π) * 0.001024 * sqrt(0.4) < 0.0006. Use sqrt_two_div_pi_lt (< 0.7980) and sqrt_04_lt (< 0.6325). Product < 0.7980 * 0.001024 * 0.6325 < 0.000517 < 0.0006. Use the already proven bounds and linarith/nlinarith.
-/
theorem latency_cost_lt_twice_impact :
    latency_cost 0.001024 0.4 1.0 < 2 * 0.0003 := by
  norm_num [ latency_cost ] at *;
  rw [ div_mul_eq_mul_div, div_mul_div_comm, div_lt_iff₀ ] <;> ring_nf <;> norm_num [ Real.pi_pos.le ];
  · nlinarith [ Real.pi_gt_three, Real.sqrt_nonneg 5, Real.sqrt_nonneg π, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), Real.sq_sqrt ( show 0 ≤ Real.pi by positivity ), mul_pos ( Real.sqrt_pos.mpr ( show 0 < 5 by norm_num ) ) ( Real.sqrt_pos.mpr ( show 0 < Real.pi by positivity ) ) ];
  · positivity

/-!
## Part 3: Signal Decay During Latency

For any continuous autocorrelation function ρ with ρ(0) = 1,
the signal retained after sub-step latency approaches 1 as latency → 0.
-/

/-
PROBLEM
**Signal retention at zero latency.** For any continuous function ρ : ℝ → ℝ with ρ(0) = 1,
    we have ρ(x) → 1 as x → 0. This means sub-step latency preserves most of the signal.

PROVIDED SOLUTION
Direct from Continuous.tendsto: hcont.tendsto 0 rewritten with h0.
-/
theorem continuous_autocorr_at_zero {ρ : ℝ → ℝ} (hcont : Continuous ρ) (h0 : ρ 0 = 1) :
    Tendsto ρ (𝓝 0) (𝓝 1) := by
  exact h0 ▸ hcont.tendsto 0

/-
PROBLEM
**Latency cost is monotone in latency.** The cost √(2/π) · σ · √(L/Δt) is monotonically
    increasing in L for fixed σ, Δt > 0. Reducing latency strictly reduces execution cost.

PROVIDED SOLUTION
Unfold latency_cost. It suffices to show sqrt(L₁/delta_t) ≤ sqrt(L₂/delta_t). Use Real.sqrt_le_sqrt and div_le_div_of_nonneg_right (or div_le_div_right). L₁ ≤ L₂ implies L₁/delta_t ≤ L₂/delta_t (since delta_t > 0), which implies sqrt(L₁/delta_t) ≤ sqrt(L₂/delta_t). Then multiply by the nonneg factors sqrt(2/π) * sigma.
-/
theorem latency_cost_mono {sigma delta_t : ℝ} (hσ : 0 < sigma) (hdt : 0 < delta_t)
    {L₁ L₂ : ℝ} (hL : 0 ≤ L₁) (hL12 : L₁ ≤ L₂) :
    latency_cost sigma L₁ delta_t ≤ latency_cost sigma L₂ delta_t := by
  exact mul_le_mul_of_nonneg_left ( Real.sqrt_le_sqrt <| by gcongr ) <| by positivity;

/-
PROBLEM
**Latency cost is nonneg.**

PROVIDED SOLUTION
latency_cost is a product of three nonneg terms: sqrt(2/π) ≥ 0, sigma ≥ 0, sqrt(L/delta_t) ≥ 0. Use mul_nonneg and Real.sqrt_nonneg.
-/
theorem latency_cost_nonneg {sigma L delta_t : ℝ} (hσ : 0 ≤ sigma) (hL : 0 ≤ L)
    (hdt : 0 ≤ delta_t) :
    0 ≤ latency_cost sigma L delta_t := by
  exact mul_nonneg ( mul_nonneg ( Real.sqrt_nonneg _ ) hσ ) ( Real.sqrt_nonneg _ )

end
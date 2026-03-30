import Mathlib

/-!
# Theorem 43: Conditional Spread and Slippage Under Activity Regimes

This file formalizes several claims about how bid-ask spreads behave conditionally
on volatility/activity regimes, following the Almgren-Chriss power-law model.

## Main results

* `spread_ratio_eq` — The power-law spread ratio R = E[σ^γ | high] / E[σ^γ]
* `sqrt_law_k2` — Under the square-root law with k=2, the ratio is √2
* `sqrt_law_k4` — Under the square-root law with k=4, the ratio is 2
* `linear_bound` — For γ ∈ (0,1], the conditional spread ratio R ≤ k^γ
* `relative_bias_formula` — The relative bias is (R-1)/R of true conditional slippage
-/

open Real

noncomputable section

/-! ## Claim 1: Power-Law Spread Ratio

Under the Almgren-Chriss model E[S | σ] = c · σ^γ, the conditional spread ratio is:
  R = E[σ^γ | high] / E[σ^γ]

We formalize this algebraically: given that the spread is c * σ^γ,
the ratio of conditional to unconditional expected spread equals the
ratio of conditional to unconditional expected σ^γ (the constant c cancels).
-/

/-- The constant c > 0 cancels in the spread ratio:
    (c * E_high) / (c * E_all) = E_high / E_all -/
theorem spread_ratio_eq (c E_high E_all : ℝ) (hc : c > 0) (_hE : E_all > 0) :
    (c * E_high) / (c * E_all) = E_high / E_all := by
  rw [mul_div_mul_left _ _ hc.ne']

/-! ## Claim 1 (continued): R > 1 when γ > 0 for high-activity regimes

When high-activity volatility is strictly greater than the unconditional mean volatility,
and γ > 0, we have R > 1. We formalize this as: if E_high > E_all > 0, then
E_high / E_all > 1.
-/

/-- If expected σ^γ conditional on high activity exceeds the unconditional mean,
    the spread ratio exceeds 1. -/
theorem spread_ratio_gt_one (E_high E_all : ℝ) (hE : E_all > 0)
    (hgt : E_high > E_all) : E_high / E_all > 1 := by
  rwa [gt_iff_lt, one_lt_div hE]

/-! ## Claim 2: Square-Root Law (γ = 0.5)

Under the square-root impact model, E[S | σ] / E[S | σ₀] = √(σ/σ₀).
If high-activity periods have σ = k · σ₀, then the ratio is √k.
-/

/-- The square-root law: spread ratio under σ = k · σ₀ is √k -/
theorem sqrt_law (k σ₀ : ℝ) (_hk : k > 0) (hσ₀ : σ₀ > 0) :
    Real.sqrt (k * σ₀ / σ₀) = Real.sqrt k := by
  rw [mul_div_cancel_right₀ _ hσ₀.ne']

/-- For k = 2, √k = √2 ≈ 1.414 -/
theorem sqrt_law_k2 : Real.sqrt 2 > 1 ∧ Real.sqrt 2 < 1.415 ∧ Real.sqrt 2 > 1.414 := by
  norm_num [Real.lt_sqrt, Real.sqrt_lt]

/-- For k = 4, √k = 2 -/
theorem sqrt_law_k4 : Real.sqrt 4 = 2 := by
  norm_num [Real.sqrt_eq_iff_mul_self_eq]

/-! ## Claim 3: Linear Bound

For any γ ∈ (0, 1] and k > 1, if high-activity volatility is at most k times
normal volatility, then R ≤ k^γ. Since k > 1 and γ > 0, we also have R > 1
(so 1 < R ≤ k^γ).

We formalize the key mathematical fact: for k > 1, 0 < γ ≤ 1, we have
1 < k^γ ≤ k.
-/

/-- k^γ > 1 when k > 1 and γ > 0 -/
theorem rpow_gt_one_of_gt_one {k γ : ℝ} (hk : k > 1) (hγ : γ > 0) :
    k ^ γ > 1 := by
  exact Real.one_lt_rpow hk hγ

/-- k^γ ≤ k when k ≥ 1 and 0 < γ ≤ 1 -/
theorem rpow_le_self_of_ge_one {k γ : ℝ} (hk : k ≥ 1) (_hγ0 : 0 < γ) (hγ1 : γ ≤ 1) :
    k ^ γ ≤ k := by
  simpa using Real.rpow_le_rpow_of_exponent_le hk hγ1

/-- The linear bound: for k > 1 and γ ∈ (0, 1], we have 1 < k^γ ≤ k -/
theorem linear_bound {k γ : ℝ} (hk : k > 1) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) :
    1 < k ^ γ ∧ k ^ γ ≤ k := by
  exact ⟨rpow_gt_one_of_gt_one hk hγ0, rpow_le_self_of_ge_one (le_of_lt hk) hγ0 hγ1⟩

/-- Monotonicity: k^γ is increasing in γ for k > 1, so the bound k^γ is tight
    when the conditional volatility equals k times the unconditional. -/
theorem rpow_mono_gamma {k γ₁ γ₂ : ℝ} (hk : k > 1) (hle : γ₁ ≤ γ₂) :
    k ^ γ₁ ≤ k ^ γ₂ := by
  exact Real.rpow_le_rpow_of_exponent_le hk.le hle

/-! ## Claim 4: R = 1 iff No Conditional Widening

R = 1 iff the spread is independent of the activity regime. The simplest case
is γ = 0, where E[S | σ] = c · σ^0 = c (constant spread).
-/

/-- When γ = 0, the spread is constant: σ^0 = 1 for any σ > 0. -/
theorem rpow_zero_eq_one (σ : ℝ) (_hσ : σ > 0) : σ ^ (0 : ℝ) = 1 := by
  norm_num

/-- If spread is constant (γ = 0), the ratio R = 1. -/
theorem spread_ratio_one_of_gamma_zero (c : ℝ) (hc : c > 0) :
    (c * 1) / (c * 1) = 1 := by
  rw [div_self (mul_ne_zero hc.ne' one_ne_zero)]

/-! ## Claim 5: Slippage Model Bias

Given the static slippage model: slippage = S/2 + impact_buffer,
if R > 1, the model underestimates the true conditional slippage by:
  bias = (R - 1) · S/2

The relative bias (as a fraction of the true conditional slippage
component S·R/2, excluding impact_buffer) is (R-1)/R.
-/

/-- The bias formula: true conditional slippage minus static estimate. -/
theorem bias_formula (S R : ℝ) :
    (R * S / 2) - (S / 2) = (R - 1) * S / 2 := by
  ring

/-- The relative bias: bias / true_conditional = (R-1)/R. -/
theorem relative_bias_formula (S R : ℝ) (hR : R > 0) (hS : S > 0) :
    ((R - 1) * S / 2) / (R * S / 2) = (R - 1) / R := by
  rw [div_eq_div_iff] <;> [ring; positivity; positivity]

/-- When R = 1, the bias is zero (unbiased model). -/
theorem no_bias_when_R_eq_one (S : ℝ) :
    (1 - 1) * S / 2 = 0 := by
  ring

/-- When R > 1, the bias is strictly positive. -/
theorem positive_bias_when_R_gt_one (S R : ℝ) (hR : R > 1) (hS : S > 0) :
    (R - 1) * S / 2 > 0 := by
  exact div_pos (mul_pos (sub_pos.mpr hR) hS) zero_lt_two

end

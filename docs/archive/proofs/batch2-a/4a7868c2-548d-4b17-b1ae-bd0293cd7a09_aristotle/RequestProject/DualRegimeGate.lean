import Mathlib

/-!
# Dual Regime Gate Properties

Formalization of the Dual Regime Gate Properties for Sortino ratio optimization
with independent and correlated binary filters.

## Background Model

- A single gate with pass rate φ and accuracy improvement factor k yields:
    Sortino_gated = k · √φ · Sortino_ungated
- From Theorem 10 (previously proved): a single gate improves Sortino iff φ ≥ 1/k².
- From Theorem 6: SNR(r) = SNR_base / √(1-r) is strictly increasing.

## Overview of Claims

1. Combined pass rate of independent filters is in (0,1).
2. Dual-gated Sortino improvement condition: k₂ · √φ₂ > 1.
3. Numerical thresholds for specific φ₂ values.
4. Positive correlation increases combined pass rate vs independence.
-/

noncomputable section

open Real

/-! ## Claim 1: Combined pass rate of independent filters -/

/-
PROBLEM
The product of two values in (0,1) is in (0,1). This captures the fact that
    for two independent binary filters with pass rates φ₁ ∈ (0,1) and φ₂ ∈ (0,1),
    the combined pass rate φ₁·φ₂ ∈ (0,1).

PROVIDED SOLUTION
Split into two parts. Positivity: mul_pos. Less than 1: mul_lt_one_of_nonneg_of_lt_one_left or similar, since 0 ≤ φ₁ < 1 and 0 < φ₂ ≤ 1 (actually φ₂ < 1), so φ₁*φ₂ < 1*φ₂ < 1.
-/
theorem combined_pass_rate_in_unit_interval
    (φ₁ φ₂ : ℝ) (h1 : 0 < φ₁) (h1' : φ₁ < 1) (h2 : 0 < φ₂) (h2' : φ₂ < 1) :
    0 < φ₁ * φ₂ ∧ φ₁ * φ₂ < 1 := by
  constructor <;> nlinarith

/-! ## Claim 2: Dual-gated Sortino improvement condition -/

/-
PROBLEM
The Sortino ratio model: Sortino_gated = k · √φ · Sortino_ungated.
    For dual gates: Sortino_dual = (k₁ · k₂) · √(φ₁ · φ₂) · Sortino_ungated.
    The dual gate beats the single gate F₁ alone iff k₂ · √φ₂ > 1.

PROVIDED SOLUTION
Unfold the lets. S_dual > S_single iff k₁*k₂*√(φ₁*φ₂)*S₀ > k₁*√φ₁*S₀. Since k₁ > 0 and S₀ > 0, divide both sides. Use sqrt_mul to get √(φ₁*φ₂) = √φ₁ * √φ₂. Then cancel √φ₁ (positive). Left with k₂*√φ₂ > 1.
-/
theorem dual_gate_beats_single_iff
    (S₀ : ℝ) (hS₀ : 0 < S₀) -- Sortino_ungated > 0
    (k₁ k₂ : ℝ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) -- accuracy improvement factors
    (φ₁ φ₂ : ℝ) (hφ₁ : 0 < φ₁) (hφ₁' : φ₁ < 1) (hφ₂ : 0 < φ₂) (hφ₂' : φ₂ < 1) :
    let S_single := k₁ * √φ₁ * S₀           -- Sortino with gate F₁ only
    let S_dual := (k₁ * k₂) * √(φ₁ * φ₂) * S₀  -- Sortino with both gates
    S_dual > S_single ↔ k₂ * √φ₂ > 1 := by
  field_simp;
  rw [ Real.sqrt_mul ( by positivity ) ] ; constructor <;> intro <;> nlinarith [ show 0 < Real.sqrt φ₁ from Real.sqrt_pos.mpr hφ₁, show 0 < Real.sqrt φ₂ from Real.sqrt_pos.mpr hφ₂, show 0 < k₂ * Real.sqrt φ₂ from mul_pos hk₂ ( Real.sqrt_pos.mpr hφ₂ ) ] ;

/-
PROBLEM
Applying the single-gate Theorem 10 threshold to the second gate:
    A single gate improves Sortino iff k · √φ ≥ 1 (equivalently φ ≥ 1/k²).
    The second gate is worth adding iff k₂ · √φ₂ > 1, i.e., k₂ > 1/√φ₂.

PROVIDED SOLUTION
Since √φ₂ > 0, k₂ * √φ₂ > 1 ↔ k₂ > 1/√φ₂ by div_lt_iff (sqrt_pos_of_pos hφ₂).
-/
theorem second_gate_threshold_equiv
    (k₂ φ₂ : ℝ) (hk₂ : 0 < k₂) (hφ₂ : 0 < φ₂) :
    k₂ * √φ₂ > 1 ↔ k₂ > 1 / √φ₂ := by
  rw [ gt_iff_lt, gt_iff_lt, div_lt_iff₀ ] ; positivity

/-! ## Claim 3: Numerical thresholds -/

/-
PROBLEM
For φ₂ = 0.5, the threshold is k₂ > 1/√0.5 = √2.
    We prove 1/√0.5 = √2.

PROVIDED SOLUTION
1/√(1/2) = √2. Since √(1/2) = 1/√2, we get 1/(1/√2) = √2. Use rw [one_div, Real.sqrt_inv, inv_inv] or similar.
-/
theorem threshold_half : 1 / √(1/2 : ℝ) = √2 := by
  norm_num [ Real.sqrt_div_self ]

/-
PROBLEM
For φ₂ = 0.7, the threshold is k₂ > 1/√0.7.
    We verify 1/√0.7 < 1.196 (i.e., approximately 1.195).

PROVIDED SOLUTION
Need 1/√(7/10) < 1196/1000. Equivalently √(7/10) > 1000/1196. Square both sides: 7/10 > 1000000/1430416. Cross multiply: 7*1430416 > 10*1000000, i.e. 10012912 > 10000000. Use norm_num and sqrt inequalities.
-/
theorem threshold_07_bound : 1 / √(7/10 : ℝ) < 1196 / 1000 := by
  field_simp;
  nlinarith [ Real.sqrt_nonneg ( 7 / 10 ), Real.sq_sqrt ( show 0 ≤ 7 / 10 by norm_num ) ]

/-
PROBLEM
For φ₂ = 0.7, the threshold is k₂ > 1/√0.7.
    We verify 1/√0.7 > 1.195.

PROVIDED SOLUTION
Need 1/√(7/10) > 1195/1000. Equivalently √(7/10) < 1000/1195. Square both sides: 7/10 < 1000000/1428025. Cross multiply: 7*1428025 < 10*1000000, i.e. 9996175 < 10000000. Use norm_num and sqrt inequalities.
-/
theorem threshold_07_lower : 1 / √(7/10 : ℝ) > 1195 / 1000 := by
  rw [ gt_iff_lt, lt_div_iff₀ ] <;> nlinarith [ Real.sqrt_nonneg ( 7 / 10 ), Real.sq_sqrt ( show 0 ≤ 7 / 10 by norm_num ) ]

/-! ## Claim 4: Positive correlation increases combined pass rate -/

/-
PROBLEM
For two binary events (indicator random variables), the joint probability satisfies
    P(F₁ ∧ F₂) = P(F₁)·P(F₂) + Cov(F₁, F₂).
    If Cov > 0 (positive correlation), then φ_combined > φ₁·φ₂.
    This means correlated gates filter LESS data than independent gates.

PROVIDED SOLUTION
From h_joint and h_pos_corr: φ_combined = φ₁*φ₂ + cov > φ₁*φ₂. Just linarith.
-/
theorem correlated_gates_higher_pass_rate
    (φ₁ φ₂ φ_combined cov : ℝ)
    (h_joint : φ_combined = φ₁ * φ₂ + cov)  -- P(F₁ ∧ F₂) = P(F₁)·P(F₂) + Cov
    (h_pos_corr : cov > 0) :                  -- positive correlation
    φ_combined > φ₁ * φ₂ := by
  linarith

/-
PROBLEM
Correlated gates preserve more data, which is beneficial for Sortino's √φ factor.
    Higher combined pass rate means the √φ_combined factor is larger.

PROVIDED SOLUTION
Use Real.sqrt_lt_sqrt or Real.sqrt_lt_sqrt with h_indep_pos and h_corr_greater.
-/
theorem correlated_gates_better_sqrt_factor
    (φ_indep φ_corr : ℝ)
    (h_indep_pos : 0 < φ_indep)
    (h_corr_greater : φ_corr > φ_indep) :
    √φ_corr > √φ_indep := by
  exact Real.sqrt_lt_sqrt h_indep_pos.le h_corr_greater

end
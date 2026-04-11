/-
  CLAIM 3: Regime gate trade-off — accuracy vs trade count.

  Sortino_gated / Sortino_ungated = k · √φ · (σ_down_ungated / σ_down_gated)

  When downside vol is unchanged: gating improves Sortino iff k · √φ > 1,
  i.e., φ > 1/k².
-/
import Mathlib

open Real

noncomputable section

/-
PROBLEM
Claim 3(b): The critical gating fraction below which Sortino decreases.
    When σ_down_gated = σ_down_ungated, the condition k · √φ ≥ 1 gives φ ≥ 1/k².
    We prove: k · √φ ≥ 1 ↔ φ ≥ 1/k² (for k > 0, φ ≥ 0).

PROVIDED SOLUTION
k * √φ ≥ 1 ↔ √φ ≥ 1/k ↔ φ ≥ (1/k)² = 1/k². Use Real.le_sqrt' and Real.sqrt_le_sqrt. The key steps: k*√φ ≥ 1 ↔ √φ ≥ 1/k (divide by k>0), and √φ ≥ 1/k ↔ φ ≥ 1/k² (square both sides, using that 1/k > 0).
-/
theorem gating_threshold (k φ : ℝ) (hk : k > 0) (hφ : φ ≥ 0) :
    k * Real.sqrt φ ≥ 1 ↔ φ ≥ 1 / k ^ 2 := by
  field_simp;
  constructor <;> intro <;> nlinarith [ show 0 ≤ k * Real.sqrt φ by positivity, Real.mul_self_sqrt hφ ]

/-
PROBLEM
Claim 3(c): For k = 1.5, φ_min = 1/2.25 = 4/9

PROVIDED SOLUTION
norm_num
-/
theorem phi_min_k_1_5 : (1 : ℝ) / (3/2) ^ 2 = 4 / 9 := by
  norm_num +zetaDelta at *

/-
PROBLEM
For k = 2, φ_min = 1/4

PROVIDED SOLUTION
norm_num
-/
theorem phi_min_k_2 : (1 : ℝ) / 2 ^ 2 = 1 / 4 := by
  norm_num +zetaDelta at *

/-
PROBLEM
4/9 ≈ 0.44 (more precisely, 4/9 > 0.44)

PROVIDED SOLUTION
norm_num
-/
theorem phi_min_approx : (4 : ℝ) / 9 > 0.44 := by
  norm_num +zetaDelta at *

end
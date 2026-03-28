import Mathlib

/-!
# Theorem 47: Optimal Observation Window from Signal-to-Noise Ratio

Formalization of the core mathematical claims from the signal decay / window optimization
analysis. We formalize:

1. The geometric series identity for Signal(W) under exponential decay of predictive power.
2. The limit of Signal(W) as W → ∞.
3. The optimal window W* that minimizes the noise denominator in the SNR expression.
-/

open Finset Real Filter Topology BigOperators

noncomputable section

/-! ## Auxiliary: exp(-2/τ) ∈ (0, 1) for τ > 0 -/

theorem exp_neg_two_div_pos (τ : ℝ) (hτ : 0 < τ) : 0 < Real.exp (-2 / τ) :=
  Real.exp_pos _

theorem exp_neg_two_div_lt_one (τ : ℝ) (hτ : 0 < τ) : Real.exp (-2 / τ) < 1 :=
  Real.exp_lt_one_iff.mpr (div_neg_of_neg_of_pos (by norm_num) hτ)

/-! ## Claim 1: Signal content with exponential decay

When |β_k| = |β₀| · exp(-k/τ), the total signal content is:
  Signal(W) = Σ_{k=0}^{W-1} |β₀|² · exp(-2k/τ)
            = |β₀|² · (1 - exp(-2W/τ)) / (1 - exp(-2/τ))
-/

/-- The geometric series identity for the exponential decay sum:
  Σ_{k=0}^{W-1} exp(-2k/τ) = (1 - exp(-2W/τ)) / (1 - exp(-2/τ)). -/
theorem geom_sum_exp_decay (τ : ℝ) (hτ : 0 < τ) (W : ℕ) :
    ∑ k ∈ range W, Real.exp (-2 * ↑k / τ) =
      (1 - Real.exp (-2 * ↑W / τ)) / (1 - Real.exp (-2 / τ)) := by
  have h_geo_series' :
      ∑ k ∈ Finset.range W, (Real.exp (-2 / τ)) ^ k =
        (1 - (Real.exp (-2 / τ)) ^ W) / (1 - Real.exp (-2 / τ)) := by
    rw [geom_sum_eq (ne_of_lt (by rw [← Real.exp_lt_exp]; ring_nf; norm_num; positivity))]
    rw [← neg_div_neg_eq, neg_sub, neg_sub]
  convert h_geo_series' using 2 <;> push_cast [← Real.exp_nat_mul] <;> ring

/-- Signal(W) = |β₀|² · (1 - exp(-2W/τ)) / (1 - exp(-2/τ)). -/
theorem signal_content_exp_decay (β₀ τ : ℝ) (hτ : 0 < τ) (W : ℕ) :
    ∑ k ∈ range W, β₀ ^ 2 * Real.exp (-2 * ↑k / τ) =
      β₀ ^ 2 * ((1 - Real.exp (-2 * ↑W / τ)) / (1 - Real.exp (-2 / τ))) := by
  rw [← geom_sum_exp_decay τ hτ W, Finset.mul_sum _ _ _]

/-! ## Claim 1 (continued): Signal(W) → |β₀|² / (1 - exp(-2/τ)) as W → ∞ -/

/-- The partial sums converge: Σ_{k=0}^{W-1} exp(-2k/τ) → 1/(1 - exp(-2/τ)) as W → ∞. -/
theorem signal_limit (τ : ℝ) (hτ : 0 < τ) :
    Tendsto (fun W : ℕ => ∑ k ∈ range W, Real.exp (-2 * ↑k / τ))
      atTop (nhds (1 / (1 - Real.exp (-2 / τ)))) := by
  have h_geo_series :
      ∀ W : ℕ, (∑ k ∈ Finset.range W, Real.exp (-2 * k / τ)) =
        (∑ k ∈ Finset.range W, (Real.exp (-2 / τ)) ^ k) :=
    fun W => Finset.sum_congr rfl fun _ _ => by rw [← Real.exp_nat_mul]; ring
  convert hasSum_geometric_of_lt_one (by positivity)
    (show Real.exp (-2 / τ) < 1 by
      rw [Real.exp_lt_one_iff]; exact div_neg_of_neg_of_pos (by norm_num) hτ)
    |>.tendsto_sum_nat using 2 <;> [skip; ring]
  aesop

/-! ## Claim 3: Optimal window minimizing noise

The noise denominator is f(W) = W · d · σ_noise² + d · σ_x² / W.
The unique minimizer over W > 0 is W* = σ_x / σ_noise.

We prove this for the simplified function g(W) = W * a + b / W where a, b > 0,
showing that g achieves its minimum at W = √(b/a).
-/

/-- The noise function g(W) = W * a + b / W for a, b > 0. -/
def noise_func (a b W : ℝ) : ℝ := W * a + b / W

/-- The noise function achieves its minimum at W* = √(b/a).
    That is, for all W > 0, g(√(b/a)) ≤ g(W). -/
theorem noise_func_minimizer (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (W : ℝ) (hW : 0 < W) :
    noise_func a b (Real.sqrt (b / a)) ≤ noise_func a b W := by
  unfold noise_func
  rw [div_eq_mul_inv]
  rw [div_eq_mul_inv, ← Real.sqrt_div_self]
  field_simp
  nlinarith [sq_nonneg (Real.sqrt (b / a) - W),
    Real.mul_self_sqrt (show 0 ≤ b / a by positivity),
    mul_div_cancel₀ b ha.ne']

/-- The minimum value of the noise function is 2√(ab). -/
theorem noise_func_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    noise_func a b (Real.sqrt (b / a)) = 2 * Real.sqrt (a * b) := by
  unfold noise_func; ring_nf; norm_num [ha.le, ha.ne', hb.le, hb.ne']; ring
  grind +ring

/-! ## Claim 3 (corollary): When σ_noise = σ_x, then W* = 1

With a = d · σ_noise² and b = d · σ_x², when σ_noise = σ_x,
W* = √(σ_x² / σ_noise²) = 1.
-/

/-- When σ_noise = σ_x, the optimal window is W* = √(σ²/σ²) = 1. -/
theorem optimal_window_equal_noise (σ : ℝ) (hσ : 0 < σ) :
    Real.sqrt (σ ^ 2 / σ ^ 2) = 1 := by
  norm_num [hσ.ne']

end

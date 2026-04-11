/-
  CLAIM 2: Class imbalance and optimal class weights.

  With K classes having probabilities p_c, inverse-frequency weights w_c = 1/p_c.
-/
import Mathlib

open Finset BigOperators

noncomputable section

/-
PROBLEM
Claim 2(a): Inverse-frequency weighting makes expected weighted contribution
    equal across classes. If w_c = 1/p_c then p_c · w_c = 1 for all c.

PROVIDED SOLUTION
p * (1/p) = 1 when p > 0. Use mul_div_cancel₀ or field_simp.
-/
theorem inverse_freq_equal_contribution {p : ℝ} (hp : p > 0) :
    p * (1 / p) = 1 := by
  grind

/-- Claim 2(b): For a majority-class-only classifier with K classes,
    weighted accuracy = 1/K.
    Here p_maj · w_maj / Σ(p_c · w_c) = 1 / K since each p_c · w_c = 1. -/
theorem weighted_accuracy_majority_classifier (K : ℕ) (hK : 0 < K) :
    (1 : ℝ) / K = 1 / K := by
  rfl

/-
PROBLEM
More substantive version: if we have K classes with positive probabilities
    summing to 1, and weights w_c = 1/p_c, then the weighted accuracy of
    a classifier that always predicts the majority class equals 1/K.

PROVIDED SOLUTION
Since w_i = 1/p_i, p_i * w_i = p_i * (1/p_i) = 1 for each i (using hp_pos). So ∑ p_i * w_i = ∑ 1 = K. And p_j * w_j = 1. Therefore p_j * w_j / ∑ (p_i * w_i) = 1 / K. Use Finset.sum_const, Finset.card_fin, and field_simp.
-/
theorem weighted_accuracy_majority {K : ℕ} (hK : 0 < K)
    (p : Fin K → ℝ) (hp_pos : ∀ i, 0 < p i)
    (hp_sum : ∑ i, p i = 1)
    (w : Fin K → ℝ) (hw : ∀ i, w i = 1 / p i)
    (j : Fin K) -- the majority class
    : p j * w j / ∑ i, (p i * w i) = 1 / K := by
  simp_all +decide [ ne_of_gt ]

/-
PROBLEM
Concrete instance: K=3, p = (0.65, 0.175, 0.175)

PROVIDED SOLUTION
Direct computation: p_flat * w_flat = 0.65 * (1/0.65) = 1, similarly for the others. So the denominator is 3 and the result is 1/3. Use norm_num or field_simp.
-/
theorem weighted_accuracy_three_class :
    let p_flat : ℝ := 0.65
    let p_long : ℝ := 0.175
    let p_short : ℝ := 0.175
    let w_flat := 1 / p_flat
    let w_long := 1 / p_long
    let w_short := 1 / p_short
    p_flat * w_flat / (p_flat * w_flat + p_long * w_long + p_short * w_short) = 1 / 3 := by
  norm_num +zetaDelta at *

end
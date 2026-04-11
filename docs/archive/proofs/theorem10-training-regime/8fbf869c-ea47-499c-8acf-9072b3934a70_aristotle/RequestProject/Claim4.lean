/-
  CLAIM 4: Model capacity bound — smaller model, fewer features.

  v8: P=692K, D=2300, N=2.5M
  v9: P=55K,  D=375,  N=2.5M
-/
import Mathlib

open Real

noncomputable section

/-
PROBLEM
Claim 4(a): If P > N, the network can memorize (P/N > 1 implies interpolation).
    We state: P/N > 1 ↔ P > N (for N > 0).

PROVIDED SOLUTION
div_lt_iff applied with hN. P/N > 1 ↔ P > 1*N ↔ P > N.
-/
theorem param_sample_ratio {P N : ℝ} (hN : N > 0) :
    P / N > 1 ↔ P > N := by
  rw [ gt_iff_lt, one_lt_div hN ]

/-
PROBLEM
Claim 4(b): P_v9/N < P_v8/N when P_v9 < P_v8 and N > 0.

PROVIDED SOLUTION
div_lt_div_right hN applied to h.
-/
theorem v9_ratio_lt_v8_ratio {P_v8 P_v9 N : ℝ} (hN : N > 0)
    (h : P_v9 < P_v8) : P_v9 / N < P_v8 / N := by
  gcongr

/-
PROBLEM
Concrete: 55000 / 2500000 < 692000 / 2500000

PROVIDED SOLUTION
norm_num
-/
theorem v9_ratio_concrete : (55000 : ℝ) / 2500000 < 692000 / 2500000 := by
  norm_num [ div_lt_div_iff₀ ] at * <;> first | linarith | aesop | assumption;

/-
PROBLEM
P_v9/N = 0.022

PROVIDED SOLUTION
norm_num
-/
theorem v9_ratio_value : (55000 : ℝ) / 2500000 = 0.022 := by
  norm_num +zetaDelta at *

/-
PROBLEM
P_v8/N = 0.2768

PROVIDED SOLUTION
norm_num
-/
theorem v8_ratio_value : (692000 : ℝ) / 2500000 = 0.2768 := by
  norm_num +zetaDelta at *

/-
PROBLEM
The ratio of ratios: 0.022 / 0.2768 < 0.08, i.e. v9 is much smaller.
    More precisely, P_v9/P_v8 = 55/692.

PROVIDED SOLUTION
norm_num
-/
theorem ratio_of_ratios : (55000 : ℝ) / 692000 = 55 / 692 := by
  grind

/-
PROBLEM
Claim 4(c): Effective complexity comparison.
    v8: min(692000, 2500000, 2300) = 2300
    v9: min(55000, 2500000, 375) = 375
    Ratio: 375/2300

PROVIDED SOLUTION
decide or native_decide or norm_num
-/
theorem effective_complexity_v8 : min (min (692000 : ℕ) 2500000) 2300 = 2300 := by
  norm_num +zetaDelta at *

/-
PROVIDED SOLUTION
decide or native_decide or norm_num
-/
theorem effective_complexity_v9 : min (min (55000 : ℕ) 2500000) 375 = 375 := by
  decide +revert

/-
PROVIDED SOLUTION
norm_num
-/
theorem complexity_ratio_bound : (375 : ℝ) / 2300 < 0.164 := by
  grind

end
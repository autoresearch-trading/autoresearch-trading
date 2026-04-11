/-
# Claim 2: Optimal fee_mult given accuracy constraint

f_opt = (2α-1)·(1-c) / c

Properties:
(a) f_opt > f_min = 1/(2α-1) when c < (2α-1)²/(1+(2α-1)²) [CORRECTED from original]
(b) f_opt is strictly increasing in α (better classifier → wider optimal barriers)
(c) f_opt is strictly decreasing in c (higher fees → tighter optimal barriers)
-/
import Mathlib

open Real

noncomputable section

/-- Optimal fee multiplier as a function of accuracy α and fee rate c -/
def f_opt (α c : ℝ) : ℝ := (2*α - 1) * (1 - c) / c

/-- Minimum fee multiplier for accuracy α -/
def f_min (α : ℝ) : ℝ := 1 / (2*α - 1)

/-
PROBLEM
The original claim (a) stated: f_opt > f_min when c < (2α-1)/(2α).
This is FALSE. Counterexample: α = 0.6, c = 0.1 satisfies c < (2α-1)/(2α) ≈ 0.167,
but f_opt = 0.2*0.9/0.1 = 1.8 < 5 = 1/0.2 = f_min.
The correct condition is c < (2α-1)² / (1 + (2α-1)²).
theorem f_opt_gt_f_min_ORIGINAL (α c : ℝ) (hα : 1/2 < α) (hc_pos : 0 < c)
(hc_bound : c < (2*α - 1)/(2*α)) :
f_min α < f_opt α c := by sorry

(a) CORRECTED: f_opt > f_min when c < (2α-1)² / (1 + (2α-1)²)

PROVIDED SOLUTION
Need f_min α < f_opt α c, i.e. 1/(2α-1) < (2α-1)(1-c)/c. Since 2α-1 > 0 and c > 0, multiply both sides by c(2α-1) to get c < (2α-1)²(1-c), i.e. c < (2α-1)² - (2α-1)²c, i.e. c(1+(2α-1)²) < (2α-1)², i.e. c < (2α-1)²/(1+(2α-1)²), which is exactly hc_bound. Work with div_lt_div and nlinarith.
-/
theorem f_opt_gt_f_min (α c : ℝ) (hα : 1/2 < α) (hc_pos : 0 < c)
    (hc_bound : c < (2*α - 1)^2 / (1 + (2*α - 1)^2)) :
    f_min α < f_opt α c := by
      unfold f_min f_opt;
      rw [ div_lt_div_iff₀ ] <;> nlinarith [ mul_div_cancel₀ ( ( 2 * α - 1 ) ^ 2 ) ( by positivity : ( 1 + ( 2 * α - 1 ) ^ 2 ) ≠ 0 ) ]

/-
PROBLEM
(b) f_opt is strictly increasing in α, for fixed c ∈ (0,1)

PROVIDED SOLUTION
f_opt α c = (2α-1)(1-c)/c. Since c ∈ (0,1), (1-c)/c > 0. So f_opt is a positive constant times (2α-1), which is strictly increasing in α. More precisely, if α₁ < α₂ then 2α₁-1 < 2α₂-1, so (2α₁-1)(1-c)/c < (2α₂-1)(1-c)/c.
-/
theorem f_opt_strictMono_alpha (c : ℝ) (hc_pos : 0 < c) (hc_lt : c < 1) :
    StrictMono (fun α => f_opt α c) := by
      unfold StrictMono f_opt;
      exact fun a b hab => by rw [ div_lt_div_iff_of_pos_right hc_pos ] ; nlinarith;

/-
PROBLEM
(c) f_opt is strictly decreasing in c, for α > 1/2

PROVIDED SOLUTION
f_opt α c = (2α-1)(1-c)/c = (2α-1)(1/c - 1). For α > 1/2, (2α-1) > 0. The function c ↦ 1/c - 1 is strictly decreasing on (0,∞). So f_opt is strictly decreasing in c on (0,∞). Concretely, for 0 < c₁ < c₂, (1-c₁)/c₁ > (1-c₂)/c₂ (cross multiply: c₂(1-c₁) > c₁(1-c₂) iff c₂ - c₁c₂ > c₁ - c₁c₂ iff c₂ > c₁ ✓). Multiply by (2α-1) > 0.
-/
theorem f_opt_strictAnti_c (α : ℝ) (hα : 1/2 < α) :
    StrictAntiOn (fun c => f_opt α c) (Set.Ioi 0) := by
      unfold f_opt;
      norm_num [ StrictAntiOn ];
      intros; rw [ div_lt_div_iff₀ ] <;> nlinarith [ mul_pos ( sub_pos.2 hα ) ( sub_pos.2 ‹_› ) ] ;

end
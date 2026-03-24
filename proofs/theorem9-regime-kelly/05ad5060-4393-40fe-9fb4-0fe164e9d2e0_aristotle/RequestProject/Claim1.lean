/-
# Claim 1: Fee-multiplier / accuracy trade-off is convex

Define α_min(f) = 1/2 + 1/(2f) for f > 0.

Properties:
(a) α_min is strictly decreasing and convex on (0, ∞)
(b) α_min(1) = 1
(c) α_min(f) → 1/2 as f → ∞
(d) For any achievable accuracy α ∈ (1/2, 1), the minimum fee_mult is f_min = 1/(2α - 1)
(e) Spot checks: α_min(3/2) = 5/6, α_min(4) = 5/8, α_min(8) = 9/16
-/
import Mathlib

open Real Filter Topology

noncomputable section

/-- Required minimum accuracy as a function of fee multiplier -/
def α_min (f : ℝ) : ℝ := 1/2 + 1/(2*f)

/-
PROBLEM
(b) α_min(1) = 1

PROVIDED SOLUTION
Unfold α_min, compute 1/2 + 1/(2*1) = 1/2 + 1/2 = 1. Use norm_num or field_simp.
-/
theorem alpha_min_at_one : α_min 1 = 1 := by
  -- By definition of α_min, we have α_min 1 = 1/2 + 1/(2*1).
  simp [α_min];
  norm_num

/-
PROBLEM
(e) Spot check: α_min(3/2) = 5/6

PROVIDED SOLUTION
Unfold α_min, compute 1/2 + 1/(2*(3/2)) = 1/2 + 1/3 = 5/6. Use field_simp and ring or norm_num.
-/
theorem alpha_min_at_three_halves : α_min (3/2) = 5/6 := by
  unfold α_min; norm_num;

/-
PROBLEM
(e) Spot check: α_min(4) = 5/8

PROVIDED SOLUTION
Unfold α_min, compute 1/2 + 1/(2*4) = 1/2 + 1/8 = 5/8. Use field_simp and norm_num.
-/
theorem alpha_min_at_four : α_min 4 = 5/8 := by
  unfold α_min; norm_num;

/-
PROBLEM
(e) Spot check: α_min(8) = 9/16

PROVIDED SOLUTION
Unfold α_min, compute 1/2 + 1/(2*8) = 1/2 + 1/16 = 9/16. Use field_simp and norm_num.
-/
theorem alpha_min_at_eight : α_min 8 = 9/16 := by
  unfold α_min; norm_num;

/-
PROBLEM
(a) α_min is strictly decreasing on (0, ∞)

PROVIDED SOLUTION
α_min f = 1/2 + 1/(2f). For f₁ < f₂ both positive, 1/(2f₁) > 1/(2f₂), so α_min f₁ > α_min f₂. Use StrictAntiOn, showing that on (0,∞), 1/(2f) is strictly decreasing. Key: for 0 < f₁ < f₂, div_lt_div_of_pos_left (by positivity) and add_lt_add_left.
-/
theorem alpha_min_strictAntiOn :
    StrictAntiOn α_min (Set.Ioi 0) := by
      exact fun f hf g hg hfg => by unfold α_min; ring_nf; gcongr; aesop;

/-
PROBLEM
(a) α_min is convex on (0, ∞)

PROVIDED SOLUTION
α_min f = 1/2 + 1/(2f). The second derivative is 1/f³ > 0 on (0,∞), so it's convex. In Lean, show ConvexOn by noting α_min = (fun f => 1/2) + (fun f => 1/(2*f)). The constant is convex. For 1/(2f), use that f ↦ 1/f is convex on (0,∞) (convexOn_inv from Mathlib) scaled by 1/2. ConvexOn.add combines them.
-/
theorem alpha_min_convexOn :
    ConvexOn ℝ (Set.Ioi 0) α_min := by
      unfold α_min;
      fapply convexOn_of_deriv2_nonneg;
      · exact convex_Ioi 0;
      · exact continuousOn_of_forall_continuousAt fun x hx => ContinuousAt.add continuousAt_const <| ContinuousAt.div continuousAt_const ( continuousAt_const.mul continuousAt_id ) <| mul_ne_zero two_ne_zero hx.out.ne';
      · exact DifferentiableOn.add ( differentiableOn_const _ ) ( DifferentiableOn.div ( differentiableOn_const _ ) ( differentiableOn_id.const_mul _ ) ( by intro x hx; aesop ) );
      · norm_num [ mul_comm ];
        exact DifferentiableOn.mul ( DifferentiableOn.inv ( differentiableOn_pow 2 ) fun x hx => ne_of_gt ( sq_pos_of_pos hx ) ) ( differentiableOn_const _ );
      · norm_num [ mul_comm ];
        intro x hx; norm_num [ hx.ne' ] ; ring_nf; norm_num [ hx.ne' ] ; positivity;

/-
PROBLEM
(c) α_min(f) → 1/2 as f → ∞

PROVIDED SOLUTION
α_min f = 1/2 + 1/(2f). As f → ∞, 1/(2f) → 0, so α_min f → 1/2. Show this using Tendsto for 1/(2*f) → 0 and then adding constant 1/2. Use tendsto_const_nhds.add for the sum, and show 1/(2*f) → 0 using tendsto_inv_atTop_zero composed with tendsto_id.mul_atTop or similar.
-/
theorem alpha_min_tendsto :
    Tendsto α_min atTop (nhds (1/2)) := by
      -- Split the limit into the sum of two limits.
      have h_split : Filter.Tendsto (fun f : ℝ => 1 / 2 + 1 / (2 * f)) Filter.atTop (nhds (1 / 2 + 0)) := by
        exact tendsto_const_nhds.add ( tendsto_const_nhds.div_atTop <| Filter.tendsto_id.const_mul_atTop zero_lt_two );
      simpa using h_split.congr fun f => by unfold α_min; ring;

/-
PROBLEM
(d) For any achievable accuracy α ∈ (1/2, 1), the minimum fee_mult is 1/(2α - 1)

PROVIDED SOLUTION
α_min(1/(2α-1)) = 1/2 + 1/(2·(1/(2α-1))) = 1/2 + (2α-1)/2 = 1/2 + α - 1/2 = α. Use field_simp and ring, noting 2α-1 > 0 from hα_lo.
-/
theorem f_min_characterization (α : ℝ) (hα_lo : 1/2 < α) (hα_hi : α < 1) :
    α_min (1/(2*α - 1)) = α := by
      unfold α_min; ring_nf; norm_num [ show α ≠ 0 by linarith ] ; ring;

/-
PROBLEM
(d) converse: if α_min(f) ≤ α then f ≥ 1/(2α-1)

PROVIDED SOLUTION
From α_min(f) ≤ α, we get 1/2 + 1/(2f) ≤ α, so 1/(2f) ≤ α - 1/2 = (2α-1)/2. Since f > 0, this gives 1 ≤ f(2α-1), so f ≥ 1/(2α-1). Use field_simp and then linarith or appropriate division lemmas.
-/
theorem f_min_is_minimum (α f : ℝ) (hα_lo : 1/2 < α) (hf : 0 < f) (h : α_min f ≤ α) :
    1/(2*α - 1) ≤ f := by
      unfold α_min at h;
      rw [ div_le_iff₀ ] <;> nlinarith [ one_div_mul_cancel ( by positivity : ( 2 * f ) ≠ 0 ) ]

end
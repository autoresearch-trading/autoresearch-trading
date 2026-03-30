import Mathlib

/-!
# Theorem B: VPIN Bounds and Adverse Selection

Formalizes properties of the Volume-Synchronized Probability of Informed Trading (VPIN),
defined as VPIN = |V_buy - V_sell| / (V_buy + V_sell).

References: Easley, López de Prado, O'Hara (2012).
-/

noncomputable section

open Real

/-- VPIN: Volume-synchronized probability of informed trading. -/
def VPIN (V_buy V_sell : ℝ) : ℝ := |V_buy - V_sell| / (V_buy + V_sell)

/-
PROBLEM
============================================================================
CLAIM B1: VPIN ∈ [0, 1]
============================================================================

B1a: VPIN is non-negative.

PROVIDED SOLUTION
VPIN = |V_buy - V_sell| / (V_buy + V_sell). Both numerator (abs value) and denominator (sum of nonneg with positive sum) are nonneg. Use div_nonneg and abs_nonneg.
-/
theorem claim_B1_nonneg {V_buy V_sell : ℝ} (hb : V_buy ≥ 0) (hs : V_sell ≥ 0)
    (hpos : V_buy + V_sell > 0) :
    VPIN V_buy V_sell ≥ 0 := by
  exact div_nonneg ( abs_nonneg _ ) hpos.le

/-
PROBLEM
B1b: VPIN is at most 1.

PROVIDED SOLUTION
VPIN = |a-b|/(a+b). Since a,b ≥ 0, |a-b| ≤ a+b (by abs_sub_le or by cases). So VPIN ≤ (a+b)/(a+b) = 1. Use div_le_one (with a+b > 0) and the triangle inequality. Specifically, |a - b| ≤ |a| + |b| = a + b since a, b ≥ 0. Or use abs_sub_abs_le_abs_sub.
-/
theorem claim_B1_le_one {V_buy V_sell : ℝ} (hb : V_buy ≥ 0) (hs : V_sell ≥ 0)
    (hpos : V_buy + V_sell > 0) :
    VPIN V_buy V_sell ≤ 1 := by
  exact div_le_one_of_le₀ ( abs_le.mpr ⟨ by linarith, by linarith ⟩ ) ( by linarith )

/-
PROBLEM
============================================================================
CLAIM B2: VPIN = 0 iff V_buy = V_sell
============================================================================

B2: VPIN equals zero if and only if buy and sell volumes are equal.

PROVIDED SOLUTION
VPIN = |a-b|/(a+b) = 0 iff |a-b| = 0 (since a+b > 0, div is 0 iff numerator is 0). And |a-b| = 0 iff a = b. Use div_eq_zero_iff, abs_eq_zero, sub_eq_zero.
-/
theorem claim_B2 {V_buy V_sell : ℝ} (hb : V_buy ≥ 0) (hs : V_sell ≥ 0)
    (hpos : V_buy + V_sell > 0) :
    VPIN V_buy V_sell = 0 ↔ V_buy = V_sell := by
  -- By definition of VPIN, we have VPIN V_buy V_sell = |V_buy - V_sell| / (V_buy + V_sell).
  have h_def : VPIN V_buy V_sell = abs (V_buy - V_sell) / (V_buy + V_sell) := by
    rfl;
  grind

/-
PROBLEM
============================================================================
CLAIM B3: VPIN = 1 iff completely one-sided flow
============================================================================

B3: VPIN equals one if and only if one side has zero volume.

PROVIDED SOLUTION
VPIN = |a-b|/(a+b) = 1 iff |a-b| = a+b (since a+b > 0, use div_eq_one_iff_eq). For a,b ≥ 0: |a-b| = max(a,b) - min(a,b) and a+b = max(a,b) + min(a,b). So equality holds iff min(a,b) = 0, i.e., a = 0 or b = 0. Alternatively, |a-b| = a+b iff (a-b = a+b or a-b = -(a+b)), i.e., b = 0 or a = 0. Use abs_eq with nonneg hypotheses.
-/
theorem claim_B3 {V_buy V_sell : ℝ} (hb : V_buy ≥ 0) (hs : V_sell ≥ 0)
    (hpos : V_buy + V_sell > 0) :
    VPIN V_buy V_sell = 1 ↔ V_buy = 0 ∨ V_sell = 0 := by
  constructor;
  · intro h_eq_one
    have h_abs : |V_buy - V_sell| = V_buy + V_sell := by
      unfold VPIN at h_eq_one; rw [ div_eq_iff ] at h_eq_one <;> linarith;
    have h_cases : V_buy = 0 ∨ V_sell = 0 := by
      cases abs_cases ( V_buy - V_sell ) <;> first | left; linarith | right; linarith;
    exact h_cases;
  · unfold VPIN; intro h; rcases h with ( rfl | rfl ) <;> norm_num [ hpos.ne' ] ;
    · rw [ abs_of_nonneg hs, div_self ( by linarith ) ];
    · rw [ abs_of_nonneg hb, div_self ( ne_of_gt ( by linarith ) ) ]

/-
PROBLEM
============================================================================
CLAIM B4: VPIN is quasi-convex
============================================================================

B4: VPIN is quasi-convex: for any threshold c ∈ [0,1], the sub-level set is convex.
    Formally, if VPIN(a₁,b₁) ≤ c and VPIN(a₂,b₂) ≤ c, then for t ∈ [0,1],
    VPIN(t·a₁+(1-t)·a₂, t·b₁+(1-t)·b₂) ≤ c.

    The key insight: |a-b|/(a+b) ≤ c iff |a-b| ≤ c·(a+b), which defines a region
    bounded by linear inequalities (1-c)a ≤ (1+c)b and (1-c)b ≤ (1+c)a, hence convex.

PROVIDED SOLUTION
VPIN(x,y) ≤ c means |x-y|/(x+y) ≤ c, i.e. |x-y| ≤ c*(x+y) since x+y > 0. This is equivalent to the pair of linear inequalities: x - y ≤ c*(x+y) and y - x ≤ c*(x+y), i.e., (1-c)*x ≤ (1+c)*y and (1-c)*y ≤ (1+c)*x.

For the convex combination: if |a₁-b₁| ≤ c*(a₁+b₁) and |a₂-b₂| ≤ c*(a₂+b₂), then for x = t*a₁+(1-t)*a₂, y = t*b₁+(1-t)*b₂:
|x - y| = |t*(a₁-b₁) + (1-t)*(a₂-b₂)| ≤ t*|a₁-b₁| + (1-t)*|a₂-b₂| ≤ t*c*(a₁+b₁) + (1-t)*c*(a₂+b₂) = c*(x+y).

Key steps: unfold VPIN, use div_le_iff (since sum > 0), then triangle inequality for the convex combo, then the hypothesis bounds, and div_le_iff back. The critical step is abs_add for the triangle inequality on t*(a₁-b₁) + (1-t)*(a₂-b₂), which actually requires ConvexOn-like reasoning.

More precisely:
1. From h₁: |a₁-b₁| ≤ c*(a₁+b₁) (using div_le_iff with hpos₁)
2. From h₂: |a₂-b₂| ≤ c*(a₂+b₂) (using div_le_iff with hpos₂)
3. |x-y| = |t*(a₁-b₁)+(1-t)*(a₂-b₂)| ≤ |t*(a₁-b₁)| + |(1-t)*(a₂-b₂)| = t*|a₁-b₁| + (1-t)*|a₂-b₂| (using abs_add, abs_mul, abs_of_nonneg for t and 1-t)
4. ≤ t*c*(a₁+b₁) + (1-t)*c*(a₂+b₂) = c*(x+y)
5. Therefore VPIN(x,y) = |x-y|/(x+y) ≤ c
-/
theorem claim_B4 {a₁ b₁ a₂ b₂ c t : ℝ}
    (ha₁ : a₁ ≥ 0) (hb₁ : b₁ ≥ 0) (hpos₁ : a₁ + b₁ > 0)
    (ha₂ : a₂ ≥ 0) (hb₂ : b₂ ≥ 0) (hpos₂ : a₂ + b₂ > 0)
    (hc₀ : c ≥ 0) (hc₁ : c ≤ 1)
    (ht₀ : 0 ≤ t) (ht₁ : t ≤ 1)
    (h₁ : VPIN a₁ b₁ ≤ c) (h₂ : VPIN a₂ b₂ ≤ c) :
    VPIN (t * a₁ + (1 - t) * a₂) (t * b₁ + (1 - t) * b₂) ≤ c := by
  unfold VPIN at *;
  rw [ div_le_iff₀ ] at *;
  · exact abs_le.mpr ⟨ by nlinarith [ abs_le.mp h₁, abs_le.mp h₂ ], by nlinarith [ abs_le.mp h₁, abs_le.mp h₂ ] ⟩;
  · linarith;
  · linarith;
  · cases lt_or_eq_of_le ht₀ <;> cases lt_or_eq_of_le ht₁ <;> nlinarith

/-
PROBLEM
============================================================================
CLAIM B5: Connection to Kyle model (simplified deterministic results)
============================================================================

B5a: For i.i.d. flow, the VPIN ratio satisfies √n / n = 1/√n.
    This captures the key scaling: E[|Σ OFI_i|]/E[Σ|OFI_i|] ≈ 1/√n.

PROVIDED SOLUTION
Real.sqrt n / n = Real.sqrt n / (Real.sqrt n * Real.sqrt n) = 1 / Real.sqrt n. Use Real.sq_sqrt (Nat.cast_nonneg) to get sqrt(n)^2 = n, then rewrite n as sqrt(n)*sqrt(n), and simplify. Or use div_eq_div_iff and cross-multiply: sqrt(n) * sqrt(n) = 1 * n = n, which holds by Real.mul_self_sqrt (Nat.cast_nonneg).
-/
theorem claim_B5a {n : ℕ} (hn : 0 < n) :
    Real.sqrt n / n = 1 / Real.sqrt n := by
  rw [ div_eq_div_iff, mul_comm ] <;> first | positivity | ring;
  exact Real.sq_sqrt <| Nat.cast_nonneg _

/-
PROBLEM
B5b: The i.i.d. VPIN scaling 1/√n approaches 0 as n → ∞.

PROVIDED SOLUTION
1/√n → 0 as n → ∞. This follows from tendsto_one_div_atTop_atTop composed with the fact that √n → ∞. Specifically, Real.tendsto_sqrt_atTop gives sqrt → ∞ on ℝ, and we compose with Nat.cast tendsto. Then use Filter.Tendsto.inv_tendsto_atTop or tendsto_const_div_atTop_nhds_0_nat or similar.
-/
theorem claim_B5b : Filter.Tendsto (fun n : ℕ => 1 / Real.sqrt n) Filter.atTop (nhds 0) := by
  simpa using tendsto_inv_atTop_nhds_zero_nat.sqrt

end
import Mathlib

/-!
# Fee-Adjusted Kelly Criterion for Triple Barrier Optimal Sizing

## Summary of findings

We formalize the mathematical claims about optimal position sizing with triple barrier
labeling and trading fees. Our analysis reveals:

1. **Claim 1 is INCORRECT as stated.** The claimed formula for f* has an error in the
   second term. The correct critical point of the growth rate function is:
     f* = (p_win - p_loss) · (1 - c) / ((p_win + p_loss) · c)
   The user claimed f* = (p_win - p_loss)/(p_win + p_loss) · (1/c) - 1/(p_win + p_loss),
   but the correct derivation gives:
     f* = (p_win - p_loss)/(p_win + p_loss) · (1/c) - (p_win - p_loss)/(p_win + p_loss)
   These differ by (1 - p_win + p_loss)/(p_win + p_loss), which is nonzero in general.
   We prove both that the corrected formula satisfies the FOC, and provide a
   counterexample showing the original formula does not.

2. **Claim 2** (Sortino ratio bound) involves informal heuristic arguments about
   distributional properties that are not rigorously derivable from the stated setup.
   We do not formalize this claim.

3. **Claim 3 is CORRECT.** The minimum accuracy α_min = 1/2 + 1/(2·f*),
   and for fee_mult = 8, this gives α_min = 9/16 = 56.25%.
   We verify this computation.
-/

noncomputable section

open Real

/-! ## Definitions -/

/-- The growth rate function (excluding the timeout term, which is constant in f).
    G(f) = p_win · log(1 + f·c - c) + p_loss · log(1 - f·c - c) -/
def growthRate (p_win p_loss c f : ℝ) : ℝ :=
  p_win * Real.log (1 + f * c - c) + p_loss * Real.log (1 - f * c - c)

/-- The CORRECTED optimal fee multiplier.
    Obtained by setting G'(f) = 0 and solving for f. -/
def optimalFeeMult (p_win p_loss c : ℝ) : ℝ :=
  (p_win - p_loss) * (1 - c) / ((p_win + p_loss) * c)

/-- The original (incorrect) claimed formula for f*. -/
def claimedFeeMult (p_win p_loss c : ℝ) : ℝ :=
  (p_win - p_loss) / (p_win + p_loss) * (1 / c) - 1 / (p_win + p_loss)

/-- The minimum accuracy required for positive expected growth after fees. -/
def minAccuracy (f_star : ℝ) : ℝ :=
  1 / 2 + 1 / (2 * f_star)

/-! ## Corrected Claim 1: The optimal fee multiplier satisfies the first-order condition -/

/-- The corrected f* satisfies the first-order optimality condition:
    p_win · (1 - f·c - c) = p_loss · (1 + f·c - c),
    which is equivalent to G'(f) = 0. -/
theorem optimalFeeMult_satisfies_foc
    (p_win p_loss c : ℝ)
    (hc : c ≠ 0) (hsum : p_win + p_loss ≠ 0) :
    let f := optimalFeeMult p_win p_loss c
    p_win * (1 - f * c - c) = p_loss * (1 + f * c - c) := by
  unfold optimalFeeMult
  grind

/-- The critical point is unique: any f satisfying the FOC equals optimalFeeMult. -/
theorem optimalFeeMult_unique
    (p_win p_loss c f : ℝ)
    (hc : c ≠ 0) (hsum : p_win + p_loss ≠ 0)
    (hfoc : p_win * (1 - f * c - c) = p_loss * (1 + f * c - c)) :
    f = optimalFeeMult p_win p_loss c := by
  unfold optimalFeeMult
  rw [eq_div_iff] <;> cases lt_or_gt_of_ne hc <;> cases lt_or_gt_of_ne hsum <;> nlinarith

/-- Counterexample: The original claimed formula does NOT satisfy the FOC
    for p_win = 3/5, p_loss = 3/10, c = 1/10. -/
theorem claimed_formula_wrong :
    let p_win : ℝ := 3 / 5
    let p_loss : ℝ := 3 / 10
    let c : ℝ := 1 / 10
    let f := claimedFeeMult p_win p_loss c
    p_win * (1 - f * c - c) ≠ p_loss * (1 + f * c - c) := by
  norm_num [claimedFeeMult]

/-! ## Claim 3: Minimum accuracy for positive expectancy after fees -/

/-- The minimum accuracy is definitionally 1/2 + 1/(2·f*). -/
theorem minAccuracy_def (f_star : ℝ) :
    minAccuracy f_star = 1 / 2 + 1 / (2 * f_star) := rfl

/-- For fee_mult = 8, the minimum accuracy is 9/16 = 56.25%. -/
theorem minAccuracy_eight : minAccuracy 8 = 9 / 16 := by
  unfold minAccuracy; norm_num

/-- The minimum accuracy exceeds 1/2 when f* > 0 (fees require better than
    coin-flip accuracy). -/
theorem minAccuracy_gt_half (f_star : ℝ) (hf : f_star > 0) :
    minAccuracy f_star > 1 / 2 := by
  unfold minAccuracy; aesop

/-- The minimum accuracy decreases as the fee multiplier increases
    (wider barriers are more forgiving of accuracy). -/
theorem minAccuracy_anti (f₁ f₂ : ℝ) (hf₁ : 0 < f₁) (_hf₂ : 0 < f₂) (h : f₁ < f₂) :
    minAccuracy f₂ < minAccuracy f₁ := by
  unfold minAccuracy; gcongr

end

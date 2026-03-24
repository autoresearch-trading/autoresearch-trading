import Mathlib

/-!
# Universal Approximation Bound for Tape Reading with Finite Features

## Analysis of Claims

We formalize the mathematical content underlying three claims about ML architecture
design for trade classification.

### Claim 1: Hybrid dominance
The **weak** version (≤) is TRUE by monotonicity of infimum over larger sets.
The **strict** version (<) is FALSE in general — we provide a counterexample.

### Claim 2: Minimum window size
We verify the numerical computation ⌈log(1/46) / log(0.95)⌉ = 75.

### Claim 3: Majority voting
We formalize structural results about ensemble majority voting.
The "correction_3class" term in the original claim is undefined, making the
exact 3-class statement ill-posed.
-/

open scoped BigOperators

/-! ## Part 1: Weak Dominance (TRUE)

If the hybrid class contains both component classes, then the infimum of
risk over the hybrid class is at most the minimum of the infima over the
component classes. This is just monotonicity of infimum w.r.t. set inclusion.
-/

/-- Weak dominance: hybrid risk ≤ min(flat risk, TCN risk).
    If the hybrid architecture can represent everything either component can,
    its optimal risk is bounded above by the better of the two components. -/
theorem hybrid_weak_dominance
    (R_flat R_tcn R_hybrid : ℝ)
    (h_flat : R_hybrid ≤ R_flat)
    (h_tcn : R_hybrid ≤ R_tcn) :
    R_hybrid ≤ min R_flat R_tcn :=
  le_min h_flat h_tcn

/-- Infimum over a superset is ≤ infimum over a subset (for real-valued functions). -/
theorem csInf_le_csInf_of_subset
    (S₁ S : Set ℝ) (h₁ : S₁ ⊆ S) (hne : S₁.Nonempty) (hbdd : BddBelow S) :
    sInf S ≤ sInf S₁ :=
  csInf_le_csInf hbdd hne h₁

/-- General weak dominance: if F_flat ⊆ F_hybrid and F_tcn ⊆ F_hybrid, then
    inf_{F_hybrid} R ≤ min(inf_{F_flat} R, inf_{F_tcn} R). -/
theorem hybrid_weak_dominance_general
    (R_flat R_tcn R_hybrid : Set ℝ)
    (h₁ : R_flat ⊆ R_hybrid) (h₂ : R_tcn ⊆ R_hybrid)
    (hne₁ : R_flat.Nonempty) (hne₂ : R_tcn.Nonempty)
    (hbdd : BddBelow R_hybrid) :
    sInf R_hybrid ≤ min (sInf R_flat) (sInf R_tcn) := by
  apply le_min
  · exact csInf_le_csInf_of_subset R_flat R_hybrid h₁ hne₁ hbdd
  · exact csInf_le_csInf_of_subset R_tcn R_hybrid h₂ hne₂ hbdd

/-! ## Part 1b: Strict Dominance is FALSE — Counterexample

We show that strict dominance does NOT hold in general.

**Counterexample**: Let X ∈ ℝ^{W·n} and Y = sign(x₁) (label depends only on
the first feature). Then F_flat already achieves Bayes risk R = 0.
Since R ≥ 0, the hybrid cannot achieve R < 0, so R_hybrid = R_flat = 0.
Strict inequality fails.
-/

/-- **Counterexample to Claim 1 (strict dominance).**
    There exist valid risk values where equality holds, disproving
    the claim that strict inequality always holds.
    Witness: R_flat = R_tcn = R_hybrid = 0. -/
theorem strict_dominance_false :
    ¬ (∀ (R_flat R_tcn R_hybrid : ℝ),
      R_hybrid ≤ R_flat → R_hybrid ≤ R_tcn →
      R_hybrid < min R_flat R_tcn) :=
  fun h => absurd (h 0 0 0 (le_refl 0) (le_refl 0)) (by norm_num)

/-- Strict dominance also fails for the set-theoretic version:
    there exist sets where inf over the superset equals inf over a subset.
    Witness: S₁ = S₂ = S = {0}. -/
theorem strict_dominance_false_sets :
    ¬ (∀ (S₁ S₂ S : Set ℝ),
      S₁ ⊆ S → S₂ ⊆ S → S₁.Nonempty → S₂.Nonempty → BddBelow S →
      sInf S < min (sInf S₁) (sInf S₂)) := by
  simp +zetaDelta
  exact ⟨{0}, {0}, {0}, by norm_num⟩

/-! ## Part 2: Minimum Window Size Computation

We verify: ⌈log(1/46) / log(0.95)⌉ = 75

Since exact real logarithm computation is difficult in Lean, we prove
the equivalent characterization: 0.95^74 > 1/46 ∧ 0.95^75 ≤ 1/46.
In integer arithmetic: 95^74 * 46 > 100^74 ∧ 95^75 * 46 ≤ 100^75.
-/

/-- 0.95^75 ≤ 1/46: the window size W=75 is large enough to detect
    the AR(1) autocorrelation with ρ=0.95 among n=46 features. -/
theorem wmin_bound_upper : (95 : ℤ)^75 * 46 ≤ 100^75 := by
  native_decide

/-- 0.95^74 > 1/46: the window size W=74 is NOT large enough,
    confirming W_min = 75 exactly. -/
theorem wmin_bound_lower : (100 : ℤ)^74 < (95 : ℤ)^74 * 46 := by
  native_decide

/-! ## Part 3: Majority Voting

### Structural results about ensemble classification

The key insight: majority voting improves accuracy only when individual
accuracy exceeds 1/2 (for binary) or 1/K (for K-class problems).
With α = 0.45 for 3-class, α > 1/3 so improvement IS possible in theory,
but the original formula with undefined "correction_3class" is ill-posed.
-/

/-- The sign of marginal improvement depends on whether α > 1/2. -/
theorem marginal_improvement_sign (α : ℝ) :
    (2 * α - 1) > 0 ↔ α > 1 / 2 := by
  constructor <;> intro <;> linarith

/-- For binary classification, the ensemble accuracy of M=5 classifiers
    with accuracy 45% is WORSE than individual accuracy.
    P(majority correct) = C(5,3)·0.45³·0.55² + C(5,4)·0.45⁴·0.55 + 0.45⁵ < 0.45. -/
theorem ensemble_M5_binary_hurts :
    10 * (45 : ℚ)^3 * 55^2 + 5 * 45^4 * 55 + 45^5 < 45 * 100^4 := by
  norm_num

/-- The exact ensemble accuracy for M=5, α=0.45 (binary) is 650997/1600000 ≈ 0.4069.
    This is the probability that at least 3 out of 5 classifiers are correct. -/
theorem ensemble_M5_value :
    (10 * (45 : ℚ)^3 * 55^2 + 5 * 45^4 * 55 + 45^5) / 100^5 =
    650997 / 1600000 := by
  norm_num

/-- For 3-class problems with α = 0.45 > 1/3, the situation is different
    from binary. The threshold for majority voting improvement is 1/3, not 1/2.
    We verify: 0.45 > 1/3, so the 3-class setting MAY benefit from ensembling. -/
theorem three_class_threshold : (45 : ℚ) / 100 > 1 / 3 := by
  norm_num

/-! ## Summary

### What is TRUE and proven:
1. **Weak dominance** (`hybrid_weak_dominance`, `hybrid_weak_dominance_general`):
   The hybrid class achieves risk ≤ min(flat risk, TCN risk), simply because
   it is a larger function class.
2. **Window size W_min = 75**: ⌈log(1/46)/log(0.95)⌉ = 75 verified via
   integer arithmetic (`wmin_bound_upper`, `wmin_bound_lower`).
3. **Majority voting structure**: The marginal improvement sign depends on
   α vs 1/2 for binary classification (`marginal_improvement_sign`).

### What is FALSE or ill-posed:
1. **Claim 1 (strict dominance)** is FALSE (`strict_dominance_false`,
   `strict_dominance_false_sets`): When one component already achieves Bayes
   risk, the hybrid cannot strictly improve. Weak dominance (≤) is the best
   one can prove in general.
2. **Claim 2 interpretation**: The formula gives W_min = 75, verified numerically.
   But the conclusion "W=50 is too small for TCN to add value" is a heuristic
   interpretation, not a rigorous theorem — it depends on the specific data
   distribution, not just AR(1) autocorrelation.
3. **Claim 3 (ensemble)**: The "correction_3class" term is undefined, making
   the claim ill-posed. For binary with α = 0.45 < 0.5, majority voting
   HURTS (`ensemble_M5_binary_hurts`): ensemble accuracy ≈ 0.407 < 0.45.
   For 3-class with α = 0.45 > 1/3 (`three_class_threshold`), improvement
   is possible in principle, but requires a multinomial (not binomial) analysis.
-/

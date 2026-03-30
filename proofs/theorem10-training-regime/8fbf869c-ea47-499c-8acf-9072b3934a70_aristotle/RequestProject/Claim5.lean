/-
  CLAIM 5: Ensemble of 5 for 3-class — majority voting analysis.

  For a 3-class problem with per-model accuracy α,
  majority voting with M=5 models: predict class c if ≥ 3 models agree on c.

  KEY FINDING: The user's Claim 5(b) states that α > 1/3 suffices for ensemble
  to beat individual. This is FALSE for majority voting (≥3 of 5).
  The correct threshold is α > 1/2 (same as binary Condorcet).

  Counterexample: α = 2/5 > 1/3, but ensembleAccuracy5(2/5) ≈ 0.317 < 0.4.

  The reason: for majority voting, we need ≥3 out of 5 to vote for the correct class.
  The number of correct votes follows Binomial(5, α) regardless of how errors split
  among wrong classes. So the threshold is the same as the binary Condorcet theorem.
-/
import Mathlib

open Finset BigOperators Real

noncomputable section

/-- Ensemble accuracy with majority voting (≥3 of 5) for any multi-class problem.
    P(≥3 correct out of 5) = C(5,3)·α³·(1-α)² + C(5,4)·α⁴·(1-α) + C(5,5)·α⁵
    This depends only on α (individual accuracy), not on the number of classes,
    because we only need to count how many models vote for the correct class. -/
def ensembleAccuracy5 (α : ℝ) : ℝ :=
  10 * α^3 * (1 - α)^2 + 5 * α^4 * (1 - α) + α^5

/-
PROBLEM
Claim 5(b) CORRECTED: For α > 1/2 (not 1/3!), ensemble accuracy exceeds
    individual accuracy with majority voting and 5 models.

PROVIDED SOLUTION
Using ensemble_accuracy_simplified, we need 6α⁵-15α⁴+10α³ > α. This is equivalent to α(6α⁴-15α³+10α²-1) > 0. Since α > 1/2 > 0, we need 6α⁴-15α³+10α²-1 > 0. Factor: 6α⁴-15α³+10α²-1 = (2α-1)²(3α²-1) + some positive term... Actually let's try: f(α) = 6α⁵-15α⁴+10α³-α. f(α) = α(6α⁴-15α³+10α²-1). We can verify that g(α) = 6α⁴-15α³+10α²-1 factors as (2α-1)²(something). g(1/2) = 6/16-15/8+10/4-1 = 3/8-15/8+20/8-8/8 = 0. So (α-1/2) is a root. Also g'(α) = 24α³-45α²+20α, g'(1/2) = 3-45/4+10 = 7/4 ≠ 0, so it's a simple root. By polynomial division: g(α) = (2α-1)(3α³-6α²+2α+1). Check: (2α-1)(3α³-6α²+2α+1) = 6α⁴-12α³+4α²+2α-3α³+6α²-2α-1 = 6α⁴-15α³+10α²-1. ✓ Now we need (2α-1)(3α³-6α²+2α+1) > 0 for 1/2 < α < 1. Since 2α-1 > 0, need 3α³-6α²+2α+1 > 0. At α=1/2: 3/8-6/4+1+1 = 3/8-3/2+2 = 3/8-12/8+16/8 = 7/8 > 0. At α=1: 3-6+2+1 = 0. So check if 3α³-6α²+2α+1 = (1-α)(something). 3(1)³-6(1)²+2(1)+1 = 0. Factor: divide by (α-1): 3α³-6α²+2α+1 = (1-α)(-3α²+3α+1). So g(α) = (2α-1)(1-α)(-3α²+3α+1). Then f(α) = α·(2α-1)·(1-α)·(-3α²+3α+1). For 1/2 < α < 1: α > 0, 2α-1 > 0, 1-α > 0. Need -3α²+3α+1 > 0, i.e., 3α²-3α-1 < 0. Discriminant: 9+12 = 21. Roots: (3±√21)/6. (3+√21)/6 ≈ (3+4.58)/6 ≈ 1.26. So 3α²-3α-1 < 0 for α < 1.26, which includes (1/2, 1). Hence f(α) > 0.
-/
theorem ensemble_beats_individual {α : ℝ} (hα_lower : 1/2 < α) (hα_upper : α < 1) :
    ensembleAccuracy5 α > α := by
  unfold ensembleAccuracy5; norm_num; nlinarith [ mul_pos ( sub_pos.2 hα_lower ) ( sub_pos.2 hα_upper ), pow_pos ( sub_pos.2 hα_lower ) 3, pow_pos ( sub_pos.2 hα_upper ) 3 ] ;

/-
PROBLEM
At α = 1/2, ensemble accuracy equals individual accuracy exactly.

PROVIDED SOLUTION
Unfold ensembleAccuracy5 and compute: 10*(1/2)^3*(1/2)^2 + 5*(1/2)^4*(1/2) + (1/2)^5 = 10/32 + 5/32 + 1/32 = 16/32 = 1/2. Use norm_num after unfolding.
-/
theorem ensemble_accuracy_at_half :
    ensembleAccuracy5 (1/2) = 1/2 := by
  unfold ensembleAccuracy5; norm_num;

/-
PROBLEM
COUNTEREXAMPLE to original Claim 5(b):
    At α = 2/5 > 1/3, ensemble accuracy < individual accuracy.
    ensembleAccuracy5(2/5) = 992/3125 < 2/5 = 1250/3125

PROVIDED SOLUTION
Unfold ensembleAccuracy5 and compute with α=2/5. norm_num after unfolding.
-/
theorem ensemble_counterexample :
    ensembleAccuracy5 (2/5) < 2/5 := by
  unfold ensembleAccuracy5; norm_num;

/-
PROBLEM
The ensemble accuracy polynomial simplifies to:
    ensembleAccuracy5 α = 6α⁵ - 15α⁴ + 10α³

PROVIDED SOLUTION
Unfold ensembleAccuracy5 and expand: 10α³(1-α)² + 5α⁴(1-α) + α⁵. Expand (1-α)² = 1-2α+α², multiply by 10α³: 10α³-20α⁴+10α⁵. Then 5α⁴-5α⁵. Then α⁵. Total: 10α³-20α⁴+10α⁵+5α⁴-5α⁵+α⁵ = 10α³-15α⁴+6α⁵. Use ring after unfolding.
-/
theorem ensemble_accuracy_simplified (α : ℝ) :
    ensembleAccuracy5 α = 6 * α^5 - 15 * α^4 + 10 * α^3 := by
  unfold ensembleAccuracy5; ring;

end
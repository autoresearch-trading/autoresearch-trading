/-
# Claim 4: Two-pass Kelly iteration converges

If T : ℝ → ℝ is a contraction mapping on a complete metric space,
then by Banach fixed point theorem:
(a) A unique fixed point f* exists
(b) The iteration converges from any starting point
(c) The convergence rate is geometric

We use Mathlib's ContractingWith API.
-/
import Mathlib

open Metric Filter Topology

noncomputable section

/-- The two-pass iteration map -/
def T_iter (p_win p_loss : ℝ → ℝ) (c : ℝ) (f : ℝ) : ℝ :=
  (p_win f - p_loss f) * (1 - c) / ((p_win f + p_loss f) * c)

/-
PROBLEM
(a) If T is a contraction on a complete nonempty metric space,
    there exists a unique fixed point.

PROVIDED SOLUTION
Use hT.fixedPoint_isFixedPt and hT.fixedPoint_unique. Existence: ⟨hT.fixedPoint, hT.fixedPoint_isFixedPt⟩. Uniqueness: for any x with T x = x, hT.fixedPoint_unique says x = fixedPoint, so they're equal.
-/
theorem contraction_has_unique_fixedPoint
    {X : Type*} [MetricSpace X] [CompleteSpace X] [Nonempty X]
    {T : X → X} {k : NNReal} (hk : k < 1) (hT : ContractingWith k T) :
    ∃! x : X, T x = x := by
      obtain ⟨x, hx⟩ : ∃ x : X, T x = x := by
        have := hT.exists_fixedPoint;
        contrapose! this;
        refine' ⟨ Classical.arbitrary X, _, _ ⟩ <;> simp_all +decide [ Function.IsFixedPt ];
        exact edist_ne_top _ _;
      refine' ⟨ x, hx, fun y hy => _ ⟩;
      have := hT.dist_le_mul y x;
      contrapose! this;
      aesop

/-
PROBLEM
(b) The iteration converges to the fixed point

PROVIDED SOLUTION
Direct from Mathlib: ContractingWith.tendsto_iterate_fixedPoint.
-/
theorem contraction_iteration_converges
    {X : Type*} [MetricSpace X] [CompleteSpace X] [Nonempty X]
    {T : X → X} {k : NNReal} (hk : k < 1) (hT : ContractingWith k T)
    (x₀ : X) :
    Tendsto (fun n => T^[n] x₀) atTop (nhds (hT.fixedPoint)) := by
      exact hT.tendsto_iterate_fixedPoint x₀

/-
PROBLEM
(c) Geometric convergence rate

PROVIDED SOLUTION
Use ContractingWith.dist_iterate_fixedPoint_le or apriori_dist_iterate_fixedPoint_le from Mathlib. The bound should be dist x₀ (T x₀) * k^n / (1 - k).
-/
theorem contraction_geometric_rate
    {X : Type*} [MetricSpace X] [CompleteSpace X] [Nonempty X]
    {T : X → X} {k : NNReal} (hk : k < 1) (hT : ContractingWith k T)
    (x₀ : X) (n : ℕ) :
    dist (T^[n] x₀) (hT.fixedPoint) ≤
      dist x₀ (T x₀) * (k : ℝ)^n / (1 - (k : ℝ)) := by
        convert hT.apriori_dist_iterate_fixedPoint_le x₀ n using 1

end
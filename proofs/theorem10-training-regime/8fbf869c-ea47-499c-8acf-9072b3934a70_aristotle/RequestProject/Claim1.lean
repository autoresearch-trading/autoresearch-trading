/-
  CLAIM 1: Focal Loss reduces to cross-entropy for well-calibrated classifiers.

  Focal loss: FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
  Cross-entropy: CE(p_t) = -α_t · log(p_t)
-/
import Mathlib

open Real

noncomputable section

/-- Focal loss function -/
def focalLoss (α_t : ℝ) (γ : ℝ) (p_t : ℝ) : ℝ :=
  -α_t * (1 - p_t) ^ γ * Real.log p_t

/-- Weighted cross-entropy -/
def weightedCE (α_t : ℝ) (p_t : ℝ) : ℝ :=
  -α_t * Real.log p_t

/-
PROBLEM
Claim 1(a): When γ = 0, focal loss reduces to weighted cross-entropy.

PROVIDED SOLUTION
Unfold focalLoss and weightedCE. When γ=0, (1-p_t)^0 = 1, so the expression simplifies to -α_t * 1 * log(p_t) = -α_t * log(p_t).
-/
theorem focal_loss_gamma_zero (α_t p_t : ℝ) :
    focalLoss α_t 0 p_t = weightedCE α_t p_t := by
  -- Since $(1 - p_t)^0 = 1$, we can simplify the expression for focal loss when $\gamma = 0$.
  simp [focalLoss, weightedCE]

/-
PROBLEM
Claim 1(b): When γ > 0 and p_t → 1, FL → 0 faster than CE.
    More precisely, FL / CE = (1 - p_t)^γ → 0 as p_t → 1.

PROVIDED SOLUTION
As p_t → 1 within (0,1), (1-p_t) → 0. Since γ > 0, (1-p_t)^γ → 0. Use continuity of rpow and the fact that 0^γ = 0 for γ > 0. The function (1-p_t)^γ is continuous and equals 0 at p_t=1.
-/
theorem focal_loss_ratio_tendsto_zero {γ : ℝ} (hγ : γ > 0) :
    Filter.Tendsto (fun p_t : ℝ => (1 - p_t) ^ γ) (nhdsWithin 1 (Set.Ioo 0 1)) (nhds 0) := by
  refine' tendsto_nhdsWithin_of_tendsto_nhds ( ContinuousAt.tendsto _ |> fun h => h.trans _ ) <;> norm_num [ hγ.ne' ];
  exact ContinuousAt.rpow ( continuousAt_const.sub continuousAt_id ) continuousAt_const <| Or.inr <| by positivity;

/-
PROBLEM
Claim 1(c): When γ > 0 and p_t → 0, (1 - p_t)^γ → 1,
    so FL approaches CE.

PROVIDED SOLUTION
As p_t → 0 within (0,1), (1-p_t) → 1. So (1-p_t)^γ → 1^γ = 1. Use continuity of rpow.
-/
theorem focal_loss_ratio_at_zero (γ : ℝ) :
    Filter.Tendsto (fun p_t : ℝ => (1 - p_t) ^ γ) (nhdsWithin 0 (Set.Ioo 0 1)) (nhds 1) := by
  convert Filter.Tendsto.rpow ( tendsto_const_nhds.sub ( Filter.tendsto_id.mono_left inf_le_left ) ) tendsto_const_nhds _ using 2 <;> norm_num

/-- The "gradient magnitude factor" (without -α_t):
    g(p) = (1-p)^γ/p - γ(1-p)^(γ-1)·log(p) -/
def gradFactor (γ : ℝ) (p : ℝ) : ℝ :=
  (1 - p) ^ γ / p - γ * (1 - p) ^ (γ - 1) * Real.log p

/-
PROBLEM
At p_t = 1/3 with γ=1: gradFactor = (2/3)/(1/3) + 1·(1)·(-log(1/3))
    = 2 + log(3)

PROVIDED SOLUTION
Unfold gradFactor. With γ=1, p=1/3: (1-1/3)^1 / (1/3) - 1*(1-1/3)^0 * log(1/3) = (2/3)/(1/3) - 1*1*log(1/3) = 2 - log(1/3) = 2 + log(3). Use log_inv and the fact that log(1/3) = -log(3).
-/
theorem grad_factor_at_third :
    gradFactor 1 (1/3) = 2 + Real.log 3 := by
  unfold gradFactor; norm_num; ring_nf; norm_num [ Real.log_div ] ;

/-
PROBLEM
At p_t = 1/2 with γ=1: gradFactor = (1/2)/(1/2) + 1·1·(-log(1/2))
    = 1 + log 2

PROVIDED SOLUTION
Unfold gradFactor. With γ=1, p=1/2: (1-1/2)^1 / (1/2) - 1*(1-1/2)^0 * log(1/2) = (1/2)/(1/2) - log(1/2) = 1 - log(1/2) = 1 + log(2). Use log_inv.
-/
theorem grad_factor_at_half :
    gradFactor 1 (1/2) = 1 + Real.log 2 := by
  unfold gradFactor; norm_num; ring_nf; norm_num [ Real.log_div ] ;

/-
PROBLEM
The gradient factor at 1/3 exceeds that at 1/2 (for γ=1),
    demonstrating that harder examples get larger gradients.

PROVIDED SOLUTION
From the previous two lemmas, gradFactor 1 (1/3) = 2 + log 3 and gradFactor 1 (1/2) = 1 + log 2. We need 2 + log 3 > 1 + log 2, i.e., 1 + log 3 - log 2 > 0, i.e., 1 + log(3/2) > 0. Since log(3/2) > 0 (as 3/2 > 1), this is clear. Use grad_factor_at_third, grad_factor_at_half, Real.log_lt_log, and the fact that log is monotone.
-/
theorem grad_factor_third_gt_half :
    gradFactor 1 (1/3) > gradFactor 1 (1/2) := by
  -- Substitute the calculated values of the gradient factors into the inequality.
  have h_ineq : (2 + Real.log 3) > (1 + Real.log 2) := by
    linarith [ Real.log_lt_log ( by norm_num ) ( by norm_num : ( 3 : ℝ ) > 2 ) ];
  exact grad_factor_at_third.symm ▸ grad_factor_at_half.symm ▸ h_ineq

end
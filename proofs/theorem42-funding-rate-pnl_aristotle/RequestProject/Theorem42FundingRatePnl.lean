import Mathlib

/-!
# Theorem 42: Funding Rate P&L Impact

We formalize the funding rate P&L for perpetual futures positions.

## Definitions

- `d ∈ {+1, -1}` is the position direction (+1 = long, -1 = short).
- `r_f` is the (constant) funding rate.
- `T` is the number of settlement periods the position is held.
- The funding P&L over T periods (constant rate) is `F = -d * r_f * T`.

## Claims

1. The sign of F is always opposite to `d * r_f`.
2. Funding drag exceeds alpha when `|E[r_f]| * T_bar * H > alpha`.
3. Numerical verification for specific parameters.
4. A balanced long/short strategy (q = 1/2) has zero expected funding cost.
-/

open Finset

noncomputable section

/-! ### Claim 1: Funding P&L formula and sign -/

/-- The funding P&L for a constant funding rate `r_f`, position direction `d`,
    held for `T` settlement periods. -/
def fundingPnL (d : ℝ) (r_f : ℝ) (T : ℕ) : ℝ := -d * r_f * T

/-
PROBLEM
For a long position (d = +1) with constant positive funding rate r_f > 0
    held for T > 0 periods, the funding P&L is negative (the long pays funding).

PROVIDED SOLUTION
Unfold fundingPnL, simplify -1 * r_f * T. Since r_f > 0 and T > 0 (cast to ℝ), the product is positive, so the negation is negative.
-/
theorem funding_pnl_long_negative {r_f : ℝ} {T : ℕ} (hr : 0 < r_f) (hT : 0 < T) :
    fundingPnL 1 r_f T < 0 := by
  unfold fundingPnL; norm_num; positivity;

/-
PROBLEM
For a short position (d = -1) with constant positive funding rate r_f > 0
    held for T > 0 periods, the funding P&L is positive (the short receives funding).

PROVIDED SOLUTION
Unfold fundingPnL, simplify -(-1) * r_f * T = r_f * T > 0.
-/
theorem funding_pnl_short_positive {r_f : ℝ} {T : ℕ} (hr : 0 < r_f) (hT : 0 < T) :
    fundingPnL (-1) r_f T > 0 := by
  unfold fundingPnL; aesop;

/-
PROBLEM
The sign of the funding P&L is always opposite to the sign of `d * r_f`,
    when `d * r_f ≠ 0` and `T > 0`.

PROVIDED SOLUTION
Unfold fundingPnL. F = -d * r_f * T = -(d * r_f) * T. Since d*r_f > 0 and T > 0 (cast to ℝ), -(d*r_f)*T < 0.
-/
theorem funding_pnl_sign_opposite {d r_f : ℝ} {T : ℕ}
    (hdr : 0 < d * r_f) (hT : 0 < T) :
    fundingPnL d r_f T < 0 := by
  exact mul_neg_of_neg_of_pos ( by nlinarith ) ( by positivity )

/-
PROBLEM
Conversely, when `d * r_f < 0`, the funding P&L is positive.

PROVIDED SOLUTION
Unfold fundingPnL. F = -(d * r_f) * T. Since d*r_f < 0, -(d*r_f) > 0, and T > 0 as a natural cast to ℝ, so F > 0.
-/
theorem funding_pnl_sign_opposite' {d r_f : ℝ} {T : ℕ}
    (hdr : d * r_f < 0) (hT : 0 < T) :
    fundingPnL d r_f T > 0 := by
  exact mul_pos ( by nlinarith ) ( Nat.cast_pos.mpr hT )

/-- The variable-rate funding P&L as a sum. -/
def fundingPnLSum (d : ℝ) (r_f : ℕ → ℝ) (T : ℕ) : ℝ :=
  ∑ t ∈ range T, (-d * r_f t)

/-
PROBLEM
The variable-rate formula reduces to the constant-rate formula when r_f is constant.

PROVIDED SOLUTION
Unfold both definitions. The sum of T copies of (-d * r_f) equals -d * r_f * T. Use Finset.sum_const and Finset.card_range.
-/
theorem fundingPnLSum_const (d r_f : ℝ) (T : ℕ) :
    fundingPnLSum d (fun _ => r_f) T = fundingPnL d r_f T := by
  unfold fundingPnLSum fundingPnL; norm_num; ring;

/-! ### Claim 2: Funding drag exceeds alpha condition -/

/-- The funding cost per trade, given expected funding rate, hold duration, and
    settlements per step. -/
def fundingCostPerTrade (E_rf : ℝ) (T_bar : ℝ) (H : ℝ) : ℝ := |E_rf| * T_bar * H

/-- The maximum hold duration before funding cost exceeds alpha. -/
def T_max (alpha : ℝ) (E_rf : ℝ) (H : ℝ) : ℝ := alpha / (|E_rf| * H)

/-
PROBLEM
When T_bar exceeds T_max, the funding cost exceeds alpha.

PROVIDED SOLUTION
Unfold fundingCostPerTrade and T_max. hExceed gives alpha / (|E_rf| * H) < T_bar. Multiply both sides by |E_rf| * H (positive). We get alpha < |E_rf| * T_bar * H.
-/
theorem funding_exceeds_alpha {alpha E_rf H T_bar : ℝ}
    (hα : 0 < alpha) (hE : 0 < |E_rf|) (hH : 0 < H) (hT : 0 < T_bar)
    (hExceed : T_max alpha E_rf H < T_bar) :
    alpha < fundingCostPerTrade E_rf T_bar H := by
  unfold fundingCostPerTrade T_max at *; rw [ div_lt_iff₀ ] at * <;> nlinarith;

/-
PROBLEM
Conversely, when T_bar is at most T_max, the funding cost does not exceed alpha.

PROVIDED SOLUTION
Unfold definitions. hWithin gives T_bar ≤ alpha / (|E_rf| * H). Multiply both sides by |E_rf| * H (positive). We get |E_rf| * T_bar * H ≤ alpha.
-/
theorem funding_within_alpha {alpha E_rf H T_bar : ℝ}
    (hα : 0 < alpha) (hE : 0 < |E_rf|) (hH : 0 < H)
    (hWithin : T_bar ≤ T_max alpha E_rf H) :
    fundingCostPerTrade E_rf T_bar H ≤ alpha := by
  unfold T_max fundingCostPerTrade at *;
  rw [ le_div_iff₀ ] at hWithin <;> nlinarith

/-! ### Claim 3: Numerical verification for specific parameters -/

/-
PROBLEM
The funding cost per trade with the given parameters is less than 0.00005 (0.5 bps).
    Parameters: E[r_f] = 0.000012, T_bar = 1200, H = 1/360.

PROVIDED SOLUTION
Unfold fundingCostPerTrade. |0.000012| = 0.000012. Compute 0.000012 * 1200 * (1/360) = 0.000012 * 1200 / 360 = 0.0144/360 = 0.00004. This is less than 0.00005. Use norm_num.
-/
theorem numerical_funding_cost_small :
    fundingCostPerTrade 0.000012 1200 (1/360) < 0.00005 := by
  unfold fundingCostPerTrade; norm_num [ abs_of_pos ] ;

/-
PROBLEM
The ratio of funding cost to alpha is less than 0.01 (1%).
    Parameters: alpha = 0.008.

PROVIDED SOLUTION
The cost is 0.00004 (as computed above), and 0.00004 / 0.008 = 0.005 < 0.01. Unfold fundingCostPerTrade and use norm_num.
-/
theorem numerical_funding_ratio_small :
    fundingCostPerTrade 0.000012 1200 (1/360) / 0.008 < 0.01 := by
  unfold fundingCostPerTrade; norm_num;

/-! ### Claim 4: Asymmetric impact on longs vs shorts -/

/-- The expected funding cost per settlement for a strategy that is long with
    probability `q` and short with probability `1 - q`. -/
def expectedFundingCost (q : ℝ) (E_rf : ℝ) : ℝ := (2 * q - 1) * E_rf

/-
PROBLEM
The expected funding cost decomposes as q * E_rf - (1-q) * E_rf.
    This formalizes the derivation from the text.

PROVIDED SOLUTION
Unfold expectedFundingCost. (2*q - 1) * E_rf = 2*q*E_rf - E_rf. And q*E_rf - (1-q)*E_rf = q*E_rf - E_rf + q*E_rf = 2*q*E_rf - E_rf. Use ring.
-/
theorem expectedFundingCost_decomposition (q E_rf : ℝ) :
    expectedFundingCost q E_rf = q * E_rf - (1 - q) * E_rf := by
  unfold expectedFundingCost; ring;

/-
PROBLEM
A balanced strategy (q = 1/2) has zero expected funding cost.

PROVIDED SOLUTION
Unfold expectedFundingCost. (2 * (1/2) - 1) * E_rf = (1 - 1) * E_rf = 0. Use simp/ring/norm_num.
-/
theorem balanced_strategy_zero_funding (E_rf : ℝ) :
    expectedFundingCost (1/2) E_rf = 0 := by
  -- Substitute q = 1/2 into the formula for expected funding cost.
  simp [expectedFundingCost]

/-
PROBLEM
A long-biased strategy (q > 1/2) with positive expected funding rate
    pays net funding.

PROVIDED SOLUTION
Unfold expectedFundingCost. Since q > 1/2, 2*q - 1 > 0. And E_rf > 0. So (2*q - 1) * E_rf > 0. Use mul_pos with linarith for the first factor.
-/
theorem long_biased_pays_funding {q E_rf : ℝ} (hq : 1/2 < q) (hE : 0 < E_rf) :
    0 < expectedFundingCost q E_rf := by
  exact mul_pos ( by linarith ) hE

/-
PROBLEM
A short-biased strategy (q < 1/2) with positive expected funding rate
    receives net funding.

PROVIDED SOLUTION
Unfold expectedFundingCost. Since q < 1/2, 2*q - 1 < 0. And E_rf > 0. So (2*q - 1) * E_rf < 0. Use mul_neg_of_neg_of_pos with linarith.
-/
theorem short_biased_receives_funding {q E_rf : ℝ} (hq : q < 1/2) (hE : 0 < E_rf) :
    expectedFundingCost q E_rf < 0 := by
  exact mul_neg_of_neg_of_pos ( by linarith ) hE

end
import Mathlib

/-!
# Theorem A: Avellaneda-Stoikov Reservation Price Properties

Formalizes key results from the Avellaneda-Stoikov (2008) market-making model.
The reservation price is r(q, τ) = s - q · γ · σ² · τ, and the optimal half-spread is
δ(τ) = γ · σ² · τ + (1/γ) · ln(1 + γ/κ).
-/

noncomputable section

open Real

/-- The reservation price: r(q, τ) = s - q · γ · σ² · τ -/
def reservationPrice (s q γ σ τ : ℝ) : ℝ := s - q * γ * σ ^ 2 * τ

/-- The optimal half-spread: δ(τ) = γ · σ² · τ + (1/γ) · ln(1 + γ/κ) -/
def optimalHalfSpread (γ σ τ κ : ℝ) : ℝ := γ * σ ^ 2 * τ + (1 / γ) * Real.log (1 + γ / κ)

/-- Bid price: r - δ -/
def bidPrice (s q γ σ τ κ : ℝ) : ℝ := reservationPrice s q γ σ τ - optimalHalfSpread γ σ τ κ

/-- Ask price: r + δ -/
def askPrice (s q γ σ τ κ : ℝ) : ℝ := reservationPrice s q γ σ τ + optimalHalfSpread γ σ τ κ

/-
PROBLEM
============================================================================
CLAIM A1: The spread δ is always positive when γ > 0 and κ > 0
============================================================================

A1a: The inventory-risk component γ·σ²·τ is non-negative.

PROVIDED SOLUTION
γ > 0, σ^2 ≥ 0 (square is nonneg), τ ≥ 0, so product is nonneg. Use mul_nonneg and sq_nonneg.
-/
theorem claim_A1a {γ σ τ : ℝ} (hγ : γ > 0) (hτ : τ ≥ 0) :
    γ * σ ^ 2 * τ ≥ 0 := by
  positivity

/-
PROBLEM
A1b: The market-making component (1/γ)·ln(1 + γ/κ) is strictly positive.

PROVIDED SOLUTION
Since γ > 0 and κ > 0, γ/κ > 0, so 1 + γ/κ > 1, so log(1 + γ/κ) > 0. Also 1/γ > 0. Product of two positives is positive. Use Real.log_pos for log(x) > 0 when x > 1, and div_pos.
-/
theorem claim_A1b {γ κ : ℝ} (hγ : γ > 0) (hκ : κ > 0) :
    (1 / γ) * Real.log (1 + γ / κ) > 0 := by
  exact mul_pos ( one_div_pos.mpr hγ ) ( Real.log_pos ( by norm_num; positivity ) )

/-
PROBLEM
A1: The full optimal half-spread is strictly positive.

PROVIDED SOLUTION
Use claim_A1a and claim_A1b: the first gives ≥ 0 and the second gives > 0, so the sum is > 0. Unfold optimalHalfSpread and use linarith with both.
-/
theorem claim_A1 {γ σ τ κ : ℝ} (hγ : γ > 0) (hτ : τ ≥ 0) (hκ : κ > 0) :
    optimalHalfSpread γ σ τ κ > 0 := by
  unfold optimalHalfSpread; exact add_pos_of_nonneg_of_pos ( mul_nonneg ( mul_nonneg hγ.le ( sq_nonneg _ ) ) hτ ) ( mul_pos ( one_div_pos.mpr hγ ) ( Real.log_pos ( by linarith [ div_pos hγ hκ ] ) ) ) ;

/-
PROBLEM
============================================================================
CLAIM A2: The spread is decreasing in κ
============================================================================

A2: The expression -1/(κ·(κ + γ)) is negative for κ, γ > 0,
    confirming ∂δ/∂κ < 0.

PROVIDED SOLUTION
κ > 0 and γ > 0, so κ*(κ+γ) > 0, so -1/(κ*(κ+γ)) < 0. Use neg_div_of_neg_of_pos or div_neg_of_neg_of_pos.
-/
theorem claim_A2_derivative_neg {κ γ : ℝ} (hκ : κ > 0) (hγ : γ > 0) :
    -1 / (κ * (κ + γ)) < 0 := by
  exact div_neg_of_neg_of_pos ( by norm_num ) ( mul_pos hκ ( add_pos hκ hγ ) )

/-
PROBLEM
A2': The spread component (1/γ)·ln(1 + γ/κ) is strictly decreasing in κ > 0.

PROVIDED SOLUTION
Since κ₁ < κ₂ and both positive, γ/κ₂ < γ/κ₁ (div_lt_div_of_pos_left). Since log is strictly monotone, log(1 + γ/κ₂) < log(1 + γ/κ₁). Since 1/γ > 0, multiplying preserves the inequality. Use Real.log_lt_log_left or strictMono_log, and mul_lt_mul_of_pos_left.
-/
theorem claim_A2_antitone {γ κ₁ κ₂ : ℝ} (hγ : γ > 0) (hκ₁ : κ₁ > 0) (hκ₂ : κ₁ < κ₂) :
    (1 / γ) * Real.log (1 + γ / κ₂) < (1 / γ) * Real.log (1 + γ / κ₁) := by
  gcongr ; nlinarith [ mul_div_cancel₀ γ ( ne_of_gt hκ₁ ), mul_div_cancel₀ γ ( ne_of_gt ( lt_trans hκ₁ hκ₂ ) ) ]

/-- A2'': The full spread is strictly decreasing in κ (fixing other parameters). -/
theorem claim_A2_spread_decreasing {γ σ τ κ₁ κ₂ : ℝ}
    (hγ : γ > 0) (hκ₁ : κ₁ > 0) (hκ₂ : κ₁ < κ₂) :
    optimalHalfSpread γ σ τ κ₂ < optimalHalfSpread γ σ τ κ₁ := by
  unfold optimalHalfSpread
  linarith [claim_A2_antitone hγ hκ₁ hκ₂]

-- ============================================================================
-- CLAIM A3: Properties of the reservation price deviation |r - s| = |q|·γ·σ²·τ
-- ============================================================================

/-- The absolute deviation of reservation price from mid-price. -/
theorem reservation_deviation {s q γ σ τ : ℝ} :
    |reservationPrice s q γ σ τ - s| = |q * γ * σ ^ 2 * τ| := by
  unfold reservationPrice
  have : s - q * γ * σ ^ 2 * τ - s = -(q * γ * σ ^ 2 * τ) := by ring
  rw [this, abs_neg]

/-
PROBLEM
When all parameters are non-negative, |r - s| = |q| · γ · σ² · τ.

PROVIDED SOLUTION
We have |reservationPrice s q γ σ τ - s| = |q * γ * σ^2 * τ| by reservation_deviation. Then show |q * γ * σ^2 * τ| = |q| * γ * σ^2 * τ using the fact that γ ≥ 0, σ^2 ≥ 0, τ ≥ 0, so γ * σ^2 * τ ≥ 0, and |a * b| = |a| * b when b ≥ 0. Use abs_mul and abs_of_nonneg.
-/
theorem reservation_deviation_nonneg {s q γ σ τ : ℝ}
    (hγ : γ ≥ 0) (hτ : τ ≥ 0) :
    |reservationPrice s q γ σ τ - s| = |q| * γ * σ ^ 2 * τ := by
  unfold reservationPrice; ring; norm_num [ abs_mul, abs_of_nonneg, hγ, hτ ] ;
  ring

/-
PROBLEM
A3a: The deviation is increasing in |q| (monotone in absolute inventory).

PROVIDED SOLUTION
|q₁| ≤ |q₂| and γ * σ^2 * τ ≥ 0 (since γ ≥ 0, σ^2 ≥ 0, τ ≥ 0). Use mul_le_mul_of_nonneg_right. Need to reassociate the multiplication.
-/
theorem claim_A3a {γ σ τ : ℝ} {q₁ q₂ : ℝ}
    (hγ : γ ≥ 0) (hτ : τ ≥ 0) (hq : |q₁| ≤ |q₂|) :
    |q₁| * γ * σ ^ 2 * τ ≤ |q₂| * γ * σ ^ 2 * τ := by
  gcongr

/-
PROBLEM
A3b: The deviation is increasing in γ (monotone in risk aversion).

PROVIDED SOLUTION
γ₁ ≤ γ₂ and |q| * σ^2 * τ ≥ 0 and τ ≥ 0. Multiply the inequality by the nonneg factor. Use nlinarith or mul_le_mul with appropriate nonneg arguments.
-/
theorem claim_A3b {q σ τ : ℝ} {γ₁ γ₂ : ℝ}
    (hγ : 0 ≤ γ₁) (hγ₂ : γ₁ ≤ γ₂) (hτ : τ ≥ 0) :
    |q| * γ₁ * σ ^ 2 * τ ≤ |q| * γ₂ * σ ^ 2 * τ := by
  gcongr

/-
PROBLEM
A3c: The deviation is increasing in σ (monotone in volatility).

PROVIDED SOLUTION
σ₁ ≤ σ₂ and 0 ≤ σ₁ implies σ₁² ≤ σ₂². Then multiply by |q|*γ*τ which is nonneg. Use sq_le_sq' or pow_le_pow_left, then gcongr or nlinarith.
-/
theorem claim_A3c {q γ τ : ℝ} {σ₁ σ₂ : ℝ}
    (hγ : γ ≥ 0) (hτ : τ ≥ 0) (hσ : 0 ≤ σ₁) (hσ₂ : σ₁ ≤ σ₂) :
    |q| * γ * σ₁ ^ 2 * τ ≤ |q| * γ * σ₂ ^ 2 * τ := by
  gcongr

/-
PROBLEM
A3d: The deviation is increasing in τ (monotone in time remaining).

PROVIDED SOLUTION
τ₁ ≤ τ₂ and |q|*γ*σ² ≥ 0, so multiply. Use gcongr.
-/
theorem claim_A3d {q γ σ : ℝ} {τ₁ τ₂ : ℝ}
    (hγ : γ ≥ 0) (hτ : 0 ≤ τ₁) (hτ₂ : τ₁ ≤ τ₂) :
    |q| * γ * σ ^ 2 * τ₁ ≤ |q| * γ * σ ^ 2 * τ₂ := by
  exact mul_le_mul_of_nonneg_left hτ₂ ( by positivity )

/-- A3e: The deviation is zero when inventory q = 0. -/
theorem claim_A3e {γ σ τ : ℝ} :
    reservationPrice s 0 γ σ τ = s := by
  unfold reservationPrice; ring

-- ============================================================================
-- CLAIM A4: Bid-ask spread symmetry properties
-- ============================================================================

/-- A4: The bid-ask spread is symmetric around r (not around s).
    Both bid and ask are equidistant from the reservation price. -/
theorem claim_A4_symmetric_around_r {s q γ σ τ κ : ℝ} :
    reservationPrice s q γ σ τ - bidPrice s q γ σ τ κ =
    askPrice s q γ σ τ κ - reservationPrice s q γ σ τ := by
  unfold bidPrice askPrice; ring

/-- A4: The asymmetry of bid and ask around the MID-PRICE s. -/
theorem claim_A4_midprice_asymmetry {s q γ σ τ κ : ℝ} :
    (s - bidPrice s q γ σ τ κ) - (askPrice s q γ σ τ κ - s) =
    2 * (q * γ * σ ^ 2 * τ) := by
  unfold bidPrice askPrice reservationPrice optimalHalfSpread; ring

/-
PROBLEM
A4: The absolute asymmetry around mid-price equals 2·|q|·γ·σ²·τ
    when parameters are non-negative.

PROVIDED SOLUTION
First rewrite using claim_A4_midprice_asymmetry to get |2*(q*γ*σ²*τ)| = 2*|q|*γ*σ²*τ. The absolute value of 2*x is 2*|x| since 2 > 0. Then |q*γ*σ²*τ| = |q|*γ*σ²*τ since γ ≥ 0, σ² ≥ 0, τ ≥ 0. Use abs_mul, abs_of_nonneg.
-/
theorem claim_A4_abs_asymmetry {s q γ σ τ κ : ℝ}
    (hγ : γ ≥ 0) (hτ : τ ≥ 0) :
    |(s - bidPrice s q γ σ τ κ) - (askPrice s q γ σ τ κ - s)| =
    2 * |q| * γ * σ ^ 2 * τ := by
  rw [ abs_eq ] <;> try positivity;
  cases abs_cases q <;> simp +decide [ *, bidPrice, askPrice, reservationPrice ] <;> ring_nf <;> norm_num

end
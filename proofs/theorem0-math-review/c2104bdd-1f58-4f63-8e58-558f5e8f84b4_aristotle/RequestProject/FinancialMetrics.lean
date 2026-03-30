import Mathlib

open Finset BigOperators Real

noncomputable section

/-!
# Financial Metrics: Formalization and Verification

We formalize five claims about financial metrics and either prove them
or provide counterexamples.

## Summary of Results
1. **Sortino Ratio Annualization**: ✅ VERIFIED — algebraic scaling identity proved.
2. **Hurst Exponent H ∈ [0,1]**: ❌ FALSE in general — counterexample provided.
   ✅ TRUE under the additional assumption 1 ≤ R/S ≤ w.
3. **Kyle's Lambda = OLS Slope**: ✅ VERIFIED — proved it minimizes sum of squared residuals.
4. **Amihud ILLIQ ≥ 0 and zero characterization**: ✅ VERIFIED — both directions proved.
5. **Triple Barrier Labeling**: ✅ VERIFIED — mutual exclusivity and exhaustiveness proved
   under the assumption tp > sl (i.e., fee_mult * fee > 0 and positive entry price).
-/

-- ============================================================
-- CLAIM 1: Sortino Ratio Annualization
-- ============================================================

/-! ### Claim 1: Sortino Ratio Annualization

The annualized Sortino ratio is defined as (μ/σ_down) · √(steps_per_day).
Under i.i.d. returns over k steps:
- Mean of sum = k · μ
- Downside deviation of sum = √k · σ_down

So the Sortino ratio of the aggregate return is:
  (k · μ) / (√k · σ_down) = (μ / σ_down) · √k

This is exactly the claimed annualization formula. We verify the algebraic identity.
-/

/-
PROBLEM
The Sortino annualization identity: (μ/σ)·√k = (k·μ)/(√k·σ).
    This justifies the annualization formula under i.i.d. returns.

PROVIDED SOLUTION
k / sqrt(k) = sqrt(k) for k > 0. So RHS = sqrt(k) * mu / sigma_down = LHS. Use field_simp and then show k / sqrt k = sqrt k by rewriting k = sqrt k * sqrt k (using Real.mul_self_sqrt or sq_sqrt).
-/
theorem sortino_annualization (mu sigma_down k : ℝ)
    (hsd : sigma_down ≠ 0) (hk : 0 < k) :
    mu / sigma_down * Real.sqrt k = k * mu / (Real.sqrt k * sigma_down) := by
  rw [ div_mul_eq_mul_div, div_eq_div_iff ] <;> ring_nf <;> norm_num [ hk.le, hk.ne', hsd ];
  ring

-- ============================================================
-- CLAIM 2: Hurst Exponent (R/S Method)
-- ============================================================

/-! ### Claim 2: Hurst Exponent

The Hurst exponent H = log(R/S) / log(w) where R is the rescaled range
and S is the standard deviation.

The claim states H ∈ [0, 1] for positive R and S. This is **FALSE** in general:
if R < S (which can happen with sample standard deviation), then R/S < 1,
so log(R/S) < 0 while log(w) > 0, giving H < 0.

Concrete counterexample: w = 2, returns = [1, 3].
- mean = 2, centered = [-1, 1], cumdev = [-1, 0]
- R = 0 - (-1) = 1
- S_sample = √2 (using w-1 = 1 in denominator)
- R/S = 1/√2 < 1
- H = log(1/√2)/log(2) = -1/2 < 0

However, H ∈ [0, 1] does hold when S ≤ R ≤ w·S (i.e., 1 ≤ R/S ≤ w).
-/

/-
PROBLEM
The claim H ∈ [0,1] for all positive R, S is FALSE.
    Counterexample: R = 1, S = 2, w = 2 gives H < 0.

PROVIDED SOLUTION
Use R = 1, S = 2, w = 2. Then R/S = 1/2 < 1, so log(1/2) < 0, and log(2) > 0, giving a negative quotient. Use norm_num and Real.log_lt_zero or similar.
-/
theorem hurst_not_always_in_unit_interval :
    ∃ R S w : ℝ, R > 0 ∧ S > 0 ∧ w > 1 ∧
      Real.log (R / S) / Real.log w < 0 := by
  exact ⟨ 1, 2, 2, by norm_num, by norm_num, by norm_num, div_neg_of_neg_of_pos ( Real.log_neg ( by norm_num ) ( by norm_num ) ) ( Real.log_pos ( by norm_num ) ) ⟩

/-
PROBLEM
H ∈ [0, 1] does hold when S ≤ R ≤ w · S and w > 1.

PROVIDED SOLUTION
Since S > 0 and S ≤ R, we have R/S ≥ 1, so log(R/S) ≥ 0. Since w > 1, log(w) > 0. So the quotient ≥ 0. For the upper bound: R ≤ w*S means R/S ≤ w, so log(R/S) ≤ log(w) (by monotonicity of log), so the quotient ≤ 1. Use div_nonneg, div_le_one, Real.log_nonneg, Real.log_le_log, Real.log_pos.
-/
theorem hurst_in_unit_interval {R S w : ℝ} (hS : 0 < S) (hw : 1 < w)
    (h_lb : S ≤ R) (h_ub : R ≤ w * S) :
    0 ≤ Real.log (R / S) / Real.log w ∧
    Real.log (R / S) / Real.log w ≤ 1 := by
  exact ⟨ div_nonneg ( Real.log_nonneg ( by rw [ le_div_iff₀ hS ] ; linarith ) ) ( Real.log_nonneg hw.le ), div_le_one_of_le₀ ( Real.log_le_log ( by rw [ lt_div_iff₀ hS ] ; linarith ) ( by rw [ div_le_iff₀ hS ] ; linarith ) ) ( Real.log_nonneg hw.le ) ⟩

-- ============================================================
-- CLAIM 3: Kyle's Lambda (OLS Slope)
-- ============================================================

/-! ### Claim 3: Kyle's Lambda

Kyle's lambda = Cov(returns, signed_flow) / Var(signed_flow) is the OLS slope
estimator for the regression returns = α + λ · signed_flow + ε.

For centered data (mean-subtracted x), this reduces to
  λ = Σ(xᵢ · yᵢ) / Σ(xᵢ²)

We prove this is the value that minimizes the sum of squared residuals
Σ(yᵢ - β · xᵢ)², which is the defining property of the OLS slope.

The interpretation "λ > 0 implies informed trading" is a financial interpretation
of the sign, not a mathematical claim.
-/

/-
PROBLEM
Kyle's lambda (= Cov/Var for centered data) minimizes the sum of
    squared residuals. This is the defining property of the OLS slope.

PROVIDED SOLUTION
The key identity: Σ(yᵢ - β·xᵢ)² = Σyᵢ² - 2β·Σ(xᵢyᵢ) + β²·Σxᵢ². This is minimized at β̂ = Σ(xᵢyᵢ)/Σxᵢ². The difference Σ(yᵢ - β·xᵢ)² - Σ(yᵢ - β̂·xᵢ)² = Σxᵢ² · (β - β̂)² ≥ 0. To prove: expand both squared sums using ring-like manipulations, take the difference, and show it equals (Σxᵢ²)·(β - β̂)². Alternatively, show each expression equals a quadratic in β and compare. Use Finset.sum_sub_sq_expand or manual ring manipulations with Finset.sum.
-/
theorem kyle_lambda_ols_minimizer {n : ℕ} (x y : Fin n → ℝ)
    (hx : 0 < ∑ i, x i ^ 2) :
    let β_hat := (∑ i, x i * y i) / (∑ i, x i ^ 2)
    ∀ β : ℝ, ∑ i, (y i - β_hat * x i) ^ 2 ≤ ∑ i, (y i - β * x i) ^ 2 := by
  norm_num [ sub_sq, Finset.sum_add_distrib, Finset.mul_sum _ _ _ ];
  norm_num [ Finset.mul_sum _ _ _, mul_pow, mul_assoc, mul_comm, mul_left_comm, Finset.sum_mul ];
  norm_num [ ← mul_assoc, ← Finset.mul_sum _ _ _, ← Finset.sum_mul ];
  intro β; norm_num [ mul_assoc, ← Finset.mul_sum _ _ _, ← Finset.sum_mul, hx.ne' ] ; nlinarith [ sq_nonneg ( β * ( ∑ i, x i ^ 2 ) - ( ∑ i, x i * y i ) ), mul_div_cancel₀ ( ∑ i, x i * y i ) hx.ne' ] ;

-- ============================================================
-- CLAIM 4: Amihud Illiquidity
-- ============================================================

/-! ### Claim 4: Amihud Illiquidity

ILLIQ = mean(|rᵢ| / volumeᵢ) is always non-negative (trivially, as a mean of
non-negative terms). Moreover, if all volumes are positive, ILLIQ = 0 if and
only if all returns are zero.
-/

/-- Amihud ILLIQ: mean of |returnᵢ| / volumeᵢ -/
def amihudILLIQ {n : ℕ} (returns volume : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, |returns i| / volume i) / n

/-
PROBLEM
ILLIQ is always non-negative when volumes are positive.

PROVIDED SOLUTION
amihudILLIQ is a sum of non-negative terms (|r_i|/v_i ≥ 0 since |r_i| ≥ 0 and v_i > 0) divided by n (cast to ℝ, which is ≥ 0). Use div_nonneg and Finset.sum_nonneg.
-/
theorem amihud_nonneg {n : ℕ} (returns volume : Fin n → ℝ)
    (hvol : ∀ i, volume i > 0) :
    0 ≤ amihudILLIQ returns volume := by
  exact div_nonneg ( Finset.sum_nonneg fun _ _ => div_nonneg ( abs_nonneg _ ) ( le_of_lt ( hvol _ ) ) ) ( Nat.cast_nonneg _ )

/-
PROBLEM
ILLIQ = 0 iff all returns are zero (given positive volumes and n > 0).

PROVIDED SOLUTION
Unfold amihudILLIQ. Forward direction: sum/n = 0 with n > 0 implies sum = 0. Sum of non-negative terms = 0 implies each term = 0 (Finset.sum_eq_zero_iff_of_nonneg). |r_i|/v_i = 0 with v_i > 0 implies |r_i| = 0, so r_i = 0. Backward: if all r_i = 0 then |r_i| = 0 and each term is 0, so sum is 0 and ILLIQ = 0.
-/
theorem amihud_zero_iff_returns_zero {n : ℕ} (hn : 0 < n)
    (returns volume : Fin n → ℝ) (hvol : ∀ i, volume i > 0) :
    amihudILLIQ returns volume = 0 ↔ ∀ i, returns i = 0 := by
  unfold amihudILLIQ;
  field_simp;
  constructor <;> intro h <;> simp_all +decide [ Finset.sum_eq_zero_iff_of_nonneg, div_eq_mul_inv ];
  exact fun i => Or.resolve_right ( h i ) ( ne_of_gt ( hvol i ) )

-- ============================================================
-- CLAIM 5: Triple Barrier Labeling
-- ============================================================

/-! ### Claim 5: Triple Barrier Labeling

Given entry price p with take-profit level tp and stop-loss level sl (where sl < tp),
the price path over T+1 time steps receives exactly one of three labels:
- Take-profit (+1): the first barrier hit is tp
- Stop-loss (-1): the first barrier hit is sl
- Timeout (0): no barrier is hit within the window

We prove mutual exclusivity and exhaustiveness under the assumption sl < tp
(which holds when fee_mult · fee > 0 and entry price > 0).
-/

/-- Take-profit is hit first: there exists a time t where price ≥ tp,
    and for all earlier times, price is strictly between sl and tp. -/
def labelIsTP {T : ℕ} (prices : Fin (T + 1) → ℝ) (tp sl : ℝ) : Prop :=
  ∃ t : Fin (T + 1), prices t ≥ tp ∧
    ∀ s : Fin (T + 1), s.val < t.val → sl < prices s ∧ prices s < tp

/-- Stop-loss is hit first: there exists a time t where price ≤ sl,
    and for all earlier times, price is strictly between sl and tp. -/
def labelIsSL {T : ℕ} (prices : Fin (T + 1) → ℝ) (tp sl : ℝ) : Prop :=
  ∃ t : Fin (T + 1), prices t ≤ sl ∧
    ∀ s : Fin (T + 1), s.val < t.val → sl < prices s ∧ prices s < tp

/-- Neither barrier is hit within the time window (timeout). -/
def labelIsTimeout {T : ℕ} (prices : Fin (T + 1) → ℝ) (tp sl : ℝ) : Prop :=
  ∀ t : Fin (T + 1), sl < prices t ∧ prices t < tp

/-
PROBLEM
The three outcomes are mutually exclusive when sl < tp.

PROVIDED SOLUTION
Three conjuncts to prove. (1) TP ∧ SL impossible: Let t₁ be the TP time and t₂ be the SL time. If t₁ = t₂, then prices t₁ ≥ tp > sl ≥ prices t₁, contradiction. If t₁ < t₂, SL's condition at s=t₁ says prices t₁ < tp, contradicting prices t₁ ≥ tp. If t₂ < t₁, TP's condition at s=t₂ says prices t₂ > sl, contradicting prices t₂ ≤ sl. (2) TP ∧ Timeout: TP gives t with prices t ≥ tp, Timeout says prices t < tp. Contradiction. (3) SL ∧ Timeout: SL gives t with prices t ≤ sl, Timeout says prices t > sl. Contradiction. Use Nat.lt_trichotomy or omega for the case split on t₁ vs t₂.
-/
theorem triple_barrier_exclusive {T : ℕ} (prices : Fin (T + 1) → ℝ)
    (tp sl : ℝ) (htp_sl : sl < tp) :
    ¬(labelIsTP prices tp sl ∧ labelIsSL prices tp sl) ∧
    ¬(labelIsTP prices tp sl ∧ labelIsTimeout prices tp sl) ∧
    ¬(labelIsSL prices tp sl ∧ labelIsTimeout prices tp sl) := by
  refine' ⟨ _, _, _ ⟩;
  · unfold labelIsTP labelIsSL;
    grind;
  · unfold labelIsTP labelIsTimeout; aesop;
  · rintro ⟨ ⟨ t, ht₁, ht₂ ⟩, ht₃ ⟩ ; linarith [ ht₃ t ]

/-
PROBLEM
The three outcomes are exhaustive when sl < tp.

PROVIDED SOLUTION
If labelIsTimeout holds, done. Otherwise, ¬labelIsTimeout means ∃ t, ¬(sl < prices t ∧ prices t < tp), i.e., prices t ≤ sl ∨ prices t ≥ tp. Among all such t, take the minimum (use Nat.find or well-ordering on Fin). Call it t₀. At t₀, either prices t₀ ≥ tp or prices t₀ ≤ sl (both can't hold since tp > sl). For all s < t₀, sl < prices s < tp (by minimality). So either labelIsTP (if prices t₀ ≥ tp) or labelIsSL (if prices t₀ ≤ sl). Use by_contra or classical logic, push_neg to negate timeout, then use Fin.exists_fin for the minimum.
-/
theorem triple_barrier_exhaustive {T : ℕ} (prices : Fin (T + 1) → ℝ)
    (tp sl : ℝ) (htp_sl : sl < tp) :
    labelIsTP prices tp sl ∨ labelIsSL prices tp sl ∨ labelIsTimeout prices tp sl := by
  by_cases h_timeout : labelIsTimeout prices tp sl;
  · exact Or.inr <| Or.inr h_timeout;
  · obtain ⟨t₀, ht₀⟩ : ∃ t₀ : Fin (T + 1), ¬(sl < prices t₀ ∧ prices t₀ < tp) ∧ ∀ t : Fin (T + 1), ¬(sl < prices t ∧ prices t < tp) → t₀.val ≤ t.val := by
      simp_all +decide [ labelIsTimeout ];
      exact ⟨ Finset.min' ( Finset.univ.filter fun t => sl < prices t → tp ≤ prices t ) ⟨ h_timeout.choose, Finset.mem_filter.mpr ⟨ Finset.mem_univ _, h_timeout.choose_spec ⟩ ⟩, Finset.mem_filter.mp ( Finset.min'_mem ( Finset.univ.filter fun t => sl < prices t → tp ≤ prices t ) ⟨ h_timeout.choose, Finset.mem_filter.mpr ⟨ Finset.mem_univ _, h_timeout.choose_spec ⟩ ⟩ ) |>.2, fun t ht => Finset.min'_le _ _ <| by aesop ⟩;
    by_cases h_case : prices t₀ ≥ tp <;> simp_all +decide [ not_and_or ];
    · refine Or.inl ⟨ t₀, h_case, fun s hs => ⟨ ?_, ?_ ⟩ ⟩ <;> contrapose! ht₀;
      · exact ⟨ s, fun _ => by linarith, hs ⟩;
      · exact ⟨ s, fun _ => ht₀, hs ⟩;
    · refine Or.inr ⟨ t₀, ht₀.1, fun s hs => ?_ ⟩;
      contrapose! ht₀;
      exact fun h => ⟨ s, ht₀, hs ⟩

end
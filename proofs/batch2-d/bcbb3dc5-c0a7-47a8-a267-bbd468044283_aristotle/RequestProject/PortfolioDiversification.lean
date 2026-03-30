import Mathlib

noncomputable section

open Real Filter Topology

/-!
# Portfolio Diversification with Correlated Strategies

We formalize the Sortino ratio scaling under diversification for N equal-weighted
strategies with pairwise correlation ρ.

## Key Definitions
- `portfolioVar`: Portfolio variance σ²·(1+(N-1)ρ)/N
- `divRatio`: Diversification ratio √(N/(1+(N-1)ρ))

## Main Results

### Claim 1 (Independence)
Under independence (ρ=0), portfolio variance = σ²/N and
Sortino scales as √N · S_single.

### Claim 2 (Correlated portfolio)
General formula Sortino = S · √(N/(1+(N-1)ρ)), with verifications:
- (a) ρ=0 ⟹ Sortino = S·√N
- (b) ρ=1 ⟹ Sortino = S (no diversification benefit)
- (c) N=25, ρ=0.3 ⟹ Sortino = S·√(125/41) ≈ S·1.746

### Claim 3 (Diversification ratio properties)
- (a) DR strictly increasing in N for ρ < 1
- (b) DR strictly decreasing in ρ for N > 1
- (c) DR → 1/√ρ as N → ∞
- (d) For ρ=0.3: max DR = 1/√0.3 = √(10/3) ≈ 1.826

### Claim 4 (Diminishing returns)
ΔDR = DR(N+1) - DR(N) → 0 as N → ∞
-/

/-- The denominator 1 + (N-1)·ρ appearing in the portfolio variance formula. -/
def denominator (N : ℕ) (ρ : ℝ) : ℝ := 1 + ((N : ℝ) - 1) * ρ

/-- Portfolio variance for N equal-weighted strategies with individual variance σ²
    and pairwise correlation ρ. -/
def portfolioVar (N : ℕ) (σ ρ : ℝ) : ℝ :=
  σ ^ 2 * (1 + ((N : ℝ) - 1) * ρ) / N

/-- The ratio N / (1 + (N-1)·ρ) inside the square root of the diversification ratio. -/
def innerRatio (N : ℕ) (ρ : ℝ) : ℝ := (N : ℝ) / denominator N ρ

/-- Diversification ratio: √(N / (1 + (N-1)·ρ)). -/
def divRatio (N : ℕ) (ρ : ℝ) : ℝ := Real.sqrt (innerRatio N ρ)

-- ============================================================
-- Helper Lemmas
-- ============================================================

lemma denominator_def (N : ℕ) (ρ : ℝ) :
    denominator N ρ = 1 + ((N : ℝ) - 1) * ρ := rfl

/-- The denominator is positive for N ≥ 1 and ρ ∈ [0,1). -/
lemma denominator_pos (N : ℕ) (hN : 0 < N) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    0 < denominator N ρ :=
  add_pos_of_pos_of_nonneg zero_lt_one (mul_nonneg (sub_nonneg.2 <| Nat.one_le_cast.2 hN) hρ0)

/-- The denominator is positive for N ≥ 1 and ρ ∈ [0,1]. -/
lemma denominator_pos' (N : ℕ) (hN : 0 < N) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    0 < denominator N ρ :=
  add_pos_of_pos_of_nonneg zero_lt_one (mul_nonneg (sub_nonneg.2 <| Nat.one_le_cast.2 hN) hρ0)

/-- The inner ratio is positive for N ≥ 1 and ρ ∈ [0,1). -/
lemma innerRatio_pos (N : ℕ) (hN : 0 < N) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    0 < innerRatio N ρ :=
  div_pos (Nat.cast_pos.mpr hN) (denominator_pos N hN hρ0 hρ1)

-- ============================================================
-- CLAIM 1: Independence case (ρ = 0)
-- ============================================================

/-- Under independence (ρ=0), portfolio variance equals σ²/N. -/
theorem claim1_independent_variance (N : ℕ) (hN : 0 < N) (σ : ℝ) :
    portfolioVar N σ 0 = σ ^ 2 / N := by
  unfold portfolioVar; ring

/-- Under independence, the Sortino ratio scales as √N times the single-strategy Sortino.
    That is, `divRatio N 0 = √N`. -/
theorem claim1_sortino_scaling (N : ℕ) (hN : 0 < N) :
    divRatio N 0 = Real.sqrt N := by
  unfold divRatio innerRatio denominator; aesop

-- ============================================================
-- CLAIM 2: General correlated case - special cases
-- ============================================================

/-- Claim 2(a): ρ = 0 gives √N (matches Claim 1). -/
theorem claim2a_rho_zero (N : ℕ) (hN : 0 < N) :
    divRatio N 0 = Real.sqrt N := claim1_sortino_scaling N hN

/-- Claim 2(b): ρ = 1 gives 1 (perfect correlation, no diversification benefit). -/
theorem claim2b_rho_one (N : ℕ) (hN : 0 < N) :
    divRatio N 1 = 1 := by
  unfold divRatio innerRatio denominator; aesop

/-- Claim 2(c): N=25, ρ=0.3 gives √(125/41).
    Note: 25/(1+24·0.3) = 25/8.2 = 250/82 = 125/41.
    Numerically √(125/41) ≈ 1.746. -/
theorem claim2c_specific :
    divRatio 25 0.3 = Real.sqrt (125 / 41) := by
  unfold divRatio innerRatio denominator
  norm_num

-- ============================================================
-- CLAIM 3: Properties of Diversification Ratio
-- ============================================================

/-- The inner ratio N/(1+(N-1)ρ) is strictly increasing in N for ρ ∈ [0,1).
    Proof: cross-multiply and use (1-ρ) > 0. -/
theorem innerRatio_strictMono_nat {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    {M N : ℕ} (hM : 0 < M) (hMN : M < N) :
    innerRatio M ρ < innerRatio N ρ := by
  unfold innerRatio; rw [div_lt_div_iff₀]
  · unfold denominator; nlinarith [(by norm_cast : (M : ℝ) < N)]
  · exact denominator_pos' M hM hρ0 hρ1.le
  · exact denominator_pos' N (pos_of_gt hMN) hρ0 hρ1.le

/-- Claim 3(a): The diversification ratio is strictly increasing in N for ρ ∈ [0,1). -/
theorem claim3a_increasing_in_N {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    {M N : ℕ} (hM : 0 < M) (hMN : M < N) :
    divRatio M ρ < divRatio N ρ :=
  Real.sqrt_lt_sqrt
    (div_nonneg (Nat.cast_nonneg _)
      (by exact denominator_pos _ hM hρ0 hρ1 |> le_of_lt |> le_trans (by norm_num)))
    (innerRatio_strictMono_nat hρ0 hρ1 hM hMN)

/-- The inner ratio is strictly decreasing in ρ for N > 1. -/
theorem innerRatio_strictAnti_rho {N : ℕ} (hN : 1 < N) {ρ₁ ρ₂ : ℝ}
    (hρ₁_nn : 0 ≤ ρ₁) (hρ₂_lt1 : ρ₂ < 1) (hlt : ρ₁ < ρ₂) :
    innerRatio N ρ₂ < innerRatio N ρ₁ := by
  unfold innerRatio denominator at *
  gcongr; nlinarith [show (N : ℝ) ≥ 2 by norm_cast]
  aesop

/-- Claim 3(b): The diversification ratio is strictly decreasing in ρ for N > 1. -/
theorem claim3b_decreasing_in_rho {N : ℕ} (hN : 1 < N) {ρ₁ ρ₂ : ℝ}
    (hρ₁_nn : 0 ≤ ρ₁) (hρ₂_lt1 : ρ₂ < 1) (hlt : ρ₁ < ρ₂) :
    divRatio N ρ₂ < divRatio N ρ₁ := by
  unfold divRatio
  rw [Real.sqrt_lt_sqrt_iff_of_pos] <;> norm_num [innerRatio]
  · unfold denominator; gcongr
    · nlinarith [show (N : ℝ) ≥ 2 by norm_cast]
    · aesop
  · exact div_pos (by positivity)
      (by rw [denominator_def]; nlinarith [show (N : ℝ) ≥ 2 by norm_cast])

/-- The inner ratio N/(1+(N-1)ρ) → 1/ρ as N → ∞. -/
theorem innerRatio_tendsto {ρ : ℝ} (hρ : 0 < ρ) (hρ1 : ρ < 1) :
    Tendsto (fun N : ℕ => innerRatio N ρ) atTop (nhds (1 / ρ)) := by
  suffices h : Tendsto (fun N => 1 / (ρ + (1 - ρ) / (N : ℝ))) Filter.atTop (nhds (1 / ρ)) by
    refine' h.comp tendsto_natCast_atTop_atTop |> fun h => h.congr' _
    filter_upwards [Filter.eventually_gt_atTop 0] with N hN
    unfold innerRatio; norm_num [hN.ne', denominator_def, hρ.ne', hρ1.ne]; ring
    rw [inv_eq_iff_eq_inv]; norm_num; ring; norm_num [hN.ne']
  exact le_trans
    (tendsto_const_nhds.div
      (tendsto_const_nhds.add (tendsto_const_nhds.div_atTop Filter.tendsto_id))
      (by linarith))
    (by norm_num)

/-- Claim 3(c): The diversification ratio converges to 1/√ρ as N → ∞. -/
theorem claim3c_limit {ρ : ℝ} (hρ : 0 < ρ) (hρ1 : ρ < 1) :
    Tendsto (fun N : ℕ => divRatio N ρ) atTop (nhds (1 / Real.sqrt ρ)) := by
  convert Tendsto.sqrt (innerRatio_tendsto hρ hρ1) using 2; ring
  rw [Real.sqrt_inv]

/-- Claim 3(d): The asymptotic maximum 1/√0.3 equals √(10/3) ≈ 1.826. -/
theorem claim3d_max_value :
    1 / Real.sqrt (0.3 : ℝ) = Real.sqrt (10 / 3) := by
  norm_num [← Real.sqrt_div_self]

/-- The N=25, ρ=0.3 case gives √(125/41) (same as Claim 2c). -/
theorem claim3d_specific : divRatio 25 0.3 = Real.sqrt (125 / 41) := claim2c_specific

-- ============================================================
-- CLAIM 4: Diminishing marginal returns
-- ============================================================

/-- Claim 4: The marginal increase ΔDR = DR(N+1) - DR(N) → 0 as N → ∞,
    i.e., adding more strategies has diminishing returns. -/
theorem claim4_diminishing_returns {ρ : ℝ} (hρ : 0 < ρ) (hρ1 : ρ < 1) :
    Tendsto (fun N : ℕ => divRatio (N + 1) ρ - divRatio N ρ) atTop (nhds 0) := by
  have hdiv := claim3c_limit hρ hρ1
  simpa using Filter.Tendsto.sub (hdiv.comp (Filter.tendsto_add_atTop_nat 1)) hdiv

end

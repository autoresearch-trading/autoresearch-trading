import Mathlib

/-!
# Theorem 44: Correlated Drawdown and Effective Diversification

We formalize the key results about portfolio variance, diversification ratio,
and effective number of independent bets for an equally-weighted portfolio
of equicorrelated assets.

## Main definitions and results

- `portfolioVariance`: σ_p² = σ² · (1 + (N-1)·ρ) / N
- `divRatio`: σ_p / σ = √((1 + (N-1)·ρ) / N)
- `nEff`: N_eff = N / (1 + (N-1)·ρ)

## Key properties proved

1. Portfolio variance formula (Claim 1)
2. Diversification ratio special cases and monotonicity (Claim 2)
3. Effective independent bets: special cases, monotonicity, positivity (Claim 3)
4. Drawdown scaling numerical bounds (Claim 5)
-/

open Real

noncomputable section

/-! ## Definitions -/

/-- Portfolio variance for N equicorrelated assets with equal weights,
    common volatility σ, and common pairwise correlation ρ. -/
def portfolioVariance (N : ℕ) (σ ρ : ℝ) : ℝ :=
  σ ^ 2 * (1 + (↑N - 1) * ρ) / ↑N

/-- Diversification ratio: σ_p / σ for the equicorrelated portfolio. -/
def divRatio (N : ℕ) (ρ : ℝ) : ℝ :=
  Real.sqrt ((1 + (↑N - 1) * ρ) / ↑N)

/-- Effective number of independent bets (Meucci 2009). -/
def nEff (N : ℕ) (ρ : ℝ) : ℝ :=
  ↑N / (1 + (↑N - 1) * ρ)

/-- Drawdown ratio: DD_portfolio / DD_single under the assumption DD ∝ σ. -/
def ddRatio (N : ℕ) (ρ : ℝ) : ℝ :=
  divRatio N ρ

/-! ## Claim 1: Portfolio Variance Formula

The portfolio variance with equal weights w_i = 1/N is:
  σ_p² = σ² · [1/N + (1 - 1/N) · ρ] = σ² · (1 + (N-1)·ρ) / N

We prove the two forms are equivalent.
-/

/-
PROBLEM
The two expressions for portfolio variance are equal:
    σ²/N + σ²·(1 - 1/N)·ρ = σ²·(1 + (N-1)·ρ)/N

PROVIDED SOLUTION
Unfold portfolioVariance, then field_simp and ring.
-/
theorem portfolioVariance_alt (N : ℕ) (hN : (N : ℝ) ≠ 0) (σ ρ : ℝ) :
    σ ^ 2 / ↑N + σ ^ 2 * (1 - 1 / ↑N) * ρ = portfolioVariance N σ ρ := by
  unfold portfolioVariance; ring;
  simpa [ hN ] using by ring;

/-! ## Claim 2: Diversification Ratio -/

/-
PROBLEM
When ρ = 0, divRatio = 1/√N (perfect diversification).

PROVIDED SOLUTION
Unfold divRatio, simplify 1 + (N-1)*0 = 1, then sqrt(1/N) = 1/sqrt(N).
-/
theorem divRatio_zero (N : ℕ) (hN : 0 < (N : ℝ)) :
    divRatio N 0 = 1 / Real.sqrt ↑N := by
  unfold divRatio; norm_num [ hN ] ;

/-
PROBLEM
When ρ = 1, divRatio = 1 (no diversification), for N ≥ 1.

PROVIDED SOLUTION
Unfold divRatio, simplify 1 + (N-1)*1 = N, then sqrt(N/N) = sqrt(1) = 1.
-/
theorem divRatio_one (N : ℕ) (hN : 0 < (N : ℝ)) :
    divRatio N 1 = 1 := by
  unfold divRatio; aesop;

/-
PROBLEM
divRatio is strictly increasing in ρ for N ≥ 2 on the valid range.

PROVIDED SOLUTION
divRatio N ρ = sqrt(f(ρ)) where f(ρ) = (1 + (N-1)·ρ) / N is an affine function with positive slope (N-1)/N > 0 since N ≥ 2. On the set Ici(-1/(N-1)), f(ρ) ≥ 0 (since f(-1/(N-1)) = 0). So for a < b in this set, we have 0 ≤ f(a) < f(b), and sqrt is strictly monotone on nonneg reals, so sqrt(f(a)) < sqrt(f(b)). Use Real.sqrt_lt_sqrt with positivity for the nonneg condition and nlinarith for the strict inequality of the arguments.
-/
theorem divRatio_strictMonoOn (N : ℕ) (hN : 2 ≤ N) :
    StrictMonoOn (divRatio N) (Set.Ici (-(1 : ℝ) / ((N : ℝ) - 1))) := by
  intros x hx
  intros y hy hxy;
  refine' Real.sqrt_lt_sqrt _ _ <;> norm_num at *;
  · exact div_nonneg ( by rw [ div_le_iff₀ ] at hx <;> nlinarith [ show ( N : ℝ ) ≥ 2 by norm_cast ] ) ( by positivity );
  · gcongr ; nlinarith [ show ( N : ℝ ) ≥ 2 by norm_cast ]

/-! ## Claim 3: Effective Number of Independent Bets -/

/-
PROBLEM
N_eff = N when ρ = 0.

PROVIDED SOLUTION
Unfold nEff, simplify 1 + (N-1)*0 = 1, then N/1 = N.
-/
theorem nEff_zero (N : ℕ) : nEff N 0 = ↑N := by
  unfold nEff; ring;

/-
PROBLEM
N_eff = 1 when ρ = 1, for N ≥ 1.

PROVIDED SOLUTION
Unfold nEff, simplify 1 + (N-1)*1 = N, then N/N = 1 using hN.
-/
theorem nEff_one (N : ℕ) (hN : 0 < (N : ℝ)) : nEff N 1 = 1 := by
  unfold nEff; aesop;

/-
PROBLEM
N_eff is strictly decreasing in ρ for N ≥ 2 and ρ in the valid range.

PROVIDED SOLUTION
nEff N ρ = N / (1 + (N-1)·ρ). In the valid range (-1/(N-1), 1), the denominator 1 + (N-1)·ρ is positive (since ρ > -1/(N-1) implies (N-1)·ρ > -1). The denominator is strictly increasing in ρ (slope N-1 > 0 since N ≥ 2). Since the numerator N > 0 is constant and the denominator is positive and strictly increasing, the quotient N/denom is strictly decreasing. Use div_lt_div_of_pos_left or similar.
-/
theorem nEff_strictAntiOn (N : ℕ) (hN : 2 ≤ N) :
    StrictAntiOn (nEff N) (Set.Ioo (-(1 : ℝ) / ((N : ℝ) - 1)) 1) := by
  -- By definition of nEff, we have nEff N ρ = N / (1 + (N-1)ρ).
  unfold StrictAntiOn nEff;
  intro a ha b hb hab; gcongr <;> nlinarith [ ha.1, ha.2, hb.1, hb.2, show ( N : ℝ ) ≥ 2 by norm_cast, mul_div_cancel₀ ( -1 : ℝ ) ( sub_ne_zero_of_ne ( by norm_cast; linarith : ( N : ℝ ) ≠ 1 ) ) ] ;

/-
PROBLEM
N_eff > 0 for all ρ in the valid range (-1/(N-1), 1).

PROVIDED SOLUTION
In the valid range, ρ > -1/(N-1) so (N-1)·ρ > -1, hence 1 + (N-1)·ρ > 0. Also N ≥ 2 > 0. So N / (1 + (N-1)·ρ) > 0 as quotient of positives.
-/
theorem nEff_pos (N : ℕ) (hN : 2 ≤ N) (ρ : ℝ)
    (hρ : ρ ∈ Set.Ioo (-(1 : ℝ) / ((N : ℝ) - 1)) 1) : 0 < nEff N ρ := by
  refine' div_pos ( by positivity ) ( by nlinarith [ hρ.1, hρ.2, show ( N : ℝ ) ≥ 2 by norm_cast, mul_div_cancel₀ ( -1 ) ( sub_ne_zero.2 <| by norm_cast; linarith : ( N : ℝ ) - 1 ≠ 0 ) ] )

/-! ## Claim 5: Drawdown Scaling Numerical Bounds

We verify the numerical claims about drawdown ratios for specific parameters.
These are stated as bounds on the squared ratio to avoid square roots.
-/

/-
PROBLEM
For N = 23, ρ = 0.282: the squared DD ratio is (1 + 22·0.282)/23.
    We verify 0.55² < (1 + 22·0.282)/23 < 0.57².

PROVIDED SOLUTION
Compute: 0.55^2 = 0.3025, and (1 + 22*0.282)/23 = (1 + 6.204)/23 = 7.204/23 ≈ 0.31322. So 0.3025 < 0.31322. Use norm_num.
-/
theorem ddRatio_squared_normal_lb :
    (0.55 : ℝ) ^ 2 < (1 + 22 * 0.282) / 23 := by
  norm_num +zetaDelta at *

/-
PROVIDED SOLUTION
Compute: 0.57^2 = 0.3249, and (1 + 22*0.282)/23 = 7.204/23 ≈ 0.31322. So 0.31322 < 0.3249. Use norm_num.
-/
theorem ddRatio_squared_normal_ub :
    (1 + 22 * (0.282 : ℝ)) / 23 < 0.57 ^ 2 := by
  norm_num +zetaDelta at *

/-
PROBLEM
For N = 23, ρ = 0.198: the squared DD ratio is (1 + 22·0.198)/23.
    We verify 0.47² < (1 + 22·0.198)/23 < 0.49².

PROVIDED SOLUTION
norm_num
-/
theorem ddRatio_squared_stress_lb :
    (0.47 : ℝ) ^ 2 < (1 + 22 * 0.198) / 23 := by
  grind

/-
PROVIDED SOLUTION
norm_num
-/
theorem ddRatio_squared_stress_ub :
    (1 + 22 * (0.198 : ℝ)) / 23 < 0.49 ^ 2 := by
  grind

/-
PROBLEM
For N = 23, ρ = 0.55: the squared DD ratio is (1 + 22·0.55)/23.
    We verify 0.74² < (1 + 22·0.55)/23 < 0.76².

PROVIDED SOLUTION
norm_num
-/
theorem ddRatio_squared_daily_lb :
    (0.74 : ℝ) ^ 2 < (1 + 22 * 0.55) / 23 := by
  norm_num +zetaDelta at *

/-
PROVIDED SOLUTION
norm_num
-/
theorem ddRatio_squared_daily_ub :
    (1 + 22 * (0.55 : ℝ)) / 23 < 0.76 ^ 2 := by
  grind

/-! ## Claim 4: Portfolio Sortino Scaling

If each asset has Sortino ratio S, the portfolio Sortino is S · √N_eff.
This follows from the observation that with equal means and DD ∝ σ,
  S_portfolio = μ / σ_p = μ / (σ · divRatio)
             = (μ/σ) · 1/divRatio
             = S · √(N / (1 + (N-1)·ρ))
             = S · √N_eff
-/

/-
PROBLEM
The Sortino scaling: 1/divRatio² = nEff/N, so 1/divRatio = √(nEff/N).
    This is the key algebraic identity underlying S_portfolio = S · √N_eff.

PROVIDED SOLUTION
Unfold nEff, field_simp and ring.
-/
theorem sortino_scaling_identity (N : ℕ) (ρ : ℝ) (hN : 0 < (N : ℝ))
    (hden : 1 + (↑N - 1) * ρ ≠ 0) :
    ↑N / ((1 + (↑N - 1) * ρ) / ↑N) = nEff N ρ * ↑N := by
  unfold nEff; ring;
  grind

/-
PROBLEM
Equivalently, divRatio² · nEff = 1, connecting diversification ratio to
    effective number of bets.

PROVIDED SOLUTION
Unfold nEff, field_simp and ring. The expression is ((1+(N-1)ρ)/N) * (N/(1+(N-1)ρ)) = 1.
-/
theorem divRatio_sq_mul_nEff (N : ℕ) (ρ : ℝ) (hN : (N : ℝ) ≠ 0)
    (hden : 1 + (↑N - 1) * ρ ≠ 0) :
    ((1 + (↑N - 1) * ρ) / ↑N) * nEff N ρ = 1 := by
  unfold nEff; aesop;

/-! ## Claim 6: Timescale Dependence (Epps Effect)

The Epps effect states that pairwise correlations measured at shorter timescales
tend to be lower. This is a qualitative/empirical observation rather than a
pure mathematical theorem, so we state it as a definitional property.

Our empirical findings are consistent:
  - ρ ≈ 0.28 at ~1 second timescale
  - ρ ≈ 0.55 at daily timescale (literature)

Since divRatio is increasing in ρ (Claim 2c), lower correlation at the trading
timescale implies BETTER diversification at the trading timescale than at the
daily risk management timescale.
-/

/-- At the trading timescale (lower ρ), diversification is better than at the
    risk management timescale (higher ρ). This follows directly from
    monotonicity of divRatio on the valid range. -/
theorem better_diversification_at_trading_timescale
    (N : ℕ) (hN : 2 ≤ N) (ρ_trade ρ_daily : ℝ)
    (h_trade : -(1 : ℝ) / ((N : ℝ) - 1) ≤ ρ_trade)
    (h_daily : -(1 : ℝ) / ((N : ℝ) - 1) ≤ ρ_daily)
    (h : ρ_trade < ρ_daily) :
    divRatio N ρ_trade < divRatio N ρ_daily :=
  divRatio_strictMonoOn N hN h_trade h_daily h

end
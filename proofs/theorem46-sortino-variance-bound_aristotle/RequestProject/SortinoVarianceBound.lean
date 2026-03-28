import Mathlib

/-!
# Theorem 46: Variance Bound for Sortino Ratio Estimator

This file formalizes the algebraic and inequality claims from the Sortino ratio
variance bound analysis. The claims involve:

1. Variance of sample mean: Var(μ̂) = σ²/N
2. Delta method partial derivatives and variance approximation
3. Dominant term analysis: when S < 1, the first variance term dominates
4. Effective sample size for correlated assets
5. Numerical verification of standard error bounds
6. Bias correction factor c₄(4) for fold variance

We formalize the exact algebraic identities and inequalities; the delta method
approximations are stated as exact equalities of the approximating expressions.
-/

open Real

noncomputable section

/-! ## Claim 1: Variance of sample mean

For i.i.d. returns with variance σ², the sample mean of N observations has
variance σ²/N. We state this as a pure algebraic identity.
-/

theorem variance_sample_mean (sigma_sq : ℝ) (N : ℝ) (hN : N > 0) :
    sigma_sq / N = sigma_sq * (1 / N) := by
  ring

/-! ## Claim 2: Delta method partial derivatives

The Sortino estimator is g(a,b) = (a/b) * √m. The partial derivatives are:
  ∂g/∂a = √m / b
  ∂g/∂b = -a * √m / b²
-/

/-
PROVIDED SOLUTION
The function a ↦ (a / sigma_d) * sqrt m is linear in a (since sigma_d and sqrt m are constants). Its derivative is sqrt m / sigma_d. Use deriv_const_mul or simp with derivative lemmas.
-/
theorem sortino_partial_a (m : ℝ) (sigma_d : ℝ) (hsd : sigma_d ≠ 0) :
    ∀ a : ℝ, deriv (fun a => (a / sigma_d) * sqrt m) a = sqrt m / sigma_d := by
  norm_num [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm ]

/-
PROVIDED SOLUTION
The function b ↦ (mu / b) * sqrt m = mu * sqrt m * b⁻¹. Its derivative at sigma_d is -mu * sqrt m / sigma_d². Use deriv_div_const or deriv of inv, and the fact that sigma_d ≠ 0.
-/
theorem sortino_partial_b (m : ℝ) (mu : ℝ) (sigma_d : ℝ) (hsd : sigma_d ≠ 0) :
    deriv (fun b => (mu / b) * sqrt m) sigma_d = -(mu * sqrt m) / sigma_d ^ 2 := by
  norm_num [ div_eq_mul_inv, hsd ] ; ring;

/-! ## Claim 4: Dominant term when S is small

When S² = m * μ² / σ_d², the first variance term dominates the second when
  m * σ² / (σ_d² * N) > S² / (2 * N)

This simplifies to 2 * σ² > μ², equivalently σ²/μ² > 1/2 (CV² > 1/2).
-/

/-
PROVIDED SOLUTION
We need m * sigma_sq / (sigma_d^2 * N) > (m * mu^2 / sigma_d^2) / (2 * N). Multiply both sides by sigma_d^2 * N * 2 (positive). This reduces to 2 * m * sigma_sq > m * mu^2, i.e., 2 * sigma_sq > mu^2, which follows from hCV: sigma_sq / mu^2 > 1/2. Use positivity of all factors and field_simp + nlinarith or similar.
-/
theorem dominant_term_simplification
    (m sigma_sq mu sigma_d N : ℝ)
    (hN : N > 0) (hsd : sigma_d > 0) (hm : m > 0) (hmu : mu ≠ 0)
    (hS_def : m * mu ^ 2 / sigma_d ^ 2 = (mu / sigma_d * sqrt m) ^ 2)
    (hCV : sigma_sq / mu ^ 2 > 1 / 2) :
    m * sigma_sq / (sigma_d ^ 2 * N) > (m * mu ^ 2 / sigma_d ^ 2) / (2 * N) := by
  field_simp;
  rw [ gt_iff_lt, div_lt_div_iff₀ ] at hCV <;> first | positivity | linarith;

/-
PROBLEM
The key algebraic equivalence: the dominant term condition reduces to CV² > 1/2.

PROVIDED SOLUTION
2 * sigma_sq > mu^2 ↔ sigma_sq / mu^2 > 1/2. Since mu ≠ 0, mu^2 > 0. Dividing both sides by 2 * mu^2 gives the equivalence. Use div_lt_div_right or constructor with field_simp and nlinarith.
-/
theorem dominant_term_iff_cv_bound
    (sigma_sq mu : ℝ) (hmu : mu ≠ 0) (hsigma_sq : sigma_sq > 0) :
    2 * sigma_sq > mu ^ 2 ↔ sigma_sq / mu ^ 2 > 1 / 2 := by
  rw [ gt_iff_lt, gt_iff_lt, lt_div_iff₀ ] <;> first | positivity | ring;
  constructor <;> intro <;> linarith

/-! ## Claim 5: Portfolio effective sample size

For K assets with pairwise correlation ρ and equal weight:
  N_eff = K / (1 + (K-1) * ρ)

Properties:
  - N_eff < K when ρ > 0
  - N_eff = K when ρ = 0
-/

def N_eff (K : ℝ) (rho : ℝ) : ℝ := K / (1 + (K - 1) * rho)

/-
PROVIDED SOLUTION
N_eff K rho = K / (1 + (K-1)*rho). Since K > 1 and rho > 0, (K-1)*rho > 0, so denominator 1 + (K-1)*rho > 1. Thus K / (1 + (K-1)*rho) < K / 1 = K. Use div_lt_self or similar with positivity of K.
-/
theorem N_eff_lt_K (K : ℝ) (rho : ℝ) (hK : K > 1) (hrho : rho > 0) :
    N_eff K rho < K := by
  exact div_lt_self ( by positivity ) ( by nlinarith )

/-
PROVIDED SOLUTION
N_eff K 0 = K / (1 + (K-1)*0) = K / 1 = K. Unfold N_eff and simplify with ring or field_simp.
-/
theorem N_eff_eq_K (K : ℝ) (hK : K > 0) :
    N_eff K 0 = K := by
  unfold N_eff; norm_num;

/-! ## Claim 6: Numerical verification

With the given parameters, verify the effective sample size computation.
-/

/-- N_eff = 9 / (1 + 8 * 0.28) = 9 / 3.24 -/
theorem numerical_N_eff :
    N_eff 9 0.28 = 9 / 3.24 := by
  unfold N_eff
  ring

/-
PROBLEM
N_eff ≈ 2.778: we verify 9 / 3.24 is in the range (2.77, 2.78)

PROVIDED SOLUTION
N_eff 9 0.28 = 9 / 3.24. We need 2.77 < 9/3.24 and 9/3.24 < 2.78. These are rational number inequalities. Unfold N_eff and use norm_num.
-/
theorem numerical_N_eff_bounds :
    2.77 < N_eff 9 0.28 ∧ N_eff 9 0.28 < 2.78 := by
  unfold N_eff; norm_num;

/-
PROBLEM
N_portfolio = N * N_eff = 28108 * (25/9) = 702700/9 ≈ 78077.78

PROVIDED SOLUTION
28108 * N_eff 9 0.28 = 28108 * (9/3.24) = 28108 * 25/9 = 702700/9. We need 78077 < 702700/9 < 78078. Unfold N_eff and use norm_num.
-/
theorem numerical_N_portfolio_bounds :
    78077 < 28108 * N_eff 9 0.28 ∧ 28108 * N_eff 9 0.28 < 78078 := by
  unfold N_eff; norm_num;

/-! ## Claim 7: Bias correction factor c₄(4)

For n i.i.d. samples from N(0, σ²), the sample standard deviation has
E[s] = σ · c₄(n), where c₄(n) = √(2/(n-1)) · Γ(n/2) / Γ((n-1)/2).

For n = 4: c₄(4) = √(2/3) · Γ(2) / Γ(3/2).

Since Γ(2) = 1! = 1 and Γ(3/2) = √π/2, we get:
  c₄(4) = √(2/3) · 2/√π = 2√2 / (√3 · √π) = 2/√(3π/2)

Numerically c₄(4) ≈ 0.9213...
We prove that c₄(4) < 1 (i.e., sample std underestimates true std).
-/

/-- The bias correction factor c₄(n) = √(2/(n-1)) · Γ(n/2) / Γ((n-1)/2) -/
noncomputable def c4 (n : ℝ) : ℝ := sqrt (2 / (n - 1)) * Gamma (n / 2) / Gamma ((n - 1) / 2)

/-- c₄(4) = √(2/3) · Γ(2) / Γ(3/2) -/
theorem c4_four_formula : c4 4 = sqrt (2 / 3) * Gamma 2 / Gamma (3 / 2) := by
  unfold c4; ring_nf

/-
PROBLEM
c₄(4) = 2 * √2 / (√3 * √π), showing the sample std underestimates by factor ≈ 0.92.
    We prove this using Γ(2) = 1 and Γ(3/2) = √π/2.

PROVIDED SOLUTION
c4 4 = sqrt(2/3) * Gamma(2) / Gamma(3/2). Gamma(2) = 1 (since Gamma(n+1) = n! for natural n, Gamma(2) = 1! = 1). Gamma(3/2) = sqrt(pi)/2 (since Gamma(1/2) = sqrt(pi) and Gamma(3/2) = (1/2)*Gamma(1/2) = sqrt(pi)/2). So c4 4 = sqrt(2/3) * 1 / (sqrt(pi)/2) = 2*sqrt(2/3)/sqrt(pi) = 2*sqrt(2)/(sqrt(3)*sqrt(pi)). Use Real.Gamma_two, Real.Gamma_three_div_two or Gamma_one_half_eq, and sqrt properties.
-/
theorem c4_four_explicit :
    c4 4 = 2 * sqrt 2 / (sqrt 3 * sqrt π) := by
  unfold c4; norm_num; ring_nf; norm_num [ Real.pi_pos.le ] ;
  rw [ show ( 3 / 2 : ℝ ) = 1 / 2 + 1 by norm_num, Real.Gamma_add_one ( by norm_num ), Real.Gamma_one_half_eq ] ; ring

/-
PROBLEM
The bias correction factor c₄(4) < 1, confirming sample std underestimates true std.

PROVIDED SOLUTION
c4 4 = 2*sqrt(2)/(sqrt(3)*sqrt(pi)). We need 2*sqrt(2) < sqrt(3)*sqrt(pi), i.e., (2*sqrt(2))^2 < (sqrt(3)*sqrt(pi))^2, i.e., 8 < 3*pi. Since pi > 3.14, 3*pi > 9.42 > 8. Use the explicit formula for c4 4 (theorem c4_four_explicit) and then show the inequality. Alternatively, show c4 4 ^2 < 1 using Gamma values. c4(4)^2 = (2/3) * Gamma(2)^2 / Gamma(3/2)^2 = (2/3) * 1 / (pi/4) = 8/(3*pi) < 1 iff 8 < 3*pi which holds since pi > 3.
-/
theorem c4_four_lt_one : c4 4 < 1 := by
  rw [ c4_four_explicit, div_lt_one ] <;> norm_num [ Real.pi_pos ];
  rw [ ← Real.sqrt_mul <| by positivity ] ; exact Real.lt_sqrt_of_sq_lt <| by nlinarith [ Real.pi_gt_three, Real.sq_sqrt <| show 0 ≤ 2 by norm_num ] ;

end
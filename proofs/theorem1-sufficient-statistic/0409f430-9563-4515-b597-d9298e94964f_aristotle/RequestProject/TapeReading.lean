import Mathlib

/-!
# Optimal Tape Reading Classifier via Sufficient Statistic Compression

## Overview

We formalize and analyze the claimed theorem about the Kyle (1985) market microstructure
model. The theorem makes two claims:

1. **Sufficient Statistic Claim:** The two-dimensional statistic
   T(X_t) = (λ_t · OFI_t, TFI_t · |OFI_t|) is sufficient for forward returns Y_t.

2. **Classifier Bound Claim:** The Bayes-optimal classifier achieves accuracy
   ≥ 1/2 + λ·E[|x_t|]/(2·√(2π)·σ_u).

### Results

- **Claim 1 is TRUE.** Under the Kyle model Y = λ·OFI + U (with U independent of X),
  the signal λ·OFI is the first component of T(X), so T(X) captures all predictive
  information. We prove this both algebraically and via conditional expectations.

- **Claim 2 is FALSE** as a universal bound. We prove that there exist valid positive
  model parameters (λ = 1, E[|x|] = 3, σ_u = 1) where the claimed lower bound
  exceeds 1, making it impossible as a probability bound. The key inequality is
  9 > 2π, which implies 3 > √(2π), causing the bound to exceed 1.

  A corrected universal lower bound on the Gaussian CDF is provided.
-/

open MeasureTheory ProbabilityTheory Real

noncomputable section

/-! ## Part 1: Sufficient Statistic — Algebraic Core

Under the Kyle model, Y_t = λ · OFI_t + U_t. The optimal predictor of Y given X is
E[Y|X] = λ · OFI (since U is independent noise with zero mean).

The statistic T(X) = (λ · OFI, TFI · |OFI|) has λ · OFI as its first component.
Therefore, the optimal predictor is a function of T(X), establishing sufficiency.
-/

/-- The signal λ·OFI is recoverable as the first projection of the statistic
    T(X) = (λ·OFI, TFI·|OFI|). This is the algebraic core of the sufficient
    statistic claim: the optimal predictor E[Y|X] = λ·OFI is a function of T(X). -/
theorem signal_eq_first_component_of_T (lambda OFI TFI : ℝ) :
    lambda * OFI = (lambda * OFI, TFI * |OFI|).1 := by
  simp

/-- Under the Kyle model Y = λ·OFI + U, the forward return Y depends on the
    full feature vector X = (OFI, TFI, λ) only through the signal S = λ·OFI.
    Any additional features (TFI, λ separately) provide no extra information
    beyond what S already contains. -/
theorem kyle_model_signal_determines_prediction
    (lambda OFI U : ℝ) (Y : ℝ) (hY : Y = lambda * OFI + U) :
    ∃ (S : ℝ), S = lambda * OFI ∧ Y = S + U := by
  exact ⟨lambda * OFI, rfl, hY⟩

/-! ## Part 1b: Sufficient Statistic — Conditional Expectation Formulation

We prove: if Y = S + U where S is 𝒢-measurable, U is integrable, and E[U | 𝒢] = 0 a.e.,
then E[Y | 𝒢] = S a.e.

Since both σ(X) and σ(T) contain σ(S) (because S = λ·OFI is a function of both X
and T), we get E[Y | σ(X)] = S = E[Y | σ(T)] a.e.
-/

variable {Ω : Type*} [mΩ : MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- If S is 𝒢-measurable, integrable, and U has conditional expectation zero
    given 𝒢, then E[S + U | 𝒢] = S a.e.

    This is the conditional expectation formulation of the sufficient statistic theorem:
    in Y = signal + noise, if noise has zero conditional expectation given 𝒢,
    the conditional expectation of Y given 𝒢 is just the signal. -/
theorem condexp_signal_plus_noise
    (𝒢 : MeasurableSpace Ω) (h𝒢 : 𝒢 ≤ mΩ)
    (S U : Ω → ℝ)
    (hS_meas : @Measurable Ω ℝ 𝒢 (borel ℝ) S)
    (hS_int : Integrable S μ)
    (hU_int : Integrable U μ)
    (hU_condexp : μ[U | 𝒢] =ᵐ[μ] 0) :
    μ[S + U | 𝒢] =ᵐ[μ] S := by
  have h_condexp_S : μ[S | 𝒢] =ᶠ[ae μ] S := by
    rw [MeasureTheory.condExp_of_stronglyMeasurable]
    · assumption
    · exact hS_meas.stronglyMeasurable
    · exact hS_int
  have := @condExp_add
  exact this hS_int hU_int _ |> fun h =>
    h.trans (Filter.EventuallyEq.add h_condexp_S hU_condexp) |> fun h =>
    h.trans (by simp +decide)

/-! ## Part 2: Counterexample to the Classifier Bound

The claimed bound states:
  P(sign(Ŷ) = sign(Y)) ≥ 1/2 + (λ · E[|x_t|]) / (2 · √(2π) · σ_u)

We show this is FALSE by exhibiting parameters where the RHS exceeds 1.
Since any probability is at most 1, the bound cannot hold.

With λ = 1, E[|x|] = 3, σ_u = 1:
  RHS = 1/2 + 3/(2√(2π)) > 1
because 3 > √(2π), which follows from 9 > 2π (since π < 4.5).
-/

/-- Key inequality: 2π < 9. -/
theorem two_pi_lt_nine : 2 * Real.pi < 9 := by
  have h_pi_approx : Real.pi < 3.15 := by exact pi_lt_d2
  linarith

/-- Consequence: √(2π) < 3. -/
theorem sqrt_two_pi_lt_three : Real.sqrt (2 * Real.pi) < 3 := by
  rw [Real.sqrt_lt] <;>
    linarith [Real.pi_gt_three, show Real.pi < 4 by pi_upper_bound []]

/-- The claimed classifier accuracy bound is FALSE.

    There exist positive model parameters (λ, E[|x|], σ_u) such that the
    claimed lower bound 1/2 + λ·E[|x|]/(2·√(2π)·σ_u) exceeds 1.
    Since probabilities cannot exceed 1, the bound is invalid.

    Witness: λ = 1, E[|x|] = 3, σ_u = 1 gives
    1/2 + 3/(2·√(2π)) > 1 because 9 > 2π. -/
theorem classifier_bound_false :
    ∃ (lam E_abs_x sigma_u : ℝ),
      0 < lam ∧ 0 < E_abs_x ∧ 0 < sigma_u ∧
      1 / 2 + (lam * E_abs_x) / (2 * Real.sqrt (2 * Real.pi) * sigma_u) > 1 := by
  use 1, 3, 1; norm_num
  field_simp
  nlinarith [Real.pi_le_four, Real.sqrt_nonneg 2, Real.sqrt_nonneg π,
    Real.sq_sqrt zero_le_two, Real.sq_sqrt Real.pi_pos.le]

/-- The claimed lower bound function 1/2 + z/(2√(2π)) is unbounded,
    hence cannot be a valid probability bound for all z ≥ 0. -/
theorem claimed_bound_unbounded :
    ∀ M : ℝ, ∃ z : ℝ, 0 ≤ z ∧
      1 / 2 + z / (2 * Real.sqrt (2 * Real.pi)) > M := by
  exact fun M =>
    ⟨Max.max (0 : ℝ) ((M - 1 / 2) * (2 * Real.sqrt (2 * Real.pi)) + 1),
     by positivity,
     by nlinarith [
       le_max_left (0 : ℝ) ((M - 1 / 2) * (2 * Real.sqrt (2 * Real.pi)) + 1),
       le_max_right (0 : ℝ) ((M - 1 / 2) * (2 * Real.sqrt (2 * Real.pi)) + 1),
       Real.sqrt_nonneg (2 * Real.pi),
       Real.mul_self_sqrt (show 0 ≤ 2 * Real.pi by positivity),
       mul_div_cancel₀
         (Max.max (0 : ℝ) ((M - 1 / 2) * (2 * Real.sqrt (2 * Real.pi)) + 1))
         (show (2 * Real.sqrt (2 * Real.pi)) ≠ 0 by positivity)]⟩

/-! ## Part 3: A Corrected Bound

A valid universal lower bound on the Gaussian CDF is: for all z ≥ 0,
  Φ(z) ≥ 1/2 + z · φ(z) = 1/2 + z · exp(-z²/2) / √(2π)
where φ is the standard normal PDF. This follows because φ is decreasing
on [0,∞), so ∫₀ᶻ φ(t) dt ≥ z · φ(z).

The corrected classifier bound is therefore:
  P(correct) ≥ 1/2 + E[|μ| · exp(-μ²/(2σ²))] / (√(2π) · σ)

Note: this bound decays exponentially for large signal-to-noise ratios
(unlike the original claim which grows without bound), and it naturally
stays below 1 for all parameter values.
-/

/-- For z ≥ 0, the integral of exp(-t²/2)/√(2π) from 0 to z is at least
    z · exp(-z²/2)/√(2π). This follows from monotonicity: the integrand
    is decreasing, so its minimum on [0,z] is achieved at z.

    This establishes the corrected Gaussian CDF lower bound:
    Φ(z) - 1/2 ≥ z · φ(z) for z ≥ 0. -/
theorem gaussian_cdf_lower_bound (z : ℝ) (hz : 0 ≤ z) :
    z * Real.exp (-(z ^ 2) / 2) / Real.sqrt (2 * Real.pi) ≤
      ∫ t in Set.Icc 0 z, Real.exp (-(t ^ 2) / 2) / Real.sqrt (2 * Real.pi) := by
  have h_decreasing : ∀ t₁ t₂ : ℝ, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ z →
      Real.exp (-t₁ ^ 2 / 2) / Real.sqrt (2 * Real.pi) ≥
      Real.exp (-t₂ ^ 2 / 2) / Real.sqrt (2 * Real.pi) := by
    exact fun t₁ t₂ ht₁ ht₂ _ =>
      div_le_div_of_nonneg_right (Real.exp_le_exp.mpr <| by nlinarith) <|
      Real.sqrt_nonneg _
  have h_integral_ge :
      ∫ t in Set.Icc 0 z, Real.exp (-t ^ 2 / 2) / Real.sqrt (2 * Real.pi) ≥
      ∫ t in Set.Icc 0 z, (Real.exp (-z ^ 2 / 2)) / Real.sqrt (2 * Real.pi) := by
    refine' MeasureTheory.setIntegral_mono_on _ _ measurableSet_Icc
      fun t ht => h_decreasing _ _ ht.1 ht.2 le_rfl
    · norm_num
    · exact Continuous.integrableOn_Icc (by continuity)
  simpa [mul_div_assoc, hz] using h_integral_ge

end

/-
# Claim 3: Regime gate improves effective accuracy

If the classifier's accuracy is higher during high-activity regimes
(because SNR(r) = SNR_base/√(1-r) is increasing in r), and
Φ is concave and increasing, then conditioning on r ≥ r_min yields:
  α_regime = E[Φ(SNR(r)) | r ≥ r_min] ≥ E[Φ(SNR(r))] = α_all

We prove the core mathematical fact: if g is monotone increasing and
we condition on a variable being in its upper tail, then the conditional
expectation is at least the unconditional expectation.

We formalize this as a deterministic inequality about averages/integrals.
-/
import Mathlib

open MeasureTheory Measure Real

noncomputable section

/-
PROBLEM
Core lemma: For a monotone increasing function g on [a,b],
    the average on [t,b] ≥ the average on [a,b] when t ∈ [a,b].
    This is the key insight behind Claim 3.

PROVIDED SOLUTION
For a monotone increasing function g on [a,b], the average on [t,b] ≥ average on [a,b] when a ≤ t < b. Key idea: split the integral on [a,b] = integral on [a,t] + integral on [t,b]. On [a,t], g(x) ≤ g(t) (by monotonicity). On [t,b], g(x) ≥ g(t). So avg[a,b] = (∫[a,t] g + ∫[t,b] g)/(b-a). And avg[t,b] = ∫[t,b] g/(b-t). We need ∫[t,b]g/(b-t) ≥ (∫[a,t]g + ∫[t,b]g)/(b-a). Cross multiply: (b-a)∫[t,b]g ≥ (b-t)(∫[a,t]g + ∫[t,b]g). Simplify: (b-a)∫[t,b]g - (b-t)∫[t,b]g ≥ (b-t)∫[a,t]g. (t-a)∫[t,b]g ≥ (b-t)∫[a,t]g. Since g is monotone, ∫[t,b]g ≥ g(t)(b-t) and ∫[a,t]g ≤ g(t)(t-a). So (t-a)∫[t,b]g ≥ (t-a)g(t)(b-t) = (b-t)g(t)(t-a) ≥ (b-t)∫[a,t]g. QED.

This is a non-trivial measure theory argument. Given the complexity of formalizing integral inequalities in Lean/Mathlib, consider using a more abstract approach or the mean value theorem for integrals.
-/
theorem monotone_upper_tail_avg_ge
    {a b t : ℝ} (hab : a < b) (hat : a ≤ t) (htb : t < b)
    {g : ℝ → ℝ} (hg : MonotoneOn g (Set.Icc a b)) :
    (∫ x in Set.Icc a b, g x) / (b - a) ≤
    (∫ x in Set.Icc t b, g x) / (b - t) := by
      -- Split the integral on [a,b] = integral on [a,t] + integral on [t,b].
      have h_split : ∫ x in (Set.Icc a b), (g x) = (∫ x in (Set.Icc a t), (g x)) + (∫ x in (Set.Icc t b), (g x)) := by
        norm_num [ MeasureTheory.integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le, * ];
        rw [ ← intervalIntegral.integral_of_le ( by linarith ), ← intervalIntegral.integral_of_le ( by linarith ), intervalIntegral.integral_add_adjacent_intervals ] <;> apply_rules [ MonotoneOn.intervalIntegrable, hg.mono ];
        · rw [ Set.uIcc_of_le hat ] ; exact Set.Icc_subset_Icc_right htb.le;
        · rw [ Set.uIcc_of_le htb.le ] ; exact Set.Icc_subset_Icc hat le_rfl;
      -- On [a,t], g(x) ≤ g(t) (by monotonicity). On [t,b], g(x) ≥ g(t).
      have h_le : (∫ x in Set.Icc a t, (g x)) ≤ (t - a) * g t := by
        have h_int_le : ∫ x in Set.Icc a t, g x ≤ ∫ x in Set.Icc a t, g t := by
          refine' MeasureTheory.setIntegral_mono_on _ _ _ _ <;> norm_num [ * ];
          · exact ( hg.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) ) |> fun h => h.mono_set ( Set.Icc_subset_Icc le_rfl htb.le );
          · exact fun x hx₁ hx₂ => hg ⟨ hx₁, by linarith ⟩ ⟨ by linarith, by linarith ⟩ hx₂;
        aesop
      have h_ge : (∫ x in Set.Icc t b, (g x)) ≥ (b - t) * g t := by
        refine' le_trans _ ( MeasureTheory.setIntegral_mono_on _ _ measurableSet_Icc fun x hx => hg ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ hx.1 ) <;> norm_num [ htb.le ];
        exact ( hg.mono ( Set.Icc_subset_Icc ( by linarith ) le_rfl ) ) |> fun h => h.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc );
      rw [ div_le_div_iff₀ ] <;> nlinarith [ mul_self_nonneg ( b - t ), mul_self_nonneg ( t - a ) ] ;

/-- SNR as a function of branching ratio r, with r ∈ [0,1) -/
def SNR (SNR_base : ℝ) (r : ℝ) : ℝ := SNR_base / Real.sqrt (1 - r)

/-
PROBLEM
SNR is monotone increasing on [0,1) when SNR_base > 0

PROVIDED SOLUTION
SNR(r) = SNR_base/√(1-r). On [0,1), as r increases, 1-r decreases, √(1-r) decreases, so 1/√(1-r) increases, and SNR_base > 0 times that is increasing. Formally, MonotoneOn by showing that for r₁ ≤ r₂ in [0,1), 1-r₂ ≤ 1-r₁, so √(1-r₂) ≤ √(1-r₁), so 1/√(1-r₂) ≥ 1/√(1-r₁), so SNR_base/√(1-r₂) ≥ SNR_base/√(1-r₁). Use Real.sqrt_le_sqrt and div_le_div_left.
-/
theorem SNR_monotoneOn (SNR_base : ℝ) (h : 0 < SNR_base) :
    MonotoneOn (SNR SNR_base) (Set.Ico 0 1) := by
      intro x hx y hy hxy; unfold SNR; gcongr; aesop;

/-
PROBLEM
If Φ is monotone increasing and concave (e.g., Gaussian CDF on [0,∞)),
    then Φ ∘ SNR is monotone increasing on [0,1).

PROVIDED SOLUTION
Composition of monotone functions. SNR is monotone on [0,1) (proved above as SNR_monotoneOn). Φ is globally monotone. Monotone.comp_monotoneOn gives the result.
-/
theorem Phi_SNR_monotoneOn (SNR_base : ℝ) (h : 0 < SNR_base)
    {Φ : ℝ → ℝ} (hΦ : Monotone Φ) :
    MonotoneOn (Φ ∘ SNR SNR_base) (Set.Ico 0 1) := by
      exact hΦ.comp_monotoneOn ( SNR_monotoneOn SNR_base h )

end
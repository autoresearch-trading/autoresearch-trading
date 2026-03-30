import Mathlib

/-!
# Normalization Preserves Sufficient Statistics

We formalize the Fisher–Neyman factorization characterization of sufficient statistics
and prove that invertible (bijective) transformations preserve sufficiency. We then
instantiate this with z-score normalization and IQR-based scaling.

## Main results

- `SufficientViaFactorization`: Definition of sufficiency via the factorization criterion.
- `sufficient_of_bijective_comp`: Claim 1 – bijective transformations preserve sufficiency.
- `zscore_bijective`: Claim 2 – z-score normalization is bijective when σ > 0.
- `zscore_preserves_sufficiency`: Claim 3 – z-score normalized features preserve sufficiency.
- `iqr_bijective`: Claim 4a – IQR scaling is bijective when IQR > 0.
- `iqr_preserves_sufficiency`: Claim 4a – IQR scaling preserves sufficiency when IQR > 0.
- `iqr_not_injective_of_zero`: Claim 4b – IQR scaling is not injective when IQR = 0.
-/

noncomputable section

open Function

/-! ## Sufficiency via Factorization -/

/-- A statistic `T : X → S` is sufficient for a family of densities `f : X → Θ → ℝ`
    (the Fisher–Neyman factorization criterion) if there exist `h : X → ℝ` and
    `k : S → Θ → ℝ` such that `f x θ = h x * k (T x) θ` for all `x, θ`. -/
def SufficientViaFactorization
    {X Θ S : Type*} (f : X → Θ → ℝ) (T : X → S) : Prop :=
  ∃ (h : X → ℝ) (k : S → Θ → ℝ), ∀ x θ, f x θ = h x * k (T x) θ

/-! ## Claim 1: Bijective transformations preserve sufficiency -/

/-
PROBLEM
If `T` is a sufficient statistic (via factorization) and `g` is bijective,
    then `g ∘ T` is also a sufficient statistic.

PROVIDED SOLUTION
From hT, get h, k such that f x θ = h x * k (T x) θ. Define k' : S' → Θ → ℝ by k' s' θ = k (g.invFun s') θ (using the inverse from bijectivity). Then f x θ = h x * k' (g (T x)) θ, since g.invFun (g (T x)) = T x by left inverse property of bijection.
-/
theorem sufficient_of_bijective_comp
    {X Θ S S' : Type*} {f : X → Θ → ℝ} {T : X → S} {g : S → S'}
    (hT : SufficientViaFactorization f T)
    (hg : Bijective g) :
    SufficientViaFactorization f (g ∘ T) := by
  -- From hT, obtain h and k such that f x θ = h x * k (T x) θ.
  obtain ⟨h, k, hk⟩ := hT;
  -- Define k' : S' → Θ → ℝ by k' s' θ = k (g.invFun s') θ.
  set k' : S' → Θ → ℝ := fun s' θ => k (hg.2 s' |> Classical.choose) θ;
  refine' ⟨ h, k', fun x θ => _ ⟩;
  have := Classical.choose_spec ( hg.2 ( g ( T x ) ) );
  have := hg.1 this; aesop;

/-! ## Claim 2: Z-score normalization is bijective -/

/-- Z-score normalization: `g(t) = (t - μ) / σ`. -/
def zscore (μ σ : ℝ) (t : ℝ) : ℝ := (t - μ) / σ

/-- Inverse of z-score normalization: `g⁻¹(z) = z * σ + μ`. -/
def zscore_inv (μ σ : ℝ) (z : ℝ) : ℝ := z * σ + μ

/-
PROBLEM
Z-score normalization is bijective when `σ > 0`.

PROVIDED SOLUTION
Show zscore_inv μ σ is a two-sided inverse of zscore μ σ. For right inverse: zscore μ σ (zscore_inv μ σ z) = ((z * σ + μ) - μ) / σ = z. For left inverse: zscore_inv μ σ (zscore μ σ t) = ((t - μ) / σ) * σ + μ = t. Use hσ to ensure σ ≠ 0 for division cancellation. Then use Function.bijective_iff_has_inverse or construct from injective+surjective.
-/
theorem zscore_bijective (μ σ : ℝ) (hσ : σ > 0) : Bijective (zscore μ σ) := by
  refine' ⟨ fun x y hxy => _, fun x => _ ⟩;
  · unfold zscore at hxy; rw [ div_eq_div_iff ] at hxy <;> nlinarith;
  · exact ⟨ x * σ + μ, by unfold zscore; rw [ div_eq_iff hσ.ne' ] ; ring ⟩

/-! ## Claim 3: Z-score normalization preserves sufficiency -/

/-- Z-score normalized features preserve the sufficient statistic property. -/
theorem zscore_preserves_sufficiency
    {X Θ : Type*} {f : X → Θ → ℝ} {T : X → ℝ}
    (hT : SufficientViaFactorization f T)
    {μ σ : ℝ} (hσ : σ > 0) :
    SufficientViaFactorization f (zscore μ σ ∘ T) :=
  sufficient_of_bijective_comp hT (zscore_bijective μ σ hσ)

/-! ## Claim 4: IQR-based scaling -/

/-- IQR-based scaling: `g(t) = (t - median) / IQR`. This is the same formula as z-score
    but with different parameters (median instead of mean, IQR instead of σ). -/
def iqrScale (median iqr : ℝ) (t : ℝ) : ℝ := (t - median) / iqr

/-
PROBLEM
IQR-based scaling is bijective when `IQR > 0`.

PROVIDED SOLUTION
iqrScale is definitionally the same as zscore (same formula). Show bijectivity using the same argument as zscore_bijective, or directly reuse it. The inverse is t = z * iqr + median. Use hiqr > 0 to get iqr ≠ 0.
-/
theorem iqr_bijective (median iqr : ℝ) (hiqr : iqr > 0) :
    Bijective (iqrScale median iqr) := by
  refine' ⟨ _, _ ⟩;
  · -- To prove injectivity, assume $iqrScale(t1) = iqrScale(t2)$ and show that $t1 = t2$.
    intro t1 t2 h_eq
    have h_eq' : (t1 - median) / iqr = (t2 - median) / iqr := by
      exact h_eq
    field_simp [hiqr] at h_eq'
    linarith;
  · exact fun x => ⟨ x * iqr + median, by unfold iqrScale; rw [ div_eq_iff hiqr.ne' ] ; ring ⟩

/-- IQR scaling preserves sufficiency when IQR > 0. -/
theorem iqr_preserves_sufficiency
    {X Θ : Type*} {f : X → Θ → ℝ} {T : X → ℝ}
    (hT : SufficientViaFactorization f T)
    {median iqr : ℝ} (hiqr : iqr > 0) :
    SufficientViaFactorization f (iqrScale median iqr ∘ T) :=
  sufficient_of_bijective_comp hT (iqr_bijective median iqr hiqr)

/-
PROBLEM
When `IQR = 0`, the IQR scaling maps everything to `0`
    (assuming `0 / 0 = 0` as in Lean), so it is **not** injective
    (hence not bijective) as soon as the domain has at least two points.

PROVIDED SOLUTION
iqrScale median 0 t = (t - median) / 0 = 0 for all t (div_zero). So iqrScale median 0 median = 0 and iqrScale median 0 (median + 1) = 0. These are equal but median ≠ median + 1. So the function is not injective. Use fun h => by have := h (show iqrScale median 0 median = iqrScale median 0 (median+1) from ...) and derive False from median = median + 1.
-/
theorem iqr_not_injective_of_zero (median : ℝ) :
    ¬ Injective (iqrScale median 0) := by
  norm_num [ Function.Injective, iqrScale ];
  exact ⟨ 0, 1, by norm_num ⟩

end
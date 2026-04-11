/-
# Claim 3: Regime-Conditional Trading — When to Use the Sufficient Statistic

The Hawkes intensity ratio r partitions time into regimes. We define the
regime-conditional SNR and prove its key properties.

Key simplification: SNR(r) = SNR_base · (1 + r/(1-r))^(1/2) = SNR_base / √(1-r)
-/
import Mathlib

open Real Set

section Claim3

/-- The regime-conditional SNR function.
    SNR(r) = SNR_base · (1 + r/(1-r))^(1/2) = SNR_base / √(1-r)
    We use the simplified form SNR_base / √(1-r). -/
noncomputable def snr_regime (SNR_base : ℝ) (r : ℝ) : ℝ :=
  SNR_base / Real.sqrt (1 - r)

/-
PROBLEM
The two definitions agree on (0,1).

PROVIDED SOLUTION
We need SNR_base / sqrt(1-r) = SNR_base * sqrt(1 + r/(1-r)). Since r ∈ (0,1), 1-r > 0. We have 1 + r/(1-r) = (1-r+r)/(1-r) = 1/(1-r). So sqrt(1 + r/(1-r)) = sqrt(1/(1-r)) = 1/sqrt(1-r). Thus both sides equal SNR_base / sqrt(1-r). Use Real.sqrt_div' or Real.sqrt_inv and algebraic rewrites.
-/
theorem snr_regime_eq_original {SNR_base r : ℝ} (hr : r ∈ Ioo 0 1) :
    snr_regime SNR_base r = SNR_base * Real.sqrt (1 + r / (1 - r)) := by
  erw [ show ( snr_regime SNR_base r ) = SNR_base / Real.sqrt ( 1 - r ) from rfl, show ( 1 + r / ( 1 - r ) ) = ( 1 - r + r ) / ( 1 - r ) from by rw [ add_div, div_self ] ; linarith [ hr.1, hr.2 ] ] ; norm_num [ hr.1.ne', hr.2.ne' ] ; ring;

/-
PROBLEM
(a) SNR(r) is strictly increasing on (0,1) for positive SNR_base

PROVIDED SOLUTION
snr_regime SNR_base r = SNR_base / sqrt(1-r). Since SNR_base > 0, this is strictly increasing iff 1/sqrt(1-r) is strictly increasing on (0,1), which holds because 1-r is strictly decreasing, sqrt is monotone, and 1/x is strictly decreasing for x > 0. Use StrictMonoOn with intro, unfold snr_regime, and show that r₁ < r₂ implies 1-r₂ < 1-r₁ implies sqrt(1-r₂) < sqrt(1-r₁) implies 1/sqrt(1-r₂) > 1/sqrt(1-r₁) implies SNR_base/sqrt(1-r₁) < SNR_base/sqrt(1-r₂). Use div_lt_div_left with sqrt_pos and sqrt_lt_sqrt.
-/
theorem snr_regime_strictMonoOn (hS : SNR_base > 0) :
    StrictMonoOn (snr_regime SNR_base) (Ioo 0 1) := by
  exact fun x hx y hy hxy => div_lt_div_of_pos_left hS ( Real.sqrt_pos.mpr <| by linarith [ hx.1, hx.2, hy.1, hy.2 ] ) <| Real.sqrt_lt_sqrt ( by linarith [ hx.1, hx.2, hy.1, hy.2 ] ) <| by linarith;

/-
PROBLEM
(b) SNR(r) → ∞ as r → 1⁻ for positive SNR_base

PROVIDED SOLUTION
snr_regime SNR_base r = SNR_base / sqrt(1-r). As r → 1⁻, 1-r → 0⁺, sqrt(1-r) → 0⁺, so SNR_base/sqrt(1-r) → +∞. Use Filter.Tendsto.div_atTop or show that sqrt(1-r) → 0 implies 1/sqrt(1-r) → ∞. The key is: Filter.Tendsto (fun r => 1 - r) (nhdsWithin 1 (Iio 1)) (nhdsWithin 0 (Ioi 0)), then sqrt tends to 0, then division by something tending to 0⁺ gives +∞.
-/
theorem snr_regime_tendsto_atTop (hS : SNR_base > 0) :
    Filter.Tendsto (fun r => snr_regime SNR_base r) (nhdsWithin 1 (Iio 1)) Filter.atTop := by
  refine' Filter.Tendsto.const_mul_atTop hS ( Filter.Tendsto.inv_tendsto_nhdsGT_zero _ );
  refine' Filter.Tendsto.inf _ _ <;> norm_num;
  exact Continuous.tendsto' ( Real.continuous_sqrt.comp <| continuous_const.sub continuous_id' ) _ _ <| by norm_num;

/-
PROBLEM
(c) There exists a unique r_min ∈ (0,1) such that SNR(r) ≥ SNR_min iff r ≥ r_min,
    when SNR_base < SNR_min and both are positive.

PROVIDED SOLUTION
Use r_min = (SNR_min² - SNR_base²)/SNR_min² from r_min_in_Ioo (which shows r_min ∈ (0,1)) and r_min_formula (which shows snr_regime SNR_base r_min = SNR_min). Then for any r ∈ (0,1), snr_regime SNR_base r ≥ SNR_min iff r ≥ r_min because snr_regime is strictly monotone on (0,1) (from snr_regime_strictMonoOn). Specifically, if r ≥ r_min, then by monotonicity snr_regime r ≥ snr_regime r_min = SNR_min. Conversely, if r < r_min then snr_regime r < snr_regime r_min = SNR_min. Use ⟨r_min, r_min_in_Ioo, ...⟩ and split the iff using the strict monotonicity.
-/
theorem exists_unique_r_min {SNR_base SNR_min : ℝ}
    (hSb : SNR_base > 0) (hSm : SNR_min > 0) (hlt : SNR_base < SNR_min) :
    ∃ r_min ∈ Ioo 0 1, ∀ r ∈ Ioo 0 1,
      snr_regime SNR_base r ≥ SNR_min ↔ r ≥ r_min := by
  -- By definition of $r_min$, we know that $snr\_regime (SNR\_base) r_min = SNR_min$.
  obtain ⟨r_min, hr_min_exists⟩ : ∃ r_min ∈ (Set.Ioo 0 1), snr_regime SNR_base r_min = SNR_min := by
    use (SNR_min ^ 2 - SNR_base ^ 2) / SNR_min ^ 2;
    unfold snr_regime;
    field_simp;
    exact ⟨ ⟨ div_pos ( by nlinarith ) ( by positivity ), by rw [ div_lt_iff₀ ( by positivity ) ] ; nlinarith ⟩, by rw [ show ( SNR_min ^ 2 - ( SNR_min ^ 2 - SNR_base ^ 2 ) ) / SNR_min ^ 2 = ( SNR_base / SNR_min ) ^ 2 by ring, Real.sqrt_sq ( by positivity ) ] ; rw [ div_div_cancel₀ <| by positivity ] ⟩;
  use r_min, hr_min_exists.1;
  have h_mono : StrictMonoOn (snr_regime SNR_base) (Set.Ioo 0 1) := by
    exact snr_regime_strictMonoOn hSb;
  exact fun x hx => ⟨ fun hx' => le_of_not_gt fun hx'' => by linarith [ h_mono ⟨ by linarith [ hx.1, hr_min_exists.1.1 ], by linarith [ hx.2, hr_min_exists.1.2 ] ⟩ ⟨ by linarith [ hx.1, hr_min_exists.1.1 ], by linarith [ hx.2, hr_min_exists.1.2 ] ⟩ hx'' ], fun hx' => by linarith [ h_mono.le_iff_le ⟨ by linarith [ hx.1, hr_min_exists.1.1 ], by linarith [ hx.2, hr_min_exists.1.2 ] ⟩ ⟨ by linarith [ hx.1, hr_min_exists.1.1 ], by linarith [ hx.2, hr_min_exists.1.2 ] ⟩ |>.2 hx' ] ⟩

/-
PROBLEM
(d) The explicit formula: r_min = (SNR_min² - SNR_base²) / SNR_min²
    when SNR_base < SNR_min.

PROVIDED SOLUTION
r_min = (SNR_min² - SNR_base²)/SNR_min². Then 1 - r_min = SNR_base²/SNR_min². So sqrt(1-r_min) = SNR_base/SNR_min (both positive). Thus snr_regime SNR_base r_min = SNR_base / (SNR_base/SNR_min) = SNR_min. Unfold snr_regime, compute 1 - r_min = 1 - (SNR_min²-SNR_base²)/SNR_min² = SNR_base²/SNR_min², then sqrt of that = SNR_base/SNR_min, then divide.
-/
theorem r_min_formula {SNR_base SNR_min : ℝ}
    (hSb : SNR_base > 0) (hSm : SNR_min > 0) (hlt : SNR_base < SNR_min) :
    let r_min := (SNR_min ^ 2 - SNR_base ^ 2) / SNR_min ^ 2
    snr_regime SNR_base r_min = SNR_min := by
  unfold snr_regime;
  field_simp;
  norm_num [ hSb.le, hSm.le, hSb.ne', hSm.ne' ]

/-
PROBLEM
r_min is in (0,1) when SNR_base < SNR_min

PROVIDED SOLUTION
r_min = (SNR_min² - SNR_base²)/SNR_min². Since 0 < SNR_base < SNR_min, we have SNR_min² - SNR_base² > 0 and SNR_min² > 0, so r_min > 0. Also SNR_base² > 0 so SNR_min² - SNR_base² < SNR_min², giving r_min < 1.
-/
theorem r_min_in_Ioo {SNR_base SNR_min : ℝ}
    (hSb : SNR_base > 0) (hSm : SNR_min > 0) (hlt : SNR_base < SNR_min) :
    let r_min := (SNR_min ^ 2 - SNR_base ^ 2) / SNR_min ^ 2
    r_min ∈ Ioo 0 1 := by
  exact ⟨ div_pos ( by nlinarith ) ( by positivity ), by rw [ div_lt_iff₀ ( by positivity ) ] ; nlinarith ⟩

end Claim3
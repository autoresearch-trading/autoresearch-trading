/-
# Claim 1: Signal-to-Noise Regime Characterization

We work with an abstract strictly increasing CDF Φ : ℝ → ℝ (modeling the standard
normal CDF) and prove that the SNR threshold for profitability is well-defined,
positive, and decreasing in the fee multiplier f*.
-/
import Mathlib

open Real Set

section Claim1

/-- (a) A strictly increasing function is injective and has a left inverse. -/
theorem phi_injective {Φ : ℝ → ℝ} (hΦ_strict : StrictMono Φ) : Function.Injective Φ :=
  StrictMono.injective hΦ_strict

/-
PROBLEM
The SNR threshold target value 1/2 + 1/(2*f) exceeds 1/2.

PROVIDED SOLUTION
1/(2*f) > 0 since f > 0, so 1/2 + 1/(2*f) > 1/2. Use positivity or linarith with div_pos.
-/
theorem snr_target_gt_half {f : ℝ} (hf : f > 0) : (1 : ℝ) / 2 + 1 / (2 * f) > 1 / 2 := by
  exact lt_add_of_pos_right _ ( by positivity )

/-
PROBLEM
(b) The SNR threshold is positive when Φ(0) = 1/2, since the target exceeds 1/2 = Φ(0).

PROVIDED SOLUTION
We have Φ(t) = 1/2 + 1/(2f) > 1/2 = Φ(0). Since Φ is strictly monotone, t > 0.
-/
theorem snr_threshold_pos {Φ : ℝ → ℝ} (hΦ_strict : StrictMono Φ) (hΦ0 : Φ 0 = 1 / 2)
    {f : ℝ} (hf : f > 0) {t : ℝ} (ht : Φ t = 1 / 2 + 1 / (2 * f)) : t > 0 := by
  exact hΦ_strict.lt_iff_lt.mp ( by linarith [ one_div_pos.mpr ( mul_pos two_pos hf ) ] )

/-
PROBLEM
(c) The threshold target 1/2 + 1/(2*f) is strictly decreasing in f for f > 0.

PROVIDED SOLUTION
For 0 < f₁ < f₂, we have 2*f₁ < 2*f₂, so 1/(2*f₁) > 1/(2*f₂), thus 1/2 + 1/(2*f₁) > 1/2 + 1/(2*f₂). Use StrictAntiOn, intro, and show 1/(2*f) is strictly decreasing using one_div_lt_one_div_of_lt or similar.
-/
theorem snr_target_antitone : StrictAntiOn (fun f => (1 : ℝ) / 2 + 1 / (2 * f)) (Ioi 0) := by
  intro f hf g hg hfg; norm_num at *; gcongr;

/-
PROBLEM
(c) The SNR threshold Φ⁻¹(1/2 + 1/(2*f)) is decreasing in f.
    More precisely: if f₁ < f₂ (both positive) and Φ(t_i) = 1/2 + 1/(2*f_i), then t₁ > t₂.
    Wider barriers → lower SNR required → easier to profit.

PROVIDED SOLUTION
Since f₁ < f₂ and both positive, 1/(2f₁) > 1/(2f₂), so 1/2 + 1/(2f₁) > 1/2 + 1/(2f₂). That means Φ(t₁) > Φ(t₂). Since Φ is strictly monotone, t₁ > t₂.
-/
theorem snr_threshold_antitone {Φ : ℝ → ℝ} (hΦ_strict : StrictMono Φ)
    {f₁ f₂ : ℝ} (hf₁ : f₁ > 0) (hf₂ : f₂ > 0) (hlt : f₁ < f₂)
    {t₁ t₂ : ℝ} (ht₁ : Φ t₁ = 1 / 2 + 1 / (2 * f₁)) (ht₂ : Φ t₂ = 1 / 2 + 1 / (2 * f₂)) :
    t₁ > t₂ := by
  exact hΦ_strict.lt_iff_lt.mp ( by rw [ ht₁, ht₂ ] ; gcongr )

end Claim1
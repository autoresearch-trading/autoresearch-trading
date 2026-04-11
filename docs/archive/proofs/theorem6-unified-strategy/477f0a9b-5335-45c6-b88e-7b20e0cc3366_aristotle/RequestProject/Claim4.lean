/-
# Claim 4: Complete Strategy Specification — Unified Optimality

This combines Claims 1-3 to show that the complete tape reading strategy
achieves positive geometric growth when all conditions are met.
-/
import Mathlib
import RequestProject.Claim1
import RequestProject.Claim2
import RequestProject.Claim3

open Real Set

section Claim4

/-- The complete strategy specification as a structure. -/
structure TapeReadingStrategy where
  /-- Kyle's lambda (price impact coefficient) -/
  lambda : ℝ
  /-- Expected absolute OFI -/
  E_abs_OFI : ℝ
  /-- Noise volatility -/
  sigma_u : ℝ
  /-- Fee rate -/
  c : ℝ
  /-- Win probability -/
  p_win : ℝ
  /-- Loss probability -/
  p_loss : ℝ
  /-- Hawkes intensity ratio α/β -/
  hawkes_ratio : ℝ
  /-- Baseline SNR -/
  SNR_base : ℝ
  -- Conditions
  h_lambda_pos : lambda > 0
  h_E_pos : E_abs_OFI > 0
  h_sigma_pos : sigma_u > 0
  h_c_pos : c > 0
  h_c_lt_one : c < 1
  h_pw_pos : p_win > 0
  h_pl_pos : p_loss > 0
  h_SNR_base_pos : SNR_base > 0
  h_hawkes : hawkes_ratio ∈ Ioo 0 1

/-- The optimal fee multiplier for a given strategy -/
noncomputable def TapeReadingStrategy.fopt (s : TapeReadingStrategy) : ℝ :=
  f_opt s.p_win s.p_loss s.c

/-- The SNR of the strategy at its Hawkes ratio -/
noncomputable def TapeReadingStrategy.snr (s : TapeReadingStrategy) : ℝ :=
  snr_regime s.SNR_base s.hawkes_ratio

/-- Main theorem: The strategy achieves positive growth when the growth product exceeds 1.
    This chains the product characterization from Claim 2 into the unified framework. -/
theorem tape_reading_profitable (s : TapeReadingStrategy)
    (h_win_arg : 1 + s.fopt * s.c - s.c > 0)
    (h_loss_arg : 1 - s.fopt * s.c - s.c > 0)
    (h_product : growth_product s.p_win s.p_loss s.c s.fopt > 1) :
    growth_rate s.p_win s.p_loss s.c s.fopt > 0 := by
  exact (growth_pos_iff_product_gt_one s.h_pw_pos s.h_pl_pos h_win_arg h_loss_arg).mpr h_product

/-
PROBLEM
The regime-conditional SNR at the Hawkes ratio exceeds baseline

PROVIDED SOLUTION
s.snr = snr_regime s.SNR_base s.hawkes_ratio = s.SNR_base / sqrt(1 - s.hawkes_ratio). Since s.hawkes_ratio ∈ (0,1), we have 0 < 1 - s.hawkes_ratio < 1, so sqrt(1-r) < 1, so SNR_base / sqrt(1-r) > SNR_base. Use div_lt_iff and sqrt_lt_one.
-/
theorem tape_reading_snr_amplified (s : TapeReadingStrategy) :
    s.snr > s.SNR_base := by
  rw [ show s.snr = s.SNR_base / Real.sqrt ( 1 - s.hawkes_ratio ) from rfl ] ; exact lt_div_iff₀ ( Real.sqrt_pos.mpr <| sub_pos.mpr s.h_hawkes.2 ) |>.2 <| by nlinarith [ Real.sqrt_nonneg ( 1 - s.hawkes_ratio ), Real.sq_sqrt ( show 0 ≤ 1 - s.hawkes_ratio by linarith [ s.h_hawkes.2 ] ), s.h_hawkes.1, s.h_hawkes.2, s.h_SNR_base_pos ] ;

/-
PROBLEM
When r ≥ r_min, the strategy's SNR exceeds the minimum threshold

PROVIDED SOLUTION
We need snr_regime s.SNR_base s.hawkes_ratio ≥ SNR_min. By r_min_formula, snr_regime s.SNR_base r_min = SNR_min where r_min = (SNR_min²-SNR_base²)/SNR_min². Since s.hawkes_ratio ≥ r_min and both are in (0,1), and snr_regime is strictly increasing on (0,1) (by snr_regime_strictMonoOn), we get snr_regime s.hawkes_ratio ≥ snr_regime r_min = SNR_min. Use the monotonicity and the formula.
-/
theorem tape_reading_snr_sufficient (s : TapeReadingStrategy)
    {SNR_min : ℝ} (hSm : SNR_min > 0) (hlt : s.SNR_base < SNR_min)
    (h_r_ge : s.hawkes_ratio ≥ (SNR_min ^ 2 - s.SNR_base ^ 2) / SNR_min ^ 2) :
    s.snr ≥ SNR_min := by
  have h_snr_ge : snr_regime s.SNR_base s.hawkes_ratio ≥ snr_regime s.SNR_base ((SNR_min ^ 2 - s.SNR_base ^ 2) / SNR_min ^ 2) := by
    apply_rules [ div_le_div_of_nonneg_left, Real.sqrt_le_sqrt ];
    · linarith [ s.h_SNR_base_pos ];
    · exact Real.sqrt_pos.mpr ( sub_pos.mpr ( s.h_hawkes.2 ) );
    · linarith;
  have h_snr_ge : snr_regime s.SNR_base ((SNR_min ^ 2 - s.SNR_base ^ 2) / SNR_min ^ 2) = SNR_min := by
    exact r_min_formula s.h_SNR_base_pos hSm hlt;
  linarith!

end Claim4
import Mathlib

/-!
# Theorem C: Optimal Spread Widening Under Adverse Selection

Formalizes the interaction between Avellaneda-Stoikov spread and VPIN-based
adverse selection adjustment.

The adjusted spread is: δ_adjusted = δ_base + lam_kyle · VPIN · E[|OFI|]
-/

noncomputable section

open Real

/-- The VPIN-adjusted spread: δ_base + lam · VPIN · expected_abs_OFI -/
def adjustedSpread (δ_base lam_kyle vpin expected_abs_OFI : ℝ) : ℝ :=
  δ_base + lam_kyle * vpin * expected_abs_OFI

/-- The VPIN adjustment term alone. -/
def vpinAdjustment (lam_kyle vpin expected_abs_OFI : ℝ) : ℝ :=
  lam_kyle * vpin * expected_abs_OFI

/-
PROBLEM
============================================================================
CLAIM C1: The adjustment is non-negative
============================================================================

C1: The VPIN adjustment is non-negative (spread only widens).

PROVIDED SOLUTION
Product of three nonneg reals is nonneg. Use mul_nonneg twice, or positivity.
-/
theorem claim_C1 {lam_kyle vpin expected_abs_OFI : ℝ}
    (hlam : lam_kyle ≥ 0) (hv : vpin ≥ 0) (hOFI : expected_abs_OFI ≥ 0) :
    vpinAdjustment lam_kyle vpin expected_abs_OFI ≥ 0 := by
  exact mul_nonneg ( mul_nonneg hlam hv ) hOFI

/-- C1': The adjusted spread is at least the base spread. -/
theorem claim_C1' {δ_base lam_kyle vpin expected_abs_OFI : ℝ}
    (hlam : lam_kyle ≥ 0) (hv : vpin ≥ 0) (hOFI : expected_abs_OFI ≥ 0) :
    adjustedSpread δ_base lam_kyle vpin expected_abs_OFI ≥ δ_base := by
  have h := claim_C1 hlam hv hOFI
  simp only [adjustedSpread, vpinAdjustment] at *
  linarith

-- ============================================================================
-- CLAIM C2: No adjustment when VPIN = 0
-- ============================================================================

/-- C2: When VPIN = 0 (no adverse selection), the adjusted spread equals the base. -/
theorem claim_C2 {δ_base lam_kyle expected_abs_OFI : ℝ} :
    adjustedSpread δ_base lam_kyle 0 expected_abs_OFI = δ_base := by
  unfold adjustedSpread; ring

/-
PROBLEM
============================================================================
CLAIM C3: Expected loss bound and zero-profit spread
============================================================================

C3a: The expected loss per unit time κ · lam · σ_x · √(2/π) is non-negative.

PROVIDED SOLUTION
Product of nonneg reals is nonneg. κ ≥ 0, lam ≥ 0, σ_x ≥ 0, and √(2/π) ≥ 0. Use positivity or mul_nonneg.
-/
theorem claim_C3a {kap lam sig_x : ℝ} (hk : kap ≥ 0) (hl : lam ≥ 0) (hs : sig_x ≥ 0) :
    kap * lam * sig_x * Real.sqrt (2 / π) ≥ 0 := by
  positivity

/-- C3b: The zero-profit spread against informed flow is 2·lam·σ_x·√(2/π),
    and this is non-negative. -/
def zeroProfitSpread (lam_kyle sig_x : ℝ) : ℝ := 2 * lam_kyle * sig_x * Real.sqrt (2 / π)

/-
PROVIDED SOLUTION
zeroProfitSpread = 2 * lam * σ_x * √(2/π). All factors are nonneg (2 > 0, lam ≥ 0, σ_x ≥ 0, √(2/π) ≥ 0). Use positivity or mul_nonneg.
-/
theorem claim_C3b {lam_kyle sig_x : ℝ} (hl : lam_kyle ≥ 0) (hs : sig_x ≥ 0) :
    zeroProfitSpread lam_kyle sig_x ≥ 0 := by
  exact mul_nonneg ( mul_nonneg ( mul_nonneg zero_le_two hl ) hs ) ( Real.sqrt_nonneg _ )

/-- C3c: The zero-profit spread is exactly twice the expected loss per unit
    informed trade (lam·σ_x·√(2/π)). -/
theorem claim_C3c {lam_kyle sig_x : ℝ} :
    zeroProfitSpread lam_kyle sig_x = 2 * (lam_kyle * sig_x * Real.sqrt (2 / π)) := by
  unfold zeroProfitSpread; ring

/-- C3d: When the spread equals the zero-profit spread, the profit from
    uninformed trades exactly covers losses to informed trades.
    Setting spread = 2·lam·σ_x·√(2/π) gives revenue = loss when κ_total = κ_informed.
    This is the break-even condition. -/
theorem claim_C3d {lam_kyle sig_x kap : ℝ} :
    zeroProfitSpread lam_kyle sig_x / 2 * kap = kap * lam_kyle * sig_x * Real.sqrt (2 / π) := by
  unfold zeroProfitSpread; ring

end
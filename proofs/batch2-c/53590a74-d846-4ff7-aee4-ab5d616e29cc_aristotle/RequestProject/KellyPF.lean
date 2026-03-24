import Mathlib

noncomputable section

open Real

/-!
# Profit Factor Decomposition and Kelly Position Sizing

We formalize and prove four claims about the relationship between
profit factor (PF), win rate, win/loss ratio, and Kelly optimal position sizing.
-/

/-
PROBLEM
============================================================================
CLAIM 1: Profit Factor Identity
============================================================================

The profit factor identity: PF = w·R/(1-w) where R = W/L.

PROVIDED SOLUTION
Field division. Cancel N (nonzero), rearrange. Use field_simp and ring.
-/
theorem pf_identity (N w W L : ℝ) (hN : N ≠ 0) (hw0 : 0 < w) (hw1 : w < 1)
    (hW : 0 < W) (hL : 0 < L) :
    (N * w * W) / (N * (1 - w) * L) = w * (W / L) / (1 - w) := by
      grind

/-
PROBLEM
The win/loss ratio R can be recovered from PF and w: R = PF·(1-w)/w.

PROVIDED SOLUTION
Substitute hPF, solve with field_simp and ring. Use that w ≠ 0 and 1-w ≠ 0.
-/
theorem R_from_PF (w R PF : ℝ) (hw0 : 0 < w) (hw1 : w < 1)
    (hPF : PF = w * R / (1 - w)) :
    R = PF * (1 - w) / w := by
      rw [ hPF, div_mul_cancel₀ _ ( by linarith ), mul_div_cancel_left₀ _ ( by linarith ) ]

/-- Numerical verification: for PF = 77/50, w = 27/50: R = 1771/1350. -/
theorem numerical_verify_R :
    (77 : ℚ) / 50 * (1 - 27 / 50) / (27 / 50) = 1771 / 1350 := by norm_num

-- ============================================================================
-- CLAIM 2: Kelly Fraction from Profit Factor
-- ============================================================================

/-- The Kelly-optimal fraction of capital to risk per trade. -/
def kelly_fraction (p PF : ℝ) : ℝ := p * (PF - 1) / PF

/-
PROBLEM
Derivation: Kelly f* = (p·b - q)/b with b = R = PF·q/p gives f* = p·(PF-1)/PF.
    Here we verify the algebraic manipulation.

PROVIDED SOLUTION
Use field_simp with p ≠ 0 (from hp) and 1-p ≠ 0 (from hp1), PF ≠ 0 (from hPF), then ring.
-/
theorem kelly_derivation (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : 0 < PF) :
    let q := 1 - p
    let b := PF * q / p
    (p * b - q) / b = p * (PF - 1) / PF := by
      grind

/-
PROBLEM
f* > 0 if and only if PF > 1 (for p ∈ (0,1)).

PROVIDED SOLUTION
kelly_fraction p PF = p * (PF - 1) / PF. Since p > 0 and PF > 0, the sign of f* is the sign of PF - 1. So f* > 0 ↔ PF - 1 > 0 ↔ PF > 1.
-/
theorem kelly_pos_iff (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : 0 < PF) :
    kelly_fraction p PF > 0 ↔ PF > 1 := by
      unfold kelly_fraction; aesop;

/-
PROBLEM
f* < 1 for all PF > 1 and p ∈ (0,1).

PROVIDED SOLUTION
kelly_fraction p PF = p * (PF - 1) / PF. We need p*(PF-1)/PF < 1, i.e. p*(PF-1) < PF, i.e. p*PF - p < PF, i.e. PF*(p-1) < p, i.e. -PF*(1-p) < p. Since PF > 0 and 1-p > 0, LHS is negative and p > 0, so this holds. Use unfold kelly_fraction, then rw div_lt_one, and linarith/nlinarith.
-/
theorem kelly_lt_one (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : PF > 1) :
    kelly_fraction p PF < 1 := by
      exact div_lt_one ( by linarith ) |>.2 ( by nlinarith )

/-
PROBLEM
f* is strictly increasing in p (for fixed PF > 1).

PROVIDED SOLUTION
kelly_fraction p PF = p * (PF - 1) / PF = p * ((PF-1)/PF). Since PF > 1, (PF-1)/PF > 0, so this is a positive scalar multiple of p, hence strictly increasing.
-/
theorem kelly_strict_mono_p (PF : ℝ) (hPF : PF > 1) :
    StrictMono (fun p => kelly_fraction p PF) := by
      unfold kelly_fraction; exact fun p q hpq => div_lt_div_iff_of_pos_right ( by positivity ) |>.2 <| by nlinarith;

/-
PROBLEM
f* is strictly increasing in PF on (0, ∞) (for fixed p > 0).

PROVIDED SOLUTION
kelly_fraction p PF = p - p/PF. For a, b ∈ Ioi 0 with a < b: kelly_fraction p a = p - p/a < p - p/b = kelly_fraction p b since a < b implies 1/a > 1/b implies p/a > p/b. Unfold kelly_fraction and use field_simp or show it directly.
-/
theorem kelly_strict_mono_PF (p : ℝ) (hp : 0 < p) :
    StrictMonoOn (fun PF => kelly_fraction p PF) (Set.Ioi 0) := by
      unfold StrictMonoOn kelly_fraction;
      intro a ha b hb hab; rw [ div_lt_div_iff₀ ] <;> nlinarith [ mul_pos hp ha.out, mul_pos hp hb.out ] ;

/-- Numerical verification: for p = 27/50, PF = 77/50: f* = 2916/7700 ≈ 0.1894. -/
theorem numerical_verify_kelly :
    kelly_fraction (27/50 : ℝ) (77/50) = 729 / 3850 := by
  unfold kelly_fraction; ring

-- ============================================================================
-- CLAIM 3: Growth Rate at Kelly Fraction
-- ============================================================================

/-- The expected log-growth rate per trade. -/
def growth_rate (p R f : ℝ) : ℝ := p * log (1 + f * R) + (1 - p) * log (1 - f)

/-- G(0) = 0. -/
theorem growth_rate_at_zero (p R : ℝ) : growth_rate p R 0 = 0 := by
  unfold growth_rate; simp [log_one]

/-
PROBLEM
The derivative of G at f = 0 is p·R - (1-p).

PROVIDED SOLUTION
G(f) = p * log(1 + f*R) + (1-p) * log(1-f). Use HasDerivAt.add, HasDerivAt.const_mul, HasDerivAt for log(1 + f*R) at f=0 with chain rule: derivative is R/(1 + 0*R) = R, so p*R. For log(1-f) at f=0: derivative is -1/(1-0) = -1, so (1-p)*(-1) = -(1-p). Total: p*R - (1-p). Use hasDerivAt_log with 1 + f*R composed with the linear map f ↦ 1 + f*R.
-/
theorem growth_rate_deriv_zero (p R : ℝ) (hR : R > 0) :
    HasDerivAt (growth_rate p R) (p * R - (1 - p)) 0 := by
      convert HasDerivAt.add ( HasDerivAt.const_mul p ( HasDerivAt.log ( HasDerivAt.add ( hasDerivAt_const _ _ ) ( HasDerivAt.mul ( hasDerivAt_id' ( 0 : ℝ ) ) ( hasDerivAt_const _ _ ) ) ) _ ) ) ( HasDerivAt.const_mul ( 1 - p ) ( HasDerivAt.log ( HasDerivAt.sub ( hasDerivAt_const _ _ ) ( hasDerivAt_id' ( 0 : ℝ ) ) ) _ ) ) using 1 <;> norm_num [ hR.ne' ] ; ring

/-
PROBLEM
When R = PF·(1-p)/p, the derivative G'(0) = (1-p)·(PF-1).

PROVIDED SOLUTION
p * (PF * (1-p) / p) - (1-p) = PF*(1-p) - (1-p) = (1-p)*(PF-1). Use field_simp and ring.
-/
theorem growth_rate_deriv_zero_eq (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : PF > 1) :
    p * (PF * (1 - p) / p) - (1 - p) = (1 - p) * (PF - 1) := by
      rw [ mul_div_cancel₀ ] <;> linarith

/-
PROBLEM
Key algebraic simplification for G(f*): the arguments of log simplify nicely.
    1 + f*·R = p + (1-p)·PF and 1 - f* = (p + (1-p)·PF)/PF
    where f* = p·(PF-1)/PF and R = PF·(1-p)/p.

PROVIDED SOLUTION
f = p*(PF-1)/PF, R = PF*(1-p)/p. f*R = p*(PF-1)/PF * PF*(1-p)/p = (PF-1)*(1-p). So 1 + f*R = 1 + (PF-1)*(1-p) = 1 + PF - PF*p - 1 + p = p + (1-p)*PF. Use unfold kelly_fraction, field_simp, ring.
-/
theorem growth_rate_kelly_arg1 (p PF : ℝ) (hp : 0 < p) (hPF : 0 < PF) :
    let f := kelly_fraction p PF
    let R := PF * (1 - p) / p
    1 + f * R = p + (1 - p) * PF := by
      unfold kelly_fraction;
      grind

/-
PROVIDED SOLUTION
f = p*(PF-1)/PF. 1 - f = 1 - p*(PF-1)/PF = (PF - p*(PF-1))/PF = (PF - p*PF + p)/PF = (PF*(1-p) + p)/PF = (p + (1-p)*PF)/PF. Use unfold kelly_fraction, field_simp, ring.
-/
theorem growth_rate_kelly_arg2 (p PF : ℝ) (hp : 0 < p) (hPF : 0 < PF) :
    let f := kelly_fraction p PF
    1 - f = (p + (1 - p) * PF) / PF := by
      unfold kelly_fraction; ring;
      linarith [ mul_inv_cancel₀ hPF.ne' ]

/-
PROBLEM
G(f*) = log(p + (1-p)·PF) - (1-p)·log(PF).

PROVIDED SOLUTION
Use growth_rate_kelly_arg1 and growth_rate_kelly_arg2 to rewrite the arguments. Then G(f*) = p * log(p + (1-p)*PF) + (1-p) * log((p + (1-p)*PF)/PF) = p*log(A) + (1-p)*(log(A) - log(PF)) where A = p + (1-p)*PF. This equals (p + 1-p)*log(A) - (1-p)*log(PF) = log(A) - (1-p)*log(PF). Use Real.log_div for the log((p+(1-p)*PF)/PF) = log(p+(1-p)*PF) - log(PF) step. Need p + (1-p)*PF > 0 (obvious since p > 0, PF > 1) and PF ≠ 0.
-/
theorem growth_rate_at_kelly (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : PF > 1) :
    let R := PF * (1 - p) / p
    let f := kelly_fraction p PF
    growth_rate p R f = log (p + (1 - p) * PF) - (1 - p) * log PF := by
      unfold growth_rate kelly_fraction; ring;
      norm_num [ sq, mul_assoc, mul_comm p, hp.ne', ne_of_gt ( zero_lt_one.trans hPF ) ] ; ring;
      rw [ show 1 - p + p * PF⁻¹ = ( PF - PF * p + p ) / PF by ring_nf; nlinarith [ mul_inv_cancel₀ ( by linarith : PF ≠ 0 ) ] ] ; rw [ Real.log_div ( by nlinarith ) ( by linarith ) ] ; ring;

/-
PROBLEM
G(f*) > 0 when PF > 1.
    Proof: By strict concavity of log on (0,∞), for p ∈ (0,1) and 1 ≠ PF:
    log(p·1 + (1-p)·PF) > p·log(1) + (1-p)·log(PF) = (1-p)·log(PF).
    Hence G(f*) = log(p + (1-p)·PF) - (1-p)·log(PF) > 0.

PROVIDED SOLUTION
Use growth_rate_at_kelly to rewrite G(f*) = log(p + (1-p)*PF) - (1-p)*log(PF). Let q = 1-p. We need log(p + q*PF) > q*log(PF). Note p + q*PF = p*1 + q*PF. By strict concavity of log on (0,∞) (strictConcaveOn_log_Ioi), since 0 < p, 0 < q, p + q = 1, and 1 ≠ PF (as PF > 1): log(p*1 + q*PF) > p*log(1) + q*log(PF) = 0 + q*log(PF) = q*log(PF). So G(f*) > 0.
-/
theorem growth_rate_kelly_pos (p PF : ℝ) (hp : 0 < p) (hp1 : p < 1) (hPF : PF > 1) :
    let R := PF * (1 - p) / p
    let f := kelly_fraction p PF
    growth_rate p R f > 0 := by
      unfold kelly_fraction growth_rate;
      -- Simplify the expression inside the logarithms.
      field_simp [mul_comm, mul_assoc, mul_left_comm] at *;
      -- By the properties of logarithms, we can simplify the expression inside the logarithm.
      have h_log_simplify : log (1 + (PF - 1) * (1 - p)) > (1 - p) * log (PF) := by
        -- We'll use that $log$ is strictly concave to show this inequality.
        have h_concave : StrictConcaveOn ℝ (Set.Ioi 0) Real.log := by
          exact strictConcaveOn_log_Ioi;
        have := h_concave.2 ( show 0 < ( 1 : ℝ ) by norm_num ) ( show 0 < PF by linarith );
        specialize this ( by linarith ) ( show 0 < p by linarith ) ( show 0 < 1 - p by linarith ) ( by linarith ) ; norm_num at this ; ring_nf at this ⊢ ; linarith;
      rw [ Real.log_div ( by nlinarith ) ( by linarith ) ] ; ring_nf at * ; nlinarith;

/-
PROBLEM
============================================================================
CLAIM 4: Partial Derivatives of Kelly Fraction
============================================================================

∂f*/∂PF = p/PF²

PROVIDED SOLUTION
kelly_fraction p x = p*(x-1)/x = p - p/x = p - p*x⁻¹. The derivative w.r.t. x at PF is 0 - p*(-PF⁻²) = p/PF². Use HasDerivAt for the composition p - p * x⁻¹. Or unfold kelly_fraction, use HasDerivAt.div_const or HasDerivAt.div.
-/
theorem kelly_deriv_PF (p PF : ℝ) (hPF : PF ≠ 0) :
    HasDerivAt (fun x => kelly_fraction p x) (p / PF ^ 2) PF := by
      convert HasDerivAt.div ( HasDerivAt.const_mul p ( hasDerivAt_id PF |> HasDerivAt.sub <| hasDerivAt_const _ _ ) ) ( hasDerivAt_id PF ) hPF using 1 ; ring;
      norm_num [ sq, mul_assoc, hPF ];
      grind

/-
PROBLEM
∂f*/∂p = (PF-1)/PF

PROVIDED SOLUTION
kelly_fraction x PF = x * (PF-1) / PF = x * ((PF-1)/PF). This is a linear function of x with slope (PF-1)/PF. Use HasDerivAt.const_mul or show HasDerivAt id 1 p and scale.
-/
theorem kelly_deriv_p (p PF : ℝ) :
    HasDerivAt (fun x => kelly_fraction x PF) ((PF - 1) / PF) p := by
      convert HasDerivAt.div_const ( HasDerivAt.mul ( hasDerivAt_id p ) ( hasDerivAt_const _ _ |> HasDerivAt.sub <| hasDerivAt_const _ _ ) ) _ using 1 ; ring!;
      norm_num ; ring

/-- ∂f*/∂PF > 0 when p > 0 and PF > 0. -/
theorem kelly_deriv_PF_pos (p PF : ℝ) (hp : 0 < p) (hPF : 0 < PF) :
    p / PF ^ 2 > 0 := by positivity

/-- ∂f*/∂p > 0 when PF > 1. -/
theorem kelly_deriv_p_pos (PF : ℝ) (hPF : PF > 1) :
    (PF - 1) / PF > 0 := by
  apply div_pos (by linarith) (by linarith)

end
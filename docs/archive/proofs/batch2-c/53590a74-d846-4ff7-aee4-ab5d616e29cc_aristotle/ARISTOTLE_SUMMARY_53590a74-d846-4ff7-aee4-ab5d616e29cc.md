# Summary of changes
All four claims have been formalized and fully proven in Lean 4 in `RequestProject/KellyPF.lean`. The file builds successfully with no `sorry` and only standard axioms (propext, Classical.choice, Quot.sound).

## Summary of what was proved

### Claim 1 — PF Identity
- `pf_identity`: (N·w·W)/(N·(1-w)·L) = w·(W/L)/(1-w)
- `R_from_PF`: If PF = w·R/(1-w) then R = PF·(1-w)/w
- `numerical_verify_R`: Exact rational verification that 77/50 · 23/50 / (27/50) = 1771/1350

### Claim 2 — Kelly Fraction from PF
- `kelly_fraction` defined as `p * (PF - 1) / PF`
- `kelly_derivation`: Algebraic derivation from the standard Kelly formula (p·b - q)/b with b = PF·q/p (requires p ∈ (0,1))
- `kelly_pos_iff`: f* > 0 ↔ PF > 1
- `kelly_lt_one`: f* < 1 for PF > 1, p ∈ (0,1)
- `kelly_strict_mono_p`: f* is strictly increasing in p (for PF > 1)
- `kelly_strict_mono_PF`: f* is strictly increasing in PF on (0,∞) (for p > 0)
- `numerical_verify_kelly`: Exact computation for p=27/50, PF=77/50

### Claim 3 — Growth Rate at Kelly
- `growth_rate` defined as `p · log(1 + f·R) + (1-p) · log(1-f)`
- `growth_rate_at_zero`: G(0) = 0
- `growth_rate_deriv_zero`: G'(0) = p·R - (1-p) (via HasDerivAt)
- `growth_rate_deriv_zero_eq`: When R = PF·(1-p)/p, G'(0) = (1-p)·(PF-1)
- `growth_rate_kelly_arg1`, `growth_rate_kelly_arg2`: Key algebraic simplifications
- `growth_rate_at_kelly`: G(f*) = log(p + (1-p)·PF) - (1-p)·log(PF)
- `growth_rate_kelly_pos`: **G(f*) > 0 when PF > 1**, proved using strict concavity of log (`strictConcaveOn_log_Ioi`)

### Claim 4 — Partial Derivatives
- `kelly_deriv_PF`: ∂f*/∂PF = p/PF² (via HasDerivAt)
- `kelly_deriv_p`: ∂f*/∂p = (PF-1)/PF (via HasDerivAt)
- `kelly_deriv_PF_pos`: p/PF² > 0 when p > 0, PF > 0
- `kelly_deriv_p_pos`: (PF-1)/PF > 0 when PF > 1

### Minor corrections
Two statements needed domain restrictions to be true:
1. `kelly_derivation` requires p < 1 (not just p > 0), because when p = 1 the odds ratio b = 0 and division is undefined.
2. `kelly_strict_mono_PF` is `StrictMonoOn` on `(0,∞)` rather than `StrictMono` on all of ℝ, because kelly_fraction is undefined (returns 0 by convention) at PF = 0.
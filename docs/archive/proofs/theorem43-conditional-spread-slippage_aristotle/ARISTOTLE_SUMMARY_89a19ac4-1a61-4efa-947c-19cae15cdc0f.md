# Summary of changes
Successfully formalized Theorem 43 (Conditional Spread and Slippage Under Activity Regimes) in `RequestProject/ConditionalSpreadSlippage.lean`. All 14 theorems are fully proved with no sorries and only standard axioms.

### What was formalized

The file covers all 5 claims from `theorem43-conditional-spread-slippage.txt`:

**Claim 1 — Power-Law Spread Ratio:**
- `spread_ratio_eq`: The constant c cancels in the Almgren-Chriss ratio: (c·E_high)/(c·E_all) = E_high/E_all
- `spread_ratio_gt_one`: R > 1 when conditional expectation exceeds unconditional

**Claim 2 — Square-Root Law (γ = 0.5):**
- `sqrt_law`: The spread ratio under σ = k·σ₀ simplifies to √k
- `sqrt_law_k2`: √2 ∈ (1.414, 1.415), verifying the ≈1.414 approximation
- `sqrt_law_k4`: √4 = 2 exactly

**Claim 3 — Linear Bound:**
- `rpow_gt_one_of_gt_one`: k^γ > 1 when k > 1 and γ > 0
- `rpow_le_self_of_ge_one`: k^γ ≤ k when k ≥ 1 and 0 < γ ≤ 1
- `linear_bound`: Combined bound 1 < k^γ ≤ k for k > 1, γ ∈ (0,1]
- `rpow_mono_gamma`: Monotonicity of k^γ in γ for k > 1

**Claim 4 — R = 1 (No Conditional Widening):**
- `rpow_zero_eq_one`: σ^0 = 1 for any σ > 0 (constant spread when γ = 0)
- `spread_ratio_one_of_gamma_zero`: R = 1 when spread is constant

**Claim 5 — Slippage Model Bias:**
- `bias_formula`: bias = (R−1)·S/2
- `relative_bias_formula`: relative bias = (R−1)/R
- `no_bias_when_R_eq_one`: bias = 0 when R = 1 (unbiased model)
- `positive_bias_when_R_gt_one`: bias > 0 when R > 1
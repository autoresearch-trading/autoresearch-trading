# Summary of changes
## Formalization Complete: Hawkes Self-Excitation Volatility Clustering

### File: `RequestProject/HawkesVolatility.lean`

### Critical Mathematical Analysis

The original "theorem" contains three claims that mix rigorous mathematics with informal heuristics and circular reasoning. After careful analysis:

**Claim 1** incorrectly labels α/β > 0.5 as "supercritical" — in Hawkes process theory, the critical threshold is α/β = 1, not 0.5. The formula for conditional expected variance also conflates intensity prediction with realized variance prediction without stating the required modeling assumptions.

**Claim 2** is circular — it assumes "the Hawkes clustering signal is not yet priced into implied vol," which is an empirical market efficiency claim, not a provable mathematical theorem.

**Claim 3** introduces "price_level_absorption" which is not a standard mathematical object, making the correlation bound claim impossible to formally state.

### What Was Formalized and Proved (8 theorems, all sorry-free)

I extracted and proved the deterministic analytic content underlying the claims:

1. **`predicted_variance_decay`**: The predicted variance formula σ²_base · (1 + rᵏ · (λ₀/μ - 1)) converges to σ²_base as k → ∞ when 0 < r < 1 (Claim 1 core).

2. **`predicted_variance_predictable`**: When λ(t) ≠ μ and σ²_base ≠ 0, the predicted variance differs from base variance for all finite k (Claim 1 predictability).

3. **`correlation_bound_range`**: The bound function 1 - exp(-r/(1-r)) lies strictly in (0,1) for r ∈ (0,1).

4. **`correlation_bound_strictly_increasing`**: The bound is strictly increasing on (0,1).

5. **`correlation_bound_tendsto_one`**: The bound approaches 1 as r → 1⁻.

6. **`correlation_bound_at_08`**: At r = 0.8, the bound ≥ 0.98 (verifying the numerical claim).

7. **`hawkes_branching_ratio_integral`**: ∫₀^∞ α·exp(-βt) dt = α/β, proving the branching ratio equals the integral of the excitation kernel.

8. **`hawkes_unconditional_intensity`**: The unconditional intensity μ/(1 - α/β) > μ when α/β < 1, confirming self-excitation amplifies baseline intensity.

All proofs use only standard axioms (propext, Classical.choice, Quot.sound). The file includes detailed documentation explaining which parts of the original claims are provable mathematics versus informal heuristics.
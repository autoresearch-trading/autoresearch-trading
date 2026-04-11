# Summary of changes
# Universal Approximation Bound — Formalization Results

The file `RequestProject/UniversalApprox.lean` contains a complete, sorry-free Lean 4 formalization analyzing all three claims. Here is what was proven:

## Claim 1: Hybrid Dominance

**Weak dominance (≤) is TRUE** — proven as `hybrid_weak_dominance` and `hybrid_weak_dominance_general`:
- If the hybrid class F_hybrid contains both F_flat and F_tcn (i.e., F_flat ⊆ F_hybrid and F_tcn ⊆ F_hybrid), then `inf_{F_hybrid} R ≤ min(inf_{F_flat} R, inf_{F_tcn} R)`. This follows from monotonicity of infimum over larger sets (`csInf_le_csInf`).

**Strict dominance (<) is FALSE** — disproven with counterexamples `strict_dominance_false` and `strict_dominance_false_sets`:
- Witness: R_flat = R_tcn = R_hybrid = 0 (or equivalently, S₁ = S₂ = S = {0}). When one component already achieves Bayes risk (e.g., the optimal classifier is a simple function of cross-sectional features alone), the hybrid cannot strictly improve. The claim "hybrid > flat > temporal alone" does not hold as a universal theorem.

## Claim 2: Minimum Window Size W_min = 75

**Verified numerically** via `wmin_bound_upper` and `wmin_bound_lower`:
- `95^75 × 46 ≤ 100^75` (i.e., 0.95^75 ≤ 1/46) — W=75 is large enough
- `100^74 < 95^74 × 46` (i.e., 0.95^74 > 1/46) — W=74 is not enough

This confirms ⌈log(1/46)/log(0.95)⌉ = 75. However, the interpretation "W=50 is too small for TCN to add value" is a heuristic, not a theorem — it depends on the specific data distribution, not just AR(1) structure.

## Claim 3: Majority Voting / Optimal Ensemble Size

**The claim is ill-posed** due to the undefined "correction_3class" term. We proved:

- `marginal_improvement_sign`: The sign of marginal improvement from adding a model depends on whether α > 1/2.
- `ensemble_M5_binary_hurts`: For **binary** classification with α = 0.45 < 0.5, majority voting with M=5 **hurts** — ensemble accuracy ≈ 0.407 < 0.45 (exact value: 650997/1600000, proven in `ensemble_M5_value`).
- `three_class_threshold`: For 3-class problems, 0.45 > 1/3, so the regime is qualitatively different from binary — ensembling *may* help, but requires multinomial analysis, not the binomial formula given.

The claim that "Δα ≈ 0.003 for M=5, α=0.45" uses a binary majority voting formula for a 3-class problem, which is invalid without the undefined correction term.

## Key Takeaway

The three claims as stated are not provable theorems: Claim 1 (strict) is false, Claim 2's interpretation is heuristic, and Claim 3 is ill-posed. The weaker, corrected versions are formalized and machine-verified in the file.
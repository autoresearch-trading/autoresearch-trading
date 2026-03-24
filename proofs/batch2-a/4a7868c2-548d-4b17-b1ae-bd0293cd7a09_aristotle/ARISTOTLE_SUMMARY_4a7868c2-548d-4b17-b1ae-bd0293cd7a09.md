# Summary of changes
Successfully formalized and proved all four claims of the Dual Regime Gate Properties in `RequestProject/DualRegimeGate.lean`. All 8 theorems compile without `sorry` and use only standard axioms.

**Claim 1** — `combined_pass_rate_in_unit_interval`: For φ₁, φ₂ ∈ (0,1), the combined pass rate φ₁·φ₂ ∈ (0,1).

**Claim 2** — Two theorems:
- `dual_gate_beats_single_iff`: Under the Sortino model (Sortino = k·√φ·S₀), the dual-gated Sortino beats the single gate F₁ alone if and only if k₂·√φ₂ > 1. The proof factors √(φ₁·φ₂) = √φ₁·√φ₂ and cancels common positive factors.
- `second_gate_threshold_equiv`: k₂·√φ₂ > 1 ⟺ k₂ > 1/√φ₂ (the equivalent threshold form from Theorem 10).

**Claim 3** — Three theorems providing numerical verification:
- `threshold_half`: 1/√(1/2) = √2 (exact identity for φ₂ = 0.5).
- `threshold_07_bound`: 1/√(7/10) < 1.196 (upper bound for φ₂ = 0.7).
- `threshold_07_lower`: 1/√(7/10) > 1.195 (lower bound, confirming ≈ 1.195).

**Claim 4** — Two theorems:
- `correlated_gates_higher_pass_rate`: If P(F₁ ∧ F₂) = P(F₁)·P(F₂) + Cov and Cov > 0, then φ_combined > φ₁·φ₂ (correlated gates filter less data).
- `correlated_gates_better_sqrt_factor`: Higher combined pass rate yields a larger √φ factor, which is beneficial for Sortino.
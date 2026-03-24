# Summary of changes
## Regime-Gated Kelly Barrier Optimization — Formalized and Proved in Lean 4

All five claims have been formalized and proved (with one correction). The project builds cleanly with no `sorry` in active code, and all theorems depend only on standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

### Files

- **`RequestProject/Claim1.lean`** — Fee-multiplier/accuracy trade-off is convex
- **`RequestProject/Claim2.lean`** — Optimal fee_mult given accuracy constraint  
- **`RequestProject/Claim3.lean`** — Regime gate improves effective accuracy
- **`RequestProject/Claim4.lean`** — Two-pass Kelly iteration converges
- **`RequestProject/Claim5.lean`** — Combined strategy optimality

### Results by Claim

**Claim 1** (9 theorems, all proved): α_min(f) = 1/2 + 1/(2f) is strictly decreasing, convex on (0,∞), equals 1 at f=1, tends to 1/2 as f→∞, with spot checks at f=3/2, 4, 8 verified. The inverse f_min = 1/(2α-1) is characterized and shown minimal.

**Claim 2** (3 theorems proved, 1 corrected):
- **(a) CORRECTED**: The original claim stated f_opt > f_min when c < (2α-1)/(2α). This is **false** — counterexample: α=0.6, c=0.1 gives f_opt=1.8 < 5=f_min despite satisfying the original condition. The correct condition is **c < (2α-1)² / (1 + (2α-1)²)**, which is proved.
- **(b)** f_opt is strictly increasing in α ✓
- **(c)** f_opt is strictly decreasing in c ✓

**Claim 3** (3 theorems proved): The core monotone-upper-tail-average inequality is proved (for a monotone function, the average on [t,b] ≥ average on [a,b]). SNR(r) = SNR_base/√(1-r) is shown monotone on [0,1), and Φ∘SNR is monotone when Φ is monotone.

**Claim 4** (3 theorems proved): Using Mathlib's `ContractingWith` API:
- Unique fixed point exists (Banach fixed point theorem)
- Iteration T^n(x₀) converges to the fixed point
- Geometric convergence rate: dist(T^n x₀, f*) ≤ dist(x₀, Tx₀) · k^n / (1-k)

**Claim 5** (5 theorems proved, formulation corrected): 
- **Key correction**: The original claim used α_min(f) = 1/2 + 1/(2f) as the threshold for log-growth positivity. This is the threshold for positive **expected value** (linear payoff), NOT positive **geometric growth** (log payoff). The log-growth threshold depends on both f and c.
- Reformulated correctly: g_inner(α) = α·log(1+fc-c) + (1-α)·log(1-fc-c) is strictly monotone and continuous in α, negative at α=0, positive at α=1, so there exists a unique threshold α* where g=0.
- The combined gating theorem is proved: if g_inner(α_all) ≤ 0 and g_inner(α_regime) > 0, then the ungated strategy has G ≤ 0 and the gated strategy has G > 0.
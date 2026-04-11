# Summary of changes
Formalized the core mathematical claims from `theorem47-optimal-window-signal-decay.txt` in `RequestProject/Theorem47.lean`. All 8 theorems are fully proved with no `sorry` and only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

### What was formalized

**Claim 1 — Geometric series for signal content with exponential decay:**
- `geom_sum_exp_decay`: Proves Σ_{k=0}^{W-1} exp(-2k/τ) = (1 - exp(-2W/τ)) / (1 - exp(-2/τ)) for τ > 0
- `signal_content_exp_decay`: Proves Signal(W) = β₀² · (1 - exp(-2W/τ)) / (1 - exp(-2/τ))
- `signal_limit`: Proves the partial sums converge to 1/(1 - exp(-2/τ)) as W → ∞

**Claim 3 — Optimal window minimizing noise:**
- `noise_func`: Defines g(W) = W·a + b/W
- `noise_func_minimizer`: Proves g(√(b/a)) ≤ g(W) for all W > 0 (AM-GM inequality)
- `noise_func_min_value`: Proves the minimum value equals 2√(ab)
- `optimal_window_equal_noise`: Proves W* = 1 when σ_noise = σ_x

**Auxiliary lemmas:**
- `exp_neg_two_div_pos`: exp(-2/τ) > 0 for τ > 0
- `exp_neg_two_div_lt_one`: exp(-2/τ) < 1 for τ > 0

Claims 4–6 from the original text are statistical/empirical in nature (chi-squared quantiles, bias-variance tradeoffs, empirical recommendations) and are not amenable to clean formalization without extensive statistical infrastructure beyond what Mathlib provides.
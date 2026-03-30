# Summary of changes
## The "Entropy Collapse" Theorem is False — Formalized Counterexamples

The purported theorem contains three claims, all of which are mathematically false. I formalized and machine-verified counterexamples in `RequestProject/EntropyCollapse.lean`. The file builds cleanly with no `sorry` and uses only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

### Claim 1 (UMP detector) — **FALSE**

The claim that (S_t, H_t) is a "uniformly most powerful" detector is false because **UMP tests do not exist for two-sided composite alternatives**. This is a standard result in mathematical statistics. The formalized proof (`no_ump_two_sided`) shows that no subset of ℝ can be simultaneously:
- **Upward-closed** (needed for optimality against μ > μ_min: reject when the statistic is large)
- **Downward-closed** (needed for optimality against μ < -μ_min: reject when the statistic is small)
- **Non-trivial** (neither ∅ nor ℝ, as required for a valid test)

The proof is elementary: given any element x in the rejection region R and any y outside R, the linear order gives x ≤ y or y ≤ x; either case forces y ∈ R by the closure properties, contradicting y ∉ R.

### Claim 2 (Optimal thresholds) — **FALSE**

The entropy threshold formula τ_S = S_max - μ²w/(2σ²) can produce **negative values** (`claim2_counterexample`). With μ = 2, σ = 1, S_max = 1, w = 2, we get τ_S = 1 - 4 = -3 < 0. Since Shannon entropy is always non-negative, a negative threshold is meaningless — it would trigger on every observation regardless of market conditions. The theorem `claim2_threshold_not_always_nonneg` proves the formula fails to be universally non-negative.

### Claim 3 (Sortino bound) — **VACUOUS**

The claimed Sortino ratio lower bound grows as √n_entries (`claim3_bound_unbounded`). With μ = σ = k = 1, the bound becomes (1/2)·√n, which exceeds any value M for large enough n. Since n_entries depends on threshold choice (lower thresholds → more entries → more false positives), but the bound includes **no false-positive penalty**, it can be made arbitrarily large even when actual performance is poor. A meaningful performance guarantee must account for the quality of entry signals, not just their quantity.
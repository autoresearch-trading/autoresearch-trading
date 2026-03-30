# Summary of changes
All four claims have been formalized and proved in Lean 4 in `RequestProject/NormSuffStat.lean`. The file builds cleanly with no `sorry` and only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

## Structure

**Definition** – `SufficientViaFactorization`: A statistic T is sufficient for a density family f if f(x,θ) = h(x) · k(T(x), θ) (Fisher–Neyman factorization criterion).

**Claim 1** – `sufficient_of_bijective_comp`: If T is sufficient and g is bijective, then g ∘ T is also sufficient. Proved by composing the factorization with g's inverse.

**Claim 2** – `zscore_bijective`: Z-score normalization g(t) = (t − μ)/σ is bijective when σ > 0, with inverse z ↦ z·σ + μ.

**Claim 3** – `zscore_preserves_sufficiency`: Direct corollary of Claims 1 and 2 — z-score normalized features preserve sufficiency.

**Claim 4a** – `iqr_bijective` and `iqr_preserves_sufficiency`: IQR-based scaling g(t) = (t − median)/IQR is bijective (and preserves sufficiency) when IQR > 0.

**Claim 4b** – `iqr_not_injective_of_zero`: When IQR = 0, the transformation maps everything to 0 (since x/0 = 0 in Lean), so it is **not** injective and sufficiency may be lost.
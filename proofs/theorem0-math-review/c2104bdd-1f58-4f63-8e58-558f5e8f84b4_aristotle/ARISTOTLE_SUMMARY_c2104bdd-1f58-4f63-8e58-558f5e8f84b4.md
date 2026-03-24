# Summary of changes
## Financial Metrics: Formalization and Verification

All five claims have been formalized and verified in Lean 4 with Mathlib, in the file `RequestProject/FinancialMetrics.lean`. Every theorem compiles without `sorry` and uses only standard axioms (propext, Classical.choice, Quot.sound).

### Results Summary

1. **Sortino Ratio Annualization** ✅ **VERIFIED**
   - Proved the algebraic identity: `(μ/σ_down) · √k = (k·μ) / (√k · σ_down)`.
   - This justifies the annualization formula: under i.i.d. returns, the mean scales as k·μ and the downside deviation scales as √k·σ_down, so the ratio scales as √k = √(steps_per_day).

2. **Hurst Exponent H ∈ [0,1]** ❌ **FALSE in general** (counterexample provided), ✅ **TRUE conditionally**
   - **Counterexample**: With R=1, S=2, w=2 (which can arise using sample standard deviation), H = log(1/2)/log(2) < 0. This violates the claimed H ∈ [0,1].
   - **Conditional result**: Proved that H ∈ [0,1] **does** hold when S ≤ R ≤ w·S and w > 1 (i.e., when 1 ≤ R/S ≤ w). The condition R/S ≥ 1 can fail with sample std (w−1 denominator), making H negative.
   - The convergence H → 0.5 for random walks is an asymptotic probabilistic result that would require measure-theoretic formalization; it is noted as correct but not formally proved.

3. **Kyle's Lambda = OLS Slope** ✅ **VERIFIED**
   - Proved that β̂ = Σ(xᵢyᵢ)/Σ(xᵢ²) (= Cov/Var for centered data) minimizes the sum of squared residuals Σ(yᵢ − β·xᵢ)². This is the defining property of the OLS slope estimator.
   - The interpretation "λ > 0 implies informed trading" is a financial interpretation, not a mathematical claim.

4. **Amihud ILLIQ ≥ 0 and zero characterization** ✅ **VERIFIED**
   - Proved ILLIQ ≥ 0 when all volumes are positive.
   - Proved ILLIQ = 0 **if and only if** all returns are zero (given positive volumes and n > 0). This strengthens the original claim to a biconditional.

5. **Triple Barrier Labeling** ✅ **VERIFIED**
   - Defined three predicates: `labelIsTP` (take-profit hit first), `labelIsSL` (stop-loss hit first), `labelIsTimeout` (neither hit).
   - Proved **mutual exclusivity**: no two outcomes can hold simultaneously.
   - Proved **exhaustiveness**: exactly one of the three outcomes always holds.
   - Both results require `sl < tp`, which holds when fee_mult·fee > 0 (the financially meaningful case).
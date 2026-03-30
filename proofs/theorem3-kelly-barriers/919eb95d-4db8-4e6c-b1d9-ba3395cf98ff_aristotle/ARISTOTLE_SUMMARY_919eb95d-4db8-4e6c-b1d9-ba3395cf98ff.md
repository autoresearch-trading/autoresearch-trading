# Summary of changes
## Summary

The formalization is in `RequestProject/KellyCriterion.lean`. Here is what the analysis found:

### Claim 1 (Optimal fee multiplier formula): **INCORRECT as stated**

The claimed formula `f* = (p_win - p_loss)/(p_win + p_loss) · (1/c) - 1/(p_win + p_loss)` is **wrong**. 

By differentiating the growth rate G(f) = p_win·log(1 + fc - c) + p_loss·log(1 - fc - c) and setting G'(f) = 0, the correct formula is:

```
f* = (p_win - p_loss) · (1 - c) / ((p_win + p_loss) · c)
```

The error is in the second term: the correct derivation gives `-(p_win - p_loss)/(p_win + p_loss)` where the claim has `-1/(p_win + p_loss)`. These differ except in the degenerate case p_win - p_loss = 1.

**What is proved in Lean:**
- `optimalFeeMult_satisfies_foc`: The corrected f* satisfies the first-order optimality condition p_win·(1 - fc - c) = p_loss·(1 + fc - c).
- `optimalFeeMult_unique`: Any f satisfying the FOC must equal the corrected formula (uniqueness).
- `claimed_formula_wrong`: A concrete counterexample (p_win=3/5, p_loss=3/10, c=1/10) showing the original formula does NOT satisfy the FOC.

### Claim 2 (Sortino ratio bound): **Not formalized**

This claim involves informal heuristic arguments about distributional properties (Sortino ratios, diminishing returns of accuracy) that are not rigorously derivable from the stated mathematical setup. It would require a full probabilistic model with measure-theoretic foundations that go beyond what the claim specifies.

### Claim 3 (Minimum accuracy): **CORRECT and proved**

The formula α_min = 1/2 + 1/(2·f*) is correct, and for fee_mult = 8, α_min = 9/16 = 56.25%.

**What is proved in Lean:**
- `minAccuracy_eight`: minAccuracy 8 = 9/16
- `minAccuracy_gt_half`: For f* > 0, the minimum accuracy exceeds 1/2 (fees always require better than coin-flip accuracy)
- `minAccuracy_anti`: The minimum accuracy decreases as the fee multiplier increases (wider barriers are more forgiving)

All 7 theorems compile without `sorry` and use only standard axioms (propext, Classical.choice, Quot.sound).
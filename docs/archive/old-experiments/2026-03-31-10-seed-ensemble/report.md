# Experiment Report: 10-seed ensemble

## Results
| Run | Name | Config | Sortino | Passing | Score |
|-----|------|--------|---------|---------|-------|
| 1 | control_5seeds | 5 seeds × 60s, budget=300 | 0.353 | 9/23 | 0.368 |
| 2 | 10seeds_300s | 10 seeds × 30s, budget=300 | 0.205 | 3/23 | 0.175 |
| 3 | 10seeds_600s | 10 seeds × 60s, budget=600 | 0.205 | 3/23 | 0.175 |

## Eval Result

| Metric | Control | Run 3 | Criterion | Status |
|--------|---------|-------|-----------|--------|
| Sortino | 0.353 | 0.205 | >= 0.353 | FAIL |
| Passing | 9/23 | 3/23 | >= 9 | FAIL |
| Alpha | 0.477 | 0.470 | > 0.5 | FAIL |

**Verdict:** DISCARD
**Reason:** Alpha < 0.5 in all runs — ensemble never forms, seed count is irrelevant.

## Analysis

Run 2 and Run 3 produced identical results despite different budgets (300s vs 600s). This revealed that training is epoch-based (25 fixed epochs), not time-based — the budget cap never triggers. Each seed trains for exactly 25 epochs regardless of wall-clock allocation.

The deeper finding: **alpha < 0.5 in ALL runs** (0.477 for 5 seeds, 0.470 for 10 seeds). The ensemble validity check (mean training accuracy > 0.5) consistently fails, forcing single-model fallback. This means:

1. The ensemble has never actually been used in recent runs
2. The reported Sortino=0.353 baseline comes from a single model, not an ensemble
3. Adding more seeds just changes which single model gets selected

The alpha threshold is a consequence of 3-class classification with ~63% flat labels — random accuracy on a balanced problem would be 0.33, but the class imbalance means a model predicting mostly flat gets ~0.63 accuracy on flat but poor accuracy on long/short, dragging overall accuracy below 0.5.

## Conclusion

Seed count is not a lever. The binding constraint is that the ensemble validity check almost always fails. The 5-seed ensemble is effectively a 1-seed model with 5x compute overhead for seed selection.

## Recommended Config
No change — keep FINAL_SEEDS=5, FINAL_BUDGET=300.

## Next Hypotheses
1. **Dropout** — might improve generalization, which could push training accuracy (and alpha) above 0.5, enabling the ensemble to actually form
2. **Investigate alpha threshold** — is 0.5 the right threshold? The model may be useful below 0.5 for imbalanced classification
3. **Residual connections** — better gradient flow could improve training accuracy

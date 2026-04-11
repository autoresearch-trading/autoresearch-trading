# Experiment Report: Dropout

## Results
| Run | Name | Dropout | Sortino | Passing | Alpha | Score |
|-----|------|---------|---------|---------|-------|-------|
| 1 | control | 0.0 | 0.353 | 9/23 | 0.477 | 0.368 |
| 2 | dropout_0.1 | 0.1 | 0.294 | 9/23 | 0.491 | 0.333 |
| 3 | dropout_0.2 | 0.2 | 0.165 | 7/23 | 0.490 | 0.221 |

## Eval Result

| Metric | Control | Best (0.1) | Criterion | Status |
|--------|---------|------------|-----------|--------|
| Sortino | 0.353 | 0.294 | >= 0.353 | FAIL |
| Passing | 9/23 | 9/23 | >= 9 | PASS |
| Alpha | 0.477 | 0.491 | > 0.5 | FAIL |

**Verdict:** DISCARD
**Reason:** Dropout hurts Sortino monotonically (0.353 → 0.294 → 0.165) while barely moving alpha (0.477 → 0.491 → 0.490).

## Analysis

Dropout did push alpha slightly upward (0.477 → 0.491 at dropout=0.1) but not enough to cross the 0.5 threshold. Meanwhile, the regularization hurt prediction quality — Sortino dropped significantly.

This confirms CLAUDE.md Key Discovery #5: "Smaller network generalizes better — the 676→64 bottleneck is a feature." The model is already implicitly regularized by its extreme bottleneck (676 inputs → 64 hidden). Adding explicit dropout on top of this architectural regularization is redundant and harmful.

The alpha is stuck around 0.47-0.49 regardless of regularization. This is a structural property of the 3-class imbalanced problem (63% flat), not a regularization issue.

## Conclusion

Dropout=0.0 confirmed optimal. The model's small size (52K params, 64-dim hidden) is its regularizer. Adding dropout hurts performance without enabling the ensemble.

## Recommended Config
No change — keep dropout=0.0.

## Next Hypotheses
1. **Residual connections** — different mechanism than regularization. Skip connections improve gradient flow, not generalization per se.
2. **GCE loss** — different loss landscape might help alpha cross 0.5.
3. **Lower alpha threshold** — the 0.5 threshold may be wrong for 3-class imbalanced classification. Relaxing it (e.g., 0.45) would enable ensemble.

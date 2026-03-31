# Experiment Report: Alpha Threshold

## Results
| Run | Name | Mode | Sortino | Passing | Trades | WR | PF | Score |
|-----|------|------|---------|---------|--------|------|------|-------|
| 1 | control | single best (alpha<0.5) | 0.353 | 9/23 | 1270 | 0.55 | 1.71 | 0.368 |
| 2 | force_ensemble | always ensemble | 0.320 | 6/23 | 1456 | 0.60 | 1.83 | 0.296 |

## Eval Result

| Metric | Control | Ensemble | Criterion | Status |
|--------|---------|----------|-----------|--------|
| Sortino | 0.353 | 0.320 | >= 0.353 | FAIL |
| Passing | 9/23 | 6/23 | >= 9 | FAIL |

**Verdict:** DISCARD
**Reason:** Theorem 10 is correct — the alpha threshold protects us. Ensembling sub-0.5 accuracy models dilutes the best model's signal.

## Analysis

The forced ensemble was more aggressive (1456 trades vs 1270) with higher win rate (0.60 vs 0.55) and profit factor (1.83 vs 1.71), but lower Sortino (0.320 vs 0.353) and fewer passing symbols (6 vs 9).

The ensemble averages out per-seed noise, but at alpha=0.48, it also averages out the signal — the best seed has learned something the others haven't, and diluting its logits with weaker models hurts more than the variance reduction helps.

## Conclusion

The alpha=0.5 threshold is correct. The model's Sortino=0.353 baseline genuinely comes from a single seed, not an ensemble. This has two implications:

1. **The reported Sortino is seed-dependent** — different runs will pick different "best seeds" with different Sortino values (explaining the variance we see)
2. **Improving the ensemble requires pushing alpha well above 0.5** — incremental improvements (0.48→0.49) won't help, we need a structural change

## Recommended Config
No change — keep alpha threshold at 0.5.

## Next Hypotheses
1. **Residual connections** — might improve training accuracy enough to push alpha above 0.5
2. **GCE loss** — different loss landscape could enable higher directional accuracy

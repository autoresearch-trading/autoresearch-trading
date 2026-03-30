# Experiment: UACE loss with proper hyperparameter tuning

## Hypothesis
The prior UACE test (Sortino=0.141) was invalid — we used lr=1e-3 tuned for focal loss. UACE produces entropy-weighted gradients with different magnitudes, so the optimal lr could be 10x higher or lower. Per the AlgoPerf benchmark protocol, a fair comparison requires re-tuning at least lr and loss-specific params.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: lr sweep with UACE
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | use_uace=True, lr=3e-4    | Lower lr (UACE may need smaller steps) |
| 2   | use_uace=True, lr=1e-3    | Same as baseline (re-confirm prior result) |
| 3   | use_uace=True, lr=3e-3    | Higher lr |
| 4   | use_uace=True, lr=1e-2    | Much higher lr |

### Phase 2: wd sweep with Phase 1 winner
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 5   | best_lr + wd=1e-4 | Add regularization |

## Decision Logic
- Phase 1 winner = highest score across lr values
- If best UACE score > 0.368 (focal baseline): UACE wins, proceed to Phase 2
- If best UACE score < 0.368 but > 0.33: marginal, try Phase 2 anyway
- If best UACE < 0.33 across all lr: UACE genuinely worse, stop

## Budget
4-5 runs × ~10 min = ~40-50 min

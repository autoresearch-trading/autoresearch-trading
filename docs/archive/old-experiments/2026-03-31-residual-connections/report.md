# Experiment Report: Residual Connections

## Results
| Run | Name | Residual | Sortino | Passing | Alpha | Score |
|-----|------|----------|---------|---------|-------|-------|
| 1 | control | False | 0.353 | 9/23 | 0.477 | 0.368 |
| 2 | residual | True | 0.167 | 3/23 | 0.469 | 0.152 |

## Eval Result

| Metric | Control | Residual | Criterion | Status |
|--------|---------|----------|-----------|--------|
| Sortino | 0.353 | 0.167 | >= 0.353 | FAIL |
| Passing | 9/23 | 3/23 | >= 9 | FAIL |

**Verdict:** DISCARD
**Reason:** Skip connections degraded both Sortino (0.353→0.167) and alpha (0.477→0.469). The plain MLP is better.

## Analysis

Despite Gorishniy et al. (2021) finding ResNet-MLP beats plain MLP on tabular benchmarks, skip connections hurt our model. The likely explanation: with only 2 residual blocks (layers 2-3), the skip connections create an identity shortcut that lets the model avoid learning useful transformations. The 64-dim hidden layers are already very constrained — adding a bypass path may cause the model to underutilize these layers.

This is consistent with the broader pattern: the model is small enough (52K params) that additional inductive biases (dropout, skip connections) interfere with the specific representations it needs to learn. The plain MLP's forced sequential processing through the bottleneck is a feature.

## Conclusion

Plain MLP confirmed optimal. Residual connections don't help at this model scale.

## Recommended Config
No change — keep residual=False.

## Next Hypotheses
1. **GCE loss (q=0.7)** — last remaining high-probability test from the untested list. Different noise-robustness mechanism than UACE. Needs lr re-sweep for fair comparison.

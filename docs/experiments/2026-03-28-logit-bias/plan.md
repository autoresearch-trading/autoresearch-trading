# Experiment: logit bias for noise-robust training

## Hypothesis
Triple barrier labels are inherently noisy — timeout labels (flat/0) carry little signal, and barrier hits near the threshold are ambiguous. The logit bias technique (arxiv 2306.05497) adds a small constant epsilon to the correct-class logit before computing loss, making the model more robust to label noise. This is a one-line change that has shown SOTA results on large-scale noisy datasets.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: logit bias sweep
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | logit_bias=0.5            | Moderate bias |
| 2   | logit_bias=1.0            | Strong bias |

Control (no bias): Sortino=0.353, 9/23, score=0.368.

## Decision Logic
- Winner = highest score
- Within 0.02: keep no bias (simpler)

## Budget
2 runs × ~10 min = ~20 min

## Implementation
In focal_loss, before computing cross-entropy:
```python
# Logit bias: add epsilon to correct-class logit (noise robustness)
if logit_bias > 0:
    logits = logits.clone()
    logits.scatter_add_(1, targets.unsqueeze(1), torch.full_like(targets.unsqueeze(1), logit_bias, dtype=logits.dtype))
```

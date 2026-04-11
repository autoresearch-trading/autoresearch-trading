# Experiment: UACE (Uncertainty-Aware Cross Entropy) loss

## Hypothesis
The focal loss already down-weights easy samples (high pt), but doesn't distinguish between "confident and correct" vs "uncertain because noisy label." UACE uses the model's own prediction entropy to identify uncertain samples and reduce their gradient contribution. Timeout labels (flat/0) are inherently noisier — UACE should automatically learn to down-weight them without explicit curriculum engineering.

Formula: `L = CE(logits, target) * (1 - H(softmax(logits)) / log(C))`
where H is entropy and C=3 classes. Confident predictions (low entropy) get full weight; uncertain predictions (high entropy) get reduced weight.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: UACE vs focal loss
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | loss=uace                 | Replace focal loss with UACE |

Control: focal loss, Sortino=0.353, 9/23.

## Budget
1 run × ~10 min

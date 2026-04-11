# Experiment: curriculum learning by label confidence

## Hypothesis
Triple barrier labels have a natural confidence hierarchy: TP/SL barrier hits (labels 1,2) are high-confidence directional signals, while timeouts (label 0) are ambiguous — the price didn't move enough in either direction. Training on noisy timeout labels from the start may confuse the model. Curriculum learning trains first on confident labels, gradually introducing uncertain ones.

Implementation: for epochs 0-9, train only on directional labels (1,2). For epochs 10-24, train on all labels (0,1,2). This lets the model learn "what a signal looks like" before learning "what absence of signal looks like."

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: curriculum vs no curriculum
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | curriculum=True (10 warm-up epochs) | Directional-first training |

Control: Sortino=0.353, 9/23, score=0.368.

## Decision Logic
- If curriculum beats baseline by >0.02 score: adopt
- If worse or within 0.02: keep standard training

## Budget
1 run × ~10 min

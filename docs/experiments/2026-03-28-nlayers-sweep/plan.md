# Experiment: nlayers sweep

## Hypothesis
nlayers=3 was chosen by Optuna alongside hdim=64. With the hdim sweep confirming 64 is optimal, the depth axis is the last untested architectural dimension. The v5 baseline used nlayers=2. With only 64 hidden units, deeper networks have less capacity per layer — maybe 2 wider-feeling layers outperform 3 narrow ones. Or maybe 4 layers would help the 676→64 compression.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: nlayers sweep
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | nlayers=2                 | Shallower (v5 baseline depth) |
| 2   | nlayers=4                 | Deeper |

Control (nlayers=3): Sortino=0.353, 9/23, score=0.368.

## Decision Logic
- Winner = highest score
- Within 0.02: keep nlayers=3

## Budget
2 runs × ~10 min = ~20 min

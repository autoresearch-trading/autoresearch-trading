# Experiment: hdim sweep

## Hypothesis
hdim=64 was chosen by Optuna on 5 symbols with a short search budget. The input is 676 dimensions (650 flat + 26 stats) compressed to 64 — a 10.6× bottleneck. A wider network (128 or 256) may capture more patterns, especially for the 14 failing symbols where the current model can't find edge. The v5 baseline used hdim=256 successfully.

Counter-argument: larger network = more parameters = more overfitting risk. But with 25 epochs and wd=0.0, the current regime may have room for more capacity.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: hdim sweep
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | hdim=128                  | 2× wider |
| 2   | hdim=256                  | v5 baseline width |

Control (hdim=64) from prior runs: Sortino=0.353, 9/23, score=0.368.

All other params: lr=1e-3, nlayers=3, batch_size=256, window=50, fee_mult=11.0, r_min=0.0, wd=0.0

## Decision Logic
- Winner = highest score
- If 128 or 256 beats 64: adopt wider network
- Within 0.02 score: keep 64 (fewer params, less overfit risk)

## Budget
2 runs × ~30 min = ~60 min

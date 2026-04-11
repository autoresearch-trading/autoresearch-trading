# Experiment: batch_size sweep

## Hypothesis
batch_size=256 was inherited from old Optuna and never re-swept in the current cost-adjusted regime (hdim=64, nlayers=3, fee_mult=11.0). With a small network trained for 60s/seed, batch size affects the bias-variance tradeoff of gradient updates. Smaller batches may improve generalization; larger batches may enable more stable convergence.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: batch_size sweep
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | batch_size=128            | Noisier gradients, more updates per epoch |
| 2   | batch_size=512            | Smoother gradients, fewer updates |

Control (batch_size=256) already run as window=50 control: Sortino=0.353, 9/23.

All other params: lr=1e-3, hdim=64, nlayers=3, window=50, fee_mult=11.0, r_min=0.0, wd=0.0, seeds=5, budget=300s

## Decision Logic
- Winner = highest score
- If 128 or 512 beats 256 by >0.02 score: adopt new batch_size
- If within 0.02: keep 256 (known quantity)

## Budget
2 runs × ~30 min = ~60 min total

## Gotchas
- Smaller batch = more gradient updates per epoch = potentially different convergence dynamics
- Larger batch = fewer updates = may undertrain in 60s budget

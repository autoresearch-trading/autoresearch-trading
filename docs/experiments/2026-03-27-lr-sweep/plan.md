# Experiment: lr sweep in cost-adjusted regime

## Hypothesis
lr=4.4e-3 was found by Optuna pre-slippage. The cost-adjusted barriers change the
label distribution (fewer directional labels with wider barriers), which may need a
different learning rate. Smaller lr = more stable convergence, larger lr = faster
escape from local minima.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: lr sweep
Base config: {hdim=64, nlayers=3, batch_size=256, fee_mult=11.0, r_min=0.0,
min_hold=1200, MAX_HOLD=300, epochs=25}

| Run | lr | Purpose |
|-----|-----|---------|
| 1 | 4.4e-3 | control (current best, Sortino=0.303) |
| 2 | 1e-3 | lower (was best for 256-dim network) |
| 3 | 2e-3 | mid-low |
| 4 | 8e-3 | higher |

## Decision Logic
Winner = highest composite score. Phase 2 refines ±50% if winner != control.

## Budget
4 runs × ~30 min = ~2 hours

# Experiment: fee_mult sweep on v11a

## Hypothesis
fee_mult is the dominant hyperparameter. Optuna top 5 all had fee_mult > 9.0 (best: 12.9).
Wider barriers = fewer but more confident trades. But too wide (>15) causes timeout-to-flat.
Sweet spot is likely 8-14. Need to validate on full 25 symbols (Optuna only screened on 5).

## Scoring
`score = mean_sortino * 0.6 + (passing / 25) * 0.4`

## Phases

### Phase 1: fee_mult sweep
Base config: {lr=4.4e-3, hdim=64, nlayers=3, batch_size=256, r_min=0.24}

| Run | fee_mult | Purpose |
|-----|----------|---------|
| 1   | 12.9     | control (Optuna winner) |
| 2   | 8.0      | lower bound |
| 3   | 10.0     | midpoint |
| 4   | 15.0     | upper bound |

### Phase 2: refine around winner
Depends on Phase 1. Narrow sweep +/- 2 around the Phase 1 winner.

## Decision Logic
- Winner = highest score across all 4 runs
- If top 2 are within 0.01 score, prefer higher passing count
- Phase 2 only if Phase 1 winner != 12.9 (if control wins, we're done)

## Budget
4 runs x ~30 min = ~2 hours for Phase 1
Up to 3 more runs for Phase 2 = ~1.5 hours

## Gotchas
- fee_mult changes barrier width: tp/sl = 2 * FEE_BPS/10000 * fee_mult
- At fee_mult=12.9: barriers = +/- 0.645% (12.9 * 2 * 5/10000)
- At fee_mult=8.0: barriers = +/- 0.4%
- At fee_mult=15.0: barriers = +/- 0.75%
- Higher fee_mult = fewer trades = less statistical power per symbol

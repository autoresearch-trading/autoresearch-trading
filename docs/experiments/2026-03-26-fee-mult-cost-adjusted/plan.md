# Experiment: fee_mult re-sweep with cost-adjusted barriers

## Hypothesis
fee_mult=11 was optimal with fee-only barriers (effective width 1.1%).
With T39 cost-adjusted barriers, effective width is now 1.8-2.9% — likely too wide.
Lower fee_mult (3-7) should perform better because it brings barriers back to the
~0.5-1.2% range where the model found alpha before.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`
(23 symbols after T40 filter)

## Phases

### Phase 1: sweep fee_mult with cost-adjusted barriers
Base config: {lr=4.4e-3, hdim=64, nlayers=3, batch_size=256, r_min=0.24}
All runs include slippage + T40 symbol filter.

| Run | fee_mult | Effective barrier (BTC) | Purpose |
|-----|----------|------------------------|---------|
| 1   | 11.0     | ~1.8%                  | control (current) |
| 2   | 3.0      | ~0.5%                  | tight |
| 3   | 5.0      | ~0.8%                  | mid-low |
| 4   | 7.0      | ~1.1%                  | mid (matches old sweet spot) |

## Decision Logic
Winner = highest composite score. Phase 2 refines ±1 around winner if != control.

## Budget
4 runs × ~30 min = ~2 hours

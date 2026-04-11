# Experiment Report: fee_mult sweep on v11a

## Results
| Run | fee_mult | Sortino | Passing | Trades | MaxDD | PF | Score |
|-----|----------|---------|---------|--------|-------|-----|-------|
| 2 | 8.0 | 0.066 | 12/25 | 2185 | 0.403 | 1.20 | 0.232 |
| 5 | 9.0 | 0.111 | 6/25 | 2189 | 0.428 | 1.34 | 0.163 |
| 3 | 10.0 | 0.043 | 13/25 | 2184 | 0.412 | 1.16 | 0.234 |
| **6** | **11.0** | **0.093** | **13/25** | **2161** | **0.460** | **1.24** | **0.264** |
| 1 | 12.9 | 0.116 | 9/25 | 2051 | 0.549 | 1.31 | 0.214 |
| 4 | 15.0 | -0.008 | 8/25 | 2138 | 0.508 | 1.17 | 0.123 |

## Analysis

### Phase 1 (fee_mult = 8, 10, 12.9, 15)
- Clear trend: fee_mult > 12 hurts passing count (barriers too wide, more DD)
- fee_mult=10 had the most passing symbols (13/25) but lower Sortino
- Optuna's 12.9 winner was optimized on 5 symbols — doesn't generalize to 25

### Phase 2 (fee_mult = 9, 11)
- fee_mult=9 has high Sortino (0.111) but only 6/25 passing — too tight, many symbols blow through DD guardrail
- fee_mult=11 hits the sweet spot: 13/25 passing + Sortino 0.093 = best composite score

### The fee_mult tradeoff
- **Lower fee_mult (8-9)**: Tighter barriers → more trades trigger → higher per-trade accuracy needed → fewer symbols pass DD guardrail
- **Higher fee_mult (13-15)**: Wider barriers → more timeouts to flat → lower Sortino, fewer decisive trades
- **Sweet spot (10-11)**: Balanced — enough barrier width to avoid noise exits, tight enough to capture real moves

## Conclusion
fee_mult=11.0 is the new best config. It balances Sortino (0.093) with broad passing (13/25) for the highest composite score (0.264).

## Recommended Config
```python
BEST_PARAMS = {
    "lr": 4.4e-3,
    "hdim": 64,
    "nlayers": 3,
    "batch_size": 256,
    "fee_mult": 11.0,
    "r_min": 0.24,
    "vpin_max_z": 0.0,
}
```

## Next Hypotheses
1. **Slippage modeling**: Research shows we're missing 3-12 bps/trade. Add slippage_bps parameter to evaluate() and re-run.
2. **min_hold sweep**: With fee_mult=11 (barriers at +/-0.55%), min_hold=800 might be too conservative. Try 400, 600.
3. **Epoch tuning**: 64-dim network with 25 epochs — try 15, 35, 50 to check if we're under/overfitting.

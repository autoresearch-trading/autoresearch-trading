# Goal-A Maker-Mode Cost-Band Sensitivity

Sensitivity sweep over maker-mode fees (bps per side). Negative = rebate; positive = fee. **Slippage assumed = 0** under maker execution (we post a limit at our chosen price; we do not cross the spread). This is the load-bearing simplification of this analysis. In reality, the equivalent risk under maker execution is **adverse selection** (we get filled when the model is wrong), which this first-cut model does not incorporate.

**Maker headroom** = (2p − 1) × |edge_bps| − 2 × maker_fee_bps. **Fill-proxy** = fraction of windows with |edge_bps| ≥ 1 bp (rough cross-symbol approximation of 'mid traverses ≥1 tick within horizon'). A cell is alive iff median headroom > 0 AND frac_positive headroom > 0.55.

**Caveats explicitly NOT modelled**: queue position, partial fills, adverse selection, symbol-specific tick sizes. A real fill-rate study needs raw limit-order event data — which we do not have.


## Accuracy = 0.550 (55%)

| maker_fee_bps | n_cells_alive | n_cells_alive (with fill-proxy) | top 5 cells by median headroom |
|---:|---:|---:|---|
| -2.0 | 288 | 288 | XPL:$1k:H500(+14.08bp), XPL:$10k:H500(+14.08bp), PUMP:$1k:H500(+14.05bp), PUMP:$10k:H500(+14.05bp), FARTCOIN:$1k:H500(+14.02bp) |
| -1.0 | 288 | 288 | XPL:$1k:H500(+12.08bp), XPL:$10k:H500(+12.08bp), PUMP:$1k:H500(+12.05bp), PUMP:$10k:H500(+12.05bp), FARTCOIN:$1k:H500(+12.02bp) |
| +0.0 | 288 | 288 | XPL:$1k:H500(+10.08bp), XPL:$10k:H500(+10.08bp), PUMP:$1k:H500(+10.05bp), PUMP:$10k:H500(+10.05bp), FARTCOIN:$1k:H500(+10.02bp) |
| +1.0 | 131 | 131 | XPL:$1k:H500(+8.08bp), XPL:$10k:H500(+8.08bp), PUMP:$1k:H500(+8.05bp), PUMP:$10k:H500(+8.05bp), FARTCOIN:$1k:H500(+8.02bp) |
| +2.0 | 53 | 53 | XPL:$1k:H500(+6.08bp), XPL:$10k:H500(+6.08bp), PUMP:$1k:H500(+6.05bp), PUMP:$10k:H500(+6.05bp), FARTCOIN:$1k:H500(+6.02bp) |
| +3.0 | 17 | 17 | XPL:$1k:H500(+4.08bp), XPL:$10k:H500(+4.08bp), PUMP:$1k:H500(+4.05bp), PUMP:$10k:H500(+4.05bp), FARTCOIN:$1k:H500(+4.02bp) |
| +4.0 | 8 | 8 | XPL:$1k:H500(+2.08bp), XPL:$10k:H500(+2.08bp), PUMP:$1k:H500(+2.05bp), PUMP:$10k:H500(+2.05bp), FARTCOIN:$1k:H500(+2.02bp) |
| +5.0 | 0 | 0 | _(none)_ |
| +6.0 | 0 | 0 | _(none)_ |

**Breakeven maker fee at 55% accuracy: ≤ +4.0 bp/side** (highest swept fee at which ≥1 cell is alive on raw headroom).


## Accuracy = 0.575 (57.5%)

| maker_fee_bps | n_cells_alive | n_cells_alive (with fill-proxy) | top 5 cells by median headroom |
|---:|---:|---:|---|
| -2.0 | 288 | 288 | XPL:$1k:H500(+19.12bp), XPL:$10k:H500(+19.12bp), PUMP:$1k:H500(+19.08bp), PUMP:$10k:H500(+19.08bp), FARTCOIN:$1k:H500(+19.04bp) |
| -1.0 | 288 | 288 | XPL:$1k:H500(+17.12bp), XPL:$10k:H500(+17.12bp), PUMP:$1k:H500(+17.08bp), PUMP:$10k:H500(+17.08bp), FARTCOIN:$1k:H500(+17.04bp) |
| +0.0 | 288 | 288 | XPL:$1k:H500(+15.12bp), XPL:$10k:H500(+15.12bp), PUMP:$1k:H500(+15.08bp), PUMP:$10k:H500(+15.08bp), FARTCOIN:$1k:H500(+15.04bp) |
| +1.0 | 188 | 188 | XPL:$1k:H500(+13.12bp), XPL:$10k:H500(+13.12bp), PUMP:$1k:H500(+13.08bp), PUMP:$10k:H500(+13.08bp), FARTCOIN:$1k:H500(+13.04bp) |
| +2.0 | 94 | 94 | XPL:$1k:H500(+11.12bp), XPL:$10k:H500(+11.12bp), PUMP:$1k:H500(+11.08bp), PUMP:$10k:H500(+11.08bp), FARTCOIN:$1k:H500(+11.04bp) |
| +3.0 | 53 | 53 | XPL:$1k:H500(+9.12bp), XPL:$10k:H500(+9.12bp), PUMP:$1k:H500(+9.08bp), PUMP:$10k:H500(+9.08bp), FARTCOIN:$1k:H500(+9.04bp) |
| +4.0 | 29 | 29 | XPL:$1k:H500(+7.12bp), XPL:$10k:H500(+7.12bp), PUMP:$1k:H500(+7.08bp), PUMP:$10k:H500(+7.08bp), FARTCOIN:$1k:H500(+7.04bp) |
| +5.0 | 15 | 15 | XPL:$1k:H500(+5.12bp), XPL:$10k:H500(+5.12bp), PUMP:$1k:H500(+5.08bp), PUMP:$10k:H500(+5.08bp), FARTCOIN:$1k:H500(+5.04bp) |
| +6.0 | 8 | 8 | XPL:$1k:H500(+3.12bp), XPL:$10k:H500(+3.12bp), PUMP:$1k:H500(+3.08bp), PUMP:$10k:H500(+3.08bp), FARTCOIN:$1k:H500(+3.04bp) |

**Breakeven maker fee at 57.5% accuracy: ≤ +6.0 bp/side** (highest swept fee at which ≥1 cell is alive on raw headroom).


## Accuracy = 0.600 (60%)

| maker_fee_bps | n_cells_alive | n_cells_alive (with fill-proxy) | top 5 cells by median headroom |
|---:|---:|---:|---|
| -2.0 | 288 | 288 | XPL:$1k:H500(+24.16bp), XPL:$10k:H500(+24.16bp), PUMP:$1k:H500(+24.10bp), PUMP:$10k:H500(+24.10bp), FARTCOIN:$1k:H500(+24.05bp) |
| -1.0 | 288 | 288 | XPL:$1k:H500(+22.16bp), XPL:$10k:H500(+22.16bp), PUMP:$1k:H500(+22.10bp), PUMP:$10k:H500(+22.10bp), FARTCOIN:$1k:H500(+22.05bp) |
| +0.0 | 288 | 288 | XPL:$1k:H500(+20.16bp), XPL:$10k:H500(+20.16bp), PUMP:$1k:H500(+20.10bp), PUMP:$10k:H500(+20.10bp), FARTCOIN:$1k:H500(+20.05bp) |
| +1.0 | 216 | 216 | XPL:$1k:H500(+18.16bp), XPL:$10k:H500(+18.16bp), PUMP:$1k:H500(+18.10bp), PUMP:$10k:H500(+18.10bp), FARTCOIN:$1k:H500(+18.05bp) |
| +2.0 | 131 | 131 | XPL:$1k:H500(+16.16bp), XPL:$10k:H500(+16.16bp), PUMP:$1k:H500(+16.10bp), PUMP:$10k:H500(+16.10bp), FARTCOIN:$1k:H500(+16.05bp) |
| +3.0 | 80 | 80 | XPL:$1k:H500(+14.16bp), XPL:$10k:H500(+14.16bp), PUMP:$1k:H500(+14.10bp), PUMP:$10k:H500(+14.10bp), FARTCOIN:$1k:H500(+14.05bp) |
| +4.0 | 53 | 53 | XPL:$1k:H500(+12.16bp), XPL:$10k:H500(+12.16bp), PUMP:$1k:H500(+12.10bp), PUMP:$10k:H500(+12.10bp), FARTCOIN:$1k:H500(+12.05bp) |
| +5.0 | 40 | 40 | XPL:$1k:H500(+10.16bp), XPL:$10k:H500(+10.16bp), PUMP:$1k:H500(+10.10bp), PUMP:$10k:H500(+10.10bp), FARTCOIN:$1k:H500(+10.05bp) |
| +6.0 | 17 | 17 | XPL:$1k:H500(+8.16bp), XPL:$10k:H500(+8.16bp), PUMP:$1k:H500(+8.10bp), PUMP:$10k:H500(+8.10bp), FARTCOIN:$1k:H500(+8.05bp) |

**Breakeven maker fee at 60% accuracy: ≤ +6.0 bp/side** (highest swept fee at which ≥1 cell is alive on raw headroom).


## Cross-reference

See `survivors.md` for the taker-mode (6 bp/side + slip) verdict. Under taker execution, zero cells survive at any of 55%/57.5%/60% accuracy.

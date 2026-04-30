# Post-Cascade Regime Sprint

**Status:** discovery-only; prior April holdout is already consumed.
**Verdict:** KILL_OR_UNDERPOWERED

## Method

Observed liquidation trades are grouped into bursts by timestamp gap. The burst sign is observed from the start-to-end burst price move. The probe then tests contrarian post-burst reversion after fixed delays and horizons, net of taker round-trip fees and a conservative slippage placeholder.

No encoder, RL, maker fills, or pre-cascade direction model is used.

## Cost assumptions

- taker fee per side: 6.00 bp
- slippage per side placeholder: 1.00 bp
- round-trip cost: 14.00 bp

## Universe summary

- rows: 15
- unique bursts: 2
- symbols: 2
- days: 2

## Horizon/delay summary

| horizon | delay_events | n_events | n_days | n_symbols | median_net_bps | mean_net_bps | frac_positive | median_gross_reversion_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0 | 2 | 2 | 2 | 13.0509 | 13.0509 | 0.5000 | 27.0509 |
| 50 | 10 | 2 | 2 | 2 | 9.6590 | 9.6590 | 0.5000 | 23.6590 |
| 50 | 50 | 2 | 2 | 2 | 1.0102 | 1.0102 | 0.5000 | 15.0102 |
| 100 | 0 | 2 | 2 | 2 | 28.0611 | 28.0611 | 1 | 42.0611 |
| 100 | 10 | 2 | 2 | 2 | 22.3014 | 22.3014 | 1 | 36.3014 |
| 100 | 50 | 2 | 2 | 2 | 17.2314 | 17.2314 | 0.5000 | 31.2314 |
| 500 | 0 | 1 | 1 | 1 | 11.2174 | 11.2174 | 1 | 25.2174 |
| 500 | 10 | 1 | 1 | 1 | -14.0000 | -14.0000 | 0 | 0.0000 |
| 500 | 50 | 1 | 1 | 1 | -56.9865 | -56.9865 | 0 | -42.9865 |

## Best cell

| horizon | delay_events | n_events | n_days | n_symbols | median_net_bps | mean_net_bps | frac_positive | median_gross_reversion_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 0 | 2 | 2 | 2 | 28.0611 | 28.0611 | 1 | 42.0611 |

## Discovery gate

Pass requires median net >= +3 bp, frac-positive >= 0.57, n_events >= 400, and n_symbols >= 5. If this gate fails, do not continue Pacifica-only modeling without a new hypothesis or fresh data.

## Interpretation

No post-cascade reversion cell clears the mechanical discovery gate. Treat this as a kill or underpowered result; do not optimize thresholds on the consumed data.

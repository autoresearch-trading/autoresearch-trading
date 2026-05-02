# Pacifica Paper-Trading Eligibility Gates

This report defines the pre-trade eligible universe for the non-HFT Pacifica paper-trading program.
It is not a strategy, alpha claim, or backtest.

Verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`
Symbols evaluated: 65
Eligible symbols: 0

## Interpretation discipline

Do not trade every collected symbol. The raw collector intentionally captures the broad live public universe; paper trading may only use symbols that pass explicit sample, liquidity, spread/cost, activity, stability, and concentration gates.
The current archive is still young, so this report is diagnostic until enough full distinct days accrue.

## Thresholds

| threshold | value |
| --- | --- |
| min_days | 30 |
| min_observations | 10000 |
| min_median_top_depth_notional | 5000 |
| max_median_spread_bps | 25 |
| min_median_trade_notional_per_min | 25 |
| min_median_bbo_updates_per_min | 10 |
| max_day_observation_concentration | 0.2500 |
| max_p95_spread_bps | 75 |
| max_mean_toxicity_score | 0.7500 |

## Gate counts

| gate | passing_symbols | total_symbols |
| --- | --- | --- |
| sample_gate_pass | 0 | 65 |
| liquidity_gate_pass | 25 | 65 |
| spread_cost_gate_pass | 60 | 65 |
| activity_gate_pass | 4 | 65 |
| stability_gate_pass | 63 | 65 |
| concentration_gate_pass | 0 | 65 |
| eligible | 0 | 65 |

## Symbol eligibility preview

| symbol | eligible | n_days | n_observations | median_top_depth_notional | median_spread_bps | median_trade_notional_per_min | max_day_observation_concentration | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | False | 2 | 917 | 152259.1170 | 0.1387 | 10310.7073 | 0.8266 | sample;concentration |
| ETH | False | 2 | 915 | 101147.3963 | 0.4507 | 22689.2234 | 0.8262 | sample;concentration |
| SOL | False | 2 | 916 | 62773.6472 | 1.2172 | 48.0164 | 0.8264 | sample;concentration |
| PUMP | False | 2 | 916 | 15129.8379 | 6.6273 | 0.0035 | 0.8264 | sample;activity;concentration |
| kPEPE | False | 2 | 917 | 14199.1728 | 7.3690 | 0.0465 | 0.8266 | sample;activity;concentration |
| CRV | False | 2 | 916 | 13889.6949 | 17.3292 | 0.0469 | 0.8264 | sample;activity;concentration |
| WLFI | False | 2 | 916 | 12743.4143 | 16.8209 | 0.0604 | 0.8264 | sample;activity;concentration |
| NEAR | False | 2 | 915 | 10892.1391 | 20.8740 | 0.1301 | 0.8262 | sample;activity;concentration |
| XRP | False | 2 | 916 | 10447.2303 | 0.8610 | 0.0411 | 0.8264 | sample;activity;concentration |
| 2Z | False | 2 | 917 | 10135.4565 | 20.0555 | 0.1634 | 0.8266 | sample;activity;concentration |
| AVAX | False | 2 | 916 | 9915.9066 | 6.2840 | 0.1817 | 0.8264 | sample;activity;concentration |
| BNB | False | 2 | 916 | 9274.6425 | 0.2304 | 1.2323 | 0.8264 | sample;activity;concentration |
| JUP | False | 2 | 915 | 9180.7159 | 13.7939 | 0.1830 | 0.8262 | sample;activity;concentration |
| HYPE | False | 2 | 917 | 7968.6920 | 0.4249 | 0.8057 | 0.8266 | sample;activity;concentration |
| AAVE | False | 2 | 916 | 7346.4557 | 5.5391 | 1.8549 | 0.8264 | sample;activity;concentration |
| FARTCOIN | False | 2 | 914 | 6964.8625 | 10.4187 | 0.0403 | 0.8260 | sample;activity;concentration |
| LTC | False | 2 | 916 | 6734.6887 | 6.8378 | 0.5533 | 0.8264 | sample;activity;concentration |
| NATGAS | False | 2 | 916 | 6675.4025 | 6.6051 | 0.2787 | 0.8264 | sample;activity;concentration |
| ICP | False | 2 | 916 | 6109.2914 | 19.2571 | 0.2386 | 0.8264 | sample;activity;concentration |
| UNI | False | 2 | 916 | 5887.3219 | 6.1780 | 0.3207 | 0.8264 | sample;activity;concentration |
| PLATINUM | False | 2 | 915 | 5210.1011 | 22.3850 | 0.1993 | 0.8262 | sample;activity;concentration |
| kBONK | False | 2 | 917 | 5043.2174 | 7.1146 | 0.0441 | 0.8266 | sample;activity;concentration |
| XPL | False | 2 | 917 | 16476.0494 | 32.2407 | 0.0923 | 0.8266 | sample;spread_cost;activity;concentration |
| CHIP | False | 2 | 915 | 8036.9460 | 29.1413 | 0.0128 | 0.8262 | sample;spread_cost;activity;concentration |
| STRK | False | 2 | 916 | 5896.0422 | 46.8500 | 0.0391 | 0.8264 | sample;spread_cost;activity;concentration |
| CL | False | 2 | 916 | 4803.1008 | 3.1741 | 189.4038 | 0.8264 | sample;liquidity;concentration |
| WLD | False | 2 | 917 | 4473.4029 | 12.3127 | 0.0242 | 0.8266 | sample;liquidity;activity;concentration |
| LDO | False | 2 | 916 | 3847.3150 | 10.8391 | 0.0369 | 0.8264 | sample;liquidity;activity;concentration |
| PENGU | False | 2 | 915 | 3687.3724 | 8.4218 | 0.0101 | 0.8262 | sample;liquidity;activity;concentration |
| ZEC | False | 2 | 916 | 3640.9816 | 1.6029 | 3.5000 | 0.8264 | sample;liquidity;activity;concentration |

## Output files

- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.
- `eligible_symbols.csv` — subset that passed all gates.
- `gate_counts.csv` — count of symbols passing each gate.
- `thresholds.csv` — fixed thresholds used for this run.

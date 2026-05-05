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
| liquidity_gate_pass | 23 | 65 |
| spread_cost_gate_pass | 60 | 65 |
| activity_gate_pass | 0 | 65 |
| stability_gate_pass | 63 | 65 |
| concentration_gate_pass | 41 | 65 |
| eligible | 0 | 65 |

## Symbol eligibility preview

| symbol | eligible | n_days | n_observations | median_top_depth_notional | median_spread_bps | median_trade_notional_per_min | max_day_observation_concentration | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | False | 6 | 5793 | 138854.2569 | 0.1281 | 0 | 0.2396 | sample;activity |
| ETH | False | 6 | 5854 | 102164.3733 | 0.4384 | 0 | 0.2388 | sample;activity |
| SOL | False | 6 | 5702 | 61233.6680 | 1.1954 | 0 | 0.2406 | sample;activity |
| WLFI | False | 6 | 5517 | 15616.0824 | 18.2983 | 0 | 0.2400 | sample;activity |
| PUMP | False | 6 | 5469 | 15474.1991 | 5.6978 | 0 | 0.2443 | sample;activity |
| CRV | False | 6 | 4956 | 14695.8536 | 17.3233 | 0 | 0.2520 | sample;activity;concentration |
| kPEPE | False | 3 | 1979 | 14604.3582 | 7.4151 | 0 | 0.4760 | sample;activity;concentration |
| 2Z | False | 6 | 4918 | 11683.0304 | 21.7983 | 0 | 0.2519 | sample;activity;concentration |
| XRP | False | 6 | 5295 | 11128.9173 | 0.9227 | 0 | 0.2457 | sample;activity |
| BNB | False | 6 | 5322 | 10369.2604 | 0.1976 | 0 | 0.2445 | sample;activity |
| NEAR | False | 6 | 4805 | 9481.1421 | 18.7793 | 0 | 0.2595 | sample;activity;concentration |
| JUP | False | 6 | 5047 | 9001.3310 | 13.1112 | 0 | 0.2453 | sample;activity |
| AVAX | False | 6 | 5261 | 8486.7002 | 6.2223 | 0 | 0.2425 | sample;activity |
| HYPE | False | 6 | 5708 | 7813.4763 | 0.4078 | 0 | 0.2398 | sample;activity |
| UNI | False | 6 | 5281 | 7386.1459 | 7.8780 | 0 | 0.2403 | sample;activity |
| FARTCOIN | False | 6 | 5601 | 7297.0898 | 9.8207 | 0 | 0.2398 | sample;activity |
| LTC | False | 6 | 4720 | 6748.5902 | 7.2173 | 0 | 0.2540 | sample;activity;concentration |
| ICP | False | 6 | 5105 | 6012.8306 | 19.8387 | 0 | 0.2447 | sample;activity |
| AAVE | False | 6 | 5284 | 5817.1516 | 5.8980 | 0 | 0.2383 | sample;activity |
| kBONK | False | 5 | 4216 | 5522.9810 | 6.9719 | 0 | 0.3107 | sample;activity;concentration |
| XPL | False | 6 | 4946 | 17598.1474 | 32.4500 | 0 | 0.2497 | sample;spread_cost;activity |
| CHIP | False | 6 | 5483 | 8955.3869 | 30.9081 | 0 | 0.2468 | sample;spread_cost;activity |
| STRK | False | 6 | 4994 | 6271.0496 | 30.4063 | 0 | 0.2497 | sample;spread_cost;activity |
| ZEC | False | 6 | 5770 | 4942.7356 | 1.7197 | 0 | 0.2414 | sample;liquidity;activity |
| PLATINUM | False | 6 | 4161 | 4666.4410 | 18.9222 | 0 | 0.2610 | sample;liquidity;activity;concentration |
| TAO | False | 6 | 5621 | 4285.8745 | 3.7550 | 0 | 0.2400 | sample;liquidity;activity |
| WLD | False | 6 | 4930 | 4285.7813 | 12.5971 | 0 | 0.2448 | sample;liquidity;activity |
| PENGU | False | 6 | 5609 | 3974.2895 | 8.1555 | 0 | 0.2393 | sample;liquidity;activity |
| XAG | False | 6 | 5022 | 3878.0666 | 1.1791 | 0 | 0.2698 | sample;liquidity;activity;concentration |
| DOGE | False | 6 | 5411 | 3851.6468 | 3.5256 | 0 | 0.2421 | sample;liquidity;activity |

## Output files

- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.
- `eligible_symbols.csv` — subset that passed all gates.
- `gate_counts.csv` — count of symbols passing each gate.
- `thresholds.csv` — fixed thresholds used for this run.

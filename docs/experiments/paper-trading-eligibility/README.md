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
| concentration_gate_pass | 63 | 65 |
| eligible | 0 | 65 |

## Symbol eligibility preview

| symbol | eligible | n_days | n_observations | median_top_depth_notional | median_spread_bps | median_trade_notional_per_min | max_day_observation_concentration | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | False | 7 | 7235 | 142073.1396 | 0.1280 | 0 | 0.1920 | sample;activity |
| ETH | False | 7 | 7301 | 104130.2619 | 0.4363 | 0 | 0.1915 | sample;activity |
| SOL | False | 7 | 7130 | 61799.9660 | 1.1942 | 0 | 0.1927 | sample;activity |
| kPEPE | False | 7 | 7006 | 15533.2194 | 7.3792 | 0 | 0.1921 | sample;activity |
| CRV | False | 7 | 6170 | 15449.5279 | 17.4419 | 0 | 0.2024 | sample;activity |
| PUMP | False | 7 | 6859 | 15187.4176 | 5.7144 | 0 | 0.1949 | sample;activity |
| WLFI | False | 7 | 6798 | 15158.2518 | 18.0343 | 0 | 0.1948 | sample;activity |
| 2Z | False | 7 | 6240 | 11625.6819 | 21.1430 | 0 | 0.2127 | sample;activity |
| XRP | False | 7 | 6687 | 10941.6015 | 0.9272 | 0 | 0.1946 | sample;activity |
| BNB | False | 7 | 6659 | 9866.7045 | 0.2116 | 0 | 0.1954 | sample;activity |
| NEAR | False | 7 | 5963 | 9535.4911 | 19.1332 | 0 | 0.2091 | sample;activity |
| AVAX | False | 7 | 6592 | 9191.7263 | 6.3199 | 0 | 0.1936 | sample;activity |
| JUP | False | 7 | 6328 | 9083.3650 | 13.3380 | 0 | 0.1956 | sample;activity |
| HYPE | False | 7 | 7132 | 7954.7270 | 0.4066 | 0 | 0.1920 | sample;activity |
| LTC | False | 7 | 5837 | 7414.2098 | 7.1762 | 0 | 0.2054 | sample;activity |
| FARTCOIN | False | 7 | 7020 | 7152.4985 | 9.8571 | 0 | 0.1920 | sample;activity |
| UNI | False | 7 | 6577 | 6995.2551 | 7.4966 | 0 | 0.1929 | sample;activity |
| ICP | False | 7 | 6485 | 6202.0686 | 20.0239 | 0 | 0.1926 | sample;activity |
| AAVE | False | 7 | 6604 | 5970.4934 | 6.1073 | 0 | 0.1906 | sample;activity |
| kBONK | False | 7 | 6862 | 5571.5925 | 7.1347 | 0 | 0.1943 | sample;activity |
| XPL | False | 7 | 6187 | 17058.2088 | 31.9732 | 0 | 0.1996 | sample;spread_cost;activity |
| CHIP | False | 7 | 6692 | 9064.3485 | 31.3640 | 0 | 0.2022 | sample;spread_cost;activity |
| STRK | False | 7 | 6243 | 6491.6553 | 36.7545 | 0 | 0.1997 | sample;spread_cost;activity |
| PLATINUM | False | 7 | 5196 | 4922.2042 | 20.5323 | 0 | 0.2090 | sample;liquidity;activity |
| ZEC | False | 7 | 7269 | 4823.1846 | 1.8523 | 0 | 0.1916 | sample;liquidity;activity |
| TAO | False | 7 | 7018 | 4572.7459 | 3.7103 | 0 | 0.1922 | sample;liquidity;activity |
| WLD | False | 7 | 6159 | 4422.2494 | 12.6024 | 0 | 0.1960 | sample;liquidity;activity |
| PENGU | False | 7 | 7024 | 4075.2968 | 8.2739 | 0 | 0.1952 | sample;liquidity;activity |
| XAG | False | 7 | 6424 | 4071.7817 | 1.5998 | 0 | 0.2109 | sample;liquidity;activity |
| DOGE | False | 7 | 6780 | 3863.2257 | 3.4212 | 0 | 0.1932 | sample;liquidity;activity |

## Output files

- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.
- `eligible_symbols.csv` — subset that passed all gates.
- `gate_counts.csv` — count of symbols passing each gate.
- `thresholds.csv` — fixed thresholds used for this run.

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
| liquidity_gate_pass | 24 | 65 |
| spread_cost_gate_pass | 60 | 65 |
| activity_gate_pass | 0 | 65 |
| stability_gate_pass | 63 | 65 |
| concentration_gate_pass | 65 | 65 |
| eligible | 0 | 65 |

## Symbol eligibility preview

| symbol | eligible | n_days | n_observations | median_top_depth_notional | median_spread_bps | median_trade_notional_per_min | max_day_observation_concentration | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | False | 8 | 9162 | 145051.0867 | 0.1280 | 0 | 0.1522 | sample;activity |
| ETH | False | 8 | 9237 | 105433.0784 | 0.4370 | 0 | 0.1513 | sample;activity |
| SOL | False | 8 | 9049 | 61977.4556 | 1.1930 | 0 | 0.1532 | sample;activity |
| WLFI | False | 8 | 8534 | 16013.0874 | 18.1984 | 0 | 0.1551 | sample;activity |
| kPEPE | False | 8 | 8904 | 15505.9906 | 7.5287 | 0 | 0.1534 | sample;activity |
| CRV | False | 8 | 7890 | 15378.7104 | 17.6367 | 0 | 0.1583 | sample;activity |
| PUMP | False | 8 | 8742 | 14999.7992 | 5.8680 | 0 | 0.1541 | sample;activity |
| XRP | False | 8 | 8473 | 10929.7180 | 0.9891 | 0 | 0.1539 | sample;activity |
| 2Z | False | 8 | 9254 | 9826.0545 | 19.7138 | 0 | 0.1520 | sample;activity |
| AVAX | False | 8 | 8419 | 9773.9771 | 6.3546 | 0 | 0.1561 | sample;activity |
| BNB | False | 8 | 8495 | 9625.0775 | 0.2333 | 0 | 0.1566 | sample;activity |
| NEAR | False | 8 | 7597 | 9549.3404 | 18.8031 | 0 | 0.1641 | sample;activity |
| JUP | False | 8 | 8138 | 9173.4728 | 13.5720 | 0 | 0.1547 | sample;activity |
| LTC | False | 8 | 7446 | 8362.1948 | 7.1523 | 0 | 0.1610 | sample;activity |
| HYPE | False | 8 | 9028 | 7823.8109 | 0.4149 | 0 | 0.1518 | sample;activity |
| FARTCOIN | False | 8 | 8931 | 7065.5289 | 9.8820 | 0 | 0.1549 | sample;activity |
| UNI | False | 8 | 8372 | 6655.3018 | 7.2711 | 0 | 0.1560 | sample;activity |
| AAVE | False | 8 | 8488 | 6420.7629 | 6.3440 | 0 | 0.1603 | sample;activity |
| ICP | False | 8 | 8375 | 6344.5812 | 20.3970 | 0 | 0.1639 | sample;activity |
| kBONK | False | 8 | 8698 | 5600.1554 | 7.1445 | 0 | 0.1533 | sample;activity |
| PLATINUM | False | 8 | 6419 | 5084.7501 | 21.6108 | 0 | 0.1692 | sample;activity |
| XPL | False | 8 | 7949 | 16014.1531 | 31.4796 | 0 | 0.1562 | sample;spread_cost;activity |
| CHIP | False | 8 | 8429 | 9329.6734 | 32.2675 | 0 | 0.1605 | sample;spread_cost;activity |
| STRK | False | 8 | 7958 | 6691.4664 | 38.7430 | 0 | 0.1601 | sample;spread_cost;activity |
| TAO | False | 8 | 8926 | 4974.8147 | 4.0631 | 0 | 0.1539 | sample;liquidity;activity |
| XAG | False | 8 | 8325 | 4807.4230 | 1.8774 | 0 | 0.1635 | sample;liquidity;activity |
| ZEC | False | 8 | 9205 | 4731.5106 | 1.8807 | 0 | 0.1520 | sample;liquidity;activity |
| DOGE | False | 8 | 8576 | 4595.4070 | 3.8398 | 0 | 0.1538 | sample;liquidity;activity |
| WLD | False | 8 | 7873 | 4500.0875 | 12.7632 | 0 | 0.1564 | sample;liquidity;activity |
| NATGAS | False | 8 | 7063 | 4445.8653 | 6.8459 | 0 | 0.1586 | sample;liquidity;activity |

## Output files

- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.
- `eligible_symbols.csv` — subset that passed all gates.
- `gate_counts.csv` — count of symbols passing each gate.
- `thresholds.csv` — fixed thresholds used for this run.

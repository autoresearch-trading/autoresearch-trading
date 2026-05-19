# Pacifica Paper-Trading Eligibility Gates

This report defines the pre-trade eligible universe for the non-HFT Pacifica paper-trading program.
It is not a strategy, alpha claim, or backtest.

Verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`
Symbols evaluated: 66
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
| sample_gate_pass | 0 | 66 |
| liquidity_gate_pass | 25 | 66 |
| spread_cost_gate_pass | 62 | 66 |
| activity_gate_pass | 0 | 66 |
| stability_gate_pass | 63 | 66 |
| concentration_gate_pass | 66 | 66 |
| eligible | 0 | 66 |

## Symbol eligibility preview

| symbol | eligible | n_days | n_observations | median_top_depth_notional | median_spread_bps | median_trade_notional_per_min | max_day_observation_concentration | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | False | 19 | 19799 | 151575.9927 | 0.1276 | 16.4833 | 0.0712 | sample;activity |
| ETH | False | 19 | 18907 | 113443.0310 | 0.4405 | 0 | 0.0745 | sample;activity |
| SOL | False | 17 | 18203 | 63787.5158 | 1.1796 | 0 | 0.0774 | sample;activity |
| WLFI | False | 17 | 18248 | 15213.0498 | 16.9975 | 0 | 0.0773 | sample;activity |
| CRV | False | 19 | 19284 | 14450.7020 | 17.7733 | 0 | 0.0731 | sample;activity |
| PUMP | False | 17 | 18220 | 14272.0452 | 5.8677 | 0 | 0.0773 | sample;activity |
| kPEPE | False | 17 | 18233 | 13981.9460 | 7.2300 | 0.0037 | 0.0773 | sample;activity |
| XRP | False | 17 | 18226 | 11648.6834 | 0.9492 | 0 | 0.0773 | sample;activity |
| AVAX | False | 19 | 20346 | 9821.1597 | 6.2406 | 0 | 0.0694 | sample;activity |
| BNB | False | 19 | 20037 | 9564.7682 | 0.2193 | 0 | 0.0704 | sample;activity |
| NEAR | False | 17 | 18274 | 8519.8401 | 16.1666 | 0 | 0.0775 | sample;activity |
| JUP | False | 17 | 18231 | 8243.8710 | 12.3743 | 0 | 0.0773 | sample;activity |
| HYPE | False | 19 | 18383 | 7513.4742 | 0.4115 | 0 | 0.0767 | sample;activity |
| LTC | False | 17 | 18261 | 7105.5894 | 6.2763 | 0 | 0.0774 | sample;activity |
| UNI | False | 17 | 18227 | 6996.1859 | 7.2229 | 0 | 0.0774 | sample;activity |
| FARTCOIN | False | 19 | 18706 | 6985.3315 | 9.4129 | 0 | 0.0753 | sample;activity |
| ICP | False | 18 | 18289 | 6400.6466 | 17.7888 | 0 | 0.0771 | sample;activity |
| AAVE | False | 19 | 20561 | 6136.1612 | 6.1429 | 0 | 0.0686 | sample;activity |
| 2Z | False | 19 | 20691 | 5985.9662 | 19.1841 | 0 | 0.0683 | sample;activity |
| DOGE | False | 19 | 19147 | 5320.2263 | 4.3884 | 0 | 0.0737 | sample;activity |
| kBONK | False | 17 | 18228 | 5266.7532 | 6.8184 | 0.0062 | 0.0774 | sample;activity |
| TAO | False | 17 | 18219 | 5042.4691 | 4.4639 | 0 | 0.0774 | sample;activity |
| XPL | False | 17 | 18261 | 10783.0110 | 28.4727 | 0 | 0.0773 | sample;spread_cost;activity |
| CHIP | False | 19 | 19832 | 8663.3082 | 31.2988 | 0 | 0.0711 | sample;spread_cost;activity |
| STRK | False | 17 | 18244 | 6254.7030 | 32.4460 | 0 | 0.0773 | sample;spread_cost;activity |
| ZEC | False | 17 | 18201 | 4673.4612 | 1.7653 | 0 | 0.0775 | sample;liquidity;activity |
| XAG | False | 17 | 18272 | 4570.3891 | 1.8873 | 0 | 0.0775 | sample;liquidity;activity |
| PENGU | False | 17 | 18229 | 4346.2797 | 7.9166 | 0 | 0.0774 | sample;liquidity;activity |
| MON | False | 17 | 18215 | 4188.4645 | 7.4762 | 0 | 0.0773 | sample;liquidity;activity |
| PLATINUM | False | 17 | 18345 | 4133.6842 | 15.0034 | 0 | 0.0772 | sample;liquidity;activity |

## Output files

- `symbol_eligibility.csv` — one row per symbol with gate metrics and failure reasons.
- `eligible_symbols.csv` — subset that passed all gates.
- `gate_counts.csv` — count of symbols passing each gate.
- `thresholds.csv` — fixed thresholds used for this run.

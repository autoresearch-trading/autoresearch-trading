# Pacifica Walk-Forward Validation

Verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`

Failure reasons: `insufficient_distinct_days;no_purged_validation_windows`

This is a strategy-neutral, non-HFT validation harness. It uses purged chronological windows and random same-frequency controls before any future strategy result can be discussed as evidence. Do not treat this as an edge claim unless sample maturity, concentration, economics, baseline, and control gates all pass.

## Interpretation discipline

- `INSUFFICIENT_SAMPLE_DIAGNOSTIC`: plumbing/sample diagnostic only.
- `EARLY_SANITY_ONLY`: at least 10 distinct days, but below provisional maturity.
- `PROVISIONAL_PASS` / `PROVISIONAL_FAIL`: at least 30 distinct days.
- `VALIDATION_GRADE_PASS` / `VALIDATION_GRADE_FAIL`: at least 60 distinct days.

## Summary

| metric | value |
| --- | --- |
| observations | 16 |
| distinct_days | 8 |
| distinct_symbols | 2 |
| max_day_concentration | 0.1250 |
| max_symbol_concentration | 0.5000 |
| distinct_oos_days | 0 |
| distinct_oos_symbols | 0 |
| max_oos_day_concentration | nan |
| max_oos_symbol_concentration | nan |
| max_window_day_concentration | nan |
| max_window_symbol_concentration | nan |
| validation_windows | 0 |
| scored_windows | 0 |
| total_test_rows | 0 |
| unique_oos_rows | 0 |
| net_pnl_bps | 0 |
| baseline_pnl_bps | 0 |
| excess_vs_baseline_bps | 0 |
| sortino | nan |
| max_drawdown_bps | nan |
| random_control_trials | 0 |
| random_controls_beaten_rate | nan |
| invalid_timestamp_rows | 0 |
| invalid_symbol_rows | 0 |
| invalid_strategy_return_rows | 0 |
| invalid_baseline_return_rows | 0 |
| invalid_eligible_rows | 0 |
| invalid_required_rows | 0 |
| filtered_ineligible_rows | 0 |

## Config

| train_days | test_days | purge_days | step_days | min_diagnostic_days | min_provisional_days | min_validation_grade_days | min_test_rows | max_day_concentration | max_symbol_concentration | random_control_trials | random_seed | min_random_control_beaten_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 7 | 1 | 7 | 10 | 30 | 60 | 20 | 0.2500 | 0.5000 | 100 | 17 | 0.5000 |

## Window scorecard preview

_No rows._

## Random same-frequency controls

_No rows._

## Artifacts

- `summary.csv`
- `config.csv`
- `windows.csv`
- `window_scorecard.csv`
- `random_controls.csv`

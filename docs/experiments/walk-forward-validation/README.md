# Pacifica Walk-Forward Validation

Verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`

Failure reasons: `insufficient_oos_distinct_days;no_oos_validation_rows;no_purged_validation_windows`

This is a strategy-neutral, non-HFT diagnostic validation harness. It uses purged chronological windows, OOS-only verdict maturity, random same-frequency controls, dumb baseline scorecards, day/symbol/hour concentration gates, invalid-input accounting, and fail-closed CLI semantics. No alpha claims: a PASS verdict is only permission to keep investigating a pre-registered strategy result, not evidence to trade live.

## Interpretation discipline

- `INSUFFICIENT_SAMPLE_DIAGNOSTIC`: plumbing/sample diagnostic only.
- `EARLY_SANITY_ONLY`: OOS-only verdict with at least 10 distinct OOS days, but below provisional maturity.
- `PROVISIONAL_PASS` / `PROVISIONAL_FAIL`: OOS-only verdict with at least 30 distinct OOS days.
- `VALIDATION_GRADE_PASS` / `VALIDATION_GRADE_FAIL`: OOS-only verdict with at least 60 distinct OOS days.

Thresholds are fixed in this harness and must not be tuned on the current sample.

## Summary

| metric | value |
| --- | --- |
| observations | 16 |
| distinct_days | 8 |
| distinct_symbols | 2 |
| max_day_concentration | 0.1250 |
| max_symbol_concentration | 0.5000 |
| max_hour_concentration | 1 |
| distinct_oos_days | 0 |
| distinct_oos_symbols | 0 |
| max_oos_day_concentration | nan |
| max_oos_symbol_concentration | nan |
| max_oos_hour_concentration | nan |
| max_window_day_concentration | nan |
| max_window_symbol_concentration | nan |
| max_window_hour_concentration | nan |
| validation_windows | 0 |
| scored_windows | 0 |
| total_test_rows | 0 |
| unique_oos_rows | 0 |
| net_pnl_bps | 0 |
| baseline_pnl_bps | 0 |
| excess_vs_baseline_bps | 0 |
| baseline_count | 1 |
| baseline_columns | baseline_return_bps |
| worst_baseline_column | baseline_return_bps |
| min_excess_vs_any_baseline_bps | 0 |
| sortino | nan |
| max_drawdown_bps | nan |
| random_control_trials | 0 |
| random_controls_beaten_rate | nan |
| invalid_timestamp_rows | 0 |
| invalid_symbol_rows | 0 |
| invalid_strategy_return_rows | 0 |
| invalid_baseline_return_rows | 0 |
| invalid_optional_baseline_return_rows | 0 |
| invalid_eligible_rows | 0 |
| invalid_required_rows | 0 |
| filtered_ineligible_rows | 0 |

## Config

| train_days | test_days | purge_days | step_days | min_diagnostic_days | min_provisional_days | min_validation_grade_days | min_test_rows | max_day_concentration | max_symbol_concentration | max_hour_concentration | random_control_trials | random_seed | min_random_control_beaten_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 7 | 1 | 7 | 10 | 30 | 60 | 20 | 0.2500 | 0.5000 | 0.5000 | 100 | 17 | 0.5000 |

## Window scorecard preview

_No rows._

## Dumb baseline scorecard

| baseline_column | oos_rows | strategy_pnl_bps | baseline_pnl_bps | excess_vs_baseline_bps | baseline_sortino | baseline_max_drawdown_bps |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_return_bps | 0 | 0 | 0 | 0 | nan | nan |

## Random same-frequency controls

_No rows._

## Artifacts

- `summary.csv`
- `config.csv`
- `windows.csv`
- `window_scorecard.csv`
- `baseline_scorecard.csv`
- `random_controls.csv`

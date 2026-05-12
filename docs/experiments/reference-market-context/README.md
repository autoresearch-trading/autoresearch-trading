# Cross-venue/reference market context

Verdict: `NO_ROWS_DIAGNOSTIC`

This is a diagnostic context layer, not a trade signal and not permission to paper/live trade.
It distinguishes Pacifica-local observations from broad crypto/reference-market states using pluggable local CSV/parquet inputs.

## Interpretation discipline

- Missing reference rows are flagged as `MISSING_REFERENCE`; they are not imputed.
- Broad risk states are fixed diagnostic labels, not optimized strategy parameters.
- Cross-venue premium/discount and funding divergence require external source-quality review before use in any strategy.

## Risk-state summary

_No rows._

## Symbol reference summary

_No rows._

## Config

| risk_on_return_bps | risk_off_return_bps | high_vol_bps | positive_premium_bps | negative_premium_bps | reference_context_version |
| --- | --- | --- | --- | --- | --- |
| 10 | -10 | 40 | 5 | -5 | pacifica_reference_context_v1_fixed_diagnostic |

## Artifacts

- `reference_context.csv`
- `risk_state_summary.csv`
- `symbol_reference_summary.csv`
- `config.csv`

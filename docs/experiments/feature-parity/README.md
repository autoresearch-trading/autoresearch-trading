# Pacifica Feature Parity

Verdict: `PARITY_FAIL_DIAGNOSTIC`

This diagnostic compares offline rebuilt features against online/offline
live-style feature snapshots. It does not authorize trading, does not claim edge,
and should gate any future use of live microbatch features in strategy adapters.

## Required metadata

available_ts, computed_at, watermark_ts, feature_version, provisional_final_flag

## Summary

| verdict | compared_rows | compared_features | mismatch_count | missing_key_count | version_mismatch_count | metadata_mismatch_count | invalid_metadata_count | invalid_feature_count | invalid_key_count | duplicate_key_count | missing_metadata_columns | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PARITY_FAIL_DIAGNOSTIC | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | offline.available_ts;offline.computed_at;offline.watermark_ts;offline.feature_version;offline.provisional_final_flag;live.available_ts;live.computed_at;live.watermark_ts;live.feature_version;live.provisional_final_flag | missing_required_columns |

## Compared feature columns

| feature |
| --- |
| avg_spread_bps |
| top_depth_notional |
| toxicity_score |

## Mismatches

_No rows._

## Missing keys

_No rows._

## Version mismatches

_No rows._

## Metadata mismatches

_No rows._

## Invalid metadata

_No rows._

## Invalid feature values

_No rows._

## Invalid keys

_No rows._

## Duplicate keys

_No rows._

## Interpretation discipline

- `PARITY_PASS_DIAGNOSTIC` only means the compared artifacts matched within tolerance.
- It is not a strategy verdict and does not override eligibility or sample-maturity gates.
- `PARITY_FAIL_DIAGNOSTIC` blocks live feature use until mismatches are explained or fixed.

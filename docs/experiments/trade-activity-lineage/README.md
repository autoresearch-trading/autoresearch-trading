# Pacifica Trade Activity Lineage Audit

This is a diagnostic audit of raw trades -> silver trades -> regime trade_count/trade_notional -> eligibility activity metrics.
It is not a strategy, alpha claim, or paper-trading permission.

Verdict: `LINEAGE_AUDIT_FAIL_DIAGNOSTIC`
Symbols audited: 10
Sparse-trade zero-median explanations: 0

## Interpretation discipline

The paper-trading eligibility activity gate currently uses the median 1-minute `trade_notional` over all regime rows for a symbol. If most otherwise-observed BBO/price/book minutes have no trades, that median can be zero even when raw and silver trades are present and correctly aggregated into regime rows.

A PASS here only means the inspected trade-activity lineage is internally consistent for the audited symbols. It does not make an edge claim and does not authorize trading.

## Failure counters

| counter | symbols |
| --- | --- |
| raw/silver mismatches | 10 |
| silver/regime trade-count mismatches | 0 |
| sparse-trade zero-median explanations | 0 |
| unexplained zero medians | 0 |

If raw/silver mismatches appear while silver/regime mismatches are zero, the first suspect is stale research artifacts: local raw cache contains trades that the current silver/regime/eligibility reports have not ingested yet. Refresh silver and regime before treating eligibility as current.

## Inputs

- Raw dir: `data/pacifica_full_fidelity`
- Silver dir: `data/pacifica_silver_partitioned`
- Regime state: `docs/experiments/non-hft-regime-state/regime_state.parquet`
- Eligibility: `docs/experiments/paper-trading-eligibility/symbol_eligibility.csv`
- Bucket: `1min`
- Target symbols: `BTC, ETH, SOL, WLFI, kPEPE, CRV, PUMP, XRP, 2Z, AVAX`

## Symbol summary preview

| symbol | raw_trade_rows | silver_trade_rows | regime_trade_count_sum | raw_silver_row_delta | silver_regime_trade_count_delta | regime_trade_active_row_share | regime_trade_notional_median_all_rows | median_trade_notional_per_min | activity_median_zero_reason | diagnostic_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2Z | 2041 | 1472 | 1472 | 569 | 0 | 0.0929 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| AVAX | 2030 | 1451 | 1451 | 579 | 0 | 0.1026 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| BTC | 11796 | 8606 | 8606 | 3190 | 0 | 0.0927 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| CRV | 1822 | 1261 | 1261 | 561 | 0 | 0.1041 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| ETH | 7506 | 5309 | 5309 | 2197 | 0 | 0.0864 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| PUMP | 2035 | 1320 | 1320 | 715 | 0 | 0.0772 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| SOL | 5834 | 2960 | 2960 | 2874 | 0 | 0.0737 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| WLFI | 6644 | 852 | 852 | 5792 | 0 | 0.0797 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| XRP | 12389 | 843 | 843 | 11546 | 0 | 0.0722 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |
| kPEPE | 9401 | 1042 | 1042 | 8359 | 0 | 0.0903 | 0 | 0 | sparse_trade_minutes_with_raw_silver_gap | raw_silver_trade_row_mismatch;raw_silver_notional_mismatch |

## Output files

- `symbol_summary.csv` — one row per audited symbol with raw/silver/regime/eligibility deltas and diagnostic notes.
- `date_summary.csv` — one row per audited symbol/date for locating lineage breaks by day.
- `README.md` — this diagnostic report.

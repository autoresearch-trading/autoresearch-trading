# Pacifica Trade Activity Lineage Audit

This is a diagnostic audit of raw trades -> silver trades -> regime trade_count/trade_notional -> eligibility activity metrics.
It is not a strategy, alpha claim, or paper-trading permission.

Verdict: `LINEAGE_AUDIT_PASS_DIAGNOSTIC`
Symbols audited: 10
Sparse-trade zero-median explanations: 8

## Interpretation discipline

The paper-trading eligibility activity gate currently uses the median 1-minute `trade_notional` over all regime rows for a symbol. If most otherwise-observed BBO/price/book minutes have no trades, that median can be zero even when raw and silver trades are present and correctly aggregated into regime rows.

A PASS here only means the inspected trade-activity lineage is internally consistent for the audited symbols. It does not make an edge claim and does not authorize trading.

## Failure counters

| counter | symbols |
| --- | --- |
| raw/silver mismatches | 0 |
| silver/regime trade-count mismatches | 0 |
| sparse-trade zero-median explanations | 8 |
| unexplained zero medians | 0 |

If raw/silver mismatches appear while silver/regime mismatches are zero, the first suspect is stale research artifacts: local raw cache contains trades that the current silver/regime/eligibility reports have not ingested yet. Refresh silver and regime before treating eligibility as current.

## Inputs

- Raw dir: `data/pacifica_full_fidelity`
- Silver dir: `data/pacifica_silver_partitioned`
- Regime state: `docs/experiments/non-hft-regime-state/regime_state.parquet`
- Eligibility: `docs/experiments/paper-trading-eligibility/symbol_eligibility.csv`
- Bucket: `1min`
- Target symbols: `BTC, ETH, SOL, WLFI, CRV, PUMP, kPEPE, XRP, AVAX, BNB`

## Symbol summary preview

| symbol | raw_trade_rows | silver_trade_rows | regime_trade_count_sum | raw_silver_row_delta | silver_regime_trade_count_delta | regime_trade_active_row_share | regime_trade_notional_median_all_rows | median_trade_notional_per_min | activity_median_zero_reason | diagnostic_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVAX | 11077 | 11077 | 11077 | 0 | 0 | 0.4369 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| BNB | 11990 | 11990 | 11990 | 0 | 0 | 0.4389 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| BTC | 56345 | 56345 | 56345 | 0 | 0 | 0.5642 | 16.4833 | 16.4833 | median_nonzero |  |
| CRV | 9653 | 9653 | 9653 | 0 | 0 | 0.4162 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| ETH | 36988 | 36988 | 36988 | 0 | 0 | 0.4935 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| PUMP | 11614 | 11614 | 11614 | 0 | 0 | 0.4540 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| SOL | 32116 | 32116 | 32116 | 0 | 0 | 0.4862 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| WLFI | 9003 | 9003 | 9003 | 0 | 0 | 0.4170 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| XRP | 15862 | 15862 | 15862 | 0 | 0 | 0.4822 | 0 | 0 | sparse_trade_minutes_not_missing_trade_lineage | median_zero_explained_by_sparse_trade_minutes |
| kPEPE | 11848 | 11848 | 11848 | 0 | 0 | 0.5281 | 0.0037 | 0.0037 | median_nonzero |  |

## Output files

- `symbol_summary.csv` — one row per audited symbol with raw/silver/regime/eligibility deltas and diagnostic notes.
- `date_summary.csv` — one row per audited symbol/date for locating lineage breaks by day.
- `README.md` — this diagnostic report.

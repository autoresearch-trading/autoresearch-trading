# Pacifica Symbol Lifecycle

Verdict: `NO_ELIGIBLE_SYMBOLS_DIAGNOSTIC`

This is a diagnostic lifecycle layer. It turns static eligibility gates into promotion/demotion states for the collect-broad/trade-selectively universe policy. It does not authorize live trading. `paper_trading_allowed_diagnostic=True` only means the symbol passed this diagnostic lifecycle snapshot and still requires the broader paper ledger, governor, parity, walk-forward, portfolio, and maturity gates.

## State counts

| lifecycle_state | symbols |
| --- | --- |
| COLLECTED | 0 |
| RESEARCHABLE | 0 |
| ELIGIBLE | 0 |
| PROBATION | 0 |
| DISABLED | 65 |
| RETIRED | 0 |

## Transition counts

| transition | symbols |
| --- | --- |
| NEW->DISABLED | 65 |

## Lifecycle preview

| symbol | previous_state | lifecycle_state | transition | reason_codes | paper_trading_allowed_diagnostic | n_days | n_observations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2Z | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 9254 |
| AAVE | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8488 |
| ADA | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7985 |
| ARB | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7737 |
| ASTER | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7627 |
| AVAX | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8419 |
| BCH | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7379 |
| BNB | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8495 |
| BP | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;spread_too_wide;insufficient_activity;unstable_feed | False | 8 | 4834 |
| BTC | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 9162 |
| CHIP | NEW | DISABLED | NEW->DISABLED | insufficient_days;spread_too_wide;insufficient_activity | False | 8 | 8429 |
| CL | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 8371 |
| COPPER | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 5874 |
| CRCL | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 8157 |
| CRV | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 7890 |
| DOGE | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 8576 |
| ENA | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 9040 |
| ETH | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 9237 |
| EURUSD | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 5258 |
| FARTCOIN | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8931 |
| GOOGL | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 6712 |
| HOOD | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7848 |
| HYPE | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 9028 |
| ICP | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8375 |
| JUP | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 8138 |
| LDO | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 8291 |
| LINK | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 7918 |
| LIT | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 9055 |
| LTC | NEW | DISABLED | NEW->DISABLED | insufficient_days;insufficient_activity | False | 8 | 7446 |
| MEGA | NEW | DISABLED | NEW->DISABLED | insufficient_days;depth_too_thin;insufficient_activity | False | 8 | 8713 |

## Artifacts

- `symbol_lifecycle.csv`
- `state_counts.csv`
- `transitions.csv`
- `config.csv`

# Pacifica Execution Simulator

Verdict: `DIAGNOSTIC_ACCOUNTING_SPINE`

This artifact defines reusable execution-cost accounting for future Pacifica
non-HFT research. It is not a strategy, not a backtest result, and not evidence
of edge. It exists so future probes, baselines, and paper-ledger runs all charge
fees, slippage/adverse selection, and funding consistently.

## Locked default assumptions

| taker_fee_bps | maker_fee_bps | funding_bps_per_hour | default_slippage_bps_per_side | default_adverse_selection_bps_per_side |
| --- | --- | --- | --- | --- |
| 4 | 1.5000 | 0 | 0 | 0 |

Interpretation:

- taker/taker round trip before slippage: `8` bps
- maker/maker round trip before adverse selection: `3` bps
- taker/maker round trip before extra costs: `5.5` bps

## Example diagnostic round trips

| symbol | side | entry_liquidity | exit_liquidity | entry_price | exit_price | quantity | initial_notional | gross_pnl | fee_bps_total | fees_paid | slippage_paid | adverse_selection_paid | funding_paid | net_pnl | net_return_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | long | taker | taker | 100 | 101 | 10 | 1000 | 10 | 8 | 0.8000 | 0.4000 | 0 | 0 | 8.8000 | 88 |
| ETH | short | maker | maker | 100 | 99 | 5 | 500 | 5 | 3 | 0.1500 | 0 | 0.3000 | 0 | 4.5500 | 91 |
| SOL | long | taker | maker | 100 | 100.5000 | 20 | 2000 | 10 | 5.5000 | 1.1000 | 0 | 0 | 0 | 8.9000 | 44.5000 |

## Required use

Any future strategy, backtest, or paper-trading ledger should call this simulator
or preserve equivalent semantics before reporting post-cost PnL, Sortino,
drawdown, or baseline deltas.

## Not yet included

- order-book-depth-aware partial fills;
- live position accounting;
- portfolio exposure caps;
- random same-frequency controls;
- chronological walk-forward validation.

Those belong in the next phases of the system level-up plan.

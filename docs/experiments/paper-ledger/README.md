# Pacifica Paper Ledger

Verdict: `DIAGNOSTIC_LEDGER_SPINE`

This is a strategy-neutral accounting artifact for future Pacifica non-HFT paper
research. It does not permit live trading and does not claim edge. Its job is to
make every future strategy account for fills, fees, funding, realized PnL,
equity, and drawdown in one reusable ledger path.

## Summary

| starting_cash | realized_pnl | fees_paid | funding_paid | net_pnl | ending_equity | max_drawdown | fills | symbols |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 7 | 0.1915 | 0.2500 | 6.5584 | 1006.5584 | -0.2650 | 5 | 2 |

## Example fills

| ts_ms | symbol | side | price | quantity | liquidity | funding_payment | notional | fee_paid | realized_pnl_on_fill | position_after_fill | cumulative_realized_pnl | cumulative_fees_paid | cumulative_funding_paid | cumulative_net_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BTC | buy | 100 | 2 | taker | 0 | 200 | 0.0800 | 0 | 2 | 0 | 0.0800 | 0 | -0.0800 |
| 2 | BTC | sell | 101 | 2 | taker | 0 | 202 | 0.0808 | 2 | 0 | 2 | 0.1608 | 0 | 1.8392 |
| 3 | SOL | buy | 10 | 10 | maker | 0 | 100 | 0.0150 | 0 | 10 | 2 | 0.1758 | 0 | 1.8242 |
| 4 | SOL | funding | 10 | 10 | taker | 0.2500 | 100 | 0 | 0 | 10 | 2 | 0.1758 | 0.2500 | 1.5742 |
| 5 | SOL | sell | 10.5000 | 10 | maker | 0 | 105 | 0.0158 | 5 | 0 | 7 | 0.1915 | 0.2500 | 6.5584 |

## Open positions after example run

| symbol | quantity | avg_cost |
| --- | --- | --- |
| BTC | 0 | 0 |
| SOL | 0 | 0 |

## Required next integrations

- wire strategy candidate outputs into this ledger;
- require eligible-symbol gates before non-diagnostic paper fills;
- add unrealized PnL from mark prices;
- add exposure and concentration reports;
- add random same-frequency controls through the walk-forward validation harness.

# Pacifica Paper-Trading Economics and Baselines

Updated: 2026-05-05

This is the pre-strategy economics/baseline contract for the non-HFT Pacifica paper-trading program. It is not an alpha claim or a backtest.

## Status

Diagnostic. The full-fidelity archive is still young, so all strategy research remains plumbing/sanity-check work until day/sample gates pass.

## Fee assumptions

Use Pacifica Tier 1 unless a live account can prove a better tier at execution time.

Source: `docs/research/pacifica-fee-schedule-2026-04-27.md`.

| Execution path | Entry fee | Exit fee | Round-trip fee |
| --- | ---: | ---: | ---: |
| taker -> taker | 4.0 bp | 4.0 bp | 8.0 bp |
| taker -> maker | 4.0 bp | 1.5 bp | 5.5 bp |
| maker -> maker | 1.5 bp | 1.5 bp | 3.0 bp |

Operational default for conservative research: taker->taker unless the simulated order type explicitly uses post-only maker logic and models non-fill/adverse-selection.

## Slippage and adverse-selection assumptions

Every paper/backtest report must show at least these scenarios:

| Scenario | Required cost model |
| --- | --- |
| Conservative taker | 8.0 bp round-trip fees + 1.0 bp slippage per side = 10.0 bp round-trip minimum |
| Maker diagnostic | 3.0 bp round-trip fees + 1.0 bp adverse-selection haircut per side = 5.0 bp round-trip minimum |
| Stress | Double the base slippage/adverse-selection haircut |

Maker fills must not be assumed free alpha. A post-only fill model must track fill probability, missed trades, and post-fill adverse drift. If those are absent, maker results are diagnostic only.

## Funding

Funding must be included when position holding windows cross funding timestamps or when the strategy explicitly holds beyond intraday buckets. Reports must state whether funding is ignored, approximated, or pulled from Pacifica public fields.

## Required baselines

No strategy result is acceptable without these controls:

1. Flat no-trade baseline.
2. Random same-frequency entry/side baseline by symbol and day.
3. Direction-shuffled baseline preserving timestamps and holding periods.
4. Always-long / always-short exposure proxy where applicable.
5. Toxicity-overlay-only no-trade baseline before adding alpha logic.
6. Same-symbol, same-day cost-only break-even threshold.

## Required acceptance checks

A candidate strategy must report:

- net PnL after fees, slippage, funding, and adverse-selection assumptions;
- Sortino and drawdown;
- number of trades;
- number of distinct trading days;
- number of distinct symbols;
- top-symbol PnL/trade concentration;
- top-day PnL/trade concentration;
- performance versus every required baseline;
- eligibility-gated universe used at decision time.

## Hard reject conditions

Reject any result if:

- it uses symbols that fail the paper-trading eligibility gates;
- it claims edge on diagnostic sample age only;
- one day or one symbol dominates without being pre-registered;
- costs are omitted or only shown gross;
- maker fills are assumed without non-fill/adverse-selection modeling;
- it depends on latency, queue-position, next-tick, or high-turnover taker execution.

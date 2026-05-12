# Event/calendar risk layer

Verdict: `NO_EVENTS_CONFIGURED_DIAGNOSTIC`

This is a diagnostic context/governor layer, not a trade signal and not permission to paper/live trade.
It marks known event-risk windows from local CSV/parquet calendar input so future strategies can skip, reduce, or annotate event windows without post-hoc tuning.

## Interpretation discipline

- Event windows come from an explicit local calendar; no hidden external API is queried here.
- `NO_KNOWN_EVENT_RISK` only means no configured event window matched the row, not that the market is safe.
- Event severity labels are fixed diagnostic annotations, not optimized alpha parameters.

## Event-risk summary

| event_risk_state | rows | row_share |
| --- | --- | --- |
| NO_KNOWN_EVENT_RISK | 519903 | 1 |

## Symbol event-risk summary

| symbol | rows | event_risk_rows | max_event_severity | event_risk_rate |
| --- | --- | --- | --- | --- |
| 2Z | 9254 | 0 | NONE | 0 |
| ETH | 9237 | 0 | NONE | 0 |
| ZEC | 9205 | 0 | NONE | 0 |
| BTC | 9162 | 0 | NONE | 0 |
| LIT | 9055 | 0 | NONE | 0 |
| SOL | 9049 | 0 | NONE | 0 |
| ENA | 9040 | 0 | NONE | 0 |
| HYPE | 9028 | 0 | NONE | 0 |
| FARTCOIN | 8931 | 0 | NONE | 0 |
| TAO | 8926 | 0 | NONE | 0 |
| kPEPE | 8904 | 0 | NONE | 0 |
| PENGU | 8854 | 0 | NONE | 0 |
| MON | 8776 | 0 | NONE | 0 |
| ZRO | 8745 | 0 | NONE | 0 |
| PUMP | 8742 | 0 | NONE | 0 |
| MEGA | 8713 | 0 | NONE | 0 |
| kBONK | 8698 | 0 | NONE | 0 |
| DOGE | 8576 | 0 | NONE | 0 |
| WLFI | 8534 | 0 | NONE | 0 |
| SUI | 8527 | 0 | NONE | 0 |
| BNB | 8495 | 0 | NONE | 0 |
| AAVE | 8488 | 0 | NONE | 0 |
| XRP | 8473 | 0 | NONE | 0 |
| CHIP | 8429 | 0 | NONE | 0 |
| AVAX | 8419 | 0 | NONE | 0 |

## Config

| event_risk_version | empty_calendar_verdict |
| --- | --- |
| pacifica_event_risk_v1_fixed_diagnostic | NO_EVENTS_CONFIGURED_DIAGNOSTIC |

## Artifacts

- `event_risk_rows.csv` (generated locally for full row-level inspection; intentionally ignored in git for full-run outputs because it can exceed GitHub blob limits)
- `event_risk_summary.csv`
- `symbol_event_risk_summary.csv`
- `config.csv`

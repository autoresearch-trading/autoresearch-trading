# Pacifica Collection Coverage Audit

**Purpose:** validate whether this repo is collecting all market-data datapoints Pacifica publicly provides.  
**Scope:** public market-data REST/WebSocket docs and local raw `data/trades` / `data/orderbook` partitions. Private account streams are intentionally out of scope.

## Executive verdict

The repo is collecting the two core high-frequency streams needed for tape research — trades and orderbook — across the expanded raw universe with near-complete symbol/date coverage through full days ending 2026-04-26.

It is **not** collecting every public Pacifica datapoint/field. Missing or lossy relative to public docs:

- `prices` stream / `/info/prices`: funding, next funding, mark, oracle, open interest, 24h volume, yesterday price.
- `bbo` stream raw fields: best bid/ask can be derived from orderbook, but order id `i` and exchange nonce `li` are not stored.
- `trades` stream raw fields: local trades do not preserve documented history id `h` or exchange-wide nonce `li`; local `trade_id` appears blank in sampled files.
- `book` stream raw fields: local orderbook does not preserve documented level order-count `n` or exchange-wide nonce `li`; `agg_level` is present but sampled values are null.
- `candle` stream: derivable from trades, but not stored as raw candles.
- `mark_price_candle` stream: not stored and not derivable without mark-price data.
- REST funding history is available in API docs but no local funding-history partition was found.

So: **good raw tape capture, not complete Pacifica market-data capture.**

## Local partition coverage

| dataset | symbols | symbol/date partitions | min_date | max_date | >=100-day symbols | <=30-day symbols | first-date cohorts |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| trades | 63 | 5349 | 2025-10-16 | 2026-04-27 | 25 | 38 | {'2025-10-16': 25, '2026-04-03': 38} |
| orderbook | 63 | 5346 | 2025-10-16 | 2026-04-27 | 25 | 38 | {'2025-10-16': 25, '2026-04-03': 38} |

## Trade/orderbook pair mismatches

Trades without orderbook: 4

| symbol | date |
| --- | --- |
| LDO | 2026-04-05 |
| MEGA | 2026-04-05 |
| UNI | 2026-04-05 |
| WLFI | 2026-04-27 |

Orderbook without trades: 1

| symbol | date |
| --- | --- |
| PAXG | 2026-04-27 |

## Live Pacifica symbol comparison

Live `/info` symbols: 65  
Live `/info/prices` symbols: 65  
Local raw symbols: 63

Live symbols not present in local raw data: 4

| symbol | local_raw | live_info | live_prices |
| --- | --- | --- | --- |
| CHIP | False | True | True |
| SOL-USDC | False | True | True |
| kBONK | False | True | True |
| kPEPE | False | True | True |

Local raw symbols not present in current live REST data: 2

| symbol | local_raw | live_info | live_prices |
| --- | --- | --- | --- |
| KBONK | True | False | False |
| KPEPE | True | False | False |

## Public websocket stream coverage

| stream | captured_locally | fields | local_status |
| --- | --- | --- | --- |
| prices | False | funding, mark, mid, next_funding, open_interest, oracle, symbol, timestamp, volume_24h, yesterday_price | No local data/prices partitions found. |
| book | True | l, a, n, p, s, t, li | Local data/orderbook stores ts_ms, symbol, bids/asks price+qty, recv_ms, agg_level; it does not preserve documented order-count n or exchange nonce li. |
| bbo | False | s, i, li, t, b, B, a, A | Best bid/ask can be derived from local orderbook snapshots, but the raw BBO stream fields order id i and last order id li are not stored. |
| trades | True | h, s, a, p, d, tc, t, li | Local data/trades stores ts_ms, symbol, trade_id, side, qty, price, recv_ms, cause, event_type, date; April+ cause/event_type are present, but documented history id h and exchange nonce li are not preserved. |
| candle | False | t, T, s, i, o, c, h, l, v, n | Trade candles are derivable from trades but raw candle stream is not stored. |
| mark_price_candle | False | t, T, s, i, o, c, h, l, v, n | Not derivable from local trades/orderbook without mark-price data; no local mark-price candle partitions found. |

## Sample local schemas

### Trades

| file | columns |
| --- | --- |
| data/trades/symbol=BTC/date=2026-04-13/trades-20260413T033511887024.parquet | ts_ms, symbol, trade_id, side, qty, price, recv_ms, cause, event_type, date |
| data/trades/symbol=ADA/date=2026-04-13/trades-20260413T033512287071.parquet | ts_ms, symbol, trade_id, side, qty, price, recv_ms, cause, event_type, date |
| data/trades/symbol=BTC/date=2025-11-01/trades-20251101T000132953930.parquet | ts_ms, symbol, trade_id, side, qty, price, recv_ms, date |

### Orderbook

| file | columns |
| --- | --- |
| data/orderbook/symbol=BTC/date=2026-04-13/orderbook-20260413T033511879570.parquet | ts_ms, symbol, bids, asks, recv_ms, agg_level, date |
| data/orderbook/symbol=ADA/date=2026-04-13/orderbook-20260413T033512205753.parquet | ts_ms, symbol, bids, asks, recv_ms, agg_level, date |
| data/orderbook/symbol=BTC/date=2025-11-01/orderbook-20251101T000413486124.parquet | ts_ms, symbol, bids, asks, recv_ms, agg_level, date |

## Liquidation cause availability spot-check

| date | cause | n | symbols |
| --- | --- | --- | --- |
| 2026-04-06 | normal | 818933 | 63 |
| 2026-04-06 | market_liquidation | 83 | 9 |
| 2026-04-13 | normal | 1287449 | 63 |
| 2026-04-13 | market_liquidation | 158 | 12 |
| 2026-04-14 | normal | 555959 | 63 |
| 2026-04-14 | market_liquidation | 74 | 4 |
| 2026-04-26 | normal | 304976 | 63 |
| 2026-04-26 | market_liquidation | 24 | 2 |
| 2026-04-27 | normal | 40 | 1 |

## Artifacts

- `trades_by_symbol.csv`
- `orderbook_by_symbol.csv`
- `trades_by_date.csv`
- `orderbook_by_date.csv`
- `live_symbol_comparison.csv`
- `april_cause_counts.csv`
- `public_ws_streams.json`

## Recommended collector fixes if the goal is complete Pacifica public market-data capture

1. Add a raw `prices` capture table/partition keyed by timestamp and symbol.
2. Preserve raw websocket fields exactly before transformation:
   - trades: `h`, `s`, `a`, `p`, `d`, `tc`, `t`, `li`, plus local `recv_ms`.
   - book: raw `l` levels including amount `a`, count `n`, price `p`, symbol `s`, timestamp `t`, exchange nonce `li`, aggregation level.
3. Decide whether to store BBO as a separate raw stream. It is partly derivable, but raw `i` and `li` are not.
4. Capture funding history / funding snapshots if any funding, carry, mark/oracle, or liquidation-economics work will continue.
5. Capture mark-price candles or mark-price snapshots if mark/oracle dislocation becomes a hypothesis.
6. Update the derived feature cache/constants separately; raw collection already has a wider universe than `tape.constants.SYMBOLS`.

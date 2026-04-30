# Pacifica full-fidelity market-data archival

This repo now has a raw archival collector for public Pacifica market data.

## What it captures

The collector stores gzip-compressed JSONL before lossy normalization under:

`data/pacifica_full_fidelity/`

It covers public market-data streams only:

- `prices`, global stream
- `trades`, per symbol
- `book`, per symbol and aggregation level
- `bbo`, per symbol
- `candle`, per symbol and interval
- `mark_price_candle`, per symbol and interval
- REST snapshots for `/info` and `/info/prices`

Private/account streams are intentionally excluded.

## Why JSONL.GZ instead of parquet first

The existing `data/trades` and `data/orderbook` parquet tables are research-friendly but lossy: they do not preserve all raw Pacifica fields such as `h`, `li`, `n`, BBO order ids, and raw message envelopes.

The archival collector writes raw event records first so we can later build derived parquet/cache tables without losing exchange IDs, nonces, order-counts, mark/oracle/funding fields, or raw payload shape.

## Run manually

From repo root:

```bash
uv sync
uv run python scripts/collect_pacifica_full_fidelity.py \
  --out-dir data/pacifica_full_fidelity
```

By default this fetches live symbols from `https://api.pacifica.fi/api/v1/info` and subscribes to all documented public market streams/intervals.

For a small smoke test:

```bash
uv run python scripts/collect_pacifica_full_fidelity.py \
  --symbols BTC,ETH \
  --intervals 1m \
  --agg-levels 1 \
  --out-dir data/pacifica_full_fidelity_smoke \
  --print-plan
```

Stop with Ctrl-C.

## macOS launchd setup

A launchd plist template is included at:

`ops/launchd/com.non-toxic.pacifica-full-fidelity.plist`

Install/start it with:

```bash
mkdir -p logs data/pacifica_full_fidelity
launchctl bootstrap "gui/$(id -u)" ops/launchd/com.non-toxic.pacifica-full-fidelity.plist
launchctl enable "gui/$(id -u)/com.non-toxic.pacifica-full-fidelity"
launchctl kickstart -k "gui/$(id -u)/com.non-toxic.pacifica-full-fidelity"
```

Check status/logs:

```bash
launchctl print "gui/$(id -u)/com.non-toxic.pacifica-full-fidelity"
tail -f logs/pacifica-full-fidelity.out.log logs/pacifica-full-fidelity.err.log
```

Stop/uninstall:

```bash
launchctl bootout "gui/$(id -u)" ops/launchd/com.non-toxic.pacifica-full-fidelity.plist
```

## Output layout

Websocket event rows:

```text
data/pacifica_full_fidelity/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/<run_id>.jsonl.gz
```

REST snapshots:

```text
data/pacifica_full_fidelity/rest/endpoint=<endpoint>/date=<YYYY-MM-DD>/hour=<HH>/<run_id>.jsonl.gz
```

Each websocket row includes:

- `recv_ms`, local receive timestamp
- `event_ts_ms`, Pacifica event timestamp when available
- `channel`
- `symbol`
- `data`, the split event payload
- `raw_message`, the original decoded websocket message
- `raw_text`, the exact websocket text frame

## Operational notes

- The collector sends heartbeat pings every 30 seconds because Pacifica closes idle websocket connections after 60 seconds.
- It reconnects automatically after websocket errors.
- Pacifica documents websocket lifetime as 24 hours; launchd `KeepAlive` restarts the process if it exits.
- Full-fidelity mode is high-cardinality: at 65 symbols and all 11 candle intervals, it creates more than 1,600 subscriptions. If Pacifica rate-limits this, run multiple symbol shards with separate plists or reduce intervals temporarily.
- `data/pacifica_full_fidelity/` should be treated as raw data, not source code. Do not commit captured archives.

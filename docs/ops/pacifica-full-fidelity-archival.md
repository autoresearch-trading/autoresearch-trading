# Pacifica full-fidelity market-data archival

This repo now has a raw archival collector for public Pacifica market data.

## What it captures

The collector stores gzip-compressed JSONL before lossy normalization under:

`data/pacifica_full_fidelity/`

It covers public market-data streams only and is intended to collect the full live Pacifica symbol universe from `/info`, not just the legacy 25-symbol research set:

- `prices`, global stream
- `trades`, per symbol
- `book`, per symbol and aggregation level
- `bbo`, per symbol
- `candle`, per symbol and interval
- `mark_price_candle`, per symbol and interval
- REST snapshots for `/info` and `/info/prices`

The reviewed public REST/API-docs surface also includes `/funding/history` and `/kline`; those are tracked in `docs/ops/pacifica-api-surface-baseline.json` and should be reviewed before expanding collection beyond the current raw collector.

Private/account streams are intentionally excluded.

## Why JSONL.GZ instead of parquet first

The existing `data/trades` and `data/orderbook` parquet tables are research-friendly but lossy: they do not preserve all raw Pacifica fields such as `h`, `li`, `n`, BBO order ids, and raw message envelopes.

The archival collector writes raw event records first so we can later build derived parquet/cache tables without losing exchange IDs, nonces, order-counts, mark/oracle/funding fields, or raw payload shape.

## Run manually

From repo root:

```bash
uv sync
uv run python scripts/collect_pacifica_full_fidelity.py \
  --out-dir data/pacifica_full_fidelity \
  --raw-payload-mode compact \
  --min-free-disk-gb 50
```

By default this fetches live symbols from `https://api.pacifica.fi/api/v1/info` and subscribes to all documented public market streams/intervals. Use the count snippet below when you need the current dynamic symbol/subscription count without printing the full subscription plan.

```bash
uv run python - <<'PY'
from scripts.collect_pacifica_full_fidelity import build_subscriptions, fetch_live_symbols
symbols = fetch_live_symbols()
print('live_symbols=', len(symbols), 'subscriptions=', len(build_subscriptions(symbols)))
PY
```

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

Websocket event rows are hour-partitioned so sealed chunks can be uploaded/pruned independently:

```text
data/pacifica_full_fidelity/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/hour=<HH>/<run_id>.jsonl.gz
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
- compact-mode raw metadata by default: per-event `raw_message`, `raw_text_sha256`, `raw_text_bytes`, and `raw_payload_mode`

Use `--raw-payload-mode full` only for short debugging runs. Full mode writes the complete parsed raw websocket message and raw text frame on every split event row. For batched/global streams like `prices`, that duplicates the full all-symbol frame into every per-symbol row and can consume disk very quickly.

## Operational notes

- The collector sends heartbeat pings every 30 seconds because Pacifica closes idle websocket connections after 60 seconds.
- It reconnects automatically after websocket errors.
- Pacifica documents websocket lifetime as 24 hours; launchd `KeepAlive` restarts the process if it exits.
- The collector refuses to write when free space under `--out-dir` is below `--min-free-disk-gb` so launchd cannot silently fill the machine. The included plist uses a 50 GiB floor. Use `--min-free-disk-gb 0` only for deliberate short smoke/debug runs.
- `scripts/pacifica_full_fidelity_storage.py` is the local/R2 lifecycle control plane. It scans sealed archive files into `data/pacifica_full_fidelity_storage.sqlite`, records deterministic R2 object keys plus SHA-256 hashes, writes JSONL manifests, uploads sealed files through rclone copy semantics, verifies remote object size plus a sibling `.sha256` sidecar, and prunes only rows already marked `verified` in the state DB.
- Raw archive upload uses copy/upload semantics for sealed files, not destructive `rclone sync` from a pruned local directory. R2 should be treated as the durable source of truth; local raw files are a short-retention spool/cache once remote verification exists.
- Recommended lifecycle sequence: scan sealed files, upload/copy to R2, verify remote object size and `.sha256` sidecar against the manifest, mark rows verified, then prune verified local copies after the retention window. Do not prune `sealed` or merely `uploaded` files.
- Full-fidelity mode is high-cardinality. Subscription count is dynamic: `1 + N_symbols * (trades + bbo + book agg levels + 2 * candle intervals)`. With the default 1 book aggregation level and 11 candle intervals, that is `1 + N_symbols * 25`. Use the count snippet above to see the current count. If Pacifica rate-limits this, run multiple symbol shards with separate plists or reduce intervals temporarily.
- `data/pacifica_full_fidelity/` should be treated as raw data, not source code. Do not commit captured archives.
- `scripts/watch_pacifica_api_surface.py` compares Pacifica's public docs/OpenAPI surface against `docs/ops/pacifica-api-surface-baseline.json` and writes `docs/ops/pacifica-api-surface-watch/`. A `CHANGED` verdict is a manual review trigger, not permission to auto-update collector subscriptions.

## Local/R2 lifecycle commands

Scan sealed local archive files into the lifecycle state DB:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --root data/pacifica_full_fidelity \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  --r2-prefix raw/pacifica/full_fidelity \
  scan
```

Run the complete R2 upload/verify workflow. This scans sealed files, uploads each data file and a `.sha256` sidecar with rclone, verifies remote size plus sidecar hash, and marks verified rows in SQLite:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  --remote-base r2:pacifica-trading-data \
  upload-verify
```

For bounded batches while testing:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  --remote-base r2:pacifica-trading-data \
  --limit 100 \
  upload-verify
```

Or run the wrapper, which does scan -> upload-verify -> prune dry-run:

```bash
scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
```

The wrapper is dry-run for local pruning by default. Actual local deletion of already verified files requires:

```bash
PACIFICA_R2_PRUNE_EXECUTE=1 scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
```

A launchd template for periodic dry-run lifecycle execution is included at:

```text
ops/launchd/com.non-toxic.pacifica-full-fidelity-r2-lifecycle.plist
```

Install it only after confirming the rclone remote and retention policy:

```bash
launchctl bootstrap "gui/$(id -u)" ops/launchd/com.non-toxic.pacifica-full-fidelity-r2-lifecycle.plist
launchctl enable "gui/$(id -u)/com.non-toxic.pacifica-full-fidelity-r2-lifecycle"
```

Write a JSONL manifest for audit tooling:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  manifest --out data/pacifica_full_fidelity_manifest.jsonl
```

Manual override for one file after independent remote verification, if a batch verify was interrupted after proving the object durable:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  mark-verified --local-path data/pacifica_full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=13/run-20260501T130000Z.jsonl.gz
```

Dry-run pruning of already remote-verified local files older than 3 days:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  prune --retention-days 3
```

Actually delete only those verified files:

```bash
uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  prune --retention-days 3 --execute
```

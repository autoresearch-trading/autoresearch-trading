# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-02 09:05 EST

## Start here

The active project is an economics-first, non-HFT Pacifica paper-trading program using full-fidelity public market-data archival across the live Pacifica symbol universe.

Fresh-session reading order:

1. `docs/NEXT_SESSION_HANDOFF.md` — this handoff.
2. `AGENTS.md` — canonical repo-level agent instructions.
3. `docs/AGENT_OPERATING_MAP.md` — Hermes/tool/skill map and archived Claude asset notes.

There is no active `CLAUDE.md` and no active root `.claude/` workflow. Hermes is primary. Do not recreate `CLAUDE.md` or route work through Claude Code unless Diego explicitly reverses that decision.

## Current commit/state

Latest functional commits before this handoff doc update:

```text
80244c7 fix: avoid uploading active Pacifica raw chunks
35c7540 chore: raise Pacifica Fly disk guard
436664f feat: run Pacifica ops watchdogs on Fly
05b7625 feat: add Pacifica API surface watcher
b3889ba feat: add Pacifica R2 retention planner
```

Current working tree at the start of this handoff update:

```text
branch: main
working tree: clean before this handoff edit
active Fly deployment: app pacifica-full-fidelity, machine e2862502a76778, region iad, version 7
active Fly image: pacifica-full-fidelity:deployment-01KQMEZKX06KB8VHSRK7SZ9TJS
```

Important latest operational fix:

- The R2 lifecycle had uploaded some current-hour raw chunks while the collector could still append to them, which produced size-mismatch verification errors.
- Commit `80244c7` fixes this by making lifecycle scans skip the current UTC hour partition and by allowing uploaded rows with verification errors to be re-uploaded.
- The fix was deployed to Fly as machine version `7`.

Relevant prior commits:

```text
f5c625a fix: tolerate active Pacifica gzip files in silver build
bc20850 docs: update next session handoff
8a5db43 chore: switch repo agent context to Hermes
```

Included in the Hermes-context switch:

- `AGENTS.md` created as the canonical repo instruction file.
- `CLAUDE.md` removed.
- tracked `.claude/` assets archived under `docs/archive/claude-code-assets/.claude/`.
- `.hermes/` added to `.gitignore` as local Hermes runtime/planning state.
- realtime Pacifica research monitor added:
  - `scripts/watch_pacifica_realtime_research.py`
  - `tests/scripts/test_watch_pacifica_realtime_research.py`
- external/current research note added:
  - `docs/research/2026-05-01-real-time-streaming-research-pass.md`
- non-HFT regime-state and toxic overlay diagnostic reports refreshed from the newer silver snapshot.

## Primary goal

Build a highly profitable paper-trading system. Sortino > 2 is a quality bar, but not the only success criterion.

A candidate strategy must show:

- positive net PnL after fees, slippage, funding, and adverse-selection assumptions;
- Sortino > 2 over a pre-registered paper window;
- enough trades and enough distinct days;
- bounded drawdown;
- no single day dominating total PnL;
- no single symbol dominating total PnL unless explicitly intended;
- performance above dumb baselines and random same-frequency controls.

## Non-HFT constraint

Diego cannot trade HFT. Do not propose latency-arb, next-tick alpha, queue-position edge, or high-turnover taker strategies.

Use full-fidelity/high-frequency data to infer slower states:

- 1m+ toxicity/no-trade regimes;
- mark/oracle/mid dislocations;
- liquidity/spread/depth stress;
- liquidation/forced-flow events;
- funding/OI crowding;
- execution-quality and data-quality filters.

## Universe policy

Collect broadly, trade selectively.

- Raw collection universe: all live public Pacifica symbols from `/info`.
- Research universe: symbols with enough clean full-fidelity data.
- Eligible trading universe: only symbols passing pre-registered liquidity, spread/cost, sample-size, stability, and concentration gates.
- Paper-traded universe: selected subset with portfolio caps.

Do not hard-code symbol counts. Pacifica's live universe changes. Counts like 63, 65, or 66 are snapshots only. Refresh dynamically before operational decisions.

Latest live `/info` check from the prior session:

```text
live_symbols=65
subscriptions=1626
```

## Data/pipeline status

### Raw collector

Script:

- `scripts/collect_pacifica_full_fidelity.py`

Primary active output is now on Fly, not the laptop:

- Fly local spool: `/data/pacifica_full_fidelity/` on machine `e2862502a76778`
- Durable remote archive: `r2:pacifica-trading-data/raw/pacifica/full_fidelity/...`
- Laptop local `data/pacifica_full_fidelity`, `data/pacifica_silver_partitioned`, and `data/pacifica_silver_partitioned_refresh` were explicitly deleted on 2026-05-02 after Diego confirmed he did not want local laptop data/cache retained. Do not assume local raw/silver exists. Rehydrate selected data from R2 when needed.

Docs/config:

- `docs/ops/pacifica-full-fidelity-archival.md`
- `ops/launchd/com.non-toxic.pacifica-full-fidelity.plist`

Storage lifecycle helper:

- `scripts/pacifica_full_fidelity_storage.py`
- collector wrapper: `scripts/run_pacifica_full_fidelity_collector.sh`
- wrapper: `scripts/run_pacifica_full_fidelity_r2_lifecycle.sh`
- health check: `scripts/check_pacifica_full_fidelity_health.py`
- API/docs surface watcher: `scripts/watch_pacifica_api_surface.py`
- Fly-side ops watchdogs: `scripts/run_pacifica_fly_ops_watchdogs.py`
- R2 inventory converter: `scripts/pacifica_r2_inventory.py`
- API/docs surface baseline/report: `docs/ops/pacifica-api-surface-baseline.json`, `docs/ops/pacifica-api-surface-watch/`
- launchd template: `ops/launchd/com.non-toxic.pacifica-full-fidelity-r2-lifecycle.plist`
- always-on Fly deployment docs/config: `docs/ops/pacifica-full-fidelity-fly.md`, `ops/fly/pacifica-full-fidelity/`
- always-on Hetzner/systemd docs/config: `docs/ops/pacifica-full-fidelity-hetzner.md`, `ops/hetzner/`, `ops/systemd/`
- state DB on Fly: `/data/pacifica_full_fidelity_storage.sqlite`
- manifest output: JSONL rows with local path, deterministic R2 object key, size, SHA-256, and upload/verification status
- R2 upload path: `r2:pacifica-trading-data/raw/pacifica/full_fidelity/...`
- upload semantics: rclone `copyto`/`rcat` only, never destructive `sync`; each data object gets a sibling `.sha256` sidecar
- verification semantics: remote object byte size plus `.sha256` sidecar hash must match before local state becomes `verified`
- lifecycle safety as of `80244c7`: scan skips current UTC hour partitions so active gzip chunks are not uploaded while still being appended; rows with verification errors are eligible for re-upload
- always-on Fly deployment is live: app `pacifica-full-fidelity`, machine `e2862502a76778`, region `iad`, 100GB volume `pacifica_full_fidelity_data` mounted at `/data`, compact-mode collector running, R2 lifecycle loop running with prune enabled for Fly spool only

Captured public data:

- global `prices` stream;
- per-symbol `trades`;
- per-symbol `book`;
- per-symbol `bbo`;
- per-symbol `candle`;
- per-symbol `mark_price_candle`;
- REST `/info` and `/info/prices` snapshots.

Latest operational check at 2026-05-02 09:00 EST:

```text
Fly app: pacifica-full-fidelity
Machine: e2862502a76778
Region: iad
Machine state: started
Machine version: 7
Image: pacifica-full-fidelity:deployment-01KQMEZKX06KB8VHSRK7SZ9TJS
Health check: ok=true, failures=[]
Free disk: 87.96 GB
Disk guard: 50 GiB floor active
Newest raw file: fresh under /data/pacifica_full_fidelity
Lifecycle DB: sealed=7224 files / ~3.19 GB; uploaded=3068 files / ~1.71 GB; verified=373 files / ~0.037 GB
Unverified backlog: ~4.56 GB
Collector process: alive with --min-free-disk-gb 50 and compact raw payload mode
Lifecycle process: running upload-verify batch after deploy
R2 raw prefix spot check: succeeded
R2 ops/watchdog prefix spot check: succeeded
```

Interpretation: Fly collection is healthy and R2 is reachable. The lifecycle was still catching up after the current-hour-skip fix; monitor that verification errors clear/re-upload and that verified/pruned counts advance over subsequent lifecycle loops.

### Fly always-on collector status

Chosen deployment path: Fly.io paid deployment, not Hetzner for now. Fly is the active always-on collector and local spool; the laptop is not the active collector.

Fly deployment details:

```text
app: pacifica-full-fidelity
machine: e2862502a76778
region: iad / Ashburn, Virginia
machine version: 7
image: pacifica-full-fidelity:deployment-01KQMEZKX06KB8VHSRK7SZ9TJS
volume: pacifica_full_fidelity_data
volume mount: /data
volume size: 100GB requested, 98G filesystem observed
latest free disk: ~87.96GB
R2 remote: r2:pacifica-trading-data
R2 prefix: raw/pacifica/full_fidelity
```

Runtime defaults on Fly:

```text
PACIFICA_USE_SYSTEM_PYTHON=1
PACIFICA_FULL_FIDELITY_ROOT=/data/pacifica_full_fidelity
PACIFICA_FULL_FIDELITY_STATE_DB=/data/pacifica_full_fidelity_storage.sqlite
PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=50
PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE=compact
PACIFICA_FULL_FIDELITY_RETENTION_DAYS=1
PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S=1800
PACIFICA_FULL_FIDELITY_BATCH_LIMIT=200
PACIFICA_R2_PRUNE_EXECUTE=1
PACIFICA_OPS_ROOT=/data/ops
PACIFICA_API_SURFACE_WATCH_INTERVAL_S=86400
PACIFICA_R2_RETENTION_PLAN_INTERVAL_S=86400
PACIFICA_OPS_R2_PREFIX=ops/pacifica/full_fidelity
PACIFICA_OPS_UPLOAD_REPORTS=1
PACIFICA_OPS_COMMAND_TIMEOUT_S=1800
```

R2 credentials were set as Fly secrets from local rclone config. Secret names: `RCLONE_CONFIG_R2_ACCESS_KEY_ID`, `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`, `RCLONE_CONFIG_R2_ENDPOINT`. Never print or store their values.

Fly process groups:

1. Collector: writes compact raw `.jsonl.gz` chunks to `/data/pacifica_full_fidelity`.
2. Lifecycle loop: every 1800s scans sealed chunks, uploads to R2, writes `.sha256` sidecars, verifies, and prunes only verified local spool older than retention.
3. Ops watchdog loop: wakes hourly; API surface watch and R2 retention planner are due-gated daily and write reports under `/data/ops`, optionally uploading reports to R2 ops prefix.

Latest observed Fly status is healthy, but monitor the R2 lifecycle backlog after commit `80244c7`:

```text
health ok: true
failures: []
free_gb: 87.96
newest_raw_age_min: fresh
sealed: 7224 files / ~3.19 GB
uploaded: 3068 files / ~1.71 GB
verified: 373 files / ~0.037 GB
unverified_gb: 4.56
```

Interpretation: collection is live and R2 is reachable. Verification had historical size-mismatch rows from active current-hour uploads; fix `80244c7` should prevent new current-hour races and re-upload failed uploaded rows. Confirm on the next lifecycle completions that `rows_with_errors` trends down and `verified` increases.

### Pacifica API/docs surface watcher

Read-only watcher:

- script: `scripts/watch_pacifica_api_surface.py`
- reviewed baseline: `docs/ops/pacifica-api-surface-baseline.json`
- latest report/output: `docs/ops/pacifica-api-surface-watch/README.md`, `current_surface.json`, `api_surface_diff.json`

Purpose: detect when Pacifica's public docs/OpenAPI surface appears to expose new collectable market-data REST paths, websocket sources, or intervals. It currently follows `https://docs.pacifica.fi/api-documentation/api`, discovers the linked OpenAPI YAML, filters out private/account/order surfaces, and compares against the reviewed baseline. It is non-mutating: a `CHANGED` verdict means manually inspect docs/API, decide if the new surface is public market data, add collector/silver/tests if useful, and only then update the baseline.

Latest run from this handoff session:

```text
python scripts/watch_pacifica_api_surface.py --out-dir docs/ops/pacifica-api-surface-watch --timeout-s 20
verdict: UNCHANGED
current REST surface: /funding/history, /info, /info/prices, /kline
```

Scheduled Hermes cron remains as a laptop/Hermes-side backup:

```text
job_id: 4bbd973f8035
name: Pacifica API surface watcher
schedule: 0 8 * * *
deliver: origin
```

Fly-side ops watchdog addition:

```text
script: scripts/run_pacifica_fly_ops_watchdogs.py
entrypoint loop: wakes hourly, runs due operations using marker files under /data/ops/.state
API surface watch: daily by default (86400s)
R2 retention/compression planner: daily by default (86400s)
reports: /data/ops locally and optionally r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest
Fly collector disk floor: PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=50
```

### R2 retention and cold-compaction policy

R2 is durable append-only raw archive for now, but remote retention gates are now documented/planned so R2 does not silently accumulate forever.

New non-destructive policy/tooling:

- policy doc: `docs/ops/pacifica-r2-retention-compaction.md`
- planner: `scripts/plan_pacifica_r2_retention.py`
- tests: `tests/scripts/test_plan_pacifica_r2_retention.py`

Default policy:

```text
0-60 days: keep raw full-fidelity .jsonl.gz and .sha256 sidecars
60+ days: require verified compacted/cold archive before raw expiry can be considered
90+ days: raw objects can become eligible for remote-expiry review only with compacted_verified=true and manifest_verified=true
```

Current live R2 size smoke check:

```text
rclone size r2:pacifica-trading-data/raw/pacifica/full_fidelity --json
count=12688
bytes=19203724868
```

Interpretation: R2 remote deletion is not enabled and no remote objects were deleted. This is intentional while the archive is young. Next durable-storage work is a compacted cold archive builder + manifest verification, then a separately approved destructive apply step if/when needed.

### Silver builder

Script:

- `scripts/build_pacifica_full_fidelity_silver.py`

Typical output when rebuilt locally:

- `data/pacifica_silver_partitioned/` or `data/pacifica_silver_partitioned_refresh/`

Important current local-data state:

- Diego approved deleting the laptop's local generated data/cache directories on 2026-05-02:
  - `data/pacifica_full_fidelity` (~17.0 GiB)
  - `data/pacifica_silver_partitioned` (~0.37 GiB)
  - `data/pacifica_silver_partitioned_refresh` (~0.42 GiB)
- Do not assume any of those local directories exists in a fresh session.
- To refresh diagnostics, first rehydrate selected raw partitions from R2 or build a bounded local cache.

Last local silver refresh before deletion completed successfully after adding active-gzip robustness:

```text
bbo: 2545618 rows
book: 8142998 rows
candle: 1551675 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned_refresh
```

Interpretation: the refreshed diagnostics below were built from the now-deleted local `data/pacifica_silver_partitioned_refresh`. Treat those reports as diagnostic historical outputs. The silver builder itself skips incomplete active gzip files from live archives instead of failing with `gzip.BadGzipFile`/CRC errors.

### Realtime research monitor

Script:

- `scripts/watch_pacifica_realtime_research.py`

Tests:

- `tests/scripts/test_watch_pacifica_realtime_research.py`

Generated output directory, gitignored via `data/`:

- `data/pacifica_realtime_research/README.md`
- `data/pacifica_realtime_research/latest_features.csv`
- `data/pacifica_realtime_research/raw_inventory.csv`
- `data/pacifica_realtime_research/warnings.json`

The monitor is read-only. It does not place trades, tune thresholds, or claim edge.

Supported sources:

- `--source silver` reads partitioned parquet from `data/pacifica_silver_partitioned`; preferred routine path after refreshing silver.
- `--source raw` reads recent/bounded raw JSONL.GZ files from `data/pacifica_full_fidelity`; fallback/debug path.

Latest silver-backed monitor verification used `data/pacifica_silver_partitioned_refresh` and wrote to `data/pacifica_realtime_research` with:

```text
warnings.json = ["silver archive stale: latest row age 54523.7s exceeds 300.0s"]
```

Interpretation: this warning is expected for the local laptop silver cache because the always-on collector has moved to Fly and local raw is no longer advancing. It is not a Fly liveness failure.

Current V1 monitor features include:

- trade count 1m;
- trade volume 1m;
- trade notional 1m;
- signed volume 1m;
- last price;
- 1m return bps;
- BBO spread bps;
- top depth notional;
- mark/oracle basis;
- mid/oracle basis;
- funding;
- open interest;
- simple stress score.

### Regime-state builder

Script:

- `scripts/build_non_hft_regime_state.py`

Report/output:

- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/non-hft-regime-state/regime_state.parquet`
- `docs/experiments/non-hft-regime-state/regime_state_preview.csv`
- `docs/experiments/non-hft-regime-state/silver_quality_summary.csv`

Latest local refreshed result:

```text
wrote 59534 regime-state rows to docs/experiments/non-hft-regime-state
Bucket: 1min
Rows: 59534
Symbols: 65
```

Liquidation classification status:

- Current code counts `trade_class == "liquidation"`.
- Current code counts `trade_class` values ending with `_liquidation`.
- Current code counts `cause in ["market_liquidation", "backstop_liquidation"]`.
- Focused test exists: `test_build_regime_state_counts_pacifica_cause_liquidations`.

Latest observed silver trade-class values from the committed diagnostic refresh:

```text
normal: 48024
market_liquidation: 29
insolvency_liquidation: 1
```

### Toxic-regime overlay probe

Script:

- `scripts/non_hft_toxic_overlay_probe.py`

Report/output:

- `docs/experiments/toxic-regime-overlay/README.md`
- `docs/experiments/toxic-regime-overlay/overlay_scorecard.csv`
- `docs/experiments/toxic-regime-overlay/symbol_summary.csv`
- `docs/experiments/toxic-regime-overlay/hour_summary.csv`
- `docs/experiments/toxic-regime-overlay/toxic_bucket_summary.csv`

Latest local refreshed result:

```text
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
Rows: 59534
Symbols: 65
Distinct dates: 2
Horizons minutes: [5, 15, 30, 60]
Toxicity cutoffs: [0.9, 0.8, 0.7]
```

Interpretation: expected diagnostic state. Two distinct dates is not edge evidence.

### Paper-trading eligibility gates

Script:

- `scripts/build_pacifica_eligibility_gates.py`

Tests:

- `tests/scripts/test_build_pacifica_eligibility_gates.py`

Output:

- `docs/experiments/paper-trading-eligibility/README.md`
- `docs/experiments/paper-trading-eligibility/symbol_eligibility.csv`
- `docs/experiments/paper-trading-eligibility/eligible_symbols.csv`
- `docs/experiments/paper-trading-eligibility/gate_counts.csv`
- `docs/experiments/paper-trading-eligibility/thresholds.csv`

Latest run:

```text
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
symbols_evaluated: 65
eligible_symbols: 0
```

Gate counts:

```text
sample_gate_pass: 0 / 65
liquidity_gate_pass: 25 / 65
spread_cost_gate_pass: 60 / 65
activity_gate_pass: 4 / 65
stability_gate_pass: 63 / 65
concentration_gate_pass: 0 / 65
eligible: 0 / 65
```

Interpretation: eligibility gates are now explicit and pre-backtester. No symbols are eligible yet because the archive only has 2 distinct dates and the day-concentration gate cannot pass. Do not loosen thresholds based on this diagnostic run.

## Interpretation discipline

The full-fidelity archive is too young to claim an edge.

Use these maturity levels:

- 1-5 days: plumbing diagnostics only;
- 10-14 days: early sanity checks;
- 30+ full days: provisional validation;
- 60+ full days: preferred serious validation.

Keep toxicity thresholds fixed while data accrues. Do not tune cutoffs based on diagnostic days.

## Verification status

Verification for latest lifecycle fix and deploy:

```bash
python -m py_compile \
  scripts/pacifica_full_fidelity_storage.py \
  scripts/check_pacifica_full_fidelity_health.py \
  scripts/collect_pacifica_full_fidelity.py

bash -n \
  scripts/run_pacifica_full_fidelity_r2_lifecycle.sh \
  ops/fly/pacifica-full-fidelity/entrypoint.sh

uv run pytest \
  tests/scripts/test_pacifica_full_fidelity_storage.py \
  tests/scripts/test_collect_pacifica_full_fidelity.py \
  tests/scripts/test_watch_pacifica_api_surface.py \
  tests/scripts/test_run_pacifica_fly_ops_watchdogs.py \
  tests/scripts/test_pacifica_r2_inventory.py -q

git diff --check
```

Result:

```text
27 passed in 0.26s
py_compile passed
bash syntax checks passed
git diff --check passed
commit: 80244c7 fix: avoid uploading active Pacifica raw chunks
deployed to Fly machine version 7
```

Also verified live operational state after deployment:

```text
fly status -a pacifica-full-fidelity: machine e2862502a76778 started in iad, version 7
health check: ok=true, failures=[]
free_gb: 87.96
newest raw file: fresh under /data/pacifica_full_fidelity
collector process: alive
lifecycle process: running upload-verify batch
R2 raw prefix spot check: succeeded
R2 ops/watchdog prefix spot check: succeeded
```

Note: the post-deploy lifecycle had not yet printed final `lifecycle complete` when Diego asked “all good?”; the health check and process/R2 checks were good, and lifecycle was actively running/catching up.

## Recommended next steps in a fresh session

1. Start with `git status --short` and `git log --oneline -5`. Latest expected commit is `80244c7 fix: avoid uploading active Pacifica raw chunks`.
2. Monitor Fly steady state. App `pacifica-full-fidelity` in region `iad` has machine `e2862502a76778`, 100GB volume `pacifica_full_fidelity_data`, R2 secrets set, collector running, and lifecycle upload/verify working.
3. Specifically confirm the R2 lifecycle catch-up after `80244c7`:
   - latest lifecycle loop prints `lifecycle complete`;
   - `rows_with_errors` in `archive_files` trends down from the observed 200 historical size-mismatch rows;
   - `verified` count increases;
   - old verified local files prune after the one-day retention window;
   - no new current-hour size mismatch errors are generated.
4. Use these Fly checks first:
   - `fly status -a pacifica-full-fidelity`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 "sh -lc 'cd /app && python scripts/check_pacifica_full_fidelity_health.py --root /data/pacifica_full_fidelity --state-db /data/pacifica_full_fidelity_storage.sqlite --min-free-gb 50 --max-newest-age-min 10'"`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 'sh -lc "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status order by status;\""'`
5. Keep the local laptop collector stopped unless intentionally used for smoke/debug collection. The always-on collector is Fly, not laptop launchd.
6. Do not expect local laptop raw/silver directories to exist. Diego approved deleting `data/pacifica_full_fidelity`, `data/pacifica_silver_partitioned`, and `data/pacifica_silver_partitioned_refresh` from the laptop on 2026-05-02. Rehydrate selected data from R2 when needed.
7. Laptop lifecycle pruning remains dry-run unless Diego explicitly enables it; Fly spool pruning is enabled because `/data` is a bounded cache.
8. Hetzner/systemd remains documented as a lower-cost fallback in `docs/ops/pacifica-full-fidelity-hetzner.md`, `ops/hetzner/`, and `ops/systemd/`, but Diego chose Fly for now.
9. Refresh silver/regime/toxic diagnostics from bounded local cache or selected R2 rehydration, without changing fixed toxicity thresholds.
10. Run or monitor the Pacifica API/docs surface watcher (`scripts/watch_pacifica_api_surface.py`). If it reports `CHANGED`, manually inspect whether any added REST path/websocket source/interval is public collectable market data before changing the collector or baseline.
11. For R2 remote growth control, build the compacted cold archive + manifest verifier before enabling any R2 raw expiry. Use `scripts/plan_pacifica_r2_retention.py` only as a non-destructive planner until a separate destructive apply step is explicitly approved.
12. Rerun `scripts/build_pacifica_eligibility_gates.py` after each mature regime-state refresh; keep thresholds fixed unless deliberately changed before reviewing outcomes.
13. Only after eligibility gates, enough full days, and simple sparse baselines exist, build the post-cost event-driven paper backtester/logger.

## Quick commands

Inspect status and dynamic live universe:

```bash
git status --short
git log --oneline -8
uv run python - <<'PY'
from scripts.collect_pacifica_full_fidelity import build_subscriptions, fetch_live_symbols
symbols = fetch_live_symbols()
print('live_symbols=', len(symbols), 'subscriptions=', len(build_subscriptions(symbols)))
PY
```

Inspect Fly always-on collector/storage:

```bash
fly status -a pacifica-full-fidelity
fly volumes list -a pacifica-full-fidelity
fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 "sh -lc 'cd /app && python scripts/check_pacifica_full_fidelity_health.py --root /data/pacifica_full_fidelity --state-db /data/pacifica_full_fidelity_storage.sqlite --min-free-gb 50 --max-newest-age-min 10'"
fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 'sh -lc "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status order by status;\""'
```

Inspect Fly R2 lifecycle errors/backlog:

```bash
fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 'sh -lc "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select count(*) as rows_with_errors from archive_files where error is not null; select status,error,object_key from archive_files where error is not null order by coalesce(uploaded_at,last_seen_at) desc limit 10;\""'
```

Inspect local cache directories; expected after 2026-05-02 cleanup is that raw/silver caches are missing unless rehydrated:

```bash
python - <<'PY'
from pathlib import Path
for p in [Path('data/pacifica_full_fidelity'), Path('data/pacifica_silver_partitioned'), Path('data/pacifica_silver_partitioned_refresh')]:
    files=0; bytes_=0
    if p.exists():
        for f in p.rglob('*'):
            if f.is_file():
                files += 1
                bytes_ += f.stat().st_size
    print(f'{p}: exists={p.exists()} files={files} gib={bytes_/1024**3:.3f}')
PY
```

Refresh silver from raw:

```bash
python scripts/build_pacifica_full_fidelity_silver.py \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir data/pacifica_silver_partitioned_refresh
```

Run silver-backed realtime monitor:

```bash
python scripts/watch_pacifica_realtime_research.py \
  --source silver \
  --silver-dir data/pacifica_silver_partitioned_refresh \
  --out-dir data/pacifica_realtime_research \
  --stale-after-s 300
```

Safe raw fallback monitor command:

```bash
python scripts/watch_pacifica_realtime_research.py \
  --source raw \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir data/pacifica_realtime_research \
  --stale-after-s 300 \
  --max-files 200 \
  --max-records-per-file 1000
```

Rebuild regime and toxic reports:

```bash
python scripts/build_non_hft_regime_state.py \
  --silver-dir data/pacifica_silver_partitioned_refresh \
  --out-dir docs/experiments/non-hft-regime-state

python scripts/non_hft_toxic_overlay_probe.py \
  --state-path docs/experiments/non-hft-regime-state/regime_state.parquet \
  --out-dir docs/experiments/toxic-regime-overlay
```

Run API/docs surface watcher:

```bash
python scripts/watch_pacifica_api_surface.py \
  --out-dir docs/ops/pacifica-api-surface-watch \
  --timeout-s 20 \
  --fail-on-change
```

Inspect R2 durable archive size and retention policy:

```bash
rclone size 'r2:pacifica-trading-data/raw/pacifica/full_fidelity' --json
python scripts/plan_pacifica_r2_retention.py \
  --inventory-csv path/to/r2_inventory.csv \
  --out-dir docs/ops/pacifica-r2-retention
```

Do not run remote R2 deletion from the planner. It is non-destructive by design.

Run focused verification:

```bash
uv run pytest tests/scripts/test_watch_pacifica_realtime_research.py \
  tests/scripts/test_build_pacifica_full_fidelity_silver.py \
  tests/scripts/test_collect_pacifica_full_fidelity.py \
  tests/scripts/test_build_non_hft_regime_state.py \
  tests/scripts/test_non_hft_toxic_overlay_probe.py -q

python -m py_compile \
  scripts/watch_pacifica_realtime_research.py \
  scripts/build_non_hft_regime_state.py \
  scripts/non_hft_toxic_overlay_probe.py \
  scripts/watch_pacifica_api_surface.py

git diff --check
```

## Files most relevant for fresh-session context

- `docs/NEXT_SESSION_HANDOFF.md` — this file.
- `AGENTS.md` — canonical repo-level agent instructions.
- `docs/AGENT_OPERATING_MAP.md` — current Hermes/tool/skill arsenal and archived Claude asset notes.
- `docs/ops/pacifica-full-fidelity-archival.md`
- `docs/ops/pacifica-full-fidelity-fly.md`
- `ops/fly/pacifica-full-fidelity/Dockerfile`
- `ops/fly/pacifica-full-fidelity/entrypoint.sh`
- `ops/fly/pacifica-full-fidelity/fly.toml`
- `scripts/pacifica_full_fidelity_storage.py`
- `scripts/run_pacifica_full_fidelity_collector.sh`
- `scripts/run_pacifica_full_fidelity_r2_lifecycle.sh`
- `scripts/check_pacifica_full_fidelity_health.py`
- `scripts/watch_pacifica_api_surface.py`
- `scripts/run_pacifica_fly_ops_watchdogs.py`
- `scripts/pacifica_r2_inventory.py`
- `docs/ops/pacifica-api-surface-baseline.json`
- `docs/ops/pacifica-api-surface-watch/README.md`
- `docs/ops/pacifica-r2-retention-compaction.md`
- `scripts/plan_pacifica_r2_retention.py`
- `tests/scripts/test_plan_pacifica_r2_retention.py`
- `tests/scripts/test_pacifica_full_fidelity_storage.py`
- `.dockerignore`
- `docs/research/2026-05-01-real-time-streaming-research-pass.md`
- `docs/research/2026-04-30-pacifica-full-fidelity-product-ideas.md`
- `docs/experiments/pacifica-full-fidelity-tradeability-filter-2026-04-30.md`
- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`
- `scripts/collect_pacifica_full_fidelity.py`
- `scripts/build_pacifica_full_fidelity_silver.py`
- `scripts/watch_pacifica_realtime_research.py`
- `scripts/build_non_hft_regime_state.py`
- `scripts/non_hft_toxic_overlay_probe.py`
- `tests/scripts/test_watch_pacifica_realtime_research.py`

## Historical context

The old 25-symbol historical parquet/cache program remains useful context, but it is not the active starting point.

Historical findings:

- v1 direction-prediction representation learning found weak signal that was fee-blocked.
- v2 cascade-onset prediction found a real signal, but direct trading economics failed.
- Maker execution was harmed by adverse selection.
- Taker-side feasibility had no strict survivors under plausible costs.
- April 14-26 cascade holdout was consumed; new clean validation requires fresh data.

The new collector matters because it preserves fields the old lossy parquet data did not: mark/oracle/funding/open_interest, BBO order IDs, book nonces/order counts, raw trade IDs/nonces, and raw message timing.

## Do not do next

- Do not launch generic RL.
- Do not optimize AUC without execution economics.
- Do not claim an edge from the 2-day diagnostic probe.
- Do not blindly trade every collected symbol.
- Do not tune toxicity thresholds on diagnostic samples.
- Do not overwrite, delete, or commit raw data archives.
- Do not configure R2 raw expiry or delete remote R2 objects until compacted cold archive + manifest gates exist and Diego explicitly approves a separate destructive apply step.
- Do not commit `.hermes/` unless explicitly intended.
- Do not recreate `CLAUDE.md` or revive Claude Code assets unless Diego explicitly decides to support Claude in this repo again.

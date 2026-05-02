# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-02 02:45 EST

## Start here

The active project is an economics-first, non-HFT Pacifica paper-trading program using full-fidelity public market-data archival across the live Pacifica symbol universe.

Fresh-session reading order:

1. `docs/NEXT_SESSION_HANDOFF.md` — this handoff.
2. `AGENTS.md` — canonical repo-level agent instructions.
3. `docs/AGENT_OPERATING_MAP.md` — Hermes/tool/skill map and archived Claude asset notes.

There is no active `CLAUDE.md` and no active root `.claude/` workflow. Hermes is primary. Do not recreate `CLAUDE.md` or route work through Claude Code unless Diego explicitly reverses that decision.

## Current commit/state

Latest committed work:

```text
32af641 feat: add Pacifica paper eligibility gates
5af8e3f docs: refresh Pacifica diagnostics after R2 lifecycle
05edb6b feat: add Pacifica R2 spool lifecycle
```

Current uncommitted work in this session:

```text
docs/ops/pacifica-r2-retention-compaction.md
scripts/plan_pacifica_r2_retention.py
tests/scripts/test_plan_pacifica_r2_retention.py
docs/NEXT_SESSION_HANDOFF.md
```

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

Latest local check in this handoff session:

```text
branch: main
latest commit: 32af641 feat: add Pacifica paper eligibility gates
working tree: uncommitted R2 retention/compaction policy doc, non-destructive planner script/tests, and this handoff update
```

Current uncommitted status snapshot at 2026-05-02 02:45 EST:

```text
 M docs/NEXT_SESSION_HANDOFF.md
?? docs/ops/pacifica-r2-retention-compaction.md
?? scripts/plan_pacifica_r2_retention.py
?? tests/scripts/test_plan_pacifica_r2_retention.py
```

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

Output:

- `data/pacifica_full_fidelity/`

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
- state DB: `data/pacifica_full_fidelity_storage.sqlite` (generated, do not commit)
- manifest output: JSONL rows with local path, deterministic R2 object key, size, SHA-256, and upload/verification status
- R2 upload path: `r2:pacifica-trading-data/raw/pacifica/full_fidelity/...`
- upload semantics: rclone `copyto`/`rcat` only, never destructive `sync`; each data object gets a sibling `.sha256` sidecar
- verification semantics: remote object byte size plus `.sha256` sidecar hash must match before local state becomes `verified`
- local canary: scanned 4,171 local files into the state DB, then uploaded+verified all 4,171 local files totaling 18,301,666,532 bytes; local pruning has not been enabled or executed
- always-on Fly deployment is live: app `pacifica-full-fidelity`, machine `e2862502a76778`, region `iad`, 100GB volume `pacifica_full_fidelity_data` mounted at `/data`, compact-mode collector running, R2 lifecycle loop running with prune enabled for Fly spool only

Captured public data:

- global `prices` stream;
- per-symbol `trades`;
- per-symbol `book`;
- per-symbol `bbo`;
- per-symbol `candle`;
- per-symbol `mark_price_candle`;
- REST `/info` and `/info/prices` snapshots.

Latest storage incident / deployment check at 2026-05-01 17:25 EST:

```text
Laptop /System/Volumes/Data: 266Gi used, 173Gi available, 61% full after cache cleanup / APFS purgeable-space recovery
prior collector log contained Errno 28 / No space left on device
laptop launchd collector should remain stopped unless intentionally smoke-testing compact mode
Fly /data volume: 98G total, 2.9G used, 90G available, 4% full
Fly lifecycle DB: sealed=4450 files / 1,835,082,475 bytes; uploaded=1638 files / 989,828,251 bytes; verified=335 files / 37,874,663 bytes
Fly local spool currently has 3,221 files under /data/pacifica_full_fidelity
old local R2 upload-verify background process completed successfully: session_id=proc_621d706409f0, pid=35414, exit code 0, uptime about 14,160s
local lifecycle DB: verified=4,171 files / 18,301,666,532 bytes; local pruning has not been enabled or executed
```

Interpretation: raw collection had recently overwhelmed disk and hit `No space left on device`. The collector was patched to default to compact raw payload rows and to refuse writes below a configurable free-space floor (`--min-free-disk-gb`, launchd plist currently 50 GiB). Current laptop free space is above the 50 GiB floor, but keep the laptop collector stopped unless intentionally resuming it for smoke/debug with compact mode. Do not paste launchd environment output in docs/reports; redact credentials if encountered.

### Fly always-on collector status

Chosen deployment path: Fly.io paid deployment, not Hetzner for now.

Fly deployment details:

```text
app: pacifica-full-fidelity
machine: e2862502a76778
region: iad / Ashburn, Virginia
volume: pacifica_full_fidelity_data
volume mount: /data
volume size: 100GB requested, 98G filesystem observed
latest /data usage: 2.9G used, 90G available, 4% full
R2 remote: r2:pacifica-trading-data
R2 prefix: raw/pacifica/full_fidelity
```

Runtime defaults on Fly:

```text
PACIFICA_USE_SYSTEM_PYTHON=1
PACIFICA_FULL_FIDELITY_ROOT=/data/pacifica_full_fidelity
PACIFICA_FULL_FIDELITY_STATE_DB=/data/pacifica_full_fidelity_storage.sqlite
PACIFICA_FULL_FIDELITY_MIN_FREE_GB=10
PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE=compact
PACIFICA_FULL_FIDELITY_RETENTION_DAYS=1
PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S=1800
PACIFICA_FULL_FIDELITY_BATCH_LIMIT=200
PACIFICA_R2_PRUNE_EXECUTE=1
```

R2 credentials were set as Fly secrets from local rclone config. Secret names: `RCLONE_CONFIG_R2_ACCESS_KEY_ID`, `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`, `RCLONE_CONFIG_R2_ENDPOINT`. Never print or store their values.

Latest observed Fly lifecycle status shows upload and verification are working, but the backlog is still moving:

```text
sealed|4450|1835082475
uploaded|1638|989828251
verified|335|37874663
```

Interpretation: Fly collection is live, R2 upload is live, `.sha256` sidecar verification is live, and some rows have reached `verified`. Continue monitoring until the steady state is clear: files should not accumulate without bound on `/data`, and older verified files should prune after the one-day retention window.

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
API surface watch interval: 86400 seconds (daily)
R2 inventory/retention-plan interval: 86400 seconds (daily)
reports root: /data/ops
report upload target: r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest
remote raw deletion: not implemented; retention reports remain non-destructive planning artifacts
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

Output:

- `data/pacifica_silver_partitioned/`

Last local silver refresh completed successfully after adding active-gzip robustness:

```text
bbo: 2545618 rows
book: 8142998 rows
candle: 1551675 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned_refresh
```

Latest filesystem freshness check at 2026-05-02 01:38 EST:

```text
data/pacifica_silver_partitioned_refresh exists=True
files=974
symbols=65
dates=2026-04-30, 2026-05-01
latest_age_s≈fresh at rebuild time
```

Interpretation: refreshed diagnostics below were built from `data/pacifica_silver_partitioned_refresh`. The canonical `data/pacifica_silver_partitioned` directory was not replaced because a cleanup command was denied; either use the refresh directory explicitly or safely replace the canonical generated silver directory later. The silver builder now skips incomplete active gzip files from the live collector instead of failing with `gzip.BadGzipFile`/CRC errors.

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

Verification in this handoff update:

```bash
uv run pytest \
  tests/scripts/test_pacifica_full_fidelity_storage.py \
  tests/scripts/test_collect_pacifica_full_fidelity.py -q

python -m py_compile \
  scripts/collect_pacifica_full_fidelity.py \
  scripts/pacifica_full_fidelity_storage.py \
  scripts/check_pacifica_full_fidelity_health.py

bash -n \
  scripts/run_pacifica_full_fidelity_collector.sh \
  scripts/run_pacifica_full_fidelity_r2_lifecycle.sh \
  ops/fly/pacifica-full-fidelity/entrypoint.sh
```

Result:

```text
47 passed in 0.67s for focused R2-retention/eligibility/storage/collector/silver/regime/toxic tests
py_compile passed
git diff --check passed
bash syntax checks passed previously for shell wrappers
```

Also verified live operational state:

```text
fly status -a pacifica-full-fidelity shows machine e2862502a76778 started in iad
Fly /data: 98G total, 2.9G used, 90G available
Fly lifecycle DB: sealed=4450, uploaded=1638, verified=335; R2 upload+sidecar verification path is active and verified advanced from 306 to 335 since the prior check
Laptop /System/Volumes/Data: 173Gi available in the prior local disk check
```

Previous broader research/diagnostic verification before the Fly deployment remained:

```text
32 passed in 0.84s
```

## Recommended next steps in a fresh session

1. Start with `git status --short`. Current uncommitted work should be the R2 retention/compaction policy doc, non-destructive planner script/tests, and this handoff update, unless already committed.
2. Monitor Fly steady state. App `pacifica-full-fidelity` in region `iad` has machine `e2862502a76778`, 100GB volume `pacifica_full_fidelity_data`, R2 secrets set, collector running, and lifecycle upload/verify working. Poll `/data` disk and lifecycle DB until `verified` grows and old verified files prune after the one-day retention window.
3. Use these Fly checks first:
   - `fly status -a pacifica-full-fidelity`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity "df -h /data" --timeout 60`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity 'sh -c "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status order by status;\""' --timeout 60`
4. The old laptop R2 upload/verify background process completed successfully: `process(action="poll", session_id="proc_621d706409f0")` returned exit code 0 with 4,166 uploaded and 4,166 verified during that run; local lifecycle DB now has 4,171 verified files totaling 18,301,666,532 bytes. No laptop pruning has been enabled or executed.
5. Keep the local laptop collector stopped unless intentionally used for smoke/debug collection. The always-on collector is Fly, not laptop launchd.
6. Laptop lifecycle pruning remains dry-run unless Diego explicitly enables `PACIFICA_R2_PRUNE_EXECUTE=1`; Fly spool pruning is enabled because `/data` is a bounded cache.
7. Hetzner/systemd remains documented as a lower-cost fallback in `docs/ops/pacifica-full-fidelity-hetzner.md`, `ops/hetzner/`, and `ops/systemd/`, but Diego chose Fly for now. Fly free capacity is not enough for the 100GB spool; this is a paid deployment.
8. Refresh silver/regime/toxic diagnostics from bounded local cache or selected R2 rehydration, without changing fixed toxicity thresholds.
9. Run or monitor the Pacifica API/docs surface watcher (`scripts/watch_pacifica_api_surface.py`). If it reports `CHANGED`, manually inspect whether any added REST path/websocket source/interval is public collectable market data before changing the collector or baseline.
10. For R2 remote growth control, build the compacted cold archive + manifest verifier before enabling any R2 raw expiry. Use `scripts/plan_pacifica_r2_retention.py` only as a non-destructive planner until a separate destructive apply step is explicitly approved.
11. Rerun `scripts/build_pacifica_eligibility_gates.py` after each mature regime-state refresh; keep thresholds fixed unless deliberately changed before reviewing outcomes.
12. Only after eligibility gates, enough full days, and simple sparse baselines exist, build the post-cost event-driven paper backtester/logger.

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
fly machine exec e2862502a76778 -a pacifica-full-fidelity "df -h /data" --timeout 60
fly machine exec e2862502a76778 -a pacifica-full-fidelity 'sh -c "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status order by status;\""' --timeout 60
```

Check completed old local background R2 upload/verify process in Hermes:

```text
process(action="poll", session_id="proc_621d706409f0")
# expected: exited, exit_code=0, uploaded=4166, verified=4166 for that run
```

Inspect local archive freshness and lifecycle inventory:

```bash
python - <<'PY'
from pathlib import Path
import time
for p in [Path('data/pacifica_full_fidelity'), Path('data/pacifica_silver_partitioned')]:
    print(p, 'exists=', p.exists())
    files=0; syms=set(); dates=set(); latest=0
    if p.exists():
        for f in p.rglob('*'):
            if f.is_file():
                files += 1
                try: latest=max(latest, f.stat().st_mtime)
                except OSError: pass
                for part in f.parts:
                    if part.startswith('symbol='): syms.add(part.split('=',1)[1])
                    if part.startswith('date='): dates.add(part.split('=',1)[1])
    age = None if latest == 0 else round(time.time()-latest, 1)
    print('files=', files, 'symbols=', len(syms), 'dates=', sorted(dates)[:3], '...', sorted(dates)[-3:] if dates else [], 'latest_age_s=', age)
PY

uv run python scripts/pacifica_full_fidelity_storage.py \
  --root data/pacifica_full_fidelity \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  --r2-prefix raw/pacifica/full_fidelity \
  scan

uv run python scripts/pacifica_full_fidelity_storage.py \
  --state-db data/pacifica_full_fidelity_storage.sqlite \
  --remote-base r2:pacifica-trading-data \
  --limit 100 \
  upload-verify
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

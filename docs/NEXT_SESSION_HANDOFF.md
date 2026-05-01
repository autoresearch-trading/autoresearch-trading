# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-01 17:30 EST

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
f5c625a fix: tolerate active Pacifica gzip files in silver build
```

Relevant prior commits:

```text
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
latest commit: f5c625a fix: tolerate active Pacifica gzip files in silver build
working tree: uncommitted storage-safety/R2/Fly deployment patch set; do not assume clean
```

Current uncommitted status snapshot at 2026-05-01 17:25 EST:

```text
 M docs/NEXT_SESSION_HANDOFF.md
 M docs/ops/pacifica-full-fidelity-archival.md
 M ops/launchd/com.non-toxic.pacifica-full-fidelity.plist
 M scripts/collect_pacifica_full_fidelity.py
 M tests/scripts/test_collect_pacifica_full_fidelity.py
?? .dockerignore
?? docs/ops/pacifica-full-fidelity-fly.md
?? docs/ops/pacifica-full-fidelity-hetzner.md
?? ops/fly/
?? ops/hetzner/
?? ops/launchd/com.non-toxic.pacifica-full-fidelity-r2-lifecycle.plist
?? ops/systemd/
?? research_cloud_provider_pacifica_collector/
?? scripts/check_pacifica_full_fidelity_health.py
?? scripts/pacifica_full_fidelity_storage.py
?? scripts/run_pacifica_full_fidelity_collector.sh
?? scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
?? tests/scripts/test_pacifica_full_fidelity_storage.py
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
Fly /data volume: 98G total, 258M used, 93G available, 1% full
Fly lifecycle DB: sealed=2049 files / 143,861,083 bytes; uploaded=50 files / 3,988,870 bytes; verified=150 files / 3,890,946 bytes
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
latest /data usage: 258M used, 93G available, 1% full
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
sealed|2049|143861083
uploaded|50|3988870
verified|150|3890946
```

Interpretation: Fly collection is live, R2 upload is live, `.sha256` sidecar verification is live, and some rows have reached `verified`. Continue monitoring until the steady state is clear: files should not accumulate without bound on `/data`, and older verified files should prune after the one-day retention window.

### Silver builder

Script:

- `scripts/build_pacifica_full_fidelity_silver.py`

Output:

- `data/pacifica_silver_partitioned/`

Last local silver refresh completed successfully after adding active-gzip robustness:

```text
bbo: 2340115 rows
book: 7762239 rows
candle: 1483561 rows
mark_price_candle: 20686075 rows
prices: 964508 rows
trades: 87637 rows
wrote silver tables to data/pacifica_silver_partitioned_refresh
```

Latest filesystem freshness check at 2026-05-01 09:54 EST:

```text
data/pacifica_silver_partitioned_refresh exists=True
files=912
symbols=65
dates=2026-04-30, 2026-05-01
latest_age_s=90.4
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
warnings.json = []
```

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
wrote 57372 regime-state rows to docs/experiments/non-hft-regime-state
Bucket: 1min
Rows: 57372
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
Rows: 57372
Symbols: 65
Distinct dates: 2
Horizons minutes: [5, 15, 30, 60]
Toxicity cutoffs: [0.9, 0.8, 0.7]
```

Interpretation: expected diagnostic state. Two distinct dates is not edge evidence.

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
15 passed in 0.04s
py_compile passed
bash syntax checks passed
```

Also verified live operational state:

```text
fly machine exec e2862502a76778 -a pacifica-full-fidelity ... df/sqlite checks passed
Fly /data: 98G total, 258M used, 93G available
Fly lifecycle DB includes verified rows, proving upload+sidecar verification path works; verified advanced to 150 rows in the latest check
Laptop /System/Volumes/Data: 173Gi available
```

Previous broader research/diagnostic verification before the Fly deployment remained:

```text
32 passed in 0.84s
```

## Recommended next steps in a fresh session

1. Start with `git status --short` and review the uncommitted storage-safety/R2/Fly deployment patch set. Do not assume a clean tree.
2. Monitor Fly steady state. App `pacifica-full-fidelity` in region `iad` has machine `e2862502a76778`, 100GB volume `pacifica_full_fidelity_data`, R2 secrets set, collector running, and lifecycle upload/verify working. Poll `/data` disk and lifecycle DB until `verified` grows and old verified files prune after the one-day retention window.
3. Use these Fly checks first:
   - `fly status -a pacifica-full-fidelity`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity "df -h /data" --timeout 60`
   - `fly machine exec e2862502a76778 -a pacifica-full-fidelity 'sh -c "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status order by status;\""' --timeout 60`
4. The old laptop R2 upload/verify background process completed successfully: `process(action="poll", session_id="proc_621d706409f0")` returned exit code 0 with 4,166 uploaded and 4,166 verified during that run; local lifecycle DB now has 4,171 verified files totaling 18,301,666,532 bytes. No laptop pruning has been enabled or executed.
5. Keep the local laptop collector stopped unless intentionally used for smoke/debug collection. The always-on collector is Fly, not laptop launchd.
6. Laptop lifecycle pruning remains dry-run unless Diego explicitly enables `PACIFICA_R2_PRUNE_EXECUTE=1`; Fly spool pruning is enabled because `/data` is a bounded cache.
7. Hetzner/systemd remains documented as a lower-cost fallback in `docs/ops/pacifica-full-fidelity-hetzner.md`, `ops/hetzner/`, and `ops/systemd/`, but Diego chose Fly for now. Fly free capacity is not enough for the 100GB spool; this is a paid deployment.
8. Then refresh silver/regime/toxic diagnostics from bounded local cache or selected R2 rehydration, without changing fixed toxicity thresholds.
9. Add explicit paper-trading eligibility gates before any strategy can trade all symbols.
10. Only after eligibility gates and simple sparse baselines exist, build the post-cost event-driven paper backtester/logger.

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
  scripts/non_hft_toxic_overlay_probe.py

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
- Do not commit `.hermes/` unless explicitly intended.
- Do not recreate `CLAUDE.md` or revive Claude Code assets unless Diego explicitly decides to support Claude in this repo again.

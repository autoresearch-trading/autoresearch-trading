# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-12 15:50 UTC

## Current state

Active program: Hermes-only Pacifica full-fidelity, non-HFT paper-trading research. Do not use Claude Code, recreate `CLAUDE.md`, propose HFT/latency strategies, tune early toxicity thresholds, or claim edge from the current young archive.

Canonical active runtime/archive:

- Fly app: `pacifica-full-fidelity`
- Machine: `e2862502a76778`
- Active R2 archive prefix: `r2:pacifica-trading-data/raw/`
- Active local lifecycle DB on Fly: `/data/pacifica_full_fidelity_storage.sqlite`
- Local research raw cache: `data/pacifica_full_fidelity/` restored from R2 for research rebuilds
- Local silver output: `data/pacifica_silver_partitioned/`

## Latest 2026-05-12 R2 freshness follow-up

Timestamp: `2026-05-12T15:50Z`

Current read-only evidence:

```text
Fly status: started, version 22, image deployment-01KRDYHQ7A79GFCM2RGR08NEXM
Lifecycle evidence from logs:
  2026-05-12T13:51:22Z fresh upload uploaded=2000 failed=0 skipped=0
  2026-05-12T15:06:28Z backlog upload uploaded=250 failed=0 skipped=0; verify verified=500 failed=0 skipped=0
  2026-05-12T15:06:30Z lifecycle complete
  2026-05-12T15:26:13Z next lifecycle scan/upload/verify/prune started
Bounded local R2 freshness check after timezone parser fix:
  checked_at=2026-05-12T15:48:37Z
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=12/run-20260512T111943Z.jsonl.gz
  latest_payload_modified=2026-05-12T13:06:41Z
  latest_payload_age_min=161.94
  payload_count=95
  sidecar_count=95
  sidecar_missing_count=0
```

Important parser fix: `rclone lsf --format t` renders timestamps in the caller's local timezone without an offset. The local laptop is `EST -0500`, so treating the string as UTC made the local bounded checker falsely report ~7.5h stale when the same object was ~2.7h old in UTC. `scripts/check_pacifica_r2_freshness.py` now parses rclone timestamps as process-local time and converts to UTC before freshness math.

Caveats:

- The uploaded watchdog artifact at `ops/pacifica/full_fidelity/watchdogs/latest/pacifica-r2-freshness/latest_status.json` was still stale at `2026-05-12T15:23:50Z`, before the latest observed hour=12 sample was visible locally.
- A direct Fly-side ad hoc SSH check was attempted but not completed because the shell-wrapped command was blocked; do not retry that exact command form. The local `TZ=UTC` run matched the expected Fly/UTC interpretation and returned `ok=true`.
- Do not start competing manual lifecycle upload/verify writers. Let the scheduled lifecycle continue and let the next hourly ops watchdog confirm the recovered R2 freshness.

Next exact check:

```text
uv run python scripts/check_pacifica_r2_freshness.py --remote-base r2:pacifica-trading-data --r2-prefix raw/pacifica/full_fidelity --stale-after-min 180 --timeout-s 45
rclone copyto r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest/pacifica-r2-freshness/latest_status.json /tmp/pacifica-r2-freshness-latest.json
python -m json.tool /tmp/pacifica-r2-freshness-latest.json
```

Expected: local checker should remain `ok=true`; uploaded watchdog should flip to `ok=true` on its next hourly run if the 15:26 lifecycle upload path stays healthy.

## Latest 2026-05-12 bounded freshness-lane/watchdog update

Timestamp: `2026-05-12T11:45Z`

Deployed Fly image/version:

```text
Image: pacifica-full-fidelity:deployment-01KRDYHQ7A79GFCM2RGR08NEXM
Machine: e2862502a76778
Version: 22
State: started
Last updated: 2026-05-12T11:19:43Z
```

Why this deployment exists: the previous newest-first lane still required a broad lifecycle scan before the upload phase and could let too-recent sealed rows consume upload limits. R2 freshness remained stale even though local collection and lifecycle progress were otherwise healthy.

What changed:

- `scan_archive_files(..., recent_hours=...)` and CLI `scan --recent-hours N` now scan only bounded recent UTC hour partitions for the freshness lane.
- The lifecycle script now runs a fast recent scan and newest-first fresh upload before the slower full-scan/backlog/verify/prune safety lane.
- Full archive scans are marker-gated by `PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S=21600` so each 15m cycle does not restat/hash the entire archive.
- Upload selection now filters rows younger than `PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS=7200` in SQL, so too-recent chunks do not spend the upload limit.
- Backlog verification remains non-destructive and bounded separately with `PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT=250` and `PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500`.
- Added `scripts/check_pacifica_r2_freshness.py`, a bounded read-only R2 sample checker that verifies latest sampled payload freshness and payload/`.sha256` pairing.
- The Fly ops watchdog now runs the bounded R2 freshness checker hourly and writes `/data/ops/pacifica-r2-freshness/latest_status.json` before durable report upload.

Current Fly env additions in `ops/fly/pacifica-full-fidelity/fly.toml`:

```text
PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_LIMIT=2000
PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT=250
PACIFICA_FULL_FIDELITY_RECENT_SCAN_HOURS=12
PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S=21600
PACIFICA_FULL_FIDELITY_UPLOAD_ORDER=newest-first
PACIFICA_R2_FRESHNESS_CHECK_INTERVAL_S=3600
PACIFICA_R2_FRESHNESS_STALE_AFTER_MIN=180
PACIFICA_R2_FRESHNESS_SAMPLE_PREFIXES=channel=bbo/symbol=BTC,channel=book/symbol=ETH,channel=trades/symbol=BTC,channel=mark_price_candle/symbol=ICP
```

Verification before/after deploy:

```text
uv run pytest tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_build_pacifica_paper_ledger.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_validate_pacifica_idea_registry.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
124 passed in 1.94s

uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
33 passed in 0.50s

python -m py_compile scripts/build_pacifica_event_risk_calendar.py scripts/build_pacifica_paper_ledger.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_regime_governor.py scripts/build_pacifica_symbol_lifecycle.py scripts/check_pacifica_feature_parity.py scripts/check_pacifica_r2_freshness.py scripts/plan_pacifica_ops_alerts.py scripts/run_pacifica_walk_forward_validation.py scripts/simulate_pacifica_execution.py scripts/validate_pacifica_idea_registry.py scripts/pacifica_full_fidelity_storage.py scripts/pacifica_r2_inventory.py scripts/run_pacifica_fly_ops_watchdogs.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/entrypoint.sh
git diff --check
# all passed
```

Caveat: `uv run pytest tests/scripts -q` timed out after 600s at about 69% progress in this working tree. The targeted new/changed test set above passed; do not report the full scripts suite as green until the slow/hanging remainder is isolated or run with a longer budget.

Post-deploy observations:

```text
Fly status at 2026-05-12T11:28Z: started, version 22, image deployment-01KRDYHQ7A79GFCM2RGR08NEXM
New lifecycle started: 2026-05-12T11:19:43Z
New ops watchdog started: 2026-05-12T11:19:43Z
Ops watchdog reported failures at 2026-05-12T11:20:03Z (expected while R2 freshness is still stale)
Recent bounded lifecycle scan completed: 2026-05-12T11:21:20Z scanned=4752
No post-version-22 fresh-upload/verify completion was visible in bounded logs by 2026-05-12T11:45Z.
```

Bounded R2 freshness check from local read-only script at `2026-05-12T11:28:54Z`:

```text
ok=false
failures=[R2_REMOTE_FRESHNESS_STALE]
latest_payload=channel=bbo/symbol=BTC/date=2026-05-12/hour=08/run-20260511T150931Z.jsonl.gz
latest_payload_modified=2026-05-12T04:03:57Z
latest_payload_age_min=444.95
payload_count=79
sidecar_count=79
sidecar_missing_count=0
listing_errors=[]
```

Interpretation: version 22 deployed and started cleanly; the new bounded recent scan is active and much smaller than the previous full scan, but R2 archive freshness has not recovered yet. Keep alert severity fail-closed until a later bounded R2 freshness check is under the 180-minute threshold and a lifecycle upload/verify line confirms the version-22 cycle reached upload completion. Do not start competing manual upload/verify writers against the active lifecycle SQLite DB.

## Latest 2026-05-11 lifecycle/freshness-lane update

Timestamp: `2026-05-11T15:15Z`

Deployed Fly image/version:

```text
Image: pacifica-full-fidelity:deployment-01KRBS9KFXCN3G9XDBE5XQJ520
Machine: e2862502a76778
Version: 21
State: started
Last updated: 2026-05-11T15:09:30Z
```

Why this deployment exists: a 12h follow-up showed the collector/lifecycle was healthy but R2 remained stale for bounded May 11 BTC prefixes. The earlier `newest-first` upload lane still ordered by `last_seen_at`, which is refreshed for all rows on every scan; when all rows shared the same scan timestamp, object-key tie-breaking could still walk lexicographic backlog instead of truly newest sealed chunks.

What changed:

- `archive_files` now has a `modified_at` column, auto-added by SQLite migration if missing.
- `scan_archive_files` records local payload mtime and reuses the saved SHA-256 for unchanged `(size_bytes, modified_at)` rows instead of rehashing every file every cycle.
- `--upload-order newest-first` now orders by `coalesce(modified_at, last_seen_at) desc`, preserving errored-uploaded repair priority before new sealed rows.
- `upload-verify` now supports split upload/verify limits, and Fly is configured with:
  - `PACIFICA_FULL_FIDELITY_UPLOAD_LIMIT=2000`
  - `PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500`
- Rationale for split limits: prioritize newest sealed payload visibility in R2 while reducing per-cycle verification read/metadata calls. Verification/prune still runs every cycle, just at a lower cap than upload catch-up.
- Tests prove these behaviors:
  - newest-first chooses the newest mtime even when object-key order would choose an older `symbol=ZZZ` file;
  - unchanged files are not rehashed on rescan;
  - upload and verify limits can differ, e.g. upload 3 rows while verifying only 1.

Verification before deploy:

```text
uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py -q
21 passed
python -m py_compile scripts/pacifica_full_fidelity_storage.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
bash -n ops/fly/pacifica-full-fidelity/entrypoint.sh
git diff --check -- scripts/pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_full_fidelity_storage.py scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/fly.toml ops/fly/pacifica-full-fidelity/entrypoint.sh
```

Post-deploy observations:

```text
Fly status: started, version 21, image deployment-01KRBS9KFXCN3G9XDBE5XQJ520
New lifecycle started: 2026-05-11T15:09:30Z
Runtime env verified over Fly SSH:
  PACIFICA_FULL_FIDELITY_BATCH_LIMIT=2000
  PACIFICA_FULL_FIDELITY_UPLOAD_LIMIT=2000
  PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500
  PACIFICA_FULL_FIDELITY_UPLOAD_ORDER=newest-first
Previous version-20 first migration scan:
  start 2026-05-11T12:39:30Z
  scanned=86666 at 2026-05-11T14:39:43Z
  reset=0 at 2026-05-11T14:39:49Z
SQLite schema includes modified_at column.
DB counts right after version-20 migration check:
  pruned   17779  7852094488 bytes
  sealed   76027  43408115236 bytes
  uploaded 1207   15834297 bytes
  verified 8000   5677461625 bytes
```

Caveat: the first post-migration scan was expensive because existing DB rows started with `modified_at = null`; it populated mtimes during the version-20 cycle. Subsequent scans should avoid full rehashing unchanged rows and should make lifecycle effective cadence closer to the configured interval. Version 21 intentionally reduces verification from 2000 to 500 rows per cycle while keeping upload at 2000 rows per cycle; this should improve R2 freshness and reduce Cloudflare read/metadata calls, but verified/pruned backlog will catch up more slowly. Do not claim R2 freshness recovered until a bounded R2 sample shows current sealed May 11 payloads and sidecars; the pre-deploy bounded BTC date=2026-05-11 sample returned zero objects for BTC bbo/mark_price_candle/trades.

Preserve these unless Diego explicitly approves deletion:

- `pacifica-full-fidelity`
- `r2:pacifica-trading-data/raw/`
- `/data/pacifica_full_fidelity_storage.sqlite`
- local research artifacts unless intentionally refreshing

## Latest operational check

Timestamp: `2026-05-08T01:55Z`

Fly status:

```text
App: pacifica-full-fidelity
Machine: e2862502a76778
Region: iad
State: started
Image: pacifica-full-fidelity:deployment-01KQW7TZFH27DWBGD0HHF6PSW6
Last updated: 2026-05-05T14:15:54Z
```

Latest observed Fly health JSON from logs:

```text
checked_at=2026-05-08T01:51:05.214052+00:00
ok=true
failures=[]
free_gb=56.45
unverified_gb=34.69
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=BP/date=2026-05-08/hour=01/run-20260505T141555Z.jsonl.gz
newest_raw_age_min=-3.19
```

Latest lifecycle DB counts from health logs:

```text
pruned|10537|3249105572
sealed|57343|37251568677
verified|1681|706023406
rows_with_errors: not directly queried in this pass; recent lifecycle logs showed upload failed=0 and verify failed=0.
```

Latest lifecycle evidence:

```text
2026-05-08T01:31:08Z scanned=59159 state_db=/data/pacifica_full_fidelity_storage.sqlite
2026-05-08T01:31:13Z reset=0 skipped_missing=0 skipped_recent=0 dry_run=false
2026-05-08T01:50:59Z upload failed=0 skipped=66 uploaded=134; verify failed=0 skipped=0 verified=134
2026-05-08T01:51:04Z lifecycle complete
```

Ops watchdog evidence:

```text
2026-05-08T00:56:54Z ops watchdog run start
2026-05-08T00:57:10Z ops watchdog run complete
```

Latest uploaded watchdog status read from R2:

```text
checked_at=2026-05-07T23:56:49.935913+00:00
ok=true
operation=noop_not_due
returncode=0
stdout_tail=No watchdog operation due.
```

Interpretation: the active Fly collector is running, lifecycle upload/verify/prune is healthy in recent logs, free disk remains above Diego's 50 GiB floor, and recent raw freshness is good. A transient WebSocket reconnect (`no close frame received or sent`) appeared at 2026-05-08T00:04Z, but subsequent lifecycle/health output remained `ok=true` with fresh raw files.

## R2 raw archive smoke check

Top-level R2 prefixes at `2026-05-08T00:29Z`:

```text
app/
funding/
ops/
raw/
```

Active raw prefix channel dirs at `r2:pacifica-trading-data/raw/pacifica/full_fidelity`:

```text
channel=bbo/
channel=book/
channel=candle/
channel=mark_price_candle/
channel=pong/
channel=prices/
channel=subscribe/
channel=trades/
rest/
```

Bounded R2 sample:

```text
r2:pacifica-trading-data/raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-07
count=42 objects
bytes=62866838
latest sampled payload=hour=20/run-20260505T141555Z.jsonl.gz
sample bytes=2252576
sha256=3f55663f4d69e34ae7d540697587b950b63d8ee3fd44b34fdb7180c4a49a0eb7
sidecar matched the payload hash
sample gzip decompressed/read successfully for at least 10 JSONL rows
```

Interpretation: this is a bounded R2 smoke check, not full-bucket proof. It independently confirms active raw payload + sidecar presence and gzip readability for a recent BTC BBO object.

## Legacy cleanup status

Done:

- Legacy Fly app `pacifica-collector` destroyed.
- Legacy GitHub Action `.github/workflows/daily_sync.yml` removed.
- Legacy helper scripts removed:
  - `scripts/sync_cloud_data.sh`
  - `scripts/sync_launch.sh`
  - `scripts/sync_remote.py`
- Removal committed as `c80a3f0 chore: remove legacy Pacifica collector GitHub Action`.
- Local laptop legacy dirs verified missing:
  - `data/prices`
  - `data/orderbook`
  - `data/trades`
  - `data/funding`
  - `data/app`

R2 legacy purge status:

- Top-level R2 prefixes at `2026-05-08T00:29Z` still included `app/` and `funding/`.
- `prices/`, `orderbook/`, and `trades/` remained absent from the top-level prefix listing.
- A bounded legacy `funding/` listing timed out in this pass, so do not infer its object count/size.

Interpretation:

```text
prices/    cleared
orderbook/ cleared
trades/    cleared
app/       still present
funding/   still present
raw/       preserved active archive
ops/       preserved/non-target
```

Do not claim R2 legacy cleanup complete until `app/` + `funding/` are verified gone/empty. Any new destructive purge needs explicit Diego approval and must preserve `raw/` and `ops/`.

## R2 raw archive / local research cache snapshot

Refreshed local research raw cache from active R2 raw prefix on 2026-05-08:

```text
rclone copy r2:pacifica-trading-data/raw/pacifica/full_fidelity data/pacifica_full_fidelity --transfers 16 --checkers 32 --fast-list --stats 30s
```

Local cache inventory after refresh:

```json
{
  "path": "data/pacifica_full_fidelity",
  "files": 34834,
  "payloads": 17417,
  "sha256_sidecars": 17417,
  "gib": 21.605,
  "symbols": 66,
  "dates": ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"]
}
```

Earlier read-only raw archive reports remain at:

- `docs/ops/pacifica-r2-raw-health-latest/README.md`
- `docs/ops/pacifica-r2-raw-health-latest/summary.json`

A long `rclone lsf ... --recursive` inventory process from an earlier session was killed after the local health summary was produced; do not rely on partial `data/pacifica_r2_raw_health/raw_lsf_pst.txt` as final inventory.

## Research refresh completed

Restored current raw archive from R2 to local `data/pacifica_full_fidelity/`, then rebuilt silver/regime/toxic/eligibility artifacts without changing thresholds or gates.

Command chain completed successfully:

```text
uv run python scripts/build_pacifica_full_fidelity_silver.py --raw-dir data/pacifica_full_fidelity --out-dir data/pacifica_silver_partitioned --layout partitioned --chunk-size 250000
uv run python scripts/build_non_hft_regime_state.py --silver-dir data/pacifica_silver_partitioned --out-dir docs/experiments/non-hft-regime-state
uv run python scripts/non_hft_toxic_overlay_probe.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/toxic-regime-overlay
uv run python scripts/build_pacifica_eligibility_gates.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/paper-trading-eligibility
```

Output:

```text
bbo: 12335891 rows
book: 12738965 rows
candle: 1723416 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned
wrote 519903 regime-state rows to docs/experiments/non-hft-regime-state
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
wrote report: docs/experiments/toxic-regime-overlay/README.md
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
symbols_evaluated: 65
eligible_symbols: 0
wrote report: docs/experiments/paper-trading-eligibility/README.md
```

Current research reports:

- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`
- `docs/experiments/paper-trading-eligibility/README.md`
- `docs/experiments/paper-trading-economics-baselines/README.md`
- `docs/experiments/execution-simulator/README.md`
- `docs/experiments/paper-ledger/README.md`
- `docs/experiments/regime-governor/README.md`
- `docs/experiments/feature-parity/README.md`

System level-up plan:

- `docs/plans/2026-05-08-pacifica-system-level-up.md`

Interpretation discipline:

- Archive has 8 distinct dates, still diagnostic.
- Toxic probe verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligibility verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligible symbols: `0`, mainly due to sample/activity gates.
- Do not tune toxicity thresholds on this sample.
- Do not treat the diagnostic toxicity/probe output as an edge claim.

## Economics/baselines contract added

Existing report:

- `docs/experiments/paper-trading-economics-baselines/README.md`

Locked baseline assumptions before strategy work:

- Taker fee: 4 bps per side.
- Maker fee: 1.5 bps per side.
- Taker/taker round trip before slippage: 8 bps.
- Maker/maker round trip before adverse selection: 3 bps.
- Taker/maker round trip before slippage/adverse selection: 5.5 bps.
- Every backtest/paper run must include fees, slippage/adverse selection, funding, dumb baselines, random same-frequency controls, drawdown, Sortino, trade/day/symbol concentration, and post-cost PnL.

## Tests/checks

Completed after the 2026-05-08 research refresh:

```text
uv run pytest tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_non_hft_toxic_overlay_probe.py tests/scripts/test_build_pacifica_eligibility_gates.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
21 passed in 0.28s

python -m py_compile scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/non_hft_toxic_overlay_probe.py scripts/build_pacifica_eligibility_gates.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed
```

## Watchdog R2 inventory fix

Completed on 2026-05-08:

- Replaced due watchdog raw inventory from full recursive `rclone lsjson` with line-oriented `rclone lsf --recursive --files-only --format pst --separator ';'` streamed to `/data/ops/r2_inventory.lsf`.
- Added `rclone_lsf_to_inventory` / `write_inventory_csv_from_lsf` conversion with LF-normalized CSV output.
- Updated watchdog summary output to record both `r2_inventory.lsf` and `r2_inventory.csv`.
- Added regression tests for line-oriented parsing, LF CSV output, and `run_once` command selection so the watchdog does not regress to `lsjson`.
- Added `pandas` to the Fly image because the existing retention planner imports pandas.
- Deployed to Fly image `pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF`; machine `e2862502a76778` reached `started` / good state at `2026-05-08T03:08:07Z`.
- Post-deploy logs showed `ops watchdog run start` at `2026-05-08T03:08:07Z` and `ops watchdog run complete` at `2026-05-08T03:08:15Z`.
- Latest uploaded R2 watchdog status after deploy was `ok=true` with `operation=noop_not_due` at `2026-05-08T03:08:08.535213+00:00`.

Verification:

```text
uv run pytest tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_plan_pacifica_r2_retention.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
13 passed in 0.40s

python -m py_compile scripts/pacifica_r2_inventory.py scripts/plan_pacifica_r2_retention.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed

git diff --check
# passed

Local bounded smoke using BTC/BBO 2026-05-07 R2 prefix:
r2_inventory_lsf ok=true returncode=0 wrote 3082 bytes
r2_inventory_csv ok=true returncode=0
r2_retention_plan ok=true returncode=0 objects=46 eligible_for_review=0 delete_command_written=False
```

Note: the first deploy attempt `deployment-01KR2RNZY91HA5TNF6NH8JDYJG` exposed a Fly image dependency gap (`pandas` missing for retention planning). The second deploy fixed it. Do not remove pandas from the Fly image unless the retention planner is rewritten to stdlib.

## 12h Fly/R2 health check — 2026-05-08T12:31Z

Live status:

- Fly app `pacifica-full-fidelity` is running image `pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF`.
- Machine `e2862502a76778` is `started` in `iad`, version `17`, last updated `2026-05-08T03:08:07Z`.
- Latest uploaded watchdog status copied from R2: `checked_at=2026-05-08T12:12:20.493277+00:00`, `ok=true`.
- Latest watchdog due operation was `api_surface_watch`, `ok=true`, `returncode=0`, `changed=False`.
- Watchdog logs continued hourly after the deploy fix: latest observed run start `2026-05-08T12:11:46Z`, complete `2026-05-08T12:12:36Z`.
- Latest complete lifecycle cycle observed in logs finished at `2026-05-08T10:52:32Z` with `upload.failed=0`, `upload.uploaded=142`, `verify.failed=0`, `verify.verified=142`.
- Latest health JSON observed after that lifecycle: `checked_at=2026-05-08T10:52:33.377594+00:00`, `ok=true`, `free_gb=54.31`, `newest_raw_age_min=-2.71`, `failures=[]`, `sealed.files=61393`, `pruned.files=11146`, `verified.files=1659`, `unverified_gb=36.99`.
- A newer lifecycle cycle had started at `2026-05-08T11:25:20Z`; by `2026-05-08T12:37:05Z` it had scanned `63836` rows and reset `0` rows, but upload/verify completion had not yet appeared in the sampled log tail.

R2 checks:

- Top-level R2 prefixes remain: `app/`, `funding/`, `ops/`, `raw/`. Legacy `app/` and `funding/` still exist; no destructive cleanup was run.
- Raw prefix dirs include `channel=bbo/`, `channel=book/`, `channel=candle/`, `channel=mark_price_candle/`, `channel=pong/`, `channel=prices/`, `channel=subscribe/`, `channel=trades/`, and `rest/`.
- Ops prefix still has `watchdogs/latest/`.
- Bounded R2 sample check on `raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-08` found `18` objects: `9` payloads and `9` `.sha256` sidecars.
- Latest sampled BTC/BBO payload: `channel=bbo/symbol=BTC/date=2026-05-08/hour=07/run-20260508T030808Z.jsonl.gz`; `.sha256` matched and gzip JSONL read 10 rows successfully.
- A full recursive local `rclone lsf` inventory of `raw/pacifica/full_fidelity` timed out at 240s during this check, so the health evidence above intentionally used bounded prefix checks plus Fly logs and uploaded watchdog status. Do not treat the timeout as data loss; treat it as confirmation that full-bucket inventories need generous watchdog timeouts or bounded/partitioned checks.

## Sample maturity / trading-readiness answer — 2026-05-08T13:07Z

Diego asked whether the data is still insufficient. Answer: yes, for edge validation and paper trading it is still insufficient.

Current refreshed research artifacts show:

- Regime-state report: `519903` 1-minute rows across `65` symbols.
- Toxic overlay report verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Toxic overlay sample: `8` distinct dates, `65` symbols.
- Toxic overlay serious-validation gate: `30` distinct days and at least `100` removed high-toxicity observations.
- Paper-trading eligibility verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Symbols evaluated: `65`; eligible symbols: `0`.
- Eligibility gates: `sample_gate_pass=0/65`, `activity_gate_pass=0/65`, `eligible=0/65`.

Interpretation:

- The data is useful for plumbing diagnostics and early sanity checks, not for claiming edge.
- Current sample is `8` days; need roughly `2` more days for 10-day early sanity, `6` more days for 14-day early sanity, `22` more days for 30-day provisional validation, and `52` more days for 60-day preferred validation.
- Do not tune toxicity cutoffs on this diagnostic sample.
- Do not launch paper trading yet; all symbols remain blocked by eligibility gates.
- Next research action is to keep collecting, then rerun the fixed silver/regime/toxic/eligibility pipeline at 10-14 days for early sanity and at 30+ days for provisional validation.

## System level-up foundation — 2026-05-08

Diego approved working on the full system level-up track. Added:

- `docs/plans/2026-05-08-pacifica-system-level-up.md` — phased implementation plan covering execution economics, paper ledger, no-trade governor, feature parity, walk-forward validation, symbol lifecycle, reference-market context, alerting, event risk, and research idea registry.
- `scripts/simulate_pacifica_execution.py` + `tests/scripts/test_simulate_pacifica_execution.py` — strategy-neutral execution-cost simulator with fees, slippage, adverse selection, funding, Markdown/CSV report output.
- `docs/experiments/execution-simulator/README.md`, `assumptions.csv`, `example_round_trips.csv`.
- `scripts/build_pacifica_paper_ledger.py` + `tests/scripts/test_build_pacifica_paper_ledger.py` — strategy-neutral paper-ledger spine with fills, positions, fees, funding, realized PnL, equity curve, drawdown, and ineligible-symbol refusal when diagnostic override is disabled.
- `docs/experiments/paper-ledger/README.md`, `fills.csv`, `positions.csv`, `equity_curve.csv`, `summary.csv`.

## System level-up Phase 3 — no-trade regime governor — 2026-05-08

Added:

- `scripts/build_pacifica_regime_governor.py` + `tests/scripts/test_build_pacifica_regime_governor.py`.
- `docs/experiments/regime-governor/README.md`, `governor_decisions.csv`, `decision_summary.csv`, `thresholds.csv`.

Current generated governor summary over the 519,903-row regime table:

```text
TRADABLE_DIAGNOSTIC              3,383 rows
REDUCE_SIZE_DIAGNOSTIC           8,344 rows
SKIP_TOXIC_REGIME                    1 row
SKIP_WIDE_SPREAD                   187 rows
SKIP_THIN_DEPTH                  1,017 rows
SKIP_STALE_DATA                506,826 rows
SKIP_MARK_DISLOCATION              145 rows
SKIP_FORCED_FLOW_AFTERSHOCK          0 rows
```

Audit note: an independent review flagged potential fail-open behavior. Fixed before handoff by requiring all safety columns, filling NaN safety metrics with conservative skip-triggering values, making missing BBO or trade activity fail closed to `SKIP_STALE_DATA`, and forcing zero-count fixed states into the summary.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
19 passed
python -m py_compile scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Interpretation:

- These artifacts are accounting/validation/governor infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is online/offline feature parity before any live microbatch feature drives decisions.

## System level-up Phase 4 — online/offline feature parity — 2026-05-08

Added:

- `scripts/check_pacifica_feature_parity.py` + `tests/scripts/test_check_pacifica_feature_parity.py`.
- `docs/experiments/feature-parity/README.md`, `summary.csv`, `mismatches.csv`, `missing_keys.csv`, `version_mismatches.csv`, `metadata_mismatches.csv`, `invalid_metadata.csv`, `invalid_features.csv`, `invalid_keys.csv`, `duplicate_keys.csv`, `feature_columns.csv`.
- Bootstrap input artifacts under `docs/experiments/feature-parity/`: `offline_bootstrap_features.parquet`, `live_bootstrap_features.csv`.

Current generated parity verdict:

```text
PARITY_FAIL_DIAGNOSTIC
failure_reasons=missing_required_columns
missing_metadata_columns=offline.available_ts;offline.computed_at;offline.watermark_ts;offline.feature_version;offline.provisional_final_flag;live.available_ts;live.computed_at;live.watermark_ts;live.feature_version;live.provisional_final_flag
```

Interpretation: the parity harness is implemented and intentionally fails closed on the current bootstrap/current-style feature inputs because the offline/live feature artifacts do not yet carry required metadata (`available_ts`, `computed_at`, `watermark_ts`, `feature_version`, `provisional_final_flag`). This blocks live microbatch feature use until the builders emit parity-ready metadata.

Audit note: independent reviews found fail-open cases during development. Fixed before handoff: parity now fails for missing metadata, metadata mismatches/nulls, empty overlaps, missing keys, duplicate keys, invalid/blank keys, feature-version mismatches, nonnumeric/null/non-finite feature values, empty feature-column configuration, and invalid tolerance (`nan`, `inf`, negative). CLI exits nonzero on `PARITY_FAIL_DIAGNOSTIC` unless `--allow-fail-diagnostic` is explicitly passed for report generation.

Verification:

```text
uv run pytest tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
33 passed
python -m py_compile scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are parity-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is walk-forward validation.

## System level-up Phase 5 — walk-forward validation — 2026-05-08

Added:

- `scripts/run_pacifica_walk_forward_validation.py` + `tests/scripts/test_run_pacifica_walk_forward_validation.py`.
- `docs/experiments/walk-forward-validation/README.md`, `summary.csv`, `config.csv`, `windows.csv`, `window_scorecard.csv`, `random_controls.csv`.
- Bootstrap input artifact: `docs/experiments/walk-forward-validation/bootstrap_strategy_returns.csv`.

Current generated walk-forward verdict:

```text
INSUFFICIENT_SAMPLE_DIAGNOSTIC
failure_reasons=insufficient_distinct_days;no_purged_validation_windows
```

Interpretation: the walk-forward harness is implemented, but the bootstrap artifact is intentionally diagnostic only. Future strategy-return inputs must pass purged chronological OOS windows, sample maturity, day/symbol concentration, post-cost PnL, baseline comparison, and random same-frequency controls before any result can be discussed as evidence.

Audit note: independent review found fail-open cases during development. Fixed before handoff: CLI `--allow-fail-diagnostic` only allows clean insufficient-sample diagnostics, not real provisional/validation failures or invalid required fields; random controls cannot be disabled for a passing verdict; invalid timestamps/symbols/eligible flags and nonnumeric/non-finite returns fail closed; string `eligible=False` is parsed as false; overlapping test windows are rejected; no-purge configs cannot pass; OOS maturity/concentration and feasible per-window concentration are enforced.

Verification:

```text
uv run pytest tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
45 passed
python -m py_compile scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are validation-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is symbol lifecycle promotion/demotion.

## System level-up Phase 6 — symbol lifecycle promotion/demotion — 2026-05-08

Added:

- `scripts/build_pacifica_symbol_lifecycle.py` + `tests/scripts/test_build_pacifica_symbol_lifecycle.py`.
- `docs/experiments/symbol-lifecycle/README.md`, `symbol_lifecycle.csv`, `state_counts.csv`, `transitions.csv`, `config.csv`.

Current generated lifecycle verdict:

```text
NO_ELIGIBLE_SYMBOLS_DIAGNOSTIC
ELIGIBLE=0
DISABLED=65
```

Interpretation: all currently evaluated symbols remain disabled by the diagnostic lifecycle, mostly because the archive is still young and activity/sample gates have not matured. `paper_trading_allowed_diagnostic=False` for every current symbol. This is expected and does not authorize paper/live trading.

Audit note: independent reviews found fail-open cases during development. Fixed before handoff: lifecycle now rejects missing required columns, duplicate/null/dirty symbols, invalid booleans, invalid numeric counts, unknown/duplicate previous lifecycle states, and dirty baseline scorecards; sticky `RETIRED` cannot be silently unretired; `eligible=True` only promotes when all upstream gates are consistent; inconsistent eligibility snapshots disable instead of promote; missing or bad post-cost baseline disables promotable symbols when a baseline scorecard is supplied, including an explicitly empty scorecard.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
56 passed
python -m py_compile scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are lifecycle-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.

## System level-up Phase 7 — cross-venue/reference market context — 2026-05-08

Added:

- `scripts/build_pacifica_reference_context.py` + `tests/scripts/test_build_pacifica_reference_context.py`.
- `docs/experiments/reference-market-context/README.md`, `reference_context.csv`, `risk_state_summary.csv`, `symbol_reference_summary.csv`, `config.csv`.

Current generated reference-context verdict:

```text
NO_ROWS_DIAGNOSTIC
reference_available_rows=0
```

Interpretation: the builder/report layer exists, but no production external reference feed has been wired yet. It starts from pluggable local CSV/parquet inputs and intentionally does not hardwire paid APIs. Missing reference rows are flagged rather than imputed. This is context infrastructure only, not a trade signal and not permission to paper/live trade.

Audit note: independent review found fail-open/misleading cases during development. Fixed before handoff: duplicate canonical keys after numeric coercion fail closed; fractional `bucket_start_ms` keys fail closed; negative volatility fails closed; high-vol positive reference returns are labeled `HIGH_VOL_RISK_ON` instead of risk-off; partial/missing BTC/ETH major-reference beta proxy coverage is explicitly labeled `PARTIAL_MAJOR_REFERENCE` or `MISSING_MAJOR_REFERENCE` rather than silently imputed/blank.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
63 passed
python -m py_compile scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are reference-context infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is external ops alerting.

## System level-up Phase 8 — external ops alerting — 2026-05-08

Added:

- `scripts/plan_pacifica_ops_alerts.py` + `tests/scripts/test_plan_pacifica_ops_alerts.py`.
- `docs/ops/pacifica-alerting/README.md`, `alert_plan.csv`, `summary.json`, `thresholds.csv`, `input_snapshot.json`.

Current generated alert-plan verdict:

```text
WARN
PAGE=0
WARN=1
OK=13
```

Interpretation: this is an alert-classification planner, not actual notification delivery. The bootstrap diagnostic snapshot is intentionally not a live check and only warns because no external delivery channel is configured in the artifact. Health facts and delivery are kept separate; no delivery credentials are stored or committed.

Alert conditions classified:

- Fly app not started.
- Raw freshness stale above 15 minutes.
- Free disk below Diego's 50 GiB floor.
- Lifecycle DB errors, upload failures, or verify failures.
- R2 raw prefix missing, remote freshness stale, or sidecar mismatch.
- Watchdog status stale or not OK.
- API surface changed.
- Archive inventory stale/timeout as WARN, not PAGE when otherwise healthy.
- Research refresh failed as WARN.
- Missing delivery channel as WARN.

Audit note: independent review passed. The planner fails closed to `PAGE` on missing or invalid required status signals, treats empty/unknown alert frames as `PAGE`, keeps inventory timeout as WARN-only unless other health facts page, and does not claim alerts have actually been delivered.

Verification:

```text
uv run pytest tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
70 passed
python -m py_compile scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open or misleading-delivery issues found.

Interpretation:

- These artifacts are alert-planning infrastructure only.
- They do not replace live Fly/R2 health checks.
- They do not send notifications until wired through Hermes cron/chat or another external delivery path.
- Next implementation phase is event/calendar risk.

## System level-up Phase 9 — event/calendar risk layer — 2026-05-08

Added:

- `scripts/build_pacifica_event_risk_calendar.py` + `tests/scripts/test_build_pacifica_event_risk_calendar.py`.
- `docs/experiments/event-risk-calendar/README.md`, `event_risk_rows.csv`, `event_risk_summary.csv`, `symbol_event_risk_summary.csv`, `config.csv`.

Current generated event-risk verdict:

```text
NO_EVENTS_CONFIGURED_DIAGNOSTIC
rows=519903
event_risk_rows=0
```

Interpretation: the event-risk builder/report layer exists and was run over the current 519,903-row regime table, but no production local event calendar is configured yet. All rows are marked `NO_KNOWN_EVENT_RISK` only because no event calendar was supplied; this does not mean the market is safe. This is context/governor infrastructure only, not a trade signal.

Input contract:

- Local CSV/parquet only; no hidden external API.
- Required event columns: `event_timestamp`, `event_type`, `pre_window_minutes`, `post_window_minutes`, `severity`, `source_note`.
- Supported severity: `LOW`, `MEDIUM`, `HIGH`.

Audit note: independent review found fail-open/misleading cases during development. Fixed before handoff: event text rejects semicolons/newlines/control characters that would corrupt semicolon-joined report fields; empty calendars produce `NO_EVENTS_CONFIGURED_DIAGNOSTIC` instead of conflating with configured-but-inactive event windows; empty state inputs now reach `NO_STATE_ROWS_DIAGNOSTIC` instead of crashing. The layer also fails closed on missing columns, dirty symbols, invalid/fractional bucket timestamps, invalid event timestamps, negative/fractional windows, noncanonical severity, blank/dirty event text, and preserves overlapping event types while taking the highest severity.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
79 passed
python -m py_compile scripts/build_pacifica_event_risk_calendar.py scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open or misleading-report issues found.

Interpretation:

- These artifacts are event-risk context infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is the research idea registry. (Completed below as Phase 10.)

## System level-up Phase 10 — research idea registry — 2026-05-08

Added:

- `docs/research/pacifica-idea-registry.md`.
- `scripts/validate_pacifica_idea_registry.py` + `tests/scripts/test_validate_pacifica_idea_registry.py`.
- `docs/research/pacifica-idea-registry-validation/README.md`, `idea_registry_validation.csv`, `summary.json`.

Current generated registry validation:

```text
registry_validation_verdict=PASS
ideas=3
errors=0
```

Registered diagnostic/pending ideas:

- `IDEA-001` toxic regime no-trade overlay — `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- `IDEA-002` event-risk no-trade overlay — `PENDING_DIAGNOSTIC`.
- `IDEA-003` reference-market dislocation governor — `PENDING_DIAGNOSTIC`.

Interpretation: the registry is a pre-registration/schema gate only. A registry schema `PASS` means the idea is falsifiable enough to test; it is not evidence of alpha, not permission to paper/live trade, and does not override the current sample/eligibility gates.

Fail-closed validator coverage:

- Requires hypothesis, mechanical label, trade/risk action, cost model, validation window, frozen parameters, kill criteria, OOS plan, and result/verdict.
- Rejects missing fields, duplicate IDs, placeholders, qualitative/visual mechanical labels, discretionary actions, missing measurable/comparison language, missing cost/OOS/kill/frozen-parameter semantics, and edge/proven-alpha claims.
- Rejects negated controls such as no/without fees, costs not modeled, no failure gates, continue retuning, parameters not fixed/can be retuned, no OOS, and `PASS; edge is proven`.
- Report columns distinguish `registration_schema_verdict` from `research_result_verdict` to avoid reading schema PASS as research PASS.

Verification:

```text
uv run pytest tests/scripts/test_validate_pacifica_idea_registry.py tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
86 passed
python -m py_compile scripts/validate_pacifica_idea_registry.py scripts/build_pacifica_event_risk_calendar.py scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Independent audit: first audit found fail-open cases around negated cost/OOS/kill/frozen controls, qualitative labels with keywords, and misleading PASS wording. Fixed and re-audited; final audit `PASS`.

Interpretation:

- These artifacts are idea-governance infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- The system level-up spine Phases 1-10 are now implemented.

## Continuation ops check — 2026-05-08 18:29 UTC

Read-only live checks run after Phase 10 completion.

Fly status:

```text
checked_at=2026-05-08T18:28:57Z
App: pacifica-full-fidelity
Machine: e2862502a76778
Region: iad
State: started
Image: pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF
Last updated: 2026-05-08T03:08:07Z
```

Latest completed lifecycle evidence from logs:

```text
2026-05-08T17:45:08Z upload failed=0 skipped=0 uploaded=200; verify failed=0 skipped=0 verified=200
2026-05-08T17:45:12Z lifecycle complete
```

Latest observed health JSON from logs:

```text
checked_at=2026-05-08T17:45:13.135039+00:00
ok=true
failures=[]
free_gb=52.81
unverified_gb=38.31
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=kBONK/date=2026-05-08/hour=17/run-20260508T030808Z.jsonl.gz
newest_raw_age_min=-2.85
db_counts: pruned=11593 files / 3678491035 bytes; sealed=63266 files / 41140405212 bytes; verified=1691 files / 602515107 bytes
```

Current caveat: a new lifecycle cycle started at `2026-05-08T18:18:10Z`; the bounded log check had not yet observed its upload/verify completion. Do not treat that as failure; use the next log/status check to confirm the next `lifecycle complete` line.

Ops watchdog evidence:

```text
2026-05-08T16:44:17Z ops watchdog run reported failures
2026-05-08T17:44:17Z ops watchdog run start
2026-05-08T17:44:41Z ops watchdog run complete
uploaded latest_status checked_at=2026-05-08T17:44:35.693259+00:00 ok=true operation=noop_not_due
```

Interpretation: the earlier watchdog failure was followed by an OK uploaded watchdog status and a later complete run. Keep watching, but latest watchdog artifact is OK.

API-surface watchdog artifact:

```text
api_surface_diff changed=false
rest_paths added=[] removed=[]
ws_sources added=[] removed=[]
intervals added=[] removed=[]
```

R2 bounded checks:

```text
top-level prefixes: app/, funding/, ops/, raw/
active raw channel dirs: channel=bbo/, channel=book/, channel=candle/, channel=mark_price_candle/, channel=pong/, channel=prices/, channel=subscribe/, channel=trades/, rest/
BTC BBO sample: raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-07/hour=23/run-20260505T141555Z.jsonl.gz
sha256 sidecar matched: b923514f6439bb47e68f4423e53d790ea3699ba98c7648fbb3e8c1f4924499d7
gzip JSONL readability: read 20 rows successfully
```

R2 legacy cleanup caveat remains: `app/` and `funding/` still exist at top level. Do not destructively purge without Diego's explicit approval. Preserve `raw/` and `ops/`.

## Alert planner refresh — 2026-05-08 18:38 UTC

Refreshed `docs/ops/pacifica-alerting/` from the latest observed non-secret health facts.

```text
overall_severity=WARN
PAGE=0
WARN=1
OK=13
```

Only WARN condition: `DELIVERY_CHANNEL_CONFIGURED` because no external notification delivery channel is configured in the repo artifact. All collector/R2/watchdog/API/research snapshot signals were classified OK using the latest bounded evidence. This remains an alert-classification artifact only; it does not send notifications.

Caveat: the refreshed snapshot uses the latest observed completed lifecycle at `2026-05-08T17:45:12Z`; the newer `2026-05-08T18:18:10Z` lifecycle cycle was still in progress/not yet complete in bounded logs.

## Continuation poll — 2026-05-08 19:23 UTC

Checked Fly logs again for the `2026-05-08T18:18:10Z` lifecycle cycle.

```text
checked_at=2026-05-08T19:23:40Z
2026-05-08T18:18:10Z lifecycle scan/upload/verify/prune start
```

No later upload/verify summary or `lifecycle complete` line was visible in that bounded log poll. The app still reported `started`, and a later ops watchdog run completed:

```text
2026-05-08T18:44:42Z ops watchdog run start
2026-05-08T18:45:08Z ops watchdog run complete
uploaded latest_status checked_at=2026-05-08T18:44:56.101339+00:00 ok=true operation=noop_not_due
```

Direct remote health probing caveat: one `flyctl ssh console ... python /app/scripts/check_pacifica_full_fidelity_health.py ...` attempt timed out after 180s, likely because it recursively scans the raw tree. A later shell-wrapped SSH diagnostic form was blocked by user approval policy (`BLOCKED: User denied. Do NOT retry.`). Do not retry that exact denied command form; use logs, bounded R2 checks, uploaded watchdog artifacts, or a safer already-deployed health/status artifact instead.

## Continuation poll — 2026-05-08 21:16 UTC

The previously pending `18:18` lifecycle cycle completed successfully in later logs:

```text
checked_at=2026-05-08T21:16:44Z
2026-05-08T18:18:10Z lifecycle scan/upload/verify/prune start
2026-05-08T19:32:28Z scanned=66133 state_db=/data/pacifica_full_fidelity_storage.sqlite
2026-05-08T19:32:34Z reset=0 skipped_missing=0 skipped_recent=0 dry_run=false
2026-05-08T19:52:47Z upload failed=0 skipped=63 uploaded=137; verify failed=0 skipped=0 verified=137
2026-05-08T19:52:54Z lifecycle complete
```

Latest observed health JSON after that cycle:

```text
checked_at=2026-05-08T19:52:55.933725+00:00
ok=true
failures=[]
free_gb=52.41
unverified_gb=38.93
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=ADA/date=2026-05-08/hour=19/run-20260508T030808Z.jsonl.gz
newest_raw_age_min=-3.5
db_counts: pruned=11795 files / 3785207612 bytes; sealed=64305 files / 41803780248 bytes; verified=1626 files / 559524101 bytes
```

A newer lifecycle cycle started at `2026-05-08T20:26:32Z`; bounded logs through `2026-05-08T21:16:44Z` had not yet shown its upload/verify completion. Latest uploaded watchdog artifact remained OK:

```text
watchdog latest_status checked_at=2026-05-08T20:45:57.374570+00:00 ok=true operation=noop_not_due
```

Interpretation: the earlier pending `18:18` cycle is now resolved healthy. Next follow-up should check completion of the `20:26` lifecycle cycle, not the already-completed `18:18` cycle.

## Continuation ops/alert refresh — 2026-05-09 15:24 UTC

Diego asked to proceed sequentially through the next ops items. Read-only checks were run without destructive R2 cleanup.

Direct Fly status/log/lifecycle probing caveat: an attempted combined Fly status/log command in this session was blocked by approval policy (`BLOCKED: User denied. Do NOT retry.`). Do not retry that exact form. The current refresh therefore uses bounded R2/watchdog/API artifacts and intentionally classifies unknown lifecycle/Fly/disk facts fail-closed rather than pretending they are OK.

Bounded R2/watchdog/API evidence:

```text
checked_at=2026-05-09T15:24:04Z
R2 top-level prefixes: app/, funding/, ops/, raw/
active raw channel dirs: channel=bbo/, channel=book/, channel=candle/, channel=mark_price_candle/, channel=pong/, channel=prices/, channel=subscribe/, channel=trades/, rest/
watchdog latest_status checked_at=2026-05-09T14:53:15.480756+00:00 ok=true operation=noop_not_due
api_surface_diff changed=false
latest sampled R2 raw payload across bounded current-day channel/symbol scan: channel=bbo/symbol=STRK/date=2026-05-09/hour=11/run-20260508T030808Z.jsonl.gz
latest sampled R2 raw modified=2026-05-09T06:24:13Z
latest sampled R2 raw age≈538.7 minutes at check time
latest sampled sidecar hash prefix matched local sha256; gzip JSONL readability read 20 rows successfully
```

Interpretation: bounded R2 evidence proves some raw uploads occurred after the previously pending `2026-05-08T20:26:32Z` lifecycle cycle start, but current bounded R2 raw freshness is stale by the alert thresholds and direct lifecycle/Fly DB health could not be verified in this pass. Treat this as `PAGE` until a safe Fly-side/status artifact confirms collector/lifecycle health or uploads resume.

Alert planner refreshed:

```text
docs/ops/pacifica-alerting/summary.json
overall_severity=PAGE
PAGE=5
WARN=1
OK=6
```

The PAGE state is intentionally conservative: invalid/unknown lifecycle counts, unknown Fly state, stale sampled raw freshness, unknown free disk, and stale sampled R2 remote freshness. `DELIVERY_CHANNEL_CONFIGURED` is now OK because a Hermes cron alert bridge exists.

Created external alert bridge:

```text
script=/Users/diego/.hermes/scripts/pacifica_ops_watchdog_alert.py
cron_job_id=e61c2f7c5593
name=pacifica-r2-watchdog-alert-bridge
schedule=every 30m
deliver=origin
mode=no_agent
```

The bridge is read-only and emits only on problems. It checks bounded R2 raw freshness, uploaded watchdog status, and API-surface diff. It does not perform Fly SSH/log lifecycle DB checks and does not delete anything. A manual run emitted a PAGE because the newest sampled R2 raw payload was ~538.7 minutes old.

Maturity decision: do not rerun the research pipeline yet. Local restored raw cache still has 8 distinct dates (`2026-04-30` through `2026-05-07`) and current artifacts remain diagnostic. First resolve/understand the collector or R2 upload freshness PAGE, then wait for at least 10-14 distinct days for early sanity reruns; keep 30+ days as provisional and 60+ days preferred serious validation.

## Continuation remediation — 2026-05-10 00:05 UTC

Diego asked to address everything. The prior PAGE was diagnosed and partially remediated.

Root cause found from safe Fly logs: the app was `started`, but collector writes were repeatedly blocked by Diego's 50 GiB free-disk guard:

```text
websocket collection error; reconnecting in 5s: free disk below safety floor: 50.0 GiB available under /data/pacifica_full_fidelity, requires at least 50.0 GiB
REST snapshot failed for /info: free disk below safety floor: 50.0 GiB available under /data/pacifica_full_fidelity, requires at least 50.0 GiB
```

Fly-side DB/disk state before remediation showed `/data` effectively at the 50 GiB floor and a large sealed backlog:

```text
volume before: 100GB
df before: /dev/vdc 98G size, 43G used, 50G avail
archive_files: pruned=13,558; sealed=71,105 (~43.8GB); verified=2,221 (~1.37GB); rows_with_errors=0
```

Actions taken:

```text
flyctl volumes extend vol_vwn2mpw8mmgwx38v -a pacifica-full-fidelity --size 200 --yes
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
```

Deployment/config now active:

```text
image=pacifica-full-fidelity:deployment-01KR7JQA20E7TFVKX6CANZ7H37
machine=e2862502a76778 version=18 state=started last_updated=2026-05-09T23:58:44Z
volume=vol_vwn2mpw8mmgwx38v size=200GB
PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S=900
PACIFICA_FULL_FIDELITY_BATCH_LIMIT=2000
PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=50
```

Verification after remediation:

```text
df after: /dev/vdc 197G size, 43G used, 145G avail
local collector fresh at 2026-05-10T00:04:38Z:
  BTC bbo current-hour file age≈0.05 min
  BTC book current-hour file age≈0.01 min
  BTC mark_price_candle current-hour file age≈0.07 min
archive_files: pruned=13,558; sealed=70,993; uploaded=112; verified=2,221; rows_with_errors=0
```

Important caveat: R2 remote freshness is still PAGE/stale until lifecycle upload/verify catches up. The alert planner was refreshed from post-remediation facts and now has only one PAGE condition:

```text
docs/ops/pacifica-alerting/alert_plan.csv
R2_REMOTE_FRESHNESS=PAGE
all other conditions=OK
```

A manual high-limit upload/verify attempt (`proc_f16ad5bb672f`) was started while the old lifecycle scan held the DB lock. It repeatedly failed with `sqlite3.OperationalError: database is locked` and exited with signal/exit code `-15` before deploy, after only a small number of rows advanced (`uploaded=112`). A post-deploy read-only DB check at `2026-05-10T00:07:32Z` showed `/data` healthy (`197G` size, `43G` used, `145G` available), `archive_files: pruned=13,558; sealed=70,993; uploaded=112; verified=2,221; rows_with_errors=0`. The redeployed lifecycle should now run every 15 minutes with batch limit 2000, but R2 catch-up still needs follow-up monitoring. Do not treat local collector freshness as proof that R2 archival has caught up.

## 24h follow-up — 2026-05-10 22:05 UTC

Diego noted it had been more than 24h since the last check. Current status remains partially healthy but not fully resolved.

Verified current facts:

```text
checked_at=2026-05-10T22:05Z
fly_state=started
image=pacifica-full-fidelity:deployment-01KR7JQA20E7TFVKX6CANZ7H37
machine=e2862502a76778 version=18
/data=/dev/vdc 197G size, 47G used, 141G available
latest local raw files fresh/current-hour around 2026-05-10T22:02-22:04Z for bbo/book/trades
rows_with_errors=0
latest watchdog status checked_at=2026-05-10T21:39:35Z ok=true age≈25.9m
api_surface_changed=false
```

Lifecycle DB direct read at the same follow-up:

```text
pruned|15,449|5,436,657,952 bytes
sealed|74,672|42,227,364,191 bytes
uploaded|421|160,694,162 bytes
verified|6,330|6,128,931,801 bytes
rows_with_errors=0
```

Change versus the post-remediation `2026-05-10T00:07Z` check:

```text
pruned_files +1,891 / pruned_bytes +1.04GB
verified_files +4,109 / verified_bytes +4.76GB
uploaded_files +309 / uploaded_bytes +114.8MB
sealed_files +3,679 / sealed_bytes -1.51GB
```

Interpretation: lifecycle is making safe non-destructive progress and verification/pruning increased materially, but the backlog is not caught up. `sealed` files net-increased because live collection kept adding sealed chunks faster than the backlog cleared in file-count terms, though sealed bytes decreased. R2 freshness is still stale by alert threshold.

Bounded R2 evidence:

```text
latest sampled R2 raw payload modified≈2026-05-10T14:09Z
latest sampled payload age≈471.2 minutes at 2026-05-10T22:00Z
sample sidecar hash matched and gzip read 20 rows
raw prefix present=true
```

Alert planner refreshed at `docs/ops/pacifica-alerting/`:

```text
overall_severity=PAGE
PAGE=1
WARN=0
OK=13
only PAGE condition: R2_REMOTE_FRESHNESS
```

Do not manually start another competing upload/verify loop while the lifecycle process is active. Continue monitoring direct DB counts and R2 freshness. Local collection and disk are healthy, but R2 archival freshness remains the unresolved PAGE.

## Remaining work

1. Monitor the version-22 bounded freshness lane deployed as `pacifica-full-fidelity:deployment-01KRDYHQ7A79GFCM2RGR08NEXM`. It has started and scanned 4,752 recent files, but no post-version-22 upload/verify completion was observed by `2026-05-12T11:45Z`.
2. Re-run the bounded R2 freshness checker after the version-22 lifecycle reaches upload completion:
   `uv run python scripts/check_pacifica_r2_freshness.py --remote-base r2:pacifica-trading-data --r2-prefix raw/pacifica/full_fidelity --stale-after-min 180 --timeout-s 45`.
   Current status is still `R2_REMOTE_FRESHNESS_STALE` with latest sampled payload age `444.95` minutes at `2026-05-12T11:28:54Z`.
3. Keep alert severity fail-closed until R2 freshness is under threshold and payload/sidecar pairing remains clean. The 2026-05-12 bounded sample had `payload_count=79`, `sidecar_count=79`, `sidecar_missing_count=0`, and no listing errors.
4. If R2 freshness remains stale after the fresh upload lane completes, inspect lifecycle/upload throughput and architecture next. Avoid competing manual SQLite writers; do not start high-limit manual upload/verify loops while the app lifecycle process is active.
5. Use Fly status/logs, bounded R2 samples, uploaded watchdog artifacts, and the Hermes cron alert bridge `e61c2f7c5593` for follow-up. Do not retry exact Fly SSH or `rclone cat` diagnostic forms that approval policy denied; use safer bounded listings/scripts instead.
6. Do not rerun research for edge claims yet. Reruns are OK only as plumbing diagnostics until distinct-day maturity reaches 10-14 days for early sanity, 30+ for provisional validation, and 60+ for preferred serious validation.
7. Do not claim edge or paper-trade until the 30-day sample gate and symbol eligibility gates pass; current eligibility artifacts remain diagnostic and previously showed `0/65` eligible.
8. Verify top-level R2 prefixes after any approved legacy purge. `app/` and `funding/` were still present in the last recorded top-level checks; `raw/` and `ops/` must remain preserved.
9. If Diego explicitly approves destructive scope, target only remaining legacy `app/` and `funding/` prefixes. Do not touch `raw/` or `ops/`.
10. System level-up Phases 1-10 plus the version-22 ops freshness lane are implemented. Keep future strategy work behind the registry, eligibility, economics, governor, parity, walk-forward, baseline, random-control, and concentration gates.

## Lifecycle freshness-lane remediation — 2026-05-11 02:36 UTC

Diego said to proceed with the next step: diagnose why lifecycle progress was not restoring R2 freshness. Root cause found:

```text
The lifecycle interval was set to 900s, but the lifecycle cycle itself was taking hours.
Observed cycle:
  2026-05-10T19:52:20Z lifecycle start
  2026-05-10T21:30:03Z scan completed, scanned=81,423
  2026-05-11T02:24:27Z upload/verify completed, uploaded=2,000 verified=2,000
  2026-05-11T02:24:46Z lifecycle complete
```

Why R2 stayed stale despite safe progress:

```text
1. scan hashes the full archive every cycle before uploading, which took ~1h38m in the observed cycle.
2. upload/verify is sequential per object/sidecar/size/cat and took ~4h54m for the 2,000-file batch.
3. upload candidate ordering was object_key ascending, so backlog catch-up walked old lexicographic partitions first instead of sending the newest eligible sealed files. This kept bounded R2 freshness stale even while verification/pruning progressed.
```

Implemented and deployed a non-destructive freshness-lane change:

```text
scripts/pacifica_full_fidelity_storage.py
  - added --upload-order {object-key,newest-first,oldest-first}
  - newest-first orders sealed upload candidates by last_seen_at desc, object_key desc
  - errored uploaded rows still remain first for repair safety

scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - passes PACIFICA_FULL_FIDELITY_UPLOAD_ORDER to upload-verify

ops/fly/pacifica-full-fidelity/fly.toml
  - PACIFICA_FULL_FIDELITY_UPLOAD_ORDER="newest-first"
```

Deployment:

```text
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
image=pacifica-full-fidelity:deployment-01KRAE6VQCAMV0P61R75YCD20Q
machine=e2862502a76778 version=19 state=started last_updated=2026-05-11T02:36:04Z
```

Post-deploy verification:

```text
upload_order=newest-first
python scripts/pacifica_full_fidelity_storage.py --help includes --upload-order
/data=/dev/vdc 197G size, 48G used, 141G available
archive_files: pruned=15,779; sealed=73,045; uploaded=48; verified=8,000; rows_with_errors=0
lifecycle scan/upload/verify/prune start at 2026-05-11T02:36:04Z
```

Tests/verification:

```text
uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py -q
19 passed

python -m py_compile scripts/pacifica_full_fidelity_storage.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
git diff --check -- scripts/pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_full_fidelity_storage.py scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/fly.toml
all passed
```

Important caveat: this changes which eligible files are uploaded first; it does not make the full scan or sequential rclone verification fast. R2 freshness should improve after the current post-deploy lifecycle reaches its upload phase. If freshness is still stale after that cycle, the next root-cause target is scan/verification architecture, not disk capacity.

## Blocked/avoid

- Do not retry exact shell/Python diagnostic forms that previously returned `BLOCKED: User denied. Do NOT retry.`
- Do not run destructive `rm -rf` cleanup chains or destructive R2 cleanup without explicit approval.
- Do not upload current/live time partitions from append-style raw collectors to R2/S3; lifecycle upload should continue to skip recent/current chunks.
- Do not claim an edge while reports remain `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.

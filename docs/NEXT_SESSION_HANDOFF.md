# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-05 14:58 UTC

## Current state

Active program: Hermes-only Pacifica full-fidelity, non-HFT paper-trading research. Do not use Claude Code, recreate `CLAUDE.md`, propose HFT/latency strategies, tune early toxicity thresholds, or claim edge from the current young archive.

Canonical active runtime/archive:

- Fly app: `pacifica-full-fidelity`
- Machine: `e2862502a76778`
- Active R2 archive prefix: `r2:pacifica-trading-data/raw/`
- Active local lifecycle DB on Fly: `/data/pacifica_full_fidelity_storage.sqlite`
- Local research raw cache: `data/pacifica_full_fidelity/` restored from R2 for research rebuilds
- Local silver output: `data/pacifica_silver_partitioned/`

Preserve these unless Diego explicitly approves deletion:

- `pacifica-full-fidelity`
- `r2:pacifica-trading-data/raw/`
- `/data/pacifica_full_fidelity_storage.sqlite`
- local research artifacts unless intentionally refreshing

## Latest operational check

Timestamp: `2026-05-05T14:53Z`

Fly lifecycle DB:

```text
pruned|5259|1369517001
sealed|37351|23082991137
verified|2714|743618723
rows_with_errors|0|0
```

Latest health JSON observed after deploy:

```text
ok=true
failures=[]
free_gb=70.11
unverified_gb=21.5
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=ENA/date=2026-05-05/hour=14/run-20260503T142803Z.jsonl.gz
```

Latest lifecycle evidence before deploy:

```text
2026-05-05T14:06:00Z upload failed=0 skipped=112 uploaded=88; verify failed=0 skipped=0 verified=95
2026-05-05T14:06:04Z lifecycle complete
```

After deploy:

```text
2026-05-05T14:15:54Z new image started
2026-05-05T14:15:54Z lifecycle scan/upload/verify/prune start
2026-05-05T14:15:54Z ops watchdog run start
2026-05-05T14:46:10Z ops watchdog run reported failures
```

Important: commit `4836fa5 fix: handle Pacifica watchdog timeout stderr bytes` has now been deployed in the latest Fly image. Post-deploy logs show watchdog failures but no observed post-deploy `TypeError: can't concat str to bytes` stack in the checked tail. The likely remaining issue is the large `rclone lsjson` inventory timing out, now reported without the bytes/string crash. Verify again on the next watchdog cycle.

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

R2 legacy purge still running:

- Process: `proc_354d8b93b915`
- Script: `/tmp/purge_legacy_pacifica_r2.sh`
- Targets only legacy prefixes:
  - `r2:pacifica-trading-data/prices`
  - `r2:pacifica-trading-data/orderbook`
  - `r2:pacifica-trading-data/trades`
  - `r2:pacifica-trading-data/funding`
  - `r2:pacifica-trading-data/app`
- Preserve: `r2:pacifica-trading-data/raw/`
- Latest observed top-level R2 prefixes at `2026-05-05T14:57Z`:

```text
app/
funding/
ops/
raw/
```

Interpretation:

```text
prices/    cleared
orderbook/ cleared
trades/    cleared
app/       still present / being purged or pending
funding/   still present / being purged or pending
raw/       preserved active archive
ops/       preserved/non-target
```

Latest visible purge progress:

```text
status=running
uptime_seconds≈96710
current stage deleting legacy BTC parquet keys
deleted≈1,444,309 files in current stage
freed≈4.957 GiB in current stage
elapsed≈8h40m in current stage
```

Do not claim R2 legacy cleanup complete until the process exits and `app/` + `funding/` are verified gone/empty.

## R2 raw archive health snapshot

A local read-only raw archive snapshot/report was generated from restored local raw cache:

- `docs/ops/pacifica-r2-raw-health-latest/README.md`
- `docs/ops/pacifica-r2-raw-health-latest/summary.json`

Summary:

```json
{
  "checked_at": "2026-05-05T14:33:55.231258+00:00",
  "root": "data/pacifica_full_fidelity",
  "jsonl_gz_files": 13348,
  "sha256_sidecars": 13348,
  "missing_sha_count": 0,
  "orphan_sha_count": 0,
  "gzip_sample_checked": 199,
  "gzip_sample_bad_count": 0,
  "bytes": 21433734139,
  "gib": 19.962,
  "symbols": 66,
  "dates": ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]
}
```

R2 size for `raw/pacifica/full_fidelity` around the same time:

```json
{"count":26696,"bytes":21435015547,"sizeless":0}
```

The small byte delta versus local summary is expected during active collection/lifecycle movement.

A long `rclone lsf ... --recursive` inventory process `proc_f8c1227771b8` was killed after the local health summary was produced; do not rely on partial `data/pacifica_r2_raw_health/raw_lsf_pst.txt` as final inventory.

## Research refresh completed

Restored current raw archive from R2 to local `data/pacifica_full_fidelity/`, then rebuilt silver/regime/toxic/eligibility artifacts.

Command chain completed successfully:

```text
uv run python scripts/build_pacifica_full_fidelity_silver.py --raw-dir data/pacifica_full_fidelity --out-dir data/pacifica_silver_partitioned --layout partitioned --chunk-size 250000
uv run python scripts/build_non_hft_regime_state.py --silver-dir data/pacifica_silver_partitioned --out-dir docs/experiments/non-hft-regime-state
uv run python scripts/non_hft_toxic_overlay_probe.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/toxic-regime-overlay
uv run python scripts/build_pacifica_eligibility_gates.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/paper-trading-eligibility
```

Output:

```text
bbo: 6824152 rows
book: 12348364 rows
candle: 1723416 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned
wrote 322782 regime-state rows to docs/experiments/non-hft-regime-state
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

Interpretation discipline:

- Archive has 6 distinct dates, still diagnostic.
- Toxic probe verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligibility verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligible symbols: `0`, mainly due to sample/activity gates.
- Do not tune toxicity thresholds on this sample.
- Do not treat the diagnostic toxicity/probe output as an edge claim.

## Economics/baselines contract added

New report:

- `docs/experiments/paper-trading-economics-baselines/README.md`

Locked baseline assumptions before strategy work:

- Taker fee: 4 bps per side.
- Maker fee: 1.5 bps per side.
- Taker/taker round trip before slippage: 8 bps.
- Maker/maker round trip before adverse selection: 3 bps.
- Taker/maker round trip before slippage/adverse selection: 5.5 bps.
- Every backtest/paper run must include fees, slippage/adverse selection, funding, dumb baselines, random same-frequency controls, drawdown, Sortino, trade/day/symbol concentration, and post-cost PnL.

## Tests/checks

Completed:

```text
uv run pytest tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_non_hft_toxic_overlay_probe.py tests/scripts/test_build_pacifica_eligibility_gates.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
21 passed in 0.33s

python -m py_compile scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/non_hft_toxic_overlay_probe.py scripts/build_pacifica_eligibility_gates.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed
```

## Remaining work

1. Continue/poll R2 legacy purge process `proc_354d8b93b915` until it exits.
2. Verify top-level R2 prefixes after purge. `app/` and `funding/` should be gone/empty; `raw/` and `ops/` should remain.
3. Verify next post-deploy ops watchdog cycle: it may still fail due to timeout, but should no longer crash with the bytes/string `TypeError`.
4. Consider changing Fly watchdog inventory from full `rclone lsjson --recursive` to a bounded or partitioned inventory, because full recursive raw inventory is now too large for the 1800s timeout.
5. Keep monitoring lifecycle DB for `rows_with_errors=0`, upload failures `0`, free disk above 50 GiB.
6. Let archive mature: 10-14 days early sanity, 30+ days provisional validation, 60+ preferred.
7. Next research step after maturity improves: rerun fixed silver/regime/toxic/eligibility pipeline without retuning thresholds, then build sparse strategy/backtest/paper logger only after gates and economics pass.

## Blocked/avoid

- Do not retry exact shell/Python diagnostic forms that previously returned `BLOCKED: User denied. Do NOT retry.`
- Do not run destructive `rm -rf` cleanup chains without explicit approval.
- Foreground commands above 600s are rejected; use background processes with `notify_on_complete=true`.
- Terminal sometimes collapses output to `1 lines output`; rely on explicitly printed values or files.
- Do not trust incomplete/timed-out inventory files.
- Do not remove active full-fidelity app/archive/DB without explicit approval.

# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-06 15:29 UTC

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

Timestamp: `2026-05-06T15:29Z`

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
checked_at=2026-05-06T15:08:22.973908+00:00
ok=true
failures=[]
free_gb=64.38
unverified_gb=26.93
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=kBONK/date=2026-05-06/hour=14/run-20260505T141555Z.jsonl.gz
newest_raw_age_min=-1.13
```

Latest lifecycle DB counts from health logs:

```text
pruned|7973|2113135724
sealed|45983|28919339676
verified|1819|724015654
rows_with_errors: not directly queried in this pass; recent lifecycle logs showed upload failed=0 and verify failed=0.
```

Latest lifecycle evidence:

```text
2026-05-06T15:08:17Z upload failed=0 skipped=65 uploaded=135; verify failed=0 skipped=0 verified=135
2026-05-06T15:08:21Z lifecycle complete
```

Ops watchdog evidence:

```text
2026-05-06T13:50:26Z ops watchdog run start
2026-05-06T13:50:41Z ops watchdog run complete
2026-05-06T14:50:41Z ops watchdog run start
2026-05-06T15:20:49Z ops watchdog run reported failures
```

Latest uploaded watchdog status read from R2:

```text
checked_at=2026-05-06T15:20:45.213452+00:00
ok=false
operation=r2_inventory_lsjson
returncode=124
stdout_tail=partial stdout retained at /data/ops/r2_inventory.lsjson
stderr_tail=Config file "/root/.config/rclone/rclone.conf" not found - using defaults; timeout after 1800s
```

Interpretation: the post-deploy bytes/string `TypeError` is fixed, but the due R2 raw inventory still times out at 1800s. Treat this as an ops-watchdog implementation issue, not evidence that the collector/archive stopped.

Collector note: logs showed transient WebSocket reconnects (`no close frame received or sent`) at 15:00Z and 15:03Z, but the lifecycle and health output after that still reported `ok=true` and fresh raw files.

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

- The old local Hermes background purge process `proc_354d8b93b915` was not present in the current session process table.
- Top-level R2 prefixes at `2026-05-06T14:54Z`:

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
app/       still present
funding/   still present
raw/       preserved active archive
ops/       preserved/non-target
```

Additional bounded check:

```text
rclone size r2:pacifica-trading-data/app --json
{"count":251941,"bytes":1067613270,"sizeless":0}
```

A size scan of `funding/` timed out in this pass, so do not infer its current object count/size. Do not claim R2 legacy cleanup complete until `app/` + `funding/` are verified gone/empty. Any new destructive purge needs explicit Diego approval and must preserve `raw/` and `ops/`.

## R2 raw archive / local research cache snapshot

Refreshed local research raw cache from active R2 raw prefix on 2026-05-06:

```text
rclone copy r2:pacifica-trading-data/raw/pacifica/full_fidelity data/pacifica_full_fidelity --transfers 16 --checkers 32 --fast-list
```

Local cache inventory after refresh:

```json
{
  "path": "data/pacifica_full_fidelity",
  "files": 30276,
  "sha256_sidecars": 15138,
  "payloads": 15138,
  "bytes": 22141118583,
  "gib": 20.621,
  "symbols": 66,
  "dates": ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06"]
}
```

Earlier read-only raw archive reports remain at:

- `docs/ops/pacifica-r2-raw-health-latest/README.md`
- `docs/ops/pacifica-r2-raw-health-latest/summary.json`

A long `rclone lsf ... --recursive` inventory process from the prior session was killed after the local health summary was produced; do not rely on partial `data/pacifica_r2_raw_health/raw_lsf_pst.txt` as final inventory.

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
bbo: 9246648 rows
book: 12361829 rows
candle: 1723416 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned
wrote 408804 regime-state rows to docs/experiments/non-hft-regime-state
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

- Archive has 7 distinct dates, still diagnostic.
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

Completed after the 2026-05-06 research refresh:

```text
uv run pytest tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_non_hft_toxic_overlay_probe.py tests/scripts/test_build_pacifica_eligibility_gates.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
21 passed in 0.31s

python -m py_compile scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/non_hft_toxic_overlay_probe.py scripts/build_pacifica_eligibility_gates.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed
```

## Remaining work

1. Fix the Fly ops watchdog R2 inventory: full `rclone lsjson --recursive` timed out at 1800s on 2026-05-06. Replace it with bounded, partitioned, or line-oriented inventory before relying on due watchdog cycles.
2. Verify top-level R2 prefixes after any approved legacy purge. `app/` and `funding/` should eventually be gone/empty; `raw/` and `ops/` should remain.
3. If Diego approves, restart/finish a targeted destructive purge only for remaining legacy `app/` and `funding/` prefixes. Do not touch `raw/` or `ops/`.
4. Keep monitoring lifecycle DB/log health: upload failures `0`, verify failures `0`, free disk above Diego's 50 GiB floor, and no persistent raw freshness failures.
5. Let archive mature: 10-14 days early sanity, 30+ days provisional validation, 60+ preferred.
6. Next research step after maturity improves: rerun fixed silver/regime/toxic/eligibility pipeline without retuning thresholds, then build sparse strategy/backtest/paper logger only after gates and economics pass.

## Blocked/avoid

- Do not retry exact shell/Python diagnostic forms that previously returned `BLOCKED: User denied. Do NOT retry.`
- Do not run destructive `rm -rf` cleanup chains or destructive R2 cleanup without explicit approval.
- Foreground commands above 600s are rejected; use background processes with `notify_on_complete=true`.
- Terminal sometimes collapses output to `1 lines output`; rely on explicitly printed values or files.
- Do not trust incomplete/timed-out inventory files.
- Do not remove active full-fidelity app/archive/DB without explicit approval.

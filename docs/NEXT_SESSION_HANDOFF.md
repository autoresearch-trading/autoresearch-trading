# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-01 09:02 EST

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
8a5db43 chore: switch repo agent context to Hermes
```

Included in that commit:

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

Git status immediately after commit was clean.

Latest local check in this handoff session:

```text
branch: main
latest commit: 8a5db43 chore: switch repo agent context to Hermes
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

Captured public data:

- global `prices` stream;
- per-symbol `trades`;
- per-symbol `book`;
- per-symbol `bbo`;
- per-symbol `candle`;
- per-symbol `mark_price_candle`;
- REST `/info` and `/info/prices` snapshots.

Latest filesystem freshness check at 2026-05-01 09:02 EST:

```text
data/pacifica_full_fidelity exists=True
files=3712
symbols=66
dates=2026-04-30, 2026-05-01
latest_age_s=18.2
```

Interpretation: raw collection appears to still be advancing. Verify launchd/process state in the fresh session before assuming it is healthy.

### Silver builder

Script:

- `scripts/build_pacifica_full_fidelity_silver.py`

Output:

- `data/pacifica_silver_partitioned/`

Last committed silver-backed reports came from this refresh:

```text
bbo: 1095714 rows
book: 4401938 rows
candle: 827339 rows
mark_price_candle: 11747782 rows
prices: 548632 rows
trades: 48054 rows
wrote silver tables to data/pacifica_silver_partitioned
```

Latest filesystem freshness check at 2026-05-01 09:02 EST:

```text
data/pacifica_silver_partitioned exists=True
files=901
symbols=65
dates=2026-04-30, 2026-05-01
latest_age_s=27712.7
```

Interpretation: raw is much fresher than silver. In a fresh session, refresh silver from raw before relying on new diagnostics.

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

Latest full silver-backed monitor verification during cleanup wrote to `/tmp/pacifica_realtime_research_full_check` with:

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

Latest committed result:

```text
wrote 32047 regime-state rows to docs/experiments/non-hft-regime-state
Bucket: 1min
Rows: 32047
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

Latest committed result:

```text
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
Rows: 32047
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

Verification after commit `8a5db43`:

```bash
uv run pytest tests/scripts/test_watch_pacifica_realtime_research.py \
  tests/scripts/test_build_pacifica_full_fidelity_silver.py \
  tests/scripts/test_collect_pacifica_full_fidelity.py \
  tests/scripts/test_build_non_hft_regime_state.py \
  tests/scripts/test_non_hft_toxic_overlay_probe.py -q
```

Result:

```text
31 passed in 0.63s
```

Compile/checks:

```bash
python -m py_compile \
  scripts/watch_pacifica_realtime_research.py \
  scripts/build_non_hft_regime_state.py \
  scripts/non_hft_toxic_overlay_probe.py

git diff --check
```

Result: passed.

Note: the commit hook ran Black/isort and reformatted the new monitor script/test before committing.

## Recommended next steps in a fresh session

1. Run `git status --short` and confirm the tree is clean, aside from any new raw data generated under gitignored `data/`.
2. Verify the raw collector/launchd job is still running and raw files are still advancing.
3. Refresh silver from raw, because raw is much fresher than the committed silver-backed reports.
4. Rerun the silver-backed realtime monitor and confirm `warnings.json` is empty.
5. Rebuild `docs/experiments/non-hft-regime-state` from refreshed silver.
6. Rerun `docs/experiments/toxic-regime-overlay` without changing thresholds.
7. If the refreshed diagnostics changed meaningfully, commit those generated report updates.
8. Add explicit paper-trading eligibility gates before any strategy can trade all symbols.
9. Only after eligibility gates and simple sparse baselines exist, build the post-cost event-driven paper backtester/logger.

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

Inspect archive freshness:

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
```

Refresh silver from raw:

```bash
python scripts/build_pacifica_full_fidelity_silver.py \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir data/pacifica_silver_partitioned
```

Run silver-backed realtime monitor:

```bash
python scripts/watch_pacifica_realtime_research.py \
  --source silver \
  --silver-dir data/pacifica_silver_partitioned \
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
  --silver-dir data/pacifica_silver_partitioned \
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

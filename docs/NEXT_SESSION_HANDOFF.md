# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-01 08:15 EST

## Start here

The active project is an economics-first, non-HFT Pacifica paper-trading program using full-fidelity public market-data archival across the live Pacifica symbol universe.

Do not resume from the old 25-symbol representation-learning branch. That work is historical context only. The current system is:

1. full-fidelity raw collection;
2. partitioned silver normalization;
3. read-only realtime/silver research monitor;
4. 1-minute non-HFT regime-state construction;
5. fixed toxic/no-trade overlay diagnostics;
6. future eligibility gates, sparse baselines, post-cost paper backtests.

## Primary goal

Build a highly profitable paper-trading system. Sortino > 2 is a high-quality bar, but not the only success criterion.

A candidate strategy must show:

- positive net PnL after fees, slippage, funding, and adverse-selection assumptions;
- Sortino > 2 over a pre-registered paper window;
- enough trades and enough distinct days to avoid one-off luck;
- bounded drawdown;
- no single day dominating total PnL;
- no single symbol dominating total PnL unless explicitly intended;
- performance above dumb baselines and random same-frequency controls.

## Non-HFT constraint

The user cannot trade HFT. Do not propose latency-arb, next-tick alpha, queue-position edge, or high-turnover taker strategies.

Use high-frequency/full-fidelity data to infer slower states:

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

Latest live `/info` check in this session:

```text
live_symbols=65
subscriptions=1626
```

## Current data/pipeline status

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

Latest pre-handoff freshness check:

```text
data/pacifica_full_fidelity files=3710 latest_age_s=0.0
```

Raw collection appeared active at handoff time.

### Silver builder

Script:

- `scripts/build_pacifica_full_fidelity_silver.py`

Output:

- `data/pacifica_silver_partitioned/`

Silver was refreshed during this session from raw:

```text
bbo: 1095714 rows
book: 4401938 rows
candle: 827339 rows
mark_price_candle: 11747782 rows
prices: 548632 rows
trades: 48054 rows
wrote silver tables to data/pacifica_silver_partitioned
```

Immediately after refresh, the silver-backed monitor reported:

```text
Files: 900
Rows: 19600683
Symbols: 65
Latest source row age seconds: 38.014
Warnings: none
```

Later pre-handoff filesystem mtime check showed raw was still advancing while silver had not been rebuilt again:

```text
data/pacifica_silver_partitioned files=901 latest_age_s=24906.3
```

So in a fresh session, assume raw may be fresher than silver. Refresh silver before relying on new diagnostics.

### Realtime research monitor

New script created this session:

- `scripts/watch_pacifica_realtime_research.py`

New tests created this session:

- `tests/scripts/test_watch_pacifica_realtime_research.py`

Generated outputs:

- `data/pacifica_realtime_research/README.md`
- `data/pacifica_realtime_research/latest_features.csv`
- `data/pacifica_realtime_research/raw_inventory.csv`
- `data/pacifica_realtime_research/warnings.json`

The monitor is read-only. It does not place trades, tune thresholds, or claim edge.

Supported sources:

- `--source silver` reads partitioned parquet from `data/pacifica_silver_partitioned` and is the preferred routine path.
- `--source raw` reads recent/bounded raw JSONL.GZ files from `data/pacifica_full_fidelity` and is a fallback/debug path.

Latest successful silver monitor command:

```bash
python scripts/watch_pacifica_realtime_research.py \
  --source silver \
  --silver-dir data/pacifica_silver_partitioned \
  --out-dir data/pacifica_realtime_research \
  --stale-after-s 300
```

Latest successful result after silver refresh:

```text
Files: 900
Rows: 19600683
Symbols: 65
Latest source row age seconds: 38.014
Warnings: none
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

Rebuilt after silver refresh:

```bash
python scripts/build_non_hft_regime_state.py \
  --silver-dir data/pacifica_silver_partitioned \
  --out-dir docs/experiments/non-hft-regime-state
```

Latest result:

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

Latest observed silver trade-class values:

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

Rerun after regime rebuild:

```bash
python scripts/non_hft_toxic_overlay_probe.py \
  --state-path docs/experiments/non-hft-regime-state/regime_state.parquet \
  --out-dir docs/experiments/toxic-regime-overlay
```

Latest result:

```text
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
Rows: 32047
Symbols: 65
Distinct dates: 2
Horizons minutes: [5, 15, 30, 60]
Toxicity cutoffs: [0.9, 0.8, 0.7]
```

This is expected. Two distinct dates is still diagnostic only, not edge evidence.

## Interpretation discipline

The current full-fidelity archive is too young to claim an edge.

Use these maturity levels:

- 1-5 days: plumbing diagnostics only;
- 10-14 days: early sanity checks;
- 30+ full days: provisional validation;
- 60+ full days: preferred serious validation.

Keep toxicity thresholds fixed while data accrues. Do not tune cutoffs based on the initial diagnostic days.

## Verification status before handoff

Focused tests passed:

```bash
pytest tests/scripts/test_watch_pacifica_realtime_research.py \
  tests/scripts/test_build_pacifica_full_fidelity_silver.py \
  tests/scripts/test_collect_pacifica_full_fidelity.py \
  tests/scripts/test_build_non_hft_regime_state.py \
  tests/scripts/test_non_hft_toxic_overlay_probe.py -q
```

Result:

```text
31 passed in 0.34s
```

Compile check passed:

```bash
python -m py_compile \
  scripts/watch_pacifica_realtime_research.py \
  scripts/build_non_hft_regime_state.py \
  scripts/non_hft_toxic_overlay_probe.py
```

Diff whitespace check passed:

```bash
git diff --check
```

Ruff is unavailable in this environment:

```text
error: Failed to spawn: `ruff`
Caused by: No such file or directory (os error 2)
```

## Current git state at handoff

Modified generated experiment reports:

```text
M docs/experiments/non-hft-regime-state/README.md
M docs/experiments/non-hft-regime-state/regime_state_preview.csv
M docs/experiments/non-hft-regime-state/silver_quality_summary.csv
M docs/experiments/toxic-regime-overlay/README.md
M docs/experiments/toxic-regime-overlay/hour_summary.csv
M docs/experiments/toxic-regime-overlay/overlay_scorecard.csv
M docs/experiments/toxic-regime-overlay/symbol_summary.csv
M docs/experiments/toxic-regime-overlay/toxic_bucket_summary.csv
```

New/untracked files to consider adding if committing this work:

```text
docs/research/2026-05-01-real-time-streaming-research-pass.md
scripts/watch_pacifica_realtime_research.py
tests/scripts/test_watch_pacifica_realtime_research.py
```

Do not add unless intentional:

```text
.hermes/
```

Do not commit credentials or local secret-bearing settings. `.claude/settings.local.json` had secret-like local content redacted earlier; do not preserve real credential values.

## Recommended next steps in a fresh session

1. Run `git status --short` and confirm the untracked/modified files above.
2. Refresh silver again from raw, because raw collection continued after the last silver build.
3. Rerun the silver-backed realtime monitor and confirm `warnings.json` is empty.
4. Rebuild `docs/experiments/non-hft-regime-state` from refreshed silver.
5. Rerun `docs/experiments/toxic-regime-overlay` without changing thresholds.
6. Commit the monitor, tests, research pass, and refreshed reports if the diff is acceptable; do not commit `.hermes/` or raw/silver data archives.
7. Add a short durable monitor usage doc if desired, because the current monitor docs are generated under `data/pacifica_realtime_research/`.
8. Build explicit paper-trading eligibility gates before any strategy trades all symbols.
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
pytest tests/scripts/test_watch_pacifica_realtime_research.py \
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
- Do not overwrite/commit raw data archives.
- Do not commit `.hermes/` unless explicitly intended.

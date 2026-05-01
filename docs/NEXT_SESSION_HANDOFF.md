# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-04-30 22:04 EST

## Start here

The active project is now an economics-first, non-HFT paper-trading program using a new full-fidelity Pacifica collector across the live symbol universe.

Do not resume by treating this as only the old 25-symbol representation-learning project. That project is historical context. The current direction is broader live archival, regime-state construction, and paper-trading validation.

## Primary goal

Build a highly profitable paper-trading system, with Sortino > 2 as a high-quality bar, not as the only success criterion.

A candidate strategy must show:

- positive net PnL after fees, slippage, funding, and adverse-selection assumptions;
- Sortino > 2 over a pre-registered paper window;
- enough trades/days to avoid one-off luck;
- bounded drawdown;
- no single day dominating total PnL;
- no single symbol dominating total PnL unless explicitly intended;
- performance above dumb baselines and random same-frequency controls.

## Active thesis

Use full-fidelity, high-frequency market data to make slower non-HFT decisions.

The user cannot trade HFT. Do not propose latency-arb, next-tick, queue-position, or high-turnover taker strategies. Use high-frequency data to infer 1m+ regime states, toxicity, dislocations, forced-flow states, and no-trade/risk filters.

The strongest current direction is not generic direction prediction. It is:

1. toxic-regime / no-trade / no-quote overlay;
2. mark/oracle/mid dislocation reversion gated by liquidity and toxicity;
3. post-liquidation absorption/stabilization event studies;
4. execution-quality filters only after a real entry edge exists.

## Universe policy

Collect broadly, trade selectively.

- Raw collection universe: all live public Pacifica symbols from `/info`; local archive currently has ~66 symbol partitions.
- Research universe: all symbols with enough clean full-fidelity data.
- Eligible trading universe: only symbols passing pre-registered liquidity, spread/cost, sample-size, stability, and concentration gates.
- Paper-traded universe: selected subset with portfolio caps.

Do not assume the system should trade all symbols. Do not restrict research to the old 25 symbols. The paper-trading engine should support all live symbols but the strategy should only trade eligible symbols.

Do not hard-code the live symbol count. Pacifica's universe changes; the collector fetches `/info` at startup. Counts like 63, 65, or 66 are snapshots only. Refresh dynamically before making operational decisions.

Suggested eligibility gates:

- minimum clean data days;
- minimum trade/bucket count;
- acceptable spread and simulated slippage;
- sufficient top-of-book/depth notional;
- stable effect across days;
- not dominated by one liquidation event;
- not dominated by one symbol/day/hour;
- post-cost expected edge exceeds a pre-registered threshold.

## New collector and pipeline

Raw collector:

- `scripts/collect_pacifica_full_fidelity.py`
- output: `data/pacifica_full_fidelity/`
- docs: `docs/ops/pacifica-full-fidelity-archival.md`
- launchd plist: `ops/launchd/com.non-toxic.pacifica-full-fidelity.plist`

Captured public streams:

- `prices` global stream;
- `trades` per symbol;
- `book` per symbol/aggregation level;
- `bbo` per symbol;
- `candle` per symbol/interval;
- `mark_price_candle` per symbol/interval;
- REST snapshots for `/info` and `/info/prices`.

Current local snapshot checked this session, for orientation only:

- `data/pacifica_full_fidelity/` exists;
- raw archive has 926 files, 66 symbol partitions, dates 2026-04-30 and 2026-05-01;
- channels include bbo, book, candle, mark_price_candle, prices, trades, plus subscribe/control rows.

Silver builder:

- `scripts/build_pacifica_full_fidelity_silver.py`
- output: `data/pacifica_silver_partitioned/`
- current silver snapshot has 406 files, 65 symbols, date 2026-04-30.

Regime-state builder:

- `scripts/build_non_hft_regime_state.py`
- report: `docs/experiments/non-hft-regime-state/README.md`
- current report: 1-minute buckets, 7,410 rows, 65 symbols.

Toxic overlay probe:

- `scripts/non_hft_toxic_overlay_probe.py`
- report: `docs/experiments/toxic-regime-overlay/README.md`
- current verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`
- current sample: 7,410 rows, 65 symbols, 1 distinct date.

## Interpretation discipline

The current full-fidelity archive is too young to claim edge. Treat early outputs as plumbing diagnostics until day gates pass.

Use these maturity levels:

- 1-5 days: plumbing diagnostics only;
- 10-14 days: early sanity checks;
- 30+ full days: provisional validation;
- 60+ full days: preferred serious validation.

Keep toxicity thresholds fixed while data accrues. Do not tune cutoffs based on the first diagnostic day.

## Current recommended next steps

1. Verify collector is still running and accumulating raw JSONL.GZ.
2. Refresh full-fidelity silver tables from raw archive.
3. Rebuild 1-minute non-HFT regime-state table across all live symbols.
4. Rerun the fixed toxic-regime overlay probe without changing thresholds.
5. Update reports with current day counts and sample gates.
6. Add paper-trading universe eligibility gates before any strategy trades all symbols.
7. Build paper backtest/trade logger only after the risk/no-trade overlay and simple sparse rules are specified.

## Commands to inspect quickly

```bash
git status --short
git log --oneline -8
uv run python - <<'PY'
from scripts.collect_pacifica_full_fidelity import build_subscriptions, fetch_live_symbols
symbols = fetch_live_symbols()
print('live_symbols=', len(symbols), 'subscriptions=', len(build_subscriptions(symbols)))
PY
python - <<'PY'
from pathlib import Path
for p in [Path('data/pacifica_full_fidelity'), Path('data/pacifica_silver_partitioned')]:
    print(p, 'exists=', p.exists())
    syms=set(); dates=set(); files=0
    if p.exists():
        for f in p.rglob('*'):
            if f.is_file(): files += 1
            for part in f.parts:
                if part.startswith('symbol='): syms.add(part.split('=',1)[1])
                if part.startswith('date='): dates.add(part.split('=',1)[1])
    print('files=', files, 'symbols=', len(syms), 'dates=', sorted(dates)[:3], '...', sorted(dates)[-3:] if dates else [])
PY
```

## Important historical context

The old 25-symbol historical parquet/cache program remains useful context, but it is not the active starting point.

Historical findings:

- v1 direction-prediction representation learning found weak signal that was fee-blocked.
- v2 cascade-onset prediction found a real signal, but direct trading economics failed.
- Maker execution was harmed by adverse selection.
- Taker-side feasibility had no strict survivors under plausible costs.
- April 14-26 cascade holdout was consumed; new clean validation requires fresh data.

The reason the new collector matters is that it preserves fields the old lossy parquet data did not: mark/oracle/funding/open_interest, BBO order IDs, book nonces/order counts, raw trade IDs/nonces, and raw message timing. These fields enable toxicity, dislocation, forced-flow, and execution-quality studies that were not possible in the old framing.

## Files most relevant for fresh-session context

- `docs/NEXT_SESSION_HANDOFF.md` — this file.
- `docs/AGENT_OPERATING_MAP.md` — current Hermes/tool/skill arsenal and status of repo-local `.claude` agents.
- `docs/ops/pacifica-full-fidelity-archival.md`
- `docs/experiments/pacifica-full-fidelity-tradeability-filter-2026-04-30.md`
- `docs/research/2026-04-30-pacifica-full-fidelity-product-ideas.md`
- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`
- `scripts/collect_pacifica_full_fidelity.py`
- `scripts/build_pacifica_full_fidelity_silver.py`
- `scripts/build_non_hft_regime_state.py`
- `scripts/non_hft_toxic_overlay_probe.py`

## Do not do next

- Do not launch generic RL.
- Do not optimize AUC without execution economics.
- Do not claim an edge from the 1-day diagnostic probe.
- Do not blindly trade every collected symbol.
- Do not tune toxicity thresholds on the initial diagnostic sample.
- Do not overwrite/commit raw data archives.

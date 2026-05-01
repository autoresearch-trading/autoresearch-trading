# Repository Guidelines — Hermes / Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-01

## Canonical fresh-session order

1. `docs/NEXT_SESSION_HANDOFF.md` — current task, pipeline state, next commands.
2. `AGENTS.md` — repo-level agent rules and active direction.
3. `docs/AGENT_OPERATING_MAP.md` — Hermes/tool/skill map and archived Claude asset notes.

There is no active `CLAUDE.md` in this repo. Do not recreate it unless Diego explicitly decides to support Claude Code again.

## Agent policy

Hermes is the primary workflow for this repo.

Do not start or delegate to Claude Code. The old `.claude` assets have been archived under `docs/archive/claude-code-assets/.claude/` and are historical context only.

Use Hermes-native tools and skills:

- file tools for reading, patching, and writing;
- terminal tools for scripts, tests, git, process checks, launchd checks;
- session search for prior cross-session context;
- delegation only when useful, and verify subagent claims before reporting success;
- web/browser only for current external facts or source-grounded research.

## Active program

Build a highly profitable, non-HFT Pacifica paper-trading system.

Sortino > 2 is the quality bar, but success also requires:

- positive net PnL after fees, slippage, funding, and adverse-selection assumptions;
- bounded drawdown;
- enough trades and enough distinct days;
- no single day, event, or symbol dominating results unless explicitly intended;
- performance above dumb baselines and random same-frequency controls.

## Current thesis

Collect full-fidelity public Pacifica market data across the live symbol universe, then aggregate it into slower decision buckets for non-HFT use.

Use high-frequency/full-fidelity data to infer:

- toxic-regime / no-trade overlays;
- mark/oracle/mid dislocations;
- forced-flow and post-liquidation stabilization states;
- liquidity, spread, depth, and execution-quality filters.

Do not propose latency arbitrage, queue-position strategies, next-tick prediction, or high-turnover taker strategies. Diego cannot do HFT.

## Universe policy

Collect broadly, trade selectively.

- Raw collection universe: all live public Pacifica symbols from `/info`.
- Research universe: symbols with enough clean full-fidelity data.
- Eligible paper-trading universe: only symbols passing pre-registered liquidity, spread/cost, sample-size, stability, and concentration gates.
- Paper-traded universe: selected subset with portfolio caps.

Do not hard-code current symbol counts. Counts in docs/reports are snapshots; refresh live `/info` or archive inventory before operational decisions.

## Active pipeline

Raw collector:

- `scripts/collect_pacifica_full_fidelity.py`
- output: `data/pacifica_full_fidelity/`
- docs: `docs/ops/pacifica-full-fidelity-archival.md`
- launchd plist: `ops/launchd/com.non-toxic.pacifica-full-fidelity.plist`
- public streams only: prices, trades, book, bbo, candle, mark_price_candle, REST `/info`, REST `/info/prices`.

Silver builder:

- `scripts/build_pacifica_full_fidelity_silver.py`
- output: `data/pacifica_silver_partitioned/`

Regime-state builder:

- `scripts/build_non_hft_regime_state.py`
- report: `docs/experiments/non-hft-regime-state/README.md`

Toxic-regime overlay probe:

- `scripts/non_hft_toxic_overlay_probe.py`
- report: `docs/experiments/toxic-regime-overlay/README.md`
- current status should remain diagnostic until enough fresh days accrue.

## Interpretation discipline

The full-fidelity archive is still young. Treat early reports as plumbing diagnostics until day/sample gates pass.

Maturity levels:

- 1-5 days: plumbing diagnostics only;
- 10-14 days: early sanity checks;
- 30+ full days: provisional validation;
- 60+ full days: preferred serious validation.

Keep toxicity thresholds fixed while data accrues. Do not tune cutoffs on the initial diagnostic sample.

## Current next steps

Follow `docs/NEXT_SESSION_HANDOFF.md` if it differs from this list. Otherwise:

1. Verify the full-fidelity collector is still running and accumulating raw JSONL.GZ.
2. Refresh silver from raw archive.
3. Rebuild the 1-minute non-HFT regime-state table across all live symbols.
4. Rerun the fixed toxic-regime overlay probe without changing thresholds.
5. Add explicit paper-trading eligibility gates before any strategy can trade all symbols.
6. Build sparse strategy/backtest/paper logger only after economics gates and baselines are specified.

## Useful inspection commands

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

## Do not do

- Do not launch generic RL.
- Do not optimize AUC without execution economics.
- Do not claim an edge from a 1-day or tiny diagnostic probe.
- Do not blindly trade every collected symbol.
- Do not tune toxicity thresholds on early diagnostic data.
- Do not overwrite, delete, or commit raw data archives.
- Do not revive the old 25-symbol representation-learning program as the default path.
- Do not recreate `CLAUDE.md` or revive Claude Code assets unless Diego explicitly decides to support Claude in this repo again.

## Historical context location

Old representation-learning, cascade-precursor, and Claude-agent details are historical context, not active guidance. Look them up only when needed in:

- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/AGENT_OPERATING_MAP.md`
- `docs/experiments/`
- `docs/archive/`
- `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`

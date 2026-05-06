# Autoresearch Trading

Active direction as of 2026-04-30: economics-first, non-HFT Pacifica paper trading from a new full-fidelity live market-data archive.

The current work collects all live public Pacifica symbols, builds a partitioned raw/silver archive, aggregates full-fidelity data into 1-minute regime states, and tests whether toxicity/no-trade overlays and sparse tradeability rules can produce highly profitable paper trading after realistic costs.

Primary objective: highly profitable paper trading, with Sortino > 2 as a quality bar, plus positive net PnL after fees/slippage/funding, bounded drawdown, enough trades/days, and no single symbol/day dominating results.

Fresh-session handoff: see `docs/NEXT_SESSION_HANDOFF.md` first. For current Hermes/tool/skill guidance, see `AGENTS.md` and `docs/AGENT_OPERATING_MAP.md`.

## Quick Start

```bash
# Install dependencies
uv sync

# Inspect collector/archive status
python scripts/validate_pacifica_collection_coverage.py --help

# Dynamic full-universe count; fetches current live symbols from /info
uv run python - <<'PY'
from scripts.collect_pacifica_full_fidelity import build_subscriptions, fetch_live_symbols
symbols = fetch_live_symbols()
print('symbols=', len(symbols), 'subscriptions=', len(build_subscriptions(symbols)))
PY

# Manual smoke plan for the full-fidelity collector
uv run python scripts/collect_pacifica_full_fidelity.py \
  --symbols BTC,ETH \
  --intervals 1m \
  --agg-levels 1 \
  --out-dir data/pacifica_full_fidelity_smoke \
  --print-plan
```

## Active Full-Fidelity Pipeline

```text
Pacifica public live streams + REST
  -> raw JSONL.GZ archive under data/pacifica_full_fidelity/
  -> partitioned silver parquet under data/pacifica_silver_partitioned/
  -> 1-minute non-HFT regime-state table
  -> fixed toxicity/no-trade overlay probe
  -> sparse tradeability rules and paper-trading harness, only after economics gates
```

Key scripts:

- `scripts/collect_pacifica_full_fidelity.py`
- `scripts/build_pacifica_full_fidelity_silver.py`
- `scripts/build_non_hft_regime_state.py`
- `scripts/non_hft_toxic_overlay_probe.py`
- `scripts/build_pacifica_eligibility_gates.py`

Key reports:

- `docs/ops/pacifica-full-fidelity-archival.md`
- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`
- `docs/experiments/paper-trading-eligibility/README.md`

Current rebuild flow:

```bash
uv run python scripts/build_pacifica_full_fidelity_silver.py \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir data/pacifica_silver_partitioned \
  --layout partitioned \
  --chunk-size 250000

uv run python scripts/build_non_hft_regime_state.py \
  --silver-dir data/pacifica_silver_partitioned \
  --out-dir docs/experiments/non-hft-regime-state

uv run python scripts/non_hft_toxic_overlay_probe.py \
  --state docs/experiments/non-hft-regime-state/regime_state.parquet \
  --out-dir docs/experiments/toxic-regime-overlay

uv run python scripts/build_pacifica_eligibility_gates.py \
  --state docs/experiments/non-hft-regime-state/regime_state.parquet \
  --out-dir docs/experiments/paper-trading-eligibility
```

## Validation Gates

Current research gates are economics-first and non-HFT:

| Gate | Requirement |
|------|-------------|
| Sample maturity | Treat 1-5 days as plumbing diagnostics, 10-14 days as early sanity, 30+ days as provisional validation, and 60+ days as preferred serious validation. |
| Execution economics | Positive net PnL after fees, slippage, funding, and adverse-selection assumptions. |
| Risk quality | Sortino > 2, bounded drawdown, and performance above dumb baselines plus random same-frequency controls. |
| Breadth | Enough trades, enough distinct days, and no single day, event, or symbol dominating results unless explicitly intended. |
| Tradeability | Paper trade only symbols passing liquidity, spread/cost, sample-size, stability, concentration, and toxicity gates. |
| Anti-overfit | Keep toxicity thresholds fixed while data accrues; do not tune cutoffs on the initial diagnostic sample. |

## Data

### Active full-fidelity live archive

- **Universe**: all live public Pacifica symbols fetched dynamically from `/info`; local archive/silver counts are snapshots and should be refreshed before operational decisions.
- **Raw archive**: `data/pacifica_full_fidelity/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/*.jsonl.gz`
- **Silver archive**: `data/pacifica_silver_partitioned/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/*.parquet`
- **Captured streams**: prices, trades, book, bbo, candle, mark_price_candle, plus REST snapshots for `/info` and `/info/prices`.
- **Trading policy**: collect all live symbols, but only paper trade symbols passing liquidity, spread/cost, sample-size, stability, and concentration gates.

### Archived material

Older implementation notes live under `docs/archive/` and `docs/superpowers/specs/`. They are not the README path for current work.

## Agent System

Hermes is the primary agent workflow for this repo. Start fresh sessions from:

1. `docs/NEXT_SESSION_HANDOFF.md`
2. `AGENTS.md`
3. `docs/AGENT_OPERATING_MAP.md`

Claude Code is no longer used in this repo. The old tracked `.claude` assets have been archived under `docs/archive/claude-code-assets/.claude/` for historical reference.

## Project Structure

```
AGENTS.md                  — Canonical working context for agents
docs/NEXT_SESSION_HANDOFF.md — Fresh-session handoff and current pipeline state
docs/AGENT_OPERATING_MAP.md — Hermes/tool/skill map and archived Claude asset notes
docs/superpowers/specs/    — Archived specs
docs/knowledge/            — Compiled wiki (concepts, decisions, experiments)
docs/council-reviews/      — Historical agent review outputs
docs/research/             — Literature surveys
docs/archive/              — Historical artifacts
scripts/                   — Collector, pipeline, and experiment scripts
data/                      — Hive-partitioned market data archives (gitignored)
.cache/                    — Local working cache files (gitignored)
```

## Stack

Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

See [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for detailed architecture documentation.

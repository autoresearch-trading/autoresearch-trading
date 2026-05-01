# Autoresearch Trading

Active direction as of 2026-04-30: economics-first, non-HFT Pacifica paper trading from a new full-fidelity live market-data archive.

The old 25-symbol representation-learning program remains historical context, but it is no longer the correct fresh-session starting point. The current work collects all live public Pacifica symbols, builds a partitioned raw/silver archive, aggregates full-fidelity data into 1-minute regime states, and tests whether toxicity/no-trade overlays and sparse tradeability rules can produce highly profitable paper trading after realistic costs.

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

For the legacy 25-symbol historical parquet/cache data only, use the R2 sync command with the cache exclusion:

```bash
rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only --exclude "cache/**"
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

Key reports:

- `docs/ops/pacifica-full-fidelity-archival.md`
- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`

## Legacy Representation-Learning Architecture

Self-supervised encoder with MEM (Masked Event Modeling) + SimCLR contrastive pretraining:

```
Raw Parquet (trades + orderbook + funding)
  -> dedup + group same-timestamp trades into order events
  -> align nearest prior OB snapshot (10 levels, ~24s cadence)
  -> compute 17 features per event (9 trade + 8 orderbook)
  -> (batch, 200, 17) windows at stride=50
  -> BatchNorm -> 6x [Conv1d + LayerNorm + ReLU] dilated 1..32
  -> concat[GlobalAvgPool, last_position] -> 256-dim embedding
  -> MEM reconstruction (weight 0.70) + NT-Xent contrastive (weight 0.30)
  -> frozen encoder -> linear probes for direction at 10/50/100/500 event horizons
```

## Features (17 per order event)

| # | Feature | Source | Notes |
|---|---------|--------|-------|
| 1 | log_return | trade | log(vwap / prev_vwap) |
| 2 | log_total_qty | trade | rolling 1000-event median normalized |
| 3 | is_open | trade | fraction of fills that are opens — DEX-specific |
| 4 | time_delta | trade | log(ts - prev_ts + 1) |
| 5 | num_fills | trade | log(fill count) |
| 6 | book_walk | trade | spread-normalized price levels consumed |
| 7 | effort_vs_result | trade | **Wyckoff master signal** — absorption detection |
| 8 | climax_score | trade | **Wyckoff phase transitions** |
| 9 | prev_seq_time_span | trade | prior window duration (no lookahead) |
| 10 | log_spread | orderbook | mid-normalized spread |
| 11 | imbalance_L1 | orderbook | notional imbalance at best bid/ask |
| 12 | imbalance_L5 | orderbook | harmonic-weighted L1:5 imbalance |
| 13 | depth_ratio | orderbook | log bid/ask notional ratio |
| 14 | trade_vs_mid | orderbook | VWAP position relative to mid |
| 15 | delta_imbalance_L1 | orderbook | change since prior snapshot |
| 16 | kyle_lambda | orderbook | per-snapshot price impact coefficient |
| 17 | cum_ofi_5 | orderbook | piecewise Cont 2014 OFI over 5 snapshots |

## Evaluation Gates (pre-registered)

| Gate | Test | Threshold |
|------|------|-----------|
| 0 | PCA + logistic regression baseline | Reference |
| 1 | Linear probe on frozen embeddings (100-event) | > 51.4% on 15+/25 symbols |
| 2 | Fine-tuned CNN vs logistic regression | Exceed by >= 0.5pp on 15+ symbols |
| 3 | Held-out symbol (AVAX) | > 51.4% at primary horizon |
| 4 | Temporal stability (months 1-4 vs 5-6) | < 3pp drop on 10+ symbols |

## Data

### Active full-fidelity live archive

- **Universe**: all live public Pacifica symbols fetched dynamically from `/info`; local archive/silver counts are snapshots and should be refreshed before operational decisions.
- **Raw archive**: `data/pacifica_full_fidelity/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/*.jsonl.gz`
- **Silver archive**: `data/pacifica_silver_partitioned/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/*.parquet`
- **Captured streams**: prices, trades, book, bbo, candle, mark_price_candle, plus REST snapshots for `/info` and `/info/prices`.
- **Trading policy**: collect all live symbols, but only paper trade symbols passing liquidity, spread/cost, sample-size, stability, and concentration gates.

### Legacy historical parquet/cache data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-04-26 (~178 days; April 14-26 consumed for cascade-precursor OOS testing)
- **Held-out symbol**: AVAX (excluded from v1 contrastive pretraining; not a clean active holdout anymore)
- **Test set note**: April 14-26 cascade holdout was deliberately consumed; future clean validation requires fresh full-fidelity accrual or a new pre-registered split
- **Pipeline**: Fly.io collector -> GitHub Actions daily sync -> Cloudflare R2 -> local

### Schema change (2026-04-01)

Collector was extended on April 1 to capture additional fields and a new dataset:

- **Trades**: added `cause` (`normal` / `market_liquidation` / `backstop_liquidation`) and `event_type` (`fulfill_taker` / `fulfill_maker`).
- **Prices** (new): per-symbol snapshots of `open_interest`, `volume_24h`, `mark`, `oracle`, `funding`, `next_funding`.

Order-event dedup rule depends on the date:
- Pre-April: `drop_duplicates(subset=['ts_ms', 'qty', 'price'])` — `side` deliberately excluded (buyer/seller fragments differ on `side`).
- April onward: filter to `event_type == 'fulfill_taker'`.

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
docs/superpowers/specs/    — Master spec (2026-04-10; legacy v1 context)
docs/knowledge/            — Compiled wiki (concepts, decisions, experiments)
docs/council-reviews/      — Historical agent review outputs
docs/research/             — Literature surveys
docs/archive/              — Historical artifacts (old supervised pipeline)
scripts/                   — Data sync, collector, pipeline, and experiment scripts
data/                      — Hive-partitioned market data archives (gitignored)
.cache/                    — Preprocessed .npz feature files (gitignored)
```

## Stack

Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

See [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for detailed architecture documentation.

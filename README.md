# Autoresearch Trading

Self-supervised representation learning on DEX perpetual futures tape data. A dilated CNN encoder (~400K params) trained on ~40GB of raw order events (160 days, 25 crypto symbols from Pacifica API) to learn meaningful tape representations — the way a human tape reader develops intuition from watching order flow. Direction prediction is a downstream probing task, not the primary objective.

## Quick Start

```bash
# Install dependencies
uv sync

# Sync data from Cloudflare R2
rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only
```

## Architecture

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

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-25 (~160 days)
- **Held-out symbol**: AVAX (excluded from pretraining)
- **Test set**: April 1-13 for probes, April 14+ untouched
- **Pipeline**: Fly.io collector -> GitHub Actions daily sync -> Cloudflare R2 -> local

## Agent System

The project uses a multi-agent system orchestrated by `lead-0`:

```
claude --agent lead-0
├── Council (parallel, advisory)
│   council-1 (eval methodology), council-2 (microstructure),
│   council-3 (information regimes), council-4 (tape reading),
│   council-5 (skeptic), council-6 (DL architecture)
└── Workers (sequential, execution)
    runpod-7, builder-8, analyst-9, reviewer-10,
    validator-11, prover-12, data-eng-13, researcher-14
```

## Project Structure

```
CLAUDE.md                  — Working context for all agents
docs/superpowers/specs/    — Master spec (2026-04-10)
docs/knowledge/            — Compiled wiki (concepts, decisions, experiments)
docs/council-reviews/      — Agent review outputs
docs/research/             — Literature surveys
docs/archive/              — Historical artifacts (old supervised pipeline)
scripts/                   — Data sync scripts
data/                      — ~40GB Hive-partitioned Parquet (gitignored)
.cache/                    — Preprocessed .npz feature files (gitignored)
```

## Stack

Python 3.12+, PyTorch, NumPy, Pandas, DuckDB

See [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for detailed architecture documentation.

# Agent Guide

## Repository Overview

This repository bundles two tightly-coupled services:

- A **Pacifica REST data collector** (`src/collector`, top-level scripts) that ingests public market data, writes columnar Parquet datasets, and offers both historical backfill and live polling modes.
- A **signal engine** (`signal-engine/`) built on Bytewax that replays those Parquet datasets (or live feeds) to compute trading signals, track market regimes, and drive the backtesting/paper-trading toolchain.

Shared configuration lives under `config/` so both services consume the same environment-driven settings. Supporting assets include Streamlit dashboards, deployment manifests, monitoring helpers, and documentation under `docs/`.

## Data Flow at a Glance

```
Pacifica REST API ──> scripts/collect_data.py / LiveRunner ──> ./data/{dataset}/symbol=…/date=…/*.parquet
        │                                        └─> Streamlit dashboards (dashboards/app.py)
        ▼
signal-engine/scripts/run_signal_pipeline.py ──> Bytewax dataflow ──> QuestDB (time-series sink)
        ▼
signal-engine/scripts/run_backtest.py ──> BacktestEngine ──> reports + metrics
```

## Key Modules and Responsibilities

### Data Collector (root `src/collector/`)
- `scripts/collect_data.py`: main CLI. Subcommands cover metadata (`market-info`), snapshots (`prices`, `orderbook`, `recent-trades`, `funding`), historical candles (`kline`, `backfill`), ad-hoc requests (`raw`), and the async `live` polling loop. Uses `APISettings.from_env()` to merge CLI flags with `.env` defaults and logs via `structlog`.
- `LiveRunner`: async polling engine using `httpx.AsyncClient`, `RateController` (global + per-endpoint throttling with `aiolimiter`), and `tenacity` retries. Persists normalized payloads through `ParquetWriter` instances per dataset (`prices`, `trades`, `orderbook`, `funding`).
- `BackfillOptions` / `KlineBackfillRunner`: batched historical fetch looping over fixed-width windows. Converts API responses into candle rows via `transform.to_candle_rows` before writing to Parquet.
- `transform.py`: canonical payload normalization. Produces validated rows with `pydantic` models (`models.py`) for prices, trades, ladders, funding, and candles while smoothing schema drift.
- `storage.py`: asynchronous Parquet writer with symbol/date partitioning, temp-file then rename semantics, auto-buffering (50k rows or 5 minutes) to cap file counts.
- `rate.py` & `utils.py`: rate limiting helpers and signal handling; `parse_duration` parses CLI-friendly intervals, `graceful_shutdown` centralizes SIGINT/SIGTERM handling.
- `config.py`: compatibility bridge onto shared `config/` package.

**Scripts**
- `scripts/collect_all_symbols.py`: interactive wrapper that discovers symbols via `/info` then invokes the CLI `live` command.
- `scripts/collect_all_symbols_cloud.py`: cloud-oriented runner with structured logging, environment-driven polling cadence, background HTTP health check (`GET /health`), and graceful stop hooks.
- `dashboards/app.py`: Streamlit dashboard that tails the Parquet tree.
- `monitoring/` & `scripts/`: shell helpers for health checks and data syncing.

**Tests**
- Located under `tests/`. Cover transformations, storage semantics, backfill scheduler, and mocked live runner loops. Run with `pytest` from the repo root.

### Shared Configuration (`config/`)
- Pydantic settings (`BaseSettings`) split across domains: `PacificaAPISettings`, `StorageSettings`, `SignalSettings`, `TradingSettings`, and `DeploymentSettings`.
- Aggregators `Settings` and `AppSettings` hydrate backwards-compatible attributes and ensure directories exist. All environment keys are documented in `.env.example` and extended docs.

### Signal Engine (`signal-engine/`)
- `src/stream/dataflow.py`: builds the Bytewax dataflow. Sources ingest trades/orderbooks/funding (typically via Parquet readers in `stream/sources.py`). Trades are partitioned by symbol and fed into:
  - `CVDCalculator` (`signals/cvd.py`) – divergence-based cumulative volume delta.
  - `TFICalculator` (`signals/tfi.py`) – trade-flow imbalance.
  - `OFICalculator` (`signals/ofi.py`) – order-flow imbalance over top-of-book deltas.
  Signals are merged and written to configured sinks. Optional sinks persist raw trades or market regimes.
- Regime detection blends minute-resolution candles, orderbook context, and funding updates into `ATRRegimeDetector` (`regime/detectors.py`) events that gate downstream trading.
- `backtest/`: event-driven `BacktestEngine` that replays chronologically ordered signals, enforces exits via `PositionManager`, aggregates entries via `SignalAggregator`, and emits metrics (`metrics.py`).
- `persistence/async_writer.py`: asynchronous QuestDB COPY-ingestion helper. `db/questdb.py` handles schema creation and inserts (SQL in `db/schema.sql`).
- `scripts/`: operational entry points (`run_signal_pipeline.py`, `run_backtest.py`, `run_paper_trading.py`, `setup_questdb*.py`, etc.).
- `tests/`: unit coverage for calculators, position management, signal router, plus integration dataflow tests.

## Dependencies and Tooling

- Python 3.12+ recommended. Dependencies listed in root `requirements.txt`; signal engine adds extras in `signal-engine/requirements.txt` (Bytewax, QuestDB client, TA-Lib, etc.).
- Structlog, httpx, aiolimiter, tenacity, pandas, pyarrow power the collector; Bytewax, numpy, TA-Lib, and async database adapters support the engine.
- Make targets (`Makefile`, `Makefile.local`) streamline virtualenv setup, linting, QuestDB provisioning, and Apple Silicon quirks (`setup-arm64.sh`, `.envrc.arm64`).

## Running & Testing

1. **Bootstrap environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r signal-engine/requirements.txt  # when working on signal engine
   ```
2. **Configure** – copy `.env.example` to `.env`, update Pacifica endpoints, storage roots, QuestDB credentials, and polling cadences.
3. **Collector examples**
   ```bash
   python scripts/collect_data.py live --symbols BTC,ETH --max-rps 3
   python scripts/collect_data.py backfill --symbols BTC --interval 1m --days 7
   python scripts/collect_all_symbols_cloud.py  # cloud-style run with health endpoint
   ```
4. **Signal pipeline & backtest**
   ```bash
   cd signal-engine
   python scripts/setup_questdb.py
   python scripts/run_signal_pipeline.py --symbols BTC --dry-run
   python scripts/run_backtest.py --symbol BTC --days 14
   ```
5. **Tests**
   ```bash
   pytest              # collector tests
   cd signal-engine && pytest
   ```

## Implementation Notes & Gotchas

- `LiveRunner` throttles per endpoint; adjust `MAX_RPS` and `*_RPS` overrides in `.env` or CLI args to avoid Pacifica rate caps (300 credits / 60s / IP).
- Parquet writers flush asynchronously; call `flush()` or terminate gracefully to ensure buffers persist (ctrl+c triggers `request_stop`).
- Candle backfill advances using the last received timestamp to guard against sparse data but still forces forward progress; expect gaps when the API omits ranges.
- Bytewax sources assume Parquet schema compatibility with collector outputs (fields like `ts_ms`, `recv_ms`, `bids`, `asks`). Schema changes should be coordinated in both collector transforms and signal-engine converters.
- Regime detector (`ATRRegimeDetector`) needs TA-Lib; ensure the shared library is installed (see `setup-arm64.sh` for macOS instructions).
- QuestDB ingestion scripts rely on the schema in `signal-engine/src/db/schema.sql`; re-run `setup_questdb.py` after schema edits.
- Health checks (`scripts/collect_all_symbols_cloud.py`) expect data under `/app/data`; adjust `OUT_ROOT` if you relocate volumes.

## Extending the System

- **New data types**: add `models` + `transform` logic, extend `LiveRunner` writer map, and update the Bytewax sources/sinks if signals depend on them.
- **Additional signals**: implement in `signal-engine/src/signals/`, export via `__init__.py`, merge within `stream/dataflow.py`, and cover with tests under `signal-engine/tests/unit/`.
- **Deployment**: Dockerfiles provided for x86, arm64, and cloud. `deploy/fly.toml`, `deploy/railway.toml`, `deploy/render.yaml` offer turnkey configs. Monitoring scripts under `monitoring/` assist with production checks.

## Reference Docs

Key guides live under `docs/`: quickstart (`SIGNAL_ENGINE_QUICKSTART.md`), deployment checklist, implementation plan, and project structure reference. Consult these for deeper operational procedures and long-form ADR-style rationale.

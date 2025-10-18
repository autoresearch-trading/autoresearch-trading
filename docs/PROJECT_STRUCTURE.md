# Project Structure Documentation

## Overview
This codebase contains a complete trading system with data collection, signal generation, backtesting, and deployment infrastructure.

## Directory Structure

```
data-collector/
├── src/collector/              # Data collection from Pacifica API
│   ├── api_client.py          # HTTP client with retry logic
│   ├── pacifica_rest.py       # Typed API wrappers
│   ├── live_runner.py         # Async polling engine
│   ├── backfill.py            # Historical data fetching
│   ├── storage.py             # Parquet writer with partitioning
│   ├── transform.py           # API response normalization
│   ├── models.py              # Pydantic models for validation
│   ├── config.py              # Settings management
│   ├── rate.py                # Rate limiting
│   └── utils.py               # Utility functions
│
├── signal-engine/             # Signal processing and backtesting
│   ├── src/
│   │   ├── signals/           # Signal calculators
│   │   │   ├── base.py       # Base types (Signal, Trade, etc.)
│   │   │   ├── cvd.py        # Cumulative Volume Delta
│   │   │   ├── tfi.py        # Trade Flow Imbalance
│   │   │   └── ofi.py        # Order Flow Imbalance
│   │   │
│   │   ├── regime/            # Market regime detection
│   │   │   └── detectors.py  # ATR-based regime classifier
│   │   │
│   │   ├── stream/            # Bytewax dataflow
│   │   │   ├── dataflow.py   # Main processing pipeline
│   │   │   └── sources.py    # Parquet input sources
│   │   │
│   │   ├── backtest/          # Backtesting engine
│   │   │   ├── engine.py     # Event-driven backtest runner
│   │   │   ├── strategy.py   # Signal aggregation logic
│   │   │   ├── position_manager.py  # Position sizing & exits
│   │   │   ├── metrics.py    # Performance calculations
│   │   │   └── reporter.py   # Results visualization
│   │   │
│   │   ├── db/                # Database layer
│   │   │   ├── questdb.py    # QuestDB client and sinks
│   │   │   └── schema.sql    # Database schema
│   │   │
│   │   └── config/            # Configuration
│   │       └── settings.py   # Environment-based settings
│   │
│   ├── scripts/               # Executable scripts
│   │   ├── run_signal_pipeline.py    # Process Parquet → QuestDB
│   │   ├── run_backtest.py           # Run historical backtests
│   │   ├── test_signals_manual.py    # Manual signal verification
│   │   └── setup_questdb.py          # Initialize database
│   │
│   └── tests/                 # Test suite
│       ├── unit/              # Fast unit tests (no external deps)
│       ├── integration/       # Integration tests (require QuestDB)
│       └── fixtures/          # Shared test data generators
│
├── dashboards/                # Real-time monitoring
│   └── app.py                # Streamlit dashboard
│
├── monitoring/                # Health checks and status
│   ├── check_collector.sh    # Cloud collector health check
│   ├── check_all_symbols.sh  # Per-symbol data verification
│   └── check-status.sh       # Quick deployment health snapshot
│
├── scripts/                   # Operational utilities
│   └── sync_cloud_data.sh    # Cloud data sync helper
│
├── tests/                     # Data collector tests
│   ├── test_storage.py
│   ├── test_transform.py
│   ├── test_backfill.py
│   └── test_live_runner.py
│
├── data/                      # Local Parquet storage (gitignored)
│   ├── trades/symbol=BTC/date=2025-10-15/*.parquet
│   ├── orderbook/symbol=ETH/date=2025-10-15/*.parquet
│   ├── prices/symbol=SOL/date=2025-10-15/*.parquet
│   └── funding/symbol=DOGE/date=2025-10-15/*.parquet
│
├── logs/                      # Application logs (gitignored)
│
├── Deployment Files
│   ├── Dockerfile             # Standard x86_64 build
│   ├── Dockerfile.arm64       # Apple Silicon build
│   ├── Dockerfile.cloud       # Production cloud build
│   ├── docker-compose.yml     # Local development stack
│   ├── fly.toml              # Fly.io deployment config
│   ├── railway.toml          # Railway.app config
│   └── render.yaml           # Render.com config
│
├── README.md                 # Main project documentation
├── docs/                     # Extended documentation bundle
│   ├── DEPLOYMENT_GUIDE.md   # Step-by-step deployment
│   ├── DEPLOY_NOW.md         # Quick deploy guide
│   ├── PROJECT_STRUCTURE.md  # This file
│   ├── deployment-checklist.txt  # Manual deployment reminders
│   ├── collection-status.md  # Current deployment status (markdown)
│   └── collection-status.txt # Terminal-friendly status snapshot
│
└── Configuration
    ├── .env.example          # Environment template
    ├── .gitignore            # Git exclusions
    ├── .cursorrules          # Cursor AI rules
    ├── .cursorignore         # Cursor AI exclusions
    ├── requirements.txt      # Python dependencies
    ├── Makefile              # Development tasks
    ├── Makefile.local        # Apple Silicon tasks
    └── pytest.ini            # Test configuration
```

## Component Responsibilities

### Data Collector
**Purpose**: Poll Pacifica API and persist market data to Parquet files  
**Entry Points**:
- `collect_data.py` - CLI for manual/historical collection
- `collect_all_symbols_cloud.py` - Production collector for all symbols

**Key Features**:
- Rate-limited API client with exponential backoff
- Parquet partitioning by symbol and date
- Atomic writes (tmp file → rename)
- Health check endpoint for monitoring
- Supports backfill and live modes

### Signal Engine
**Purpose**: Process market data into trading signals and backtest strategies  
**Entry Points**:
- `scripts/run_signal_pipeline.py` - Process Parquet → QuestDB
- `scripts/run_backtest.py` - Backtest signal strategies

**Key Features**:
- Bytewax streaming dataflow (stateful operators)
- Three signal types: CVD (divergence), TFI (imbalance), OFI (order flow)
- ATR-based regime detection
- Event-driven backtesting with position management
- QuestDB for time-series storage

**Setup Guide**: See [`SIGNAL_ENGINE_QUICKSTART.md`](./SIGNAL_ENGINE_QUICKSTART.md) for complete setup instructions.

### Deployment
**Purpose**: Run data collector 24/7 in the cloud  
**Primary Platform**: Fly.io (free tier: 3GB storage, 512MB RAM)

**Deployment Flow**:
1. Build with `Dockerfile.cloud`
2. Deploy to Fly.io with persistent volume
3. Collector runs `collect_all_symbols_cloud.py`
4. Health check on port 8080
5. Weekly data sync to local machine

## Data Flow

```
Pacifica API
    ↓
Data Collector (collect_all_symbols_cloud.py)
    ↓
Parquet Files (partitioned: symbol=X/date=Y/*.parquet)
    ↓
Signal Engine (run_signal_pipeline.py)
    ↓
QuestDB (time-series database)
    ↓
Backtesting Engine (run_backtest.py)
    ↓
Performance Reports
```

## Development Workflow

> **Quick Start:** For a faster signal engine setup, see [`SIGNAL_ENGINE_QUICKSTART.md`](./SIGNAL_ENGINE_QUICKSTART.md) - complete QuestDB and signal processing setup in 5 minutes.

### 1. Local Development Setup
```bash
# Clone and setup
git clone <repo>
cd data-collector

# Create virtual environment (use Python 3.13+)
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
cd signal-engine && pip install -r requirements.txt

# Install TA-Lib (required for regime detection)
# macOS: brew install ta-lib
# Linux: apt-get install ta-lib

# Setup QuestDB (for signal engine)
make questdb-local  # or `make questdb-docker`

# Initialize database schema
cd signal-engine
python scripts/setup_questdb.py
```

### 2. Running Collectors Locally
```bash
# Collect specific symbols
python collect_data.py live --symbols BTC,ETH --max-rps 2

# Collect all available symbols
python collect_all_symbols.py

# Backfill historical data
python collect_data.py backfill --symbols BTC --interval 1m --days 30
```

### 3. Testing Signals
```bash
cd signal-engine

# Manual signal verification (no QuestDB needed)
python scripts/test_signals_manual.py --symbol BTC --date 2025-10-15

# Run signal pipeline in dry-run mode
python scripts/run_signal_pipeline.py --symbols BTC --dry-run --skip-regime

# Run full test suite
pytest tests/ -v
```

### 4. Backtesting
```bash
cd signal-engine

# Run backtest on 30 days of data
python scripts/run_backtest.py --symbol BTC --days 30

# Custom backtest parameters
python scripts/run_backtest.py \
  --symbol BTC \
  --days 14 \
  --position-size 0.05 \
  --stop-loss 0.015 \
  --take-profit 0.025 \
  --min-confidence 0.6
```

### 5. Deployment
```bash
# Deploy to Fly.io
flyctl auth login
flyctl apps create pacifica-collector
flyctl volumes create data_volume --size 3 --region sjc
flyctl deploy

# Monitor deployment
flyctl logs
./monitoring/check_collector.sh
```

## Testing Strategy

### Unit Tests (Fast, No External Dependencies)
- Location: `signal-engine/tests/unit/`
- Run: `pytest signal-engine/tests/unit/ -v`
- Mock QuestDB with `SKIP_QUESTDB_TESTS=true`
- Test signal logic, regime detection, position management

### Integration Tests (Require QuestDB)
- Location: `signal-engine/tests/integration/`
- Run: `pytest signal-engine/tests/integration/ -v`
- Require running QuestDB instance
- Test dataflow pipelines, database writes

### Manual Verification
- Script: `scripts/test_signals_manual.py`
- Validates signals against real Parquet data
- Useful for parameter tuning

## Configuration Management

### Environment Variables
All configuration via environment variables (see `.env.example`):

**Shared**:
- `PACIFICA_NETWORK`, `PACIFICA_API_BASE_URL`, `PACIFICA_API_TIMEOUT`, `PACIFICA_MAX_RETRIES`, `PACIFICA_API_KEY`
- `DATA_ROOT`, `LOGS_ROOT`, `RETENTION_DAYS`, `PARQUET_BUFFER_MAX_ROWS`, `PARQUET_BUFFER_MAX_SECONDS`, `ARCHIVE_ENABLED`
- `QUESTDB_HOST`, `QUESTDB_PORT`, `QUESTDB_USER`, `QUESTDB_PASSWORD`

**Data Collector**:
- `MAX_RPS`: Rate limit (requests per second)
- `POLL_TRADES`, `POLL_ORDERBOOK`, `POLL_PRICES`, `POLL_FUNDING`: Intervals

**Signal Engine / Trading**:
- `SYMBOLS`, `INITIAL_CAPITAL`, `POSITION_SIZE_PCT`, `MIN_CONFIDENCE`, `MIN_SIGNALS_AGREE`
- `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `MAX_HOLD_SECONDS`
- `MAX_DAILY_LOSS_PCT`, `MAX_DAILY_TRADES`, `MAX_CONSECUTIVE_LOSSES`
- `MAX_TOTAL_EXPOSURE_PCT`, `MAX_CONCENTRATION_PCT`
- `CVD_LOOKBACK_PERIODS`, `CVD_DIVERGENCE_THRESHOLD`
- `TFI_WINDOW_SECONDS`, `TFI_SIGNAL_THRESHOLD`
- `OFI_SIGNAL_THRESHOLD_SIGMA`
- `ATR_PERIOD`, `ATR_THRESHOLD_MULTIPLIER`, `SPREAD_THRESHOLD_BPS`, `MIN_DEPTH_THRESHOLD`, `EXTREME_FUNDING_THRESHOLD`

### Configuration Files
- `.env`: Local overrides (gitignored, copy from `.env.example`)
- `config/`: Shared Pydantic settings (`api.py`, `storage.py`, `signals.py`, `trading.py`, `deployment.py`)
- `fly.toml`: Cloud deployment configuration

## Common Tasks

### Add a New Signal
1. Create `signal-engine/src/signals/new_signal.py`
2. Inherit from `signals.base.Signal`
3. Implement `process_trade()` or `process_snapshot()`
4. Add unit tests in `tests/unit/test_new_signal.py`
5. Add to `stream/dataflow.py` pipeline
6. Update `signals/__init__.py` exports

### Add a New Data Type
1. Define Pydantic model in `src/collector/models.py`
2. Add transform function in `src/collector/transform.py`
3. Add to `LiveRunner` in `src/collector/live_runner.py`
4. Update QuestDB schema if persisting to database

### Deploy Configuration Changes
1. Modify `fly.toml` or `collect_all_symbols_cloud.py`
2. Test locally: `docker build -f Dockerfile.cloud .`
3. Deploy: `flyctl deploy`
4. Monitor: `flyctl logs` and `./monitoring/check_collector.sh`

## Troubleshooting

### No Data Being Collected
```bash
# Check collector logs
flyctl logs --app pacifica-collector

# Verify API connection
python -c "from collector.api_client import APIClient; from collector.config import APISettings; print(APIClient(APISettings.from_env()).get('/info'))"

# Check file writes
flyctl ssh console -C "ls -lth /app/data/trades/symbol=BTC/date=$(date +%Y-%m-%d)/"
```

### Signals Not Generating
```bash
# Test manually with real data
cd signal-engine
python scripts/test_signals_manual.py --symbol BTC

# Check QuestDB connection
python scripts/setup_questdb.py

# Verify Parquet data exists
ls -lh ../data/trades/symbol=BTC/
```

### Backtest Failures
```bash
# Ensure QuestDB has data
# Run signal pipeline first to populate database
python scripts/run_signal_pipeline.py --symbols BTC

# Then run backtest
python scripts/run_backtest.py --symbol BTC --days 7
```

## Resources

- **Pacifica API Docs**: https://docs.pacifica.fi/api-documentation/api
- **Bytewax Docs**: https://docs.bytewax.io/
- **QuestDB Docs**: https://questdb.io/docs/
- **Fly.io Docs**: https://fly.io/docs/
- **TA-Lib Docs**: https://mrjbq7.github.io/ta-lib/

## Architecture Decisions

### Why Parquet over Database for Raw Data?
- **Immutable**: Once written, never modified
- **Columnar**: Efficient queries on time ranges
- **Portable**: Easy to backup, sync, and analyze
- **Partition**: Symbol+date partitioning enables parallel processing
- **Cost**: No database hosting costs for raw data

### Why QuestDB for Signals?
- **Time-series optimized**: Fast aggregations on timestamp columns
- **SQL interface**: Easy to query and analyze
- **COPY command**: Bulk inserts for throughput
- **Indexing**: Automatic time-based indexing
- **PostgreSQL wire protocol**: Standard tooling compatibility

### Why Bytewax for Stream Processing?
- **Python-native**: No JVM overhead
- **Stateful operators**: Built-in state management for signals
- **Testable**: Can replay Parquet files deterministically
- **Performant**: Rust core with Python API
- **Recovery**: Automatic checkpointing and recovery

### Why Fly.io for Deployment?
- **Free tier**: 3GB storage sufficient for 7-14 days
- **Persistent volumes**: Data survives restarts
- **Global**: Deploy close to Panama (sjc region)
- **Simple**: One-command deployments
- **SSH access**: Easy debugging and monitoring

## Future Enhancements

- [ ] Add LLM-based regime detection (Phase 2)
- [ ] Implement paper trading system
- [ ] Add WebSocket support for lower latency
- [ ] Create Grafana dashboards for monitoring
- [ ] Add Telegram/Discord alerts
- [ ] Implement portfolio backtesting (multi-symbol)
- [ ] Add machine learning model training pipeline
- [ ] Create web UI for backtest parameter tuning
```

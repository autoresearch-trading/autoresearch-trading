# Signal Engine Quick Start Guide

## Overview
This guide walks through setting up and running the signal engine from scratch. Follow these steps to process historical trade data and generate trading signals using QuestDB.

**Time to Complete:** ~5 minutes  
**Prerequisites:** Docker, Python 3.11+, Homebrew (macOS)

---

## One-Shot Setup Script

If you want to run everything in one go:

```bash
# Start QuestDB
make questdb-docker

# Install dependencies (Python 3.11+)
cd signal-engine && python3.11 -m pip install -r requirements.txt --user

# Setup QuestDB schema
python3.11 scripts/setup_questdb_local.py

# Run signal pipeline (dry-run mode)
python3.11 scripts/run_signal_pipeline.py --symbols BTC ETH SOL --date 2025-10-17 --dry-run --skip-regime
```

---

## Step-by-Step Instructions

### Step 1: Start QuestDB Instance

QuestDB is used to store signals, trades, and regime data.

```bash
cd /Users/diego/Dev/data-collector
make questdb-docker
```

**What this does:**
- Starts QuestDB container in Docker
- Exposes HTTP console on `http://localhost:9000`
- Exposes PostgreSQL wire protocol on `localhost:8812`
- Waits for QuestDB to be ready

**Verify it's running:**
```bash
# Check container status
docker ps | grep questdb

# Test HTTP endpoint
curl http://localhost:9000/status

# Test PostgreSQL port
nc -zv localhost 8812
```

**Access Points:**
- **Web Console:** http://localhost:9000
- **PostgreSQL:** `localhost:8812` (user: `admin`, password: `quest`)

---

### Step 2: Install Signal Engine Dependencies

The signal engine requires Python 3.11+ because of `hftbacktest>=2.0.0`.

**Important:** Must use Python 3.11 or higher. The system Python 3.9 will fail.

```bash
cd signal-engine

# Check Python version
/opt/homebrew/bin/python3.11 --version  # Should be 3.11.x or higher

# Install requirements
/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt --user
```

**Key Dependencies Installed:**
- `pydantic>=2.8.0` - Configuration and validation
- `bytewax>=0.20.0` - Stream processing framework
- `pandas>=2.2.0` - Data manipulation
- `TA-Lib>=0.4.28` - Technical analysis indicators
- `psycopg>=3.1.0` - PostgreSQL client for QuestDB
- `hftbacktest>=2.0.0` - High-frequency backtesting
- `questdb>=1.1.0` - QuestDB Python client
- `rich>=13.7.0` - Terminal output formatting

**Troubleshooting:**

If you get `hftbacktest` version errors:
```bash
# Check your Python version
python3 --version

# If < 3.11, install Python 3.11 via Homebrew
brew install python@3.11
```

If you get permission errors:
- Add `--user` flag to pip install
- Or run with `required_permissions: ['all']`

---

### Step 3: Set Up QuestDB Schema

Create the necessary tables in QuestDB for signals, trades, and regimes.

```bash
cd signal-engine
/opt/homebrew/bin/python3.11 scripts/setup_questdb_local.py
```

**Expected Output:**
```
Connecting to local QuestDB at localhost:8812...
Connected successfully. Creating tables...
✓ QuestDB tables created successfully!

You can access QuestDB at:
  HTTP Console: http://localhost:9000
  PostgreSQL:   localhost:8812
```

**What Tables Are Created:**
- `signals` - Trading signals (CVD, TFI, OFI)
- `trades` - Processed trade data
- `regimes` - Market regime classifications

**Troubleshooting:**

If connection fails:
```bash
# Ensure QuestDB is running
docker ps | grep questdb

# Check if port 8812 is open
lsof -i :8812
```

If `psycopg` import error:
```bash
# Reinstall with binary support
pip3.11 install 'psycopg[binary]>=3.1.0' --user
```

---

### Step 4: Run Signal Pipeline

Process historical trade data and generate signals.

#### Option A: Dry-Run Mode (Console Output Only)

Test the pipeline without writing to QuestDB:

```bash
cd signal-engine
/opt/homebrew/bin/python3.11 scripts/run_signal_pipeline.py \
  --symbols BTC ETH SOL \
  --date 2025-10-17 \
  --dry-run \
  --skip-regime
```

**What You'll See:**
- Trade files being loaded from `data/trades/`
- Signals generated in real-time
- Signal output printed to console (TFI signals with metadata)

#### Option B: Write to QuestDB

Process and save signals to the database:

```bash
cd signal-engine
/opt/homebrew/bin/python3.11 scripts/run_signal_pipeline.py \
  --symbols BTC ETH SOL \
  --date 2025-10-17 \
  --skip-regime
```

**Query Signals in QuestDB:**
```sql
-- Open http://localhost:9000 and run:

-- View latest signals
SELECT * FROM signals 
ORDER BY ts DESC 
LIMIT 100;

-- Count signals by type
SELECT signal_type, COUNT(*) as count 
FROM signals 
GROUP BY signal_type;

-- Analyze BTC signals
SELECT 
    symbol,
    signal_type,
    direction,
    AVG(confidence) as avg_confidence,
    COUNT(*) as signal_count
FROM signals
WHERE symbol = 'BTC'
GROUP BY symbol, signal_type, direction;
```

---

## Command Reference

### Pipeline Script Options

```bash
python3.11 scripts/run_signal_pipeline.py --help
```

**Available Flags:**

| Flag | Description | Example |
|------|-------------|---------|
| `--date` | Specific date to process (YYYY-MM-DD) | `--date 2025-10-17` |
| `--symbols` | List of symbols to process | `--symbols BTC ETH SOL` |
| `--skip-orderbook` | Disable OFI signal (no orderbook data) | `--skip-orderbook` |
| `--skip-regime` | Disable regime detection | `--skip-regime` |
| `--dry-run` | Print signals to console (don't write to DB) | `--dry-run` |

**Default Behavior:**
- If `--date` not provided: Uses latest available data per symbol
- If `--symbols` not provided: Uses symbols from `SYMBOLS` environment variable
- Orderbook data included by default (for OFI signals)
- Regime detection enabled by default

---

## Signal Types Generated

### 1. TFI (Trade Flow Imbalance)
Measures buy vs. sell volume imbalance over a time window.

**Signal Output:**
```python
Signal(
    ts=datetime(2025, 10, 17, 11, 33, 9),
    symbol='BTC',
    signal_type='tfi',
    value=-1.0,                    # -1=bearish, +1=bullish
    confidence=1.0,                # 0.0 to 1.0
    direction='bearish',
    price=106013.0,
    metadata={
        'buy_volume': 0.0,
        'sell_volume': 7.92,
        'trade_count': 215,
        'window_seconds': 60
    }
)
```

### 2. CVD (Cumulative Volume Delta)
Tracks cumulative buy/sell volume divergence from price.

### 3. OFI (Order Flow Imbalance)
Analyzes orderbook depth changes (requires `--skip-orderbook` NOT set).

---

## Data Location

The pipeline reads Parquet files from:

```
data/
├── trades/
│   ├── symbol=BTC/
│   │   └── date=2025-10-17/
│   │       └── trades-*.parquet
│   ├── symbol=ETH/
│   └── symbol=SOL/
│
└── orderbook/
    ├── symbol=BTC/
    └── ...
```

**Data Requirements:**
- Trade data: **Required** (pipeline will fail if missing)
- Orderbook data: Optional (skip with `--skip-orderbook`)

**Check Available Data:**
```bash
# List available symbols
ls data/trades/

# Check latest date for BTC
ls data/trades/symbol=BTC/ | tail -1

# Count files for a specific date
find data/trades/symbol=BTC/date=2025-10-17/ -name "*.parquet" | wc -l
```

---

## Environment Variables

Configure via `.env` file in `signal-engine/`:

```env
# Data location
DATA_ROOT=../data

# QuestDB connection
QUESTDB_HOST=localhost
QUESTDB_PORT=8812
QUESTDB_USER=admin
QUESTDB_PASSWORD=quest

# Symbols to process (JSON array or comma-separated)
SYMBOLS=["BTC","ETH","SOL"]
# Or: SYMBOLS=BTC,ETH,SOL

# Signal parameters
CVD_LOOKBACK_PERIODS=20
CVD_DIVERGENCE_THRESHOLD=0.15

TFI_WINDOW_SECONDS=60
TFI_SIGNAL_THRESHOLD=0.3

OFI_SIGNAL_THRESHOLD_SIGMA=2.0

# Regime detection
ATR_PERIOD=14
ATR_THRESHOLD_MULTIPLIER=1.5
SPREAD_THRESHOLD_BPS=15
MIN_DEPTH_THRESHOLD=10.0
EXTREME_FUNDING_THRESHOLD=0.001
```

---

## Useful Make Commands

From the root directory:

```bash
# Start QuestDB with Docker
make questdb-docker

# Stop QuestDB
make docker-down

# Install all dependencies
make install

# Run tests (without QuestDB)
make test-no-questdb

# Run integration tests (requires QuestDB)
make test-integration

# Run signal pipeline with sample data
make signal-pipeline
```

---

## Troubleshooting

### QuestDB Won't Start

```bash
# Check if port is already in use
lsof -i :9000
lsof -i :8812

# View container logs
docker logs data-collector-questdb-1 -f

# Restart QuestDB
docker-compose restart questdb
```

### Python Version Issues

```bash
# Verify Python 3.11 is installed
which python3.11
/opt/homebrew/bin/python3.11 --version

# If not installed
brew install python@3.11

# Create alias (optional)
alias python3.11='/opt/homebrew/bin/python3.11'
```

### Missing Dependencies

```bash
# Reinstall all requirements
cd signal-engine
pip3.11 install -r requirements.txt --user --force-reinstall

# Check for TA-Lib system dependency
brew install ta-lib
```

### No Trade Data Found

```bash
# Verify data directory structure
ls -R data/trades/ | head -20

# Check permissions
ls -la data/

# Run data collector first
python3 collect_data.py live --symbols BTC,ETH --max-rps 2
```

### Signal Pipeline Fails

```bash
# Run with verbose output
cd signal-engine
python3.11 scripts/run_signal_pipeline.py \
  --symbols BTC \
  --date 2025-10-17 \
  --dry-run \
  --skip-regime \
  --skip-orderbook  # If no orderbook data

# Check specific date has data
find data/trades/symbol=BTC/date=2025-10-17/ -name "*.parquet"
```

---

## Performance Tips

### Processing Large Date Ranges

For backtesting over multiple days, process in batches:

```bash
# Process one symbol at a time
for symbol in BTC ETH SOL; do
  python3.11 scripts/run_signal_pipeline.py \
    --symbols $symbol \
    --date 2025-10-17 \
    --skip-regime
done

# Or use make command
make signal-pipeline
```

### Memory Usage

The pipeline loads all data for a date into memory. For large datasets:

1. Process one symbol at a time
2. Use `--skip-orderbook` if not needed
3. Process in smaller date ranges
4. Monitor with: `docker stats data-collector-questdb-1`

### Disk Space

QuestDB data persists in Docker volume:

```bash
# Check volume usage
docker system df -v

# Clean up old data (careful!)
docker volume prune
```

---

## Next Steps

After running the signal engine:

1. **View Signals:** Open http://localhost:9000 and query the `signals` table
2. **Run Backtest:** Use `scripts/run_backtest.py` to test trading strategies
3. **Paper Trading:** Enable paper trading with `scripts/paper_trading.py`
4. **Deploy:** Follow `docs/DEPLOY_NOW.md` for production deployment

---

## Related Documentation

- **Project Structure:** `docs/PROJECT_STRUCTURE.md`
- **Deployment Guide:** `docs/DEPLOY_NOW.md`
- **Data Collection:** Main `README.md`
- **Contributing:** `docs/CONTRIBUTING.md`
- **Schema Reference:** `signal-engine/src/db/schema.sql`

---

## Quick Reference Card

```bash
# Complete setup from scratch (copy-paste ready)
cd /Users/diego/Dev/data-collector
make questdb-docker
cd signal-engine
/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt --user
/opt/homebrew/bin/python3.11 scripts/setup_questdb_local.py
/opt/homebrew/bin/python3.11 scripts/run_signal_pipeline.py \
  --symbols BTC ETH SOL \
  --date 2025-10-17 \
  --dry-run \
  --skip-regime

# Access QuestDB web console
open http://localhost:9000

# Stop QuestDB when done
cd ..
make docker-down
```

---

**Last Updated:** October 17, 2025  
**Tested On:** macOS (arm64), Python 3.11, Docker Desktop


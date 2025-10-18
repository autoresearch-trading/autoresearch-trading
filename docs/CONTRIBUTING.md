# Contributing to Pacifica Trading Bot

## Development Setup

### Prerequisites
- Python 3.13 or higher
- TA-Lib system library
- QuestDB (for signal engine development)
- Docker (optional, for containerized QuestDB)

### First Time Setup

```bash
# Clone repository
git clone <repository-url>
cd data-collector

# Setup virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
cd signal-engine
pip install -r requirements.txt
cd ..

# Install TA-Lib
# macOS:
brew install ta-lib

# Linux:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Apple Silicon (M1/M2/M3):
./setup-arm64.sh
```

### Running Tests

```bash
# Run all tests
make test

# Run only unit tests (fast, no QuestDB required)
make test-unit

# Run integration tests (requires QuestDB)
make test-integration

# Run specific test file
pytest signal-engine/tests/unit/test_cvd.py -v
```

### Code Style

We follow PEP 8 with these specifics:
- Line length: 100 characters
- Use type hints on all functions
- Import order: stdlib, third-party, local (separated by blank lines)
- Prefer dataclasses over plain dicts
- Use pathlib.Path instead of string paths

```bash
# Check code style
make lint

# Auto-fix style issues
make lint-fix
```

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests first (TDD approach)
   - Keep commits atomic and well-described
   - Update documentation if needed

3. **Test your changes**
   ```bash
   make test
   ```

4. **Submit a pull request**
   - Describe what changed and why
   - Reference any related issues
   - Ensure CI passes

### Project Organization Rules

#### Data Collector (`src/collector/`)
- **api_client.py**: Low-level HTTP client - no business logic
- **pacifica_rest.py**: Typed wrappers for Pacifica endpoints
- **live_runner.py**: Async polling coordinator - handles scheduling only
- **storage.py**: Parquet writing - handles partitioning and atomicity
- **transform.py**: Pure functions for normalizing API responses
- **models.py**: Pydantic models - validation only, no business logic

**Rule**: Each module has ONE responsibility. No circular imports.

#### Signal Engine (`signal-engine/src/`)
- **signals/**: Each signal in its own file, inherits from base.Signal
- **regime/**: Market regime detection only
- **stream/**: Bytewax dataflow - coordinate signals, don't compute them
- **backtest/**: Deterministic backtesting - no randomness allowed
- **db/**: Database client only - no business logic

**Rule**: Signals must be stateless or use explicit state management.

### Adding New Features

#### Adding a New Signal

```python
# 1. Create signal-engine/src/signals/your_signal.py
from signals.base import Signal, SignalDirection, SignalType, Trade

class YourSignalCalculator:
    def __init__(self, symbol: str, **config):
        self.symbol = symbol
        # Initialize parameters
        
    def process_trade(self, trade: Trade) -> Signal | None:
        # Calculate signal logic
        # Return Signal or None
        pass

# 2. Add tests in signal-engine/tests/unit/test_your_signal.py
def test_your_signal_basic():
    calc = YourSignalCalculator(symbol="BTC")
    trade = make_trade(ts=datetime.now(), side="buy", price=100, qty=1)
    signal = calc.process_trade(trade)
    assert signal is not None
    # More assertions...

# 3. Add to dataflow in signal-engine/src/stream/dataflow.py
# (Follow existing CVD/TFI/OFI patterns)

# 4. Export in signal-engine/src/signals/__init__.py
__all__ = [..., "YourSignalCalculator"]
```

#### Adding a New Data Type

```python
# 1. Define model in src/collector/models.py
class NewDataRow(BaseModel):
    ts_ms: int
    symbol: str
    # ... other fields

# 2. Add transform in src/collector/transform.py
def to_new_data_rows(payload: dict, recv_ms: int) -> list[dict]:
    # Normalize API response
    return [NewDataRow(**item).dict() for item in payload["data"]]

# 3. Add to LiveRunner in src/collector/live_runner.py
async def _poll_new_data(self, symbols, interval):
    # Implementation...
```

### Testing Guidelines

#### Unit Tests
- Fast (< 1 second per test)
- No external dependencies (mock everything)
- Test one thing per test
- Use descriptive names: `test_cvd_detects_bullish_divergence`

#### Integration Tests
- Test interaction between components
- May use real QuestDB instance
- Clean up after yourself (delete test data)
- Mark with `@pytest.mark.integration`

#### Test Data
- Use `tests/fixtures/sample_data.py` for generating test data
- Keep test data minimal but realistic
- Don't commit large test files

### Debugging Tips

#### Data Collection Issues
```bash
# Check what's being collected
ls -lh data/trades/symbol=BTC/date=$(date +%Y-%m-%d)/

# Tail logs
tail -f logs/collector.log

# Test API manually
python -c "from collector.api_client import APIClient; from collector.config import APISettings; print(APIClient(APISettings.from_env()).get('/info'))"
```

#### Signal Issues
```bash
# Test signals manually with real data
cd signal-engine
python scripts/test_signals_manual.py --symbol BTC --date 2025-10-15

# Check QuestDB has data
python -c "from db.questdb import QuestDBClient; from config import Settings; s = Settings(); c = QuestDBClient(s.questdb_host, s.questdb_port, s.questdb_user, s.questdb_password); print(c.query_signals('BTC', ...))"
```

#### Backtest Issues
```bash
# Run with verbose output
python scripts/run_backtest.py --symbol BTC --days 7 --min-confidence 0.4

# Check position exits
grep "exit_reason" logs/backtest.log
```

### Performance Considerations

- **Parquet Writes**: Buffer 5000 rows before flushing
- **QuestDB Inserts**: Use COPY command for bulk (with INSERT fallback)
- **Signal Calculation**: Keep lookback windows small (<100 periods)
- **Memory**: Process data in chunks, don't load entire datasets
- **Rate Limiting**: Respect API limits (default 3 RPS)

### Security Checklist

Before committing:
- [ ] No API keys in code
- [ ] No hardcoded credentials
- [ ] All secrets in .env (which is gitignored)
- [ ] Validated all external inputs
- [ ] Used parameterized database queries
- [ ] No sensitive data in logs

### Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite: `make test`
4. Tag release: `git tag v1.0.0`
5. Deploy to production: `flyctl deploy`
6. Monitor for 24 hours

### Getting Help

- Check existing issues on GitHub
- Read documentation in `/docs`
- Ask in team chat (if applicable)
- Review similar implementations in codebase

## Common Patterns

### Error Handling
```python
# Use structlog for structured logging
import structlog
log = structlog.get_logger(__name__)

try:
    result = risky_operation()
except SpecificException as e:
    log.error("operation_failed", error=str(e), context=extra_info)
    # Handle or re-raise
```

### Configuration
```python
# Always use Settings class
from config import Settings

settings = Settings()  # Loads from environment
value = settings.some_config_value
```

### Database Operations
```python
# Always use context managers
with psycopg.connect(conn_string) as conn:
    conn.execute(sql, params)
    conn.commit()
```

### File Operations
```python
# Always use atomic writes
from pathlib import Path

tmp_path = Path("file.tmp")
final_path = Path("file.parquet")

# Write to temp
tmp_path.write_bytes(data)

# Atomic rename
tmp_path.rename(final_path)
```
```

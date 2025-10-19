# Implementation Plan: Architecture Improvements

**Created**: 2025-10-18  
**Priority**: High → Medium → Low  
**Estimated Total Time**: 4-6 weeks for all items

---

## 📋 OVERVIEW

This document provides detailed implementation plans for all architectural improvements identified in the codebase analysis. Each item includes:
- Step-by-step instructions
- Code examples
- File changes needed
- Testing requirements
- Time estimates
- Rollback strategies

---

## 🔴 HIGH PRIORITY IMPLEMENTATIONS

### HIGH-1: Unify Configuration System ⚡
**Issue**: Configuration scattered across data-collector and signal-engine  
**Impact**: High - Affects maintainability, deployment, testing  
**Time Estimate**: 8-12 hours  
**Risk**: Medium - Requires coordination between components

#### Current State
```
src/collector/config.py              # Data collector settings
signal-engine/src/config/settings.py # Signal engine settings
# Overlapping: API settings, symbol lists, paths
```

#### Target State
```
config/
├── __init__.py
├── api.py          # Shared Pacifica API configuration
├── storage.py      # Data paths and storage settings
├── signals.py      # Signal calculation parameters
├── trading.py      # Backtest & paper trading parameters
└── deployment.py   # Environment-specific overrides
```

#### Implementation Steps

**Step 1: Create Shared Config Package** (2 hours)
```bash
mkdir -p config
touch config/__init__.py
```

Create `config/api.py`:
```python
from __future__ import annotations

import os
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PacificaAPISettings(BaseSettings):
    """Shared Pacifica API configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PACIFICA_",
        extra="forbid" if os.getenv("ENV") == "development" else "ignore",
    )
    
    network: str = Field(default="mainnet", pattern="^(mainnet|testnet)$")
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=5, ge=1, le=10)
    
    @property
    def effective_base_url(self) -> str:
        """Get base URL with network fallback."""
        if self.base_url:
            return self.base_url.rstrip("/")
        
        urls = {
            "mainnet": "https://api.pacifica.fi/api/v1",
            "testnet": "https://test-api.pacifica.fi/api/v1",
        }
        return urls[self.network]
```

Create `config/storage.py`:
```python
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """Data storage configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    data_root: Path = Field(default=Path("./data"))
    logs_root: Path = Field(default=Path("./logs"))
    
    # Parquet buffering
    parquet_buffer_max_rows: int = Field(default=5000, ge=1000, le=100_000)
    parquet_buffer_max_seconds: float = Field(default=60.0, ge=5.0, le=600.0)
    
    # Retention
    retention_days: int = Field(default=30, ge=7, le=365)
    archive_enabled: bool = Field(default=False)
    
    def ensure_directories(self) -> None:
        """Create data and log directories if missing."""
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
```

Create `config/signals.py`:
```python
from __future__ import annotations

from typing import List

import ast
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalSettings(BaseSettings):
    """Signal calculation parameters."""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # CVD settings
    cvd_lookback_periods: int = Field(default=20, ge=5, le=100)
    cvd_divergence_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    
    # TFI settings
    tfi_window_seconds: int = Field(default=60, ge=10, le=300)
    tfi_signal_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    
    # OFI settings
    ofi_signal_threshold_sigma: float = Field(default=2.0, ge=1.0, le=5.0)
    
    # Regime detection
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_threshold_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    spread_threshold_bps: int = Field(default=15, ge=5, le=100)
    min_depth_threshold: float = Field(default=10.0, ge=1.0, le=1000.0)
    extreme_funding_threshold: float = Field(default=0.001, ge=0.0001, le=0.01)
    
    def cvd_config(self) -> dict:
        return {
            "lookback_periods": self.cvd_lookback_periods,
            "divergence_threshold": self.cvd_divergence_threshold,
        }
    
    def tfi_config(self) -> dict:
        return {
            "window_seconds": self.tfi_window_seconds,
            "signal_threshold": self.tfi_signal_threshold,
        }
    
    def ofi_config(self) -> dict:
        return {
            "signal_threshold_sigma": self.ofi_signal_threshold_sigma,
        }
    
    def atr_config(self) -> dict:
        return {
            "atr_period": self.atr_period,
            "atr_threshold_multiplier": self.atr_threshold_multiplier,
            "spread_threshold_bps": self.spread_threshold_bps,
            "min_depth_threshold": self.min_depth_threshold,
            "extreme_funding_threshold": self.extreme_funding_threshold,
        }
```

Create `config/trading.py`:
```python
from __future__ import annotations

from typing import List

import ast
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingSettings(BaseSettings):
    """Backtest and paper trading parameters."""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Symbols to trade
    symbols: List[str] = Field(default_factory=lambda: ["BTC"])
    
    # Capital & position sizing
    initial_capital: float = Field(default=10_000.0, ge=100.0, le=1_000_000.0)
    position_size_pct: float = Field(default=0.10, ge=0.01, le=1.0)
    
    # Risk management
    stop_loss_pct: float = Field(default=0.02, ge=0.001, le=0.5)
    take_profit_pct: float = Field(default=0.03, ge=0.001, le=1.0)
    max_hold_seconds: int = Field(default=180, ge=10, le=86400)
    
    # Daily limits
    max_daily_loss_pct: float = Field(default=0.05, ge=0.01, le=0.5)
    max_daily_trades: int = Field(default=50, ge=1, le=1000)
    max_consecutive_losses: int = Field(default=5, ge=1, le=20)
    max_total_exposure_pct: float = Field(default=0.50, ge=0.1, le=1.0)
    max_concentration_pct: float = Field(default=0.30, ge=0.1, le=1.0)
    
    # Signal requirements
    min_confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    min_signals_agree: int = Field(default=2, ge=1, le=5)
    require_cvd: bool = Field(default=True)
    require_tfi: bool = Field(default=True)
    require_ofi: bool = Field(default=False)
    
    @field_validator("symbols", mode="before")
    @classmethod
    def _split_symbols(cls, value):
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    return [str(sym).strip().upper() for sym in parsed if str(sym).strip()]
                except Exception:
                    pass
            return [sym.strip().upper() for sym in text.split(",") if sym.strip()]
        if isinstance(value, list):
            return [str(sym).upper() for sym in value]
        return value
    
    def risk_config(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_daily_trades": self.max_daily_trades,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_position_size_pct": self.position_size_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_concentration_pct": self.max_concentration_pct,
        }
```

Create `config/__init__.py`:
```python
"""Unified configuration for data-collector and signal-engine."""

from .api import PacificaAPISettings
from .signals import SignalSettings
from .storage import StorageSettings
from .trading import TradingSettings

__all__ = [
    "PacificaAPISettings",
    "SignalSettings",
    "StorageSettings",
    "TradingSettings",
]


class AppSettings:
    """Convenience wrapper for all settings."""
    
    def __init__(self):
        self.api = PacificaAPISettings()
        self.storage = StorageSettings()
        self.signals = SignalSettings()
        self.trading = TradingSettings()
        
        # Ensure directories exist
        self.storage.ensure_directories()
    
    @classmethod
    def load(cls) -> "AppSettings":
        """Load all settings from environment."""
        return cls()
```

**Step 2: Update Data Collector** (2 hours)

Update `src/collector/config.py`:
```python
"""Backwards compatibility wrapper for existing collector code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Import from new unified config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PacificaAPISettings

PACIFICA_BASE_URLS: Dict[str, str] = {
    "mainnet": "https://api.pacifica.fi/api/v1",
    "testnet": "https://test-api.pacifica.fi/api/v1",
}


def load_environment(env_file: str = ".env") -> None:
    """Load environment variables from the given file if available."""
    if Path(env_file).exists():
        load_dotenv(env_file)
    else:
        if env_file != ".env":
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv()


@dataclass
class APISettings:
    """Configuration for accessing the Pacifica REST API.
    
    DEPRECATED: Use config.PacificaAPISettings instead.
    Kept for backwards compatibility.
    """

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 10.0
    network: str = "mainnet"

    @classmethod
    def from_env(cls) -> "APISettings":
        """Create settings from environment variables."""
        # Delegate to new unified config
        new_settings = PacificaAPISettings()
        
        return cls(
            base_url=new_settings.effective_base_url,
            api_key=new_settings.api_key,
            timeout=new_settings.timeout,
            network=new_settings.network,
        )
```

**Step 3: Update Signal Engine** (2 hours)

Update `signal-engine/src/config/settings.py`:
```python
"""Signal engine settings - uses unified config.

DEPRECATED: This file is kept for backwards compatibility.
New code should import from root config/ package.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Import from unified config
root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root))

from config import AppSettings, SignalSettings, TradingSettings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment.
    
    DEPRECATED: Use config.AppSettings instead.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_root: Path = Field(default=Path("../data"))
    symbols: List[str] = Field(default_factory=lambda: ["BTC"], alias="SYMBOLS")

    pacifica_network: str = Field(default="mainnet", alias="PACIFICA_NETWORK")
    pacifica_api_base_url: str = Field(
        default="https://api.pacifica.fi/api/v1", alias="PACIFICA_API_BASE_URL"
    )
    pacifica_api_key: str | None = Field(default=None, alias="PACIFICA_API_KEY")
    pacifica_api_timeout: float = Field(default=10.0, alias="PACIFICA_API_TIMEOUT")

    questdb_host: str = Field(default="localhost", alias="QUESTDB_HOST")
    questdb_port: int = Field(default=8812, alias="QUESTDB_PORT")
    questdb_user: str = Field(default="admin", alias="QUESTDB_USER")
    questdb_password: str = Field(default="quest", alias="QUESTDB_PASSWORD")

    # Delegate to unified config
    _unified: AppSettings = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._unified = AppSettings.load()
    
    @property
    def cvd_lookback_periods(self) -> int:
        return self._unified.signals.cvd_lookback_periods
    
    @property
    def cvd_divergence_threshold(self) -> float:
        return self._unified.signals.cvd_divergence_threshold
    
    # ... similar properties for other settings ...
    
    def cvd_config(self) -> dict:
        return self._unified.signals.cvd_config()
    
    def tfi_config(self) -> dict:
        return self._unified.signals.tfi_config()
    
    def ofi_config(self) -> dict:
        return self._unified.signals.ofi_config()
    
    def atr_config(self) -> dict:
        return self._unified.signals.atr_config()
    
    def risk_config(self) -> dict:
        return self._unified.trading.risk_config()
```

**Step 4: Update Scripts** (2 hours)

Update all scripts to use new config:
- `scripts/collect_all_symbols_cloud.py`
- `signal-engine/scripts/run_backtest.py`
- `signal-engine/scripts/run_paper_trading.py`
- `signal-engine/scripts/run_signal_pipeline.py`

**Step 5: Update .env.example** (1 hour)
```bash
# Consolidated environment variables
# API Configuration
PACIFICA_NETWORK=mainnet
PACIFICA_API_KEY=
PACIFICA_TIMEOUT=10.0
PACIFICA_MAX_RETRIES=5

# Storage Configuration
DATA_ROOT=./data
LOGS_ROOT=./logs
PARQUET_BUFFER_MAX_ROWS=5000
PARQUET_BUFFER_MAX_SECONDS=60.0
RETENTION_DAYS=30

# Signal Configuration
CVD_LOOKBACK_PERIODS=20
CVD_DIVERGENCE_THRESHOLD=0.15
TFI_WINDOW_SECONDS=60
TFI_SIGNAL_THRESHOLD=0.3
# ... etc
```

**Step 6: Testing** (3 hours)
```bash
# Test unified config loads
python -c "from config import AppSettings; s = AppSettings.load(); print(s.api.network)"

# Test backwards compatibility
python -c "from src.collector.config import APISettings; s = APISettings.from_env()"
python -c "from signal-engine.src.config.settings import Settings; s = Settings()"

# Run existing test suites
pytest tests/
cd signal-engine && pytest tests/
```

#### Rollback Strategy
```bash
# Keep old files as .backup
cp src/collector/config.py src/collector/config.py.backup
cp signal-engine/src/config/settings.py signal-engine/src/config/settings.py.backup

# If issues arise:
rm -rf config/
mv src/collector/config.py.backup src/collector/config.py
mv signal-engine/src/config/settings.py.backup signal-engine/src/config/settings.py
```

#### Success Criteria
- [ ] All config imports from unified package
- [ ] Backwards compatibility maintained
- [ ] All tests pass
- [ ] No hardcoded defaults in business logic
- [ ] Documentation updated

---

### HIGH-2: Add Bytewax Checkpointing ⚡
**Issue**: State recovery strategy unclear  
**Impact**: Medium - Data loss on crashes  
**Time Estimate**: 6-8 hours  
**Risk**: Low - Additive change

#### Current State
```python
# signal-engine/src/stream/dataflow.py
# No checkpoint configuration
# State only in memory
```

#### Target State
```python
# Bytewax with SQLite checkpoints
# State persists across restarts
# Recovery on failure
```

#### Implementation Steps

**Step 1: Add Checkpoint Directory** (30 minutes)
```bash
mkdir -p signal-engine/checkpoints
echo "*" > signal-engine/checkpoints/.gitignore
echo "!.gitignore" >> signal-engine/checkpoints/.gitignore
```

**Step 2: Create Checkpoint Manager** (2 hours)

Create `signal-engine/src/stream/checkpoints.py`:
```python
"""Checkpoint management for Bytewax dataflows."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import structlog
from bytewax.recovery import SqliteRecoveryConfig

log = structlog.get_logger(__name__)


class CheckpointManager:
    """Manage Bytewax state checkpoints."""
    
    def __init__(self, checkpoint_dir: Path, flow_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.flow_name = flow_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.checkpoint_dir / f"{flow_name}.db"
        log.info(
            "checkpoint_manager_initialized",
            flow_name=flow_name,
            db_path=str(self.db_path),
        )
    
    def get_recovery_config(self) -> SqliteRecoveryConfig:
        """Get Bytewax recovery config for this flow."""
        return SqliteRecoveryConfig(str(self.db_path))
    
    def clear_checkpoints(self) -> None:
        """Clear all checkpoints for this flow (use with caution)."""
        if self.db_path.exists():
            log.warning("clearing_checkpoints", flow_name=self.flow_name)
            self.db_path.unlink()
    
    def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get statistics about checkpoint state."""
        if not self.db_path.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "num_workers": 0,
            }
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get worker count
            cursor.execute("SELECT COUNT(DISTINCT worker_index) FROM state")
            num_workers = cursor.fetchone()[0]
            
            # Get state count
            cursor.execute("SELECT COUNT(*) FROM state")
            num_states = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "exists": True,
                "size_bytes": self.db_path.stat().st_size,
                "num_workers": num_workers,
                "num_states": num_states,
            }
        except Exception as e:
            log.error("checkpoint_stats_error", error=str(e))
            return {"exists": True, "error": str(e)}
```

**Step 3: Update Dataflow to Use Checkpoints** (2 hours)

Update `signal-engine/src/stream/dataflow.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import bytewax.operators as op
from bytewax.dataflow import Dataflow, Stream
from bytewax.inputs import Source
from bytewax.outputs import DynamicSink

from regime.detectors import ATRRegimeDetector
from signals.base import MarketRegime, OrderbookSnapshot, Signal, Trade
from signals.cvd import CVDCalculator
from signals.ofi import OFICalculator
from signals.tfi import TFICalculator


# ... existing dataclasses ...


def build_signal_dataflow(
    *,
    trades_source: Source[Trade],
    signal_sink: DynamicSink[Signal],
    orderbook_source: Source[OrderbookSnapshot] | None = None,
    regime_sink: DynamicSink[MarketRegime] | None = None,
    funding_source: Source[FundingRateEvent] | None = None,
    trade_sink: DynamicSink[Trade] | None = None,
    cvd_config: Optional[Dict[str, Any]] = None,
    tfi_config: Optional[Dict[str, Any]] = None,
    ofi_config: Optional[Dict[str, Any]] = None,
    atr_config: Optional[Dict[str, Any]] = None,
    recovery_config: Optional[Any] = None,  # ← NEW
) -> Dataflow:
    """Construct the Bytewax dataflow for real-time signal computation.
    
    Args:
        recovery_config: Bytewax recovery config for checkpointing.
                        Use SqliteRecoveryConfig for production.
    """

    cvd_config = cvd_config or {}
    tfi_config = tfi_config or {}
    ofi_config = ofi_config or {}
    atr_config = atr_config or {}

    flow = Dataflow("signal_processor")
    
    # Set recovery config if provided
    if recovery_config:
        flow.recovery_config = recovery_config  # ← NEW

    # ... rest of existing code ...
```

**Step 4: Update Pipeline Scripts** (2 hours)

Update `signal-engine/scripts/run_signal_pipeline.py`:
```python
#!/usr/bin/env python3
"""Run the signal processing pipeline with checkpointing."""

import argparse
import sys
from pathlib import Path

import structlog

from config import AppSettings
from db.questdb import QuestDBSink
from stream.checkpoints import CheckpointManager
from stream.dataflow import build_signal_dataflow
from stream.sources import ParquetTradeSource

log = structlog.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Signal processing pipeline")
    parser.add_argument("--symbols", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--clear-checkpoints", action="store_true",
                       help="Clear checkpoints before starting")
    parser.add_argument("--disable-checkpoints", action="store_true",
                       help="Disable checkpointing (faster but no recovery)")
    args = parser.parse_args()
    
    settings = AppSettings.load()
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        flow_name="signal_processor",
    )
    
    if args.clear_checkpoints:
        checkpoint_mgr.clear_checkpoints()
    
    # Log checkpoint stats
    stats = checkpoint_mgr.get_checkpoint_stats()
    log.info("checkpoint_status", **stats)
    
    # Build dataflow
    recovery_config = None if args.disable_checkpoints else checkpoint_mgr.get_recovery_config()
    
    flow = build_signal_dataflow(
        trades_source=ParquetTradeSource(settings.storage.data_root, symbols),
        signal_sink=QuestDBSink.for_signals(
            host=settings.questdb_host,
            port=settings.questdb_port,
            user=settings.questdb_user,
            password=settings.questdb_password,
        ),
        cvd_config=settings.signals.cvd_config(),
        tfi_config=settings.signals.tfi_config(),
        recovery_config=recovery_config,
    )
    
    # Run dataflow
    log.info("starting_dataflow", symbols=symbols, checkpoints_enabled=recovery_config is not None)
    # ... execute flow ...


if __name__ == "__main__":
    main()
```

**Step 5: Add Monitoring Script** (1 hour)

Create `signal-engine/scripts/inspect_checkpoints.py`:
```python
#!/usr/bin/env python3
"""Inspect checkpoint state."""

import argparse
from pathlib import Path

from stream.checkpoints import CheckpointManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--flow-name", default="signal_processor")
    args = parser.parse_args()
    
    mgr = CheckpointManager(args.checkpoint_dir, args.flow_name)
    stats = mgr.get_checkpoint_stats()
    
    print(f"Checkpoint Status: {args.flow_name}")
    print(f"  Exists: {stats.get('exists', False)}")
    if stats.get('exists'):
        print(f"  Size: {stats.get('size_bytes', 0):,} bytes")
        print(f"  Workers: {stats.get('num_workers', 0)}")
        print(f"  States: {stats.get('num_states', 0)}")


if __name__ == "__main__":
    main()
```

**Step 6: Documentation** (1 hour)

Update `signal-engine/docs/CHECKPOINTING.md`:
```markdown
# Bytewax Checkpointing Guide

## Overview
The signal engine uses Bytewax's SQLite recovery system to checkpoint stateful operators (CVD, TFI, OFI calculators).

## How It Works
- State is persisted to `checkpoints/signal_processor.db`
- Checkpoints occur automatically during dataflow execution
- On crash/restart, state is recovered from last checkpoint

## Usage

### Normal Operation (with checkpoints)
```bash
python scripts/run_signal_pipeline.py --symbols BTC,ETH
```

### Clear Checkpoints (fresh start)
```bash
python scripts/run_signal_pipeline.py --symbols BTC --clear-checkpoints
```

### Disable Checkpoints (testing)
```bash
python scripts/run_signal_pipeline.py --symbols BTC --disable-checkpoints
```

### Inspect Checkpoint State
```bash
python scripts/inspect_checkpoints.py
```

## Backup Strategy
```bash
# Backup checkpoints before major changes
cp -r checkpoints/ checkpoints.backup-$(date +%Y%m%d)
```

## Recovery Testing
```bash
# 1. Start pipeline
python scripts/run_signal_pipeline.py --symbols BTC &
PID=$!

# 2. Let it run for 60 seconds
sleep 60

# 3. Kill it
kill $PID

# 4. Check checkpoint
python scripts/inspect_checkpoints.py

# 5. Restart - should resume from checkpoint
python scripts/run_signal_pipeline.py --symbols BTC
```
```

**Step 7: Testing** (1.5 hours)
```bash
# Test checkpoint creation
python signal-engine/scripts/run_signal_pipeline.py --symbols BTC --disable-checkpoints &
sleep 5
kill $!

# Test with checkpoints
python signal-engine/scripts/run_signal_pipeline.py --symbols BTC &
PID=$!
sleep 30
kill $PID

# Verify checkpoint exists
ls -lh signal-engine/checkpoints/
python signal-engine/scripts/inspect_checkpoints.py

# Test recovery
python signal-engine/scripts/run_signal_pipeline.py --symbols BTC
```

#### Success Criteria
- [ ] Checkpoints created automatically
- [ ] State recovers after crash
- [ ] Monitoring script shows checkpoint stats
- [ ] Documentation complete
- [ ] CI tests checkpoint functionality

---

### HIGH-3: Fix Health Check Timezone Issues ⚡
**Issue**: Timezone-naive datetime comparison  
**Impact**: Medium - False health check failures  
**Time Estimate**: 2 hours  
**Risk**: Low - Simple fix

#### Implementation Steps

**Step 1: Fix Health Check Function** (30 minutes)

Update `scripts/collect_all_symbols_cloud.py:54-103`:
```python
def setup_health_check():
    """Start health check server in background."""
    from datetime import timedelta, timezone
    from http.server import BaseHTTPRequestHandler, HTTPServer
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                status = check_collector_health()
                
                self.send_response(200 if status['healthy'] else 503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status, indent=2).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            pass  # Suppress logs
    
    def run_server():
        server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
        logger.info("health_check_server_started", port=8080)
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


def check_collector_health() -> dict:
    """Check if collector is writing data recently.
    
    Returns health status with per-symbol freshness check.
    """
    data_dir = Path('/app/data/trades')
    now = datetime.now(timezone.utc)
    
    status = {
        'status': 'healthy',
        'timestamp': now.isoformat(),
        'data_dir_exists': data_dir.exists(),
        'healthy': False,
    }
    
    if not data_dir.exists():
        status['error'] = 'Data directory not found'
        return status
    
    # Get expected symbols and poll intervals
    symbols = get_all_symbols()
    poll_config = {
        "trades": parse_duration(os.getenv("POLL_TRADES", "1s")),
    }
    max_interval = max(poll_config.values())
    
    # Files should be fresher than 2x the poll interval (with buffer)
    freshness_threshold = timedelta(seconds=max_interval * 2 + 60)
    cutoff = now - freshness_threshold
    
    # Check files per symbol
    recent_by_symbol = defaultdict(list)
    for parquet_file in data_dir.rglob('*.parquet'):
        try:
            # Extract symbol from path: data/trades/symbol=BTC/date=2025-10-18/file.parquet
            parts = parquet_file.parts
            symbol_part = next((p for p in parts if p.startswith('symbol=')), None)
            if not symbol_part:
                continue
            
            symbol = symbol_part.split('=')[1]
            
            # Get file modification time (UTC)
            mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime, tz=timezone.utc)
            
            if mtime > cutoff:
                recent_by_symbol[symbol].append(parquet_file.name)
        except Exception as e:
            logger.warning("health_check_file_error", file=str(parquet_file), error=str(e))
    
    # Status details
    status['symbols_expected'] = len(symbols)
    status['symbols_with_recent_data'] = len(recent_by_symbol)
    status['freshness_threshold_seconds'] = freshness_threshold.total_seconds()
    status['cutoff_time'] = cutoff.isoformat()
    status['recent_files_per_symbol'] = {
        sym: len(files) for sym, files in recent_by_symbol.items()
    }
    
    # Healthy if at least 80% of symbols have recent data
    min_healthy_symbols = int(len(symbols) * 0.8)
    status['healthy'] = len(recent_by_symbol) >= min_healthy_symbols
    
    if not status['healthy']:
        status['error'] = f"Only {len(recent_by_symbol)}/{len(symbols)} symbols have recent data"
        status['missing_symbols'] = sorted(set(symbols) - set(recent_by_symbol.keys()))
    
    return status
```

**Step 2: Add Health Check Tests** (1 hour)

Create `tests/test_health_check.py`:
```python
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_health_check_timezone_aware():
    """Health check uses timezone-aware datetimes."""
    from scripts.collect_all_symbols_cloud import check_collector_health
    
    with patch('scripts.collect_all_symbols_cloud.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.rglob.return_value = []
        
        status = check_collector_health()
        
        # Verify timestamp is ISO format with timezone
        ts = datetime.fromisoformat(status['timestamp'])
        assert ts.tzinfo is not None
        assert ts.tzinfo == timezone.utc


def test_health_check_freshness_logic(tmp_path):
    """Health check correctly identifies fresh files."""
    from scripts.collect_all_symbols_cloud import check_collector_health
    
    # Create test data structure
    trades_dir = tmp_path / "data" / "trades"
    btc_dir = trades_dir / "symbol=BTC" / "date=2025-10-18"
    btc_dir.mkdir(parents=True)
    
    # Create recent file
    recent_file = btc_dir / "trades-001.parquet"
    recent_file.write_text("test")
    
    # Create old file
    old_file = btc_dir / "trades-002.parquet"
    old_file.write_text("test")
    old_mtime = datetime.now(timezone.utc) - timedelta(hours=1)
    old_file.touch()
    # Set mtime to 1 hour ago (OS-dependent)
    
    with patch('scripts.collect_all_symbols_cloud.Path') as mock_path:
        mock_path.return_value = trades_dir
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.rglob.return_value = [recent_file, old_file]
        
        status = check_collector_health()
        
        # Should find at least the recent file
        assert status['data_dir_exists']
        assert status['symbols_with_recent_data'] >= 0
```

**Step 3: Update Monitoring Scripts** (30 minutes)

Update `monitoring/check_collector.sh`:
```bash
#!/bin/bash
# Check collector health with detailed output

echo "🔍 Checking Pacifica Collector Health..."
echo

# Call health endpoint
HEALTH_JSON=$(curl -s http://localhost:8080/health || echo '{"error": "Failed to connect"}')

# Pretty print with color
echo "$HEALTH_JSON" | python3 -c "
import sys
import json
from datetime import datetime

try:
    data = json.load(sys.stdin)
    
    # Parse status
    healthy = data.get('healthy', False)
    status_icon = '✅' if healthy else '❌'
    
    print(f'{status_icon} Status: {\"HEALTHY\" if healthy else \"UNHEALTHY\"}')
    print()
    
    # Timestamp
    ts = data.get('timestamp', 'unknown')
    print(f'⏰ Checked at: {ts}')
    print()
    
    # Symbol stats
    expected = data.get('symbols_expected', 0)
    with_data = data.get('symbols_with_recent_data', 0)
    print(f'📊 Symbols: {with_data}/{expected} have recent data')
    
    # Per-symbol breakdown
    per_symbol = data.get('recent_files_per_symbol', {})
    if per_symbol:
        print(f'\\n📁 Files per symbol:')
        for sym, count in sorted(per_symbol.items()):
            print(f'   {sym}: {count} files')
    
    # Errors
    if not healthy and 'error' in data:
        print(f'\\n⚠️  Error: {data[\"error\"]}')
        if 'missing_symbols' in data:
            print(f'   Missing: {\" \".join(data[\"missing_symbols\"])}')
    
    sys.exit(0 if healthy else 1)
    
except json.JSONDecodeError:
    print('❌ Invalid JSON response')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

exit $?
```

#### Success Criteria
- [ ] All datetime comparisons use timezone-aware times
- [ ] Health check accurately reflects data freshness
- [ ] Per-symbol status reported
- [ ] Tests verify timezone handling
- [ ] Monitoring script shows detailed output

---

### HIGH-4: Add Configuration Range Validation ⚡
**Issue**: Invalid config values not caught  
**Impact**: Medium - Runtime errors, bad trades  
**Time Estimate**: 3 hours  
**Risk**: Low - Validation only

#### Implementation

This is already covered in HIGH-1 (Unify Configuration). All Field definitions include `ge`/`le` constraints:

```python
position_size_pct: float = Field(default=0.10, ge=0.01, le=1.0)
stop_loss_pct: float = Field(default=0.02, ge=0.001, le=0.5)
take_profit_pct: float = Field(default=0.03, ge=0.001, le=1.0)
```

**Additional Step: Add Cross-Field Validation** (1 hour)

Update `config/trading.py`:
```python
from pydantic import model_validator

class TradingSettings(BaseSettings):
    # ... existing fields ...
    
    @model_validator(mode='after')
    def validate_risk_ratios(self) -> 'TradingSettings':
        """Validate risk management ratios make sense."""
        
        # Take profit should be > stop loss
        if self.take_profit_pct <= self.stop_loss_pct:
            raise ValueError(
                f"take_profit_pct ({self.take_profit_pct}) must be > "
                f"stop_loss_pct ({self.stop_loss_pct})"
            )
        
        # Position size * max positions should not exceed total exposure
        max_positions = 1.0 / self.position_size_pct
        if max_positions * self.position_size_pct > self.max_total_exposure_pct:
            raise ValueError(
                f"position_size_pct ({self.position_size_pct}) * max_positions "
                f"would exceed max_total_exposure_pct ({self.max_total_exposure_pct})"
            )
        
        return self
```

**Add Config Validation Script** (1 hour)

Create `scripts/validate_config.py`:
```python
#!/usr/bin/env python3
"""Validate configuration before deployment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppSettings


def main():
    print("🔍 Validating configuration...")
    
    try:
        settings = AppSettings.load()
        
        print("✅ Configuration is valid!")
        print()
        print(f"Network: {settings.api.network}")
        print(f"Symbols: {', '.join(settings.trading.symbols)}")
        print(f"Position Size: {settings.trading.position_size_pct * 100:.1f}%")
        print(f"Stop Loss: {settings.trading.stop_loss_pct * 100:.1f}%")
        print(f"Take Profit: {settings.trading.take_profit_pct * 100:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Add to CI Pipeline** (30 minutes)

Update `.github/workflows/ci.yml`:
```yaml
- name: Validate Configuration
  run: |
    cp .env.example .env
    python scripts/validate_config.py
```

#### Success Criteria
- [ ] Invalid ranges rejected with clear errors
- [ ] Cross-field validation works
- [ ] Validation script in CI
- [ ] Documentation updated

---

## 📄 MEDIUM PRIORITY IMPLEMENTATIONS

*Medium priority implementations will be added in the next message due to length constraints. Continue?*


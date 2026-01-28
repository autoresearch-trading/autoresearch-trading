from __future__ import annotations

"""Simple Settings class for signal-engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class Settings:
    """Simple settings class for signal-engine compatibility."""

    # Basic paths - relative to project root
    data_root: Path = Path(__file__).parent.parent.parent.parent / "data"
    logs_root: Path = Path("/tmp/logs")

    # QuestDB settings
    questdb_host: str = "localhost"
    questdb_port: int = 8812
    questdb_user: str = "admin"
    questdb_password: str = "quest"

    # API settings
    pacifica_network: str = "mainnet"
    pacifica_api_base_url: str = "https://api.pacifica.finance"
    pacifica_api_key: str | None = None
    pacifica_api_timeout: float = 30.0
    pacifica_api_max_retries: int = 3

    # Trading settings
    symbols: list[str] | None = None
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_confidence: float = 0.6
    min_signals_agree: int = 2
    require_cvd: bool = True
    require_tfi: bool = True
    require_ofi: bool = True
    max_hold_seconds: int = 180

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC", "ETH", "SOL"]

    def as_iterable(self):
        """Return key/value pairs for compatibility."""
        yield from {
            "data_root": self.data_root,
            "logs_root": self.logs_root,
            "questdb_host": self.questdb_host,
            "questdb_port": self.questdb_port,
            "symbols": self.symbols,
        }.items()

    def risk_config(self) -> Dict[str, Any]:
        """Return risk management configuration."""
        return {
            "initial_capital": self.initial_capital,
            "max_daily_loss_pct": 0.05,
            "max_daily_trades": 10,
            "max_consecutive_losses": 3,
            "max_total_exposure_pct": 0.2,
            "max_concentration_pct": 0.1,
            "min_confidence": self.min_confidence,
            "min_signals_agree": self.min_signals_agree,
            "require_cvd": self.require_cvd,
            "require_tfi": self.require_tfi,
            "require_ofi": self.require_ofi,
        }

    def cvd_config(self) -> Dict[str, Any]:
        """Return CVD signal configuration."""
        return {
            "lookback_periods": 20,
            "divergence_threshold": 0.1,
        }

    def tfi_config(self) -> Dict[str, Any]:
        """Return TFI signal configuration.

        Tuned for ~0.76 ticks/sec (46 trades/min) data density.
        60s window = ~45 trades, aligned with 30-180s holding period.
        """
        return {
            "window_seconds": 60,
            "signal_threshold": 0.5,
        }

    def ofi_config(self) -> Dict[str, Any]:
        """Return OFI signal configuration."""
        return {
            "signal_threshold_sigma": 2.0,
        }

    def atr_config(self) -> Dict[str, Any]:
        """Return ATR configuration."""
        return {
            "period": 14,
            "threshold_multiplier": 1.5,
        }


__all__ = ["Settings"]

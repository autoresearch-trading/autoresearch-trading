from __future__ import annotations

"""Simple Settings class for signal-engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class Settings:
    """Simple settings class for signal-engine compatibility."""

    # Basic paths
    data_root: Path = Path("/tmp/data")
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
    symbols: list[str] = None
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04

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


__all__ = ["Settings"]

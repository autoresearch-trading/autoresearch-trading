from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .api import PacificaAPISettings
from .signals import SignalSettings
from .storage import StorageSettings
from .trading import TradingSettings


@dataclass(frozen=True)
class _QuestDBConfig:
    host: str
    port: int
    user: str
    password: str


class Settings:
    """Backwards compatible facade aggregating all settings."""

    def __init__(
        self,
        *,
        api: PacificaAPISettings | None = None,
        storage: StorageSettings | None = None,
        signals: SignalSettings | None = None,
        trading: TradingSettings | None = None,
    ) -> None:
        self.api = api or PacificaAPISettings()
        self.storage = storage or StorageSettings()
        self.signals = signals or SignalSettings()
        self.trading = trading or TradingSettings()

        self.storage.ensure_directories()
        self._hydrate()

    def _hydrate(self) -> None:
        storage = self.storage
        trading = self.trading
        signals = self.signals
        api = self.api

        # Storage
        self.data_root: Path = storage.data_root
        self.logs_root: Path = storage.logs_root
        self._questdb = _QuestDBConfig(
            host=storage.questdb_host,
            port=storage.questdb_port,
            user=storage.questdb_user,
            password=storage.questdb_password,
        )

        # Trading
        self.symbols: list[str] = list(trading.symbols)
        self.initial_capital: float = trading.initial_capital
        self.position_size_pct: float = trading.position_size_pct
        self.stop_loss_pct: float = trading.stop_loss_pct
        self.take_profit_pct: float = trading.take_profit_pct
        self.max_hold_seconds: int = trading.max_hold_seconds
        self.max_daily_loss_pct: float = trading.max_daily_loss_pct
        self.max_daily_trades: int = trading.max_daily_trades
        self.max_consecutive_losses: int = trading.max_consecutive_losses
        self.max_total_exposure_pct: float = trading.max_total_exposure_pct
        self.max_concentration_pct: float = trading.max_concentration_pct
        self.min_confidence: float = trading.min_confidence
        self.min_signals_agree: int = trading.min_signals_agree
        self.require_cvd: bool = trading.require_cvd
        self.require_tfi: bool = trading.require_tfi
        self.require_ofi: bool = trading.require_ofi

        # Signals
        self.cvd_lookback_periods: int = signals.cvd_lookback_periods
        self.cvd_divergence_threshold: float = signals.cvd_divergence_threshold
        self.tfi_window_seconds: int = signals.tfi_window_seconds
        self.tfi_signal_threshold: float = signals.tfi_signal_threshold
        self.ofi_signal_threshold_sigma: float = signals.ofi_signal_threshold_sigma
        self.atr_period: int = signals.atr_period
        self.atr_threshold_multiplier: float = signals.atr_threshold_multiplier
        self.spread_threshold_bps: int = signals.spread_threshold_bps
        self.min_depth_threshold: float = signals.min_depth_threshold
        self.extreme_funding_threshold: float = signals.extreme_funding_threshold

        # API
        self.pacifica_network: str = api.network
        self.pacifica_api_base_url: str = api.effective_base_url
        self.pacifica_api_key: str | None = api.api_key
        self.pacifica_api_timeout: float = api.timeout
        self.pacifica_api_max_retries: int = api.max_retries

    @property
    def questdb_host(self) -> str:
        return self._questdb.host

    @property
    def questdb_port(self) -> int:
        return self._questdb.port

    @property
    def questdb_user(self) -> str:
        return self._questdb.user

    @property
    def questdb_password(self) -> str:
        return self._questdb.password

    def cvd_config(self) -> dict[str, float | int]:
        return self.signals.cvd_config()

    def tfi_config(self) -> dict[str, float | int]:
        return self.signals.tfi_config()

    def ofi_config(self) -> dict[str, float]:
        return self.signals.ofi_config()

    def atr_config(self) -> dict[str, float | int]:
        return self.signals.atr_config()

    def risk_config(self) -> dict[str, float | int]:
        return self.trading.risk_config()

    def as_iterable(self) -> Iterable[tuple[str, object]]:
        """Return key/value pairs for quick diagnostics."""
        yield from {
            "data_root": self.data_root,
            "logs_root": self.logs_root,
            "symbols": list(self.symbols),
            "pacifica_network": self.pacifica_network,
            "pacifica_api_base_url": self.pacifica_api_base_url,
            "pacifica_api_timeout": self.pacifica_api_timeout,
            "questdb_host": self.questdb_host,
            "questdb_port": self.questdb_port,
        }.items()

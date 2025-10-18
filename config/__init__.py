from __future__ import annotations

"""Unified configuration layer shared by data-collector and signal-engine."""

from .api import DEFAULT_PACIFICA_BASE_URLS, PacificaAPISettings
from .deployment import DeploymentSettings
from .settings import Settings
from .signals import SignalSettings
from .storage import StorageSettings
from .trading import TradingSettings

__all__ = [
    "AppSettings",
    "DEFAULT_PACIFICA_BASE_URLS",
    "Settings",
    "PacificaAPISettings",
    "DeploymentSettings",
    "SignalSettings",
    "StorageSettings",
    "TradingSettings",
]


class AppSettings:
    """Aggregate view over all settings domains."""

    def __init__(
        self,
        *,
        api: PacificaAPISettings | None = None,
        storage: StorageSettings | None = None,
        signals: SignalSettings | None = None,
        trading: TradingSettings | None = None,
        deployment: DeploymentSettings | None = None,
    ) -> None:
        self.api = api or PacificaAPISettings()
        self.storage = storage or StorageSettings()
        self.signals = signals or SignalSettings()
        self.trading = trading or TradingSettings()
        self.deployment = deployment or DeploymentSettings()

        # Ensure data/log directories exist early to fail fast if permissions are missing.
        self.storage.ensure_directories()

    @classmethod
    def load(cls) -> "AppSettings":
        """Convenience helper mirroring previous Settings.load() APIs."""
        return cls()

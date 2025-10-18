from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """Data and log storage configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_root: Path = Field(default=Path("./data"), validation_alias="DATA_ROOT")
    logs_root: Path = Field(default=Path("./logs"), validation_alias="LOGS_ROOT")

    parquet_buffer_max_rows: int = Field(
        default=5000,
        ge=1000,
        le=100_000,
        validation_alias="PARQUET_BUFFER_MAX_ROWS",
    )
    parquet_buffer_max_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        validation_alias="PARQUET_BUFFER_MAX_SECONDS",
    )

    retention_days: int = Field(
        default=30,
        ge=7,
        le=365,
        validation_alias="RETENTION_DAYS",
    )
    archive_enabled: bool = Field(
        default=False,
        validation_alias="ARCHIVE_ENABLED",
    )
    questdb_host: str = Field(
        default="localhost",
        validation_alias="QUESTDB_HOST",
    )
    questdb_port: int = Field(
        default=8812,
        ge=1,
        le=65_535,
        validation_alias="QUESTDB_PORT",
    )
    questdb_user: str = Field(
        default="admin",
        validation_alias="QUESTDB_USER",
    )
    questdb_password: str = Field(
        default="quest",
        validation_alias="QUESTDB_PASSWORD",
    )

    def ensure_directories(self) -> None:
        """Make sure data and log directories exist."""
        self.data_root.expanduser().mkdir(parents=True, exist_ok=True)
        self.logs_root.expanduser().mkdir(parents=True, exist_ok=True)

    def questdb_connection_kwargs(self) -> dict[str, object]:
        """Return kwargs suitable for QuestDB Python clients."""
        return {
            "host": self.questdb_host,
            "port": self.questdb_port,
            "user": self.questdb_user,
            "password": self.questdb_password,
        }

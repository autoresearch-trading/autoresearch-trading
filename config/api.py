from __future__ import annotations

import os
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PACIFICA_BASE_URLS: dict[str, str] = {
    "mainnet": "https://api.pacifica.fi/api/v1",
    "testnet": "https://test-api.pacifica.fi/api/v1",
}


class PacificaAPISettings(BaseSettings):
    """Shared Pacifica REST API configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid" if os.getenv("ENV") == "development" else "ignore",
    )

    network: str = Field(
        default="mainnet",
        validation_alias=AliasChoices("PACIFICA_NETWORK", "PACIFICA_ENV"),
    )
    base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "PACIFICA_API_BASE_URL",
            "PACIFICA_BASE_URL",
            "API_BASE_URL",
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("PACIFICA_API_KEY", "API_KEY"),
    )
    timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        validation_alias=AliasChoices(
            "PACIFICA_API_TIMEOUT",
            "PACIFICA_TIMEOUT",
            "API_TIMEOUT",
        ),
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        validation_alias=AliasChoices(
            "PACIFICA_MAX_RETRIES",
            "PACIFICA_API_MAX_RETRIES",
            "API_MAX_RETRIES",
        ),
    )

    @field_validator("network")
    @classmethod
    def _validate_network(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in DEFAULT_PACIFICA_BASE_URLS:
            raise ValueError(
                f"Unsupported PACIFICA network '{value}'. "
                f"Choose from {', '.join(sorted(DEFAULT_PACIFICA_BASE_URLS))}."
            )
        return normalized

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: Optional[str]) -> Optional[str]:
        if value:
            return value.rstrip("/")
        return value

    @property
    def effective_base_url(self) -> str:
        """REST base URL including version suffix."""
        if self.base_url:
            return self._ensure_version_suffix(self.base_url)
        return DEFAULT_PACIFICA_BASE_URLS[self.network]

    @property
    def host_base_url(self) -> str:
        """Base host URL without the REST suffix."""
        base = self.base_url or DEFAULT_PACIFICA_BASE_URLS[self.network]
        # Drop trailing /api/v1 if present.
        if base.endswith("/api/v1"):
            return base[: -len("/api/v1")]
        return base

    @staticmethod
    def _ensure_version_suffix(value: str) -> str:
        trimmed = value.rstrip("/")
        if trimmed.endswith("/api/v1"):
            return trimmed
        if trimmed.endswith("/api"):
            return f"{trimmed}/v1"
        return f"{trimmed}/api/v1"

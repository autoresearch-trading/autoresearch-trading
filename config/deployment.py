from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploymentSettings(BaseSettings):
    """Environment-specific overrides for deployments."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: str = Field(default="local", validation_alias="ENV")
    enable_health_checks: bool = Field(
        default=True, validation_alias="ENABLE_HEALTH_CHECKS"
    )
    metrics_endpoint: str | None = Field(
        default=None, validation_alias="METRICS_ENDPOINT"
    )
    region: str | None = Field(default=None, validation_alias="REGION")

    @property
    def is_cloud(self) -> bool:
        return self.environment in {"cloud", "prod", "production"}

    @property
    def is_local(self) -> bool:
        return not self.is_cloud

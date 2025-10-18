from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalSettings(BaseSettings):
    """Signal calculation parameters shared across services."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    cvd_lookback_periods: int = Field(
        default=20,
        ge=5,
        le=100,
        validation_alias="CVD_LOOKBACK_PERIODS",
    )
    cvd_divergence_threshold: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        validation_alias="CVD_DIVERGENCE_THRESHOLD",
    )

    tfi_window_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        validation_alias="TFI_WINDOW_SECONDS",
    )
    tfi_signal_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        validation_alias="TFI_SIGNAL_THRESHOLD",
    )

    ofi_signal_threshold_sigma: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        validation_alias="OFI_SIGNAL_THRESHOLD_SIGMA",
    )

    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        validation_alias="ATR_PERIOD",
    )
    atr_threshold_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        validation_alias="ATR_THRESHOLD_MULTIPLIER",
    )
    spread_threshold_bps: int = Field(
        default=15,
        ge=5,
        le=100,
        validation_alias="SPREAD_THRESHOLD_BPS",
    )
    min_depth_threshold: float = Field(
        default=10.0,
        ge=1.0,
        le=1000.0,
        validation_alias="MIN_DEPTH_THRESHOLD",
    )
    extreme_funding_threshold: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        validation_alias="EXTREME_FUNDING_THRESHOLD",
    )

    def cvd_config(self) -> dict[str, float | int]:
        return {
            "lookback_periods": self.cvd_lookback_periods,
            "divergence_threshold": self.cvd_divergence_threshold,
        }

    def tfi_config(self) -> dict[str, float | int]:
        return {
            "window_seconds": self.tfi_window_seconds,
            "signal_threshold": self.tfi_signal_threshold,
        }

    def ofi_config(self) -> dict[str, float]:
        return {
            "signal_threshold_sigma": self.ofi_signal_threshold_sigma,
        }

    def atr_config(self) -> dict[str, float | int]:
        return {
            "atr_period": self.atr_period,
            "atr_threshold_multiplier": self.atr_threshold_multiplier,
            "spread_threshold_bps": self.spread_threshold_bps,
            "min_depth_threshold": self.min_depth_threshold,
            "extreme_funding_threshold": self.extreme_funding_threshold,
        }

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_root: Path = Field(default=Path("../data"))
    symbols: List[str] = Field(default_factory=lambda: ["BTC"], alias="SYMBOLS")

    questdb_host: str = Field(default="localhost", alias="QUESTDB_HOST")
    questdb_port: int = Field(default=8812, alias="QUESTDB_PORT")
    questdb_user: str = Field(default="admin", alias="QUESTDB_USER")
    questdb_password: str = Field(default="quest", alias="QUESTDB_PASSWORD")

    cvd_lookback_periods: int = Field(default=20, alias="CVD_LOOKBACK_PERIODS")
    cvd_divergence_threshold: float = Field(default=0.15, alias="CVD_DIVERGENCE_THRESHOLD")

    tfi_window_seconds: int = Field(default=60, alias="TFI_WINDOW_SECONDS")
    tfi_signal_threshold: float = Field(default=0.3, alias="TFI_SIGNAL_THRESHOLD")

    ofi_signal_threshold_sigma: float = Field(default=2.0, alias="OFI_SIGNAL_THRESHOLD_SIGMA")

    atr_period: int = Field(default=14, alias="ATR_PERIOD")
    atr_threshold_multiplier: float = Field(default=1.5, alias="ATR_THRESHOLD_MULTIPLIER")
    spread_threshold_bps: int = Field(default=15, alias="SPREAD_THRESHOLD_BPS")
    min_depth_threshold: float = Field(default=10.0, alias="MIN_DEPTH_THRESHOLD")
    extreme_funding_threshold: float = Field(default=0.001, alias="EXTREME_FUNDING_THRESHOLD")

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

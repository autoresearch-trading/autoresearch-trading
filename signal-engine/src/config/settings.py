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

    initial_capital: float = Field(default=10_000.0, alias="INITIAL_CAPITAL")
    position_size_pct: float = Field(default=0.10, alias="POSITION_SIZE_PCT")
    stop_loss_pct: float = Field(default=0.02, alias="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.03, alias="TAKE_PROFIT_PCT")
    max_hold_seconds: int = Field(default=180, alias="MAX_HOLD_SECONDS")

    max_daily_loss_pct: float = Field(default=0.05, alias="MAX_DAILY_LOSS_PCT")
    max_daily_trades: int = Field(default=50, alias="MAX_DAILY_TRADES")
    max_consecutive_losses: int = Field(default=5, alias="MAX_CONSECUTIVE_LOSSES")
    max_total_exposure_pct: float = Field(default=0.50, alias="MAX_TOTAL_EXPOSURE_PCT")
    max_concentration_pct: float = Field(default=0.30, alias="MAX_CONCENTRATION_PCT")

    min_confidence: float = Field(default=0.5, alias="MIN_CONFIDENCE")
    min_signals_agree: int = Field(default=2, alias="MIN_SIGNALS_AGREE")
    require_cvd: bool = Field(default=True, alias="REQUIRE_CVD")
    require_tfi: bool = Field(default=True, alias="REQUIRE_TFI")
    require_ofi: bool = Field(default=False, alias="REQUIRE_OFI")

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

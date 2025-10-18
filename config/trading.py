from __future__ import annotations

import ast
from typing import List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingSettings(BaseSettings):
    """Backtest and paper-trading parameters."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    symbols: List[str] = Field(
        default_factory=lambda: ["BTC"],
        validation_alias=AliasChoices("TRADING_SYMBOLS", "SYMBOLS"),
    )

    initial_capital: float = Field(
        default=10_000.0,
        ge=100.0,
        le=1_000_000.0,
        validation_alias="INITIAL_CAPITAL",
    )
    position_size_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        validation_alias="POSITION_SIZE_PCT",
    )

    stop_loss_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.5,
        validation_alias="STOP_LOSS_PCT",
    )
    take_profit_pct: float = Field(
        default=0.03,
        ge=0.001,
        le=1.0,
        validation_alias="TAKE_PROFIT_PCT",
    )
    max_hold_seconds: int = Field(
        default=180,
        ge=10,
        le=86_400,
        validation_alias="MAX_HOLD_SECONDS",
    )

    max_daily_loss_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        validation_alias="MAX_DAILY_LOSS_PCT",
    )
    max_daily_trades: int = Field(
        default=50,
        ge=1,
        le=1000,
        validation_alias="MAX_DAILY_TRADES",
    )
    max_consecutive_losses: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias="MAX_CONSECUTIVE_LOSSES",
    )
    max_total_exposure_pct: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        validation_alias="MAX_TOTAL_EXPOSURE_PCT",
    )
    max_concentration_pct: float = Field(
        default=0.30,
        ge=0.1,
        le=1.0,
        validation_alias="MAX_CONCENTRATION_PCT",
    )

    min_confidence: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        validation_alias="MIN_CONFIDENCE",
    )
    min_signals_agree: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias="MIN_SIGNALS_AGREE",
    )
    require_cvd: bool = Field(default=True, validation_alias="REQUIRE_CVD")
    require_tfi: bool = Field(default=True, validation_alias="REQUIRE_TFI")
    require_ofi: bool = Field(default=False, validation_alias="REQUIRE_OFI")

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value):
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    return [str(item).strip().upper() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip().upper() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip().upper() for item in value if str(item).strip()]
        return value

    def risk_config(self) -> dict[str, float | int]:
        return {
            "initial_capital": self.initial_capital,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_daily_trades": self.max_daily_trades,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_position_size_pct": self.position_size_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_concentration_pct": self.max_concentration_pct,
        }


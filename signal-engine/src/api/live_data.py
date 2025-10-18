from __future__ import annotations

import asyncio
from typing import Any, Optional

import structlog

from config import Settings

try:  # Import lazily to allow running without collector package installed.
    from collector.api_client import APIClient
    from collector.config import APISettings
    from collector.pacifica_rest import PacificaREST
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIClient = None  # type: ignore[assignment]
    APISettings = None  # type: ignore[assignment]
    PacificaREST = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)


class LiveDataClient:
    """Lightweight adapter that fetches latest prices from Pacifica's REST API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rest: PacificaREST | None = None

        if APIClient and APISettings and PacificaREST:
            api = settings.api
            api_settings = APISettings(
                base_url=api.effective_base_url,
                api_key=api.api_key,
                timeout=api.timeout,
                network=api.network,
                max_retries=api.max_retries,
            )
            client = APIClient(api_settings)
            self._rest = PacificaREST(client)
        else:
            log.debug("collector_api_client_unavailable_falling_back_to_signals")

    async def fetch_price(self, symbol: str) -> Optional[float]:
        if not self._rest:
            return None

        return await asyncio.to_thread(self._fetch_price_sync, symbol.upper())

    def _fetch_price_sync(self, symbol: str) -> Optional[float]:
        try:
            payload = self._rest.get_recent_trades(symbol=symbol)
        except Exception as exc:  # pragma: no cover - network failure path
            log.warning("live_price_request_failed", symbol=symbol, error=str(exc))
            return None

        price = self._extract_price(payload)
        if price is None:
            log.debug("live_price_parse_failed", symbol=symbol, payload=payload)
        return price

    @staticmethod
    def _extract_price(payload: Any) -> Optional[float]:
        if payload is None:
            return None

        data: Any = payload
        if isinstance(payload, dict):
            # Pacifica API typically wraps results under "data" or "result".
            data = payload.get("data") or payload.get("result") or payload

        trades: Any = data
        if isinstance(data, dict):
            # Some endpoints return {"trades": [...]}
            trades = data.get("trades") or data.get("items") or data

        if isinstance(trades, dict):
            # A dict keyed by trade ids.
            trades = list(trades.values())

        if isinstance(trades, list):
            for entry in reversed(trades):
                if not isinstance(entry, dict):
                    continue
                price = entry.get("price")
                if price is None:
                    continue
                try:
                    value = float(price)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value

        if isinstance(trades, (int, float)) and trades > 0:
            return float(trades)

        return None

from __future__ import annotations

from typing import Any, Dict, Optional

from .api_client import APIClient


class PacificaREST:
    """Convenience wrapper around Pacifica's public REST endpoints."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_market_info(self) -> Dict[str, Any]:
        return self._fetch("/info")

    def get_prices(self) -> Dict[str, Any]:
        return self._fetch("/info/prices")

    def get_kline(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "start_time": start_time,
        }
        if end_time is not None:
            params["end_time"] = end_time
        return self._fetch("/kline", params=params)

    def get_orderbook(
        self, *, symbol: str, agg_level: Optional[int] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol}
        if agg_level is not None:
            params["agg_level"] = agg_level
        return self._fetch("/book", params=params)

    def get_recent_trades(self, *, symbol: str) -> Dict[str, Any]:
        return self._fetch("/trades", params={"symbol": symbol})

    def get_historical_funding(
        self,
        *,
        symbol: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._fetch("/funding_rate/history", params=params)

    def _fetch(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = self.client.get(endpoint, params=params)
        if isinstance(payload, dict) and not payload.get("success", True):
            error = payload.get("error") or payload.get("code") or "Unknown error"
            raise RuntimeError(f"Pacifica API request failed: {error}")
        return payload

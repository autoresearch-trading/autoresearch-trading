from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import APISettings

logger = logging.getLogger(__name__)


class APIClient:
    """Lightweight HTTP client tailored for JSON APIs."""

    def __init__(self, settings: APISettings, session: Optional[Session] = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(self._default_headers())
        retry = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _default_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "pacifica-data-collector/1.0",
        }
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _build_url(self, endpoint: str) -> str:
        return urljoin(f"{self.settings.base_url}/", endpoint.lstrip("/"))

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._execute("GET", endpoint, params=params)
        return self._parse_json(response)

    def _execute(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Response:
        url = self._build_url(endpoint)
        logger.debug("Preparing %s request to %s with params=%s json=%s", method, url, params, json)
        response = self.session.request(
            method=method,
            url=url,
            timeout=self.settings.timeout,
            params=params,
            json=json,
        )
        logger.info(
            "Completed %s request to %s with status=%s in %.3fs",
            method,
            url,
            response.status_code,
            response.elapsed.total_seconds() if response.elapsed else 0.0,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _parse_json(response: Response) -> Dict[str, Any]:
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise ValueError("Expected JSON response.")
        payload = response.json()
        logger.debug("Parsed JSON response with top-level keys: %s", list(payload.keys()) if isinstance(payload, dict) else type(payload))
        return payload

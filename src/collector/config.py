from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

PACIFICA_BASE_URLS: Dict[str, str] = {
    "mainnet": "https://api.pacifica.fi/api/v1",
    "testnet": "https://test-api.pacifica.fi/api/v1",
}


def load_environment(env_file: str = ".env") -> None:
    """Load environment variables from the given file if available."""
    if Path(env_file).exists():
        load_dotenv(env_file)
    else:
        # Fall back to default .env if a custom path was provided but missing.
        if env_file != ".env":
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv()


@dataclass
class APISettings:
    """Configuration for accessing the Pacifica REST API."""

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 10.0
    network: str = "mainnet"

    @classmethod
    def from_env(cls) -> "APISettings":
        """Create settings from environment variables."""
        load_environment()

        network = os.getenv("PACIFICA_NETWORK", "mainnet").strip().lower()
        if network not in PACIFICA_BASE_URLS:
            raise ValueError(
                f"Unsupported PACIFICA_NETWORK '{network}'. "
                f"Valid options: {', '.join(sorted(PACIFICA_BASE_URLS))}."
            )

        base_url = os.getenv("API_BASE_URL") or PACIFICA_BASE_URLS[network]
        api_key = os.getenv("PACIFICA_API_KEY") or os.getenv("API_KEY")
        timeout_raw = os.getenv("API_TIMEOUT", os.getenv("PACIFICA_API_TIMEOUT", "10"))

        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("API_TIMEOUT must be numeric.") from exc

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            network=network,
        )

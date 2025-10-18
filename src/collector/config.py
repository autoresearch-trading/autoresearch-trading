from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from config import DEFAULT_PACIFICA_BASE_URLS, PacificaAPISettings

PACIFICA_BASE_URLS: Dict[str, str] = dict(DEFAULT_PACIFICA_BASE_URLS)


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
    """Configuration for accessing the Pacifica REST API.

    Prefer :class:`config.PacificaAPISettings` for new code. This wrapper is kept
    for backwards compatibility with existing collector scripts.
    """

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 10.0
    network: str = "mainnet"
    max_retries: int = 5

    @classmethod
    def from_env(cls) -> "APISettings":
        """Create settings from environment variables."""
        load_environment()

        new_settings = PacificaAPISettings()
        return cls(
            base_url=new_settings.effective_base_url,
            api_key=new_settings.api_key,
            timeout=new_settings.timeout,
            network=new_settings.network,
            max_retries=new_settings.max_retries,
        )

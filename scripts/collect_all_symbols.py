#!/usr/bin/env python3
"""
Script to collect data from all available symbols on Pacifica API.
This script dynamically fetches all available symbols and collects data from them.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from config import PacificaAPISettings


def get_all_symbols(api: PacificaAPISettings | None = None) -> List[str]:
    """Fetch all available symbols from Pacifica API."""
    api = api or PacificaAPISettings()
    try:
        response = requests.get(f"{api.effective_base_url}/info", timeout=api.timeout)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError(f"API returned success=False: {data}")

        symbols = [item["symbol"] for item in data["data"]]
        print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
        return symbols

    except Exception as e:
        print(f"Error fetching symbols: {e}")
        sys.exit(1)


def collect_data_for_symbols(symbols: List[str], **kwargs) -> None:
    """Collect data for the given symbols."""

    # Default parameters
    params = {
        "poll_funding": kwargs.get("poll_funding", "60s"),
        "book_depth": kwargs.get("book_depth", 25),
        "out_root": kwargs.get("out_root", "./data"),
        "max_rps": kwargs.get("max_rps", 2),  # Conservative rate limit
    }

    # Create the command
    symbols_str = ",".join(symbols)
    collector_script = PROJECT_ROOT / "scripts" / "collect_data.py"
    cmd = [
        "python3",
        str(collector_script),
        "live",
        "--symbols",
        symbols_str,
        "--poll-funding",
        params["poll_funding"],
        "--book-depth",
        str(params["book_depth"]),
        "--out-root",
        params["out_root"],
        "--max-rps",
        str(params["max_rps"]),
    ]

    print(f"Starting data collection for {len(symbols)} symbols...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Rate limit: {params['max_rps']} RPS")
    print("Press Ctrl+C to stop")

    try:
        # Run the collector
        subprocess.run(cmd, check=True)
        print("Data collection completed successfully!")

    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Data collection failed with exit code {e.returncode}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    """Main function."""
    print("Pacifica Data Collector - All Symbols")
    print("=" * 50)

    # Get all available symbols
    api_config = PacificaAPISettings()
    symbols = get_all_symbols(api_config)

    if not symbols:
        print("No symbols found!")
        sys.exit(1)

    # Ask user for confirmation
    print(f"\nThis will collect data for {len(symbols)} symbols:")
    print(f"Symbols: {', '.join(symbols)}")

    response = input("\nProceed? (y/N): ").strip().lower()
    if response not in ["y", "yes"]:
        print("Aborted by user")
        sys.exit(0)

    # Collect data
    collect_data_for_symbols(symbols)


if __name__ == "__main__":
    main()

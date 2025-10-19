#!/usr/bin/env python3
"""
Cloud-optimized collector that fetches all available symbols
and collects data with proper rate limiting.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import requests
import structlog

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from collector.config import APISettings
from collector.live_runner import LiveRunner
from collector.utils import parse_duration
from config import PacificaAPISettings

logger = structlog.get_logger(__name__)


def get_all_symbols(api: PacificaAPISettings | None = None) -> list[str]:
    """Fetch all available symbols from Pacifica API."""
    api = api or PacificaAPISettings()
    try:
        response = requests.get(f"{api.effective_base_url}/info", timeout=api.timeout)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError(f"API returned success=False: {data}")

        symbols = [item["symbol"] for item in data["data"]]
        logger.info("fetched_symbols", count=len(symbols), symbols=symbols)
        return symbols

    except Exception as e:
        logger.error(
            "failed_to_fetch_symbols",
            error=str(e),
            base_url=api.effective_base_url,
        )
        # Fallback to common symbols
        return ["BTC", "ETH", "SOL"]


def setup_health_check():
    """Start health check server in background."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                # Check if data is being written
                data_dir = Path("/app/data/trades")
                status = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_dir_exists": data_dir.exists(),
                }

                if data_dir.exists():
                    from datetime import timedelta

                    recent_files = []
                    cutoff = datetime.utcnow() - timedelta(minutes=5)

                    for file in data_dir.rglob("*.parquet"):
                        try:
                            mtime = datetime.fromtimestamp(file.stat().st_mtime)
                            if mtime > cutoff:
                                recent_files.append(str(file.name))
                        except OSError:
                            pass

                    status["recent_files_count"] = len(recent_files)
                    status["healthy"] = len(recent_files) > 0

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress logs

    def run_server():
        server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
        logger.info("health_check_server_started", port=8080)
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


def configure_logging():
    """Configure structured logging."""
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


async def main():
    configure_logging()
    logger.info("starting_cloud_collector")

    # Start health check server
    setup_health_check()

    # Get all symbols
    api_config = PacificaAPISettings()
    symbols = get_all_symbols(api_config)
    if not symbols:
        logger.error("no_symbols_available")
        sys.exit(1)

    logger.info("symbols_to_collect", count=len(symbols), symbols=symbols)

    # Load settings from environment
    settings = APISettings.from_env()

    # Configure poll intervals from environment
    poll_config = {
        "prices": parse_duration(os.getenv("POLL_PRICES", "2s")),
        "trades": parse_duration(os.getenv("POLL_TRADES", "1s")),
        "orderbook": parse_duration(os.getenv("POLL_ORDERBOOK", "3s")),
        "funding": parse_duration(os.getenv("POLL_FUNDING", "60s")),
    }

    max_rps = int(os.getenv("MAX_RPS", "3"))
    book_depth = int(os.getenv("BOOK_DEPTH", "25"))
    out_root = os.getenv("OUT_ROOT", "/app/data")

    logger.info(
        "collector_config",
        symbols=len(symbols),
        max_rps=max_rps,
        poll_config=poll_config,
        out_root=out_root,
    )

    # Create runner
    runner = LiveRunner(
        settings=settings,
        max_rps=max_rps,
        out_root=out_root,
        book_depth=book_depth,
        agg_level=None,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        runner.request_stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run collector
    try:
        logger.info("starting_live_collection")
        await runner.run(symbols, poll_config)
    except Exception as e:
        logger.exception("collector_error", error=str(e))
        raise
    finally:
        logger.info("collector_stopped")


if __name__ == "__main__":
    asyncio.run(main())

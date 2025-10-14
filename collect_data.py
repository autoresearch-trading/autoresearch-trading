#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from collector.api_client import APIClient
from collector.config import PACIFICA_BASE_URLS, APISettings
from collector.live_runner import LiveRunner
from collector.pacifica_rest import PacificaREST
from collector.utils import parse_duration

logger = structlog.get_logger(__name__)


def parse_params(pairs: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid param '{pair}'. Use key=value syntax.")
        key, value = pair.split("=", 1)
        parsed[key] = value
    return parsed


def parse_timestamp(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse timestamp '{raw}'. "
            "Use milliseconds since epoch or ISO-8601 (e.g. 2024-07-01T00:00:00Z)."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_symbols_arg(raw: str) -> List[str]:
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one symbol must be provided to --symbols.")
    return sorted(dict.fromkeys(symbols))


def resolve_settings(args: argparse.Namespace) -> APISettings:
    settings = APISettings.from_env()
    network = (args.network or settings.network).lower()
    if network not in PACIFICA_BASE_URLS:
        raise ValueError(
            f"Unsupported network '{network}'. "
            f"Valid options: {', '.join(sorted(PACIFICA_BASE_URLS))}."
        )

    base_url = settings.base_url
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    elif args.network:
        base_url = PACIFICA_BASE_URLS[network]

    timeout = args.timeout if args.timeout is not None else settings.timeout
    api_key = args.api_key or settings.api_key

    return replace(settings, base_url=base_url, timeout=timeout, api_key=api_key, network=network)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pacifica REST data collector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--network",
        choices=sorted(PACIFICA_BASE_URLS),
        help="Override the Pacifica environment inferred from PACIFICA_NETWORK.",
    )
    parser.add_argument("--base-url", help="Override the REST base URL.")
    parser.add_argument("--timeout", type=float, help="Request timeout in seconds.")
    parser.add_argument("--api-key", help="API key for private endpoints (if enabled).")
    parser.add_argument(
        "--out",
        dest="output_path",
        help="Optional file path to write the JSON response.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for the collector.",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path to append collector logs.",
    )
    parser.add_argument(
        "--log-payload",
        action="store_true",
        help="Log the full JSON response payload through the logger.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    market_info = subparsers.add_parser("market-info", help="List exchange metadata for all symbols.")
    market_info.set_defaults(handler=run_market_info)

    prices = subparsers.add_parser("prices", help="Get current pricing snapshot for all symbols.")
    prices.set_defaults(handler=run_prices)

    kline = subparsers.add_parser("kline", help="Fetch historical candle data.")
    kline.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTC.")
    kline.add_argument(
        "--interval",
        required=True,
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"],
        help="Candlestick interval.",
    )
    kline.add_argument(
        "--start",
        required=True,
        help="Start time in ms since epoch or ISO-8601 (UTC assumed if no timezone).",
    )
    kline.add_argument(
        "--end",
        help="End time in ms since epoch or ISO-8601. Defaults to now when omitted.",
    )
    kline.set_defaults(handler=run_kline)

    orderbook = subparsers.add_parser("orderbook", help="Fetch current order book snapshot.")
    orderbook.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTC.")
    orderbook.add_argument(
        "--agg-level",
        type=int,
        help="Aggregation level for price grouping. Leave unset for default (1).",
    )
    orderbook.set_defaults(handler=run_orderbook)

    trades = subparsers.add_parser("recent-trades", help="Fetch recent public trades for a symbol.")
    trades.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTC.")
    trades.set_defaults(handler=run_recent_trades)

    funding = subparsers.add_parser("funding", help="Fetch historical funding rates.")
    funding.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTC.")
    funding.add_argument("--limit", type=int, help="Number of records to fetch (max 4000).")
    funding.add_argument("--offset", type=int, help="Pagination offset.")
    funding.set_defaults(handler=run_funding)

    raw = subparsers.add_parser(
        "raw",
        help="Perform an ad-hoc GET request against the REST API. Useful for experimentation.",
    )
    raw.add_argument("endpoint", help="Endpoint path, e.g. /trades?symbol=BTC or /info.")
    raw.add_argument(
        "--param",
        dest="params",
        action="append",
        default=[],
        help="Query parameter in key=value form. Repeat for multiple parameters.",
    )
    raw.set_defaults(handler=run_raw_request)

    live = subparsers.add_parser(
        "live",
        help="Continuously poll Pacifica endpoints and persist results to Parquet.",
    )
    live.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of symbols (e.g. BTC,ETH).",
    )
    live.add_argument(
        "--poll-prices",
        default="2s",
        help="Polling cadence for the price snapshot endpoint.",
    )
    live.add_argument(
        "--poll-trades",
        default="1s",
        help="Polling cadence for the recent trades endpoint.",
    )
    live.add_argument(
        "--poll-orderbook",
        default="3s",
        help="Polling cadence for the order book endpoint.",
    )
    live.add_argument(
        "--poll-funding",
        default="60s",
        help="Polling cadence for funding history.",
    )
    live.add_argument(
        "--book-depth",
        type=int,
        default=25,
        help="How many price levels to retain per side of the order book.",
    )
    live.add_argument(
        "--agg-level",
        type=int,
        help="Aggregation level forwarded to the order book endpoint.",
    )
    live.add_argument(
        "--out-root",
        default="data",
        help="Path where Parquet datasets will be written.",
    )
    live.add_argument(
        "--max-rps",
        type=int,
        default=4,
        help="Global rate limit in requests per second.",
    )
    live.add_argument(
        "--prices-rps",
        type=int,
        help="Optional per-endpoint RPS override for prices polling.",
    )
    live.add_argument(
        "--trades-rps",
        type=int,
        help="Optional per-endpoint RPS override for trades polling.",
    )
    live.add_argument(
        "--orderbook-rps",
        type=int,
        help="Optional per-endpoint RPS override for order book polling.",
    )
    live.add_argument(
        "--funding-rps",
        type=int,
        help="Optional per-endpoint RPS override for funding polling.",
    )
    live.set_defaults(handler=run_live)

    return parser


def run_market_info(rest: PacificaREST, _: APIClient, __: argparse.Namespace) -> Dict[str, Any]:
    return rest.get_market_info()


def run_prices(rest: PacificaREST, _: APIClient, __: argparse.Namespace) -> Dict[str, Any]:
    return rest.get_prices()


def run_kline(rest: PacificaREST, _: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end) if args.end else None
    return rest.get_kline(symbol=args.symbol, interval=args.interval, start_time=start, end_time=end)


def run_orderbook(rest: PacificaREST, _: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    return rest.get_orderbook(symbol=args.symbol, agg_level=args.agg_level)


def run_recent_trades(rest: PacificaREST, _: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    return rest.get_recent_trades(symbol=args.symbol)


def run_funding(rest: PacificaREST, _: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    return rest.get_historical_funding(symbol=args.symbol, limit=args.limit, offset=args.offset)


def run_raw_request(rest: PacificaREST, client: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    params = parse_params(args.params)
    # Allow users to pass inline query strings, otherwise rely on --param flags.
    endpoint = args.endpoint
    payload = client.get(endpoint, params=params or None)
    if isinstance(payload, dict) and not payload.get("success", True):
        error = payload.get("error") or payload.get("code") or "Unknown error"
        raise RuntimeError(f"Pacifica API request failed: {error}")
    return payload


def run_live(_: PacificaREST, client: APIClient, args: argparse.Namespace) -> Dict[str, Any]:
    settings = client.settings
    symbols = parse_symbols_arg(args.symbols)
    poll_config = {
        "prices": parse_duration(args.poll_prices),
        "trades": parse_duration(args.poll_trades),
        "orderbook": parse_duration(args.poll_orderbook),
        "funding": parse_duration(args.poll_funding),
    }
    per_endpoint = {
        key: value
        for key, value in {
            "prices": args.prices_rps,
            "trades": args.trades_rps,
            "orderbook": args.orderbook_rps,
            "funding": args.funding_rps,
        }.items()
        if value
    }
    runner = LiveRunner(
        settings=settings,
        max_rps=max(1, args.max_rps),
        out_root=args.out_root,
        book_depth=args.book_depth,
        agg_level=args.agg_level,
        per_endpoint_rps=per_endpoint or None,
    )
    logger.info("starting_live_run", symbols=symbols, poll_config=poll_config)
    try:
        asyncio.run(runner.run(symbols, poll_config))
    except KeyboardInterrupt:
        runner.request_stop()
        logger.info("live_run_interrupted")
    return {"success": True, "mode": "live", "symbols": symbols}


def emit(payload: Dict[str, Any], output_path: Optional[str]) -> None:
    formatted = json.dumps(payload, indent=2)
    size_bytes = len(formatted.encode("utf-8"))
    if output_path:
        path = Path(output_path).expanduser()
        path.write_text(formatted)
        print(f"Wrote response to {path}")
        logger.info("payload_written", destination=str(path), bytes=size_bytes)
    else:
        print(formatted)
        logger.info("payload_emitted_stdout", bytes=size_bytes)
    logger.debug("payload_preview", preview=formatted[:500])


def configure_logging(level_name: str, log_file: Optional[str]) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=processors,
    )

    handlers: List[logging.Handler] = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    structlog.configure(
        processors=processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()

    configure_logging(args.log_level, args.log_file)
    safe_args = {
        key: value for key, value in vars(args).items() if key not in {"handler", "api_key"}
    }
    logger.debug("parsed_arguments", arguments=safe_args)

    try:
        settings = resolve_settings(args)
        client = APIClient(settings=settings)
        rest = PacificaREST(client)
        logger.info("executing_command", command=args.command, options=safe_args)
        payload = args.handler(rest, client, args)
        logger.info("command_completed", command=args.command)
        if args.log_payload:
            logger.info("response_payload", payload=payload)
    except Exception as exc:
        logger.exception("command_failed", command=getattr(args, "command", "unknown"), error=str(exc))
        print(f"[collector] Error: {exc}", file=sys.stderr)
        sys.exit(1)

    emit(payload, args.output_path)


if __name__ == "__main__":
    main()

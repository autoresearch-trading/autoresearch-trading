# scripts/collect_pacifica_full_fidelity.py
"""Full-fidelity public Pacifica market-data archival collector.

This collector intentionally stores raw websocket/REST payloads as JSONL.GZ before
any lossy transformation.  It is meant to run alongside the existing derived
`data/trades` and `data/orderbook` collectors, not replace them.

Public websocket streams covered:

- prices, global stream
- trades, per symbol
- book, per symbol and aggregation level
- bbo, per symbol
- candle, per symbol and interval
- mark_price_candle, per symbol and interval

Private/account streams are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import signal
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

REST_BASE = "https://api.pacifica.fi/api/v1"
DEFAULT_WS_URL = "wss://ws.pacifica.fi/ws"
DEFAULT_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "8h",
    "12h",
    "1d",
)
DEFAULT_AGG_LEVELS = (1,)
DEFAULT_OUT_DIR = Path("data/pacifica_full_fidelity")
PUBLIC_REST_ENDPOINTS = ("/info", "/info/prices")

JsonObject = dict[str, Any]


def utc_ms() -> int:
    return int(time.time() * 1000)


def utc_dt_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC)


def parse_symbol_filter(value: str | None) -> list[str]:
    if not value:
        return []
    seen: set[str] = set()
    symbols: list[str] = []
    for raw in value.split(","):
        symbol = raw.strip()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def parse_csv_values(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    parsed = parse_symbol_filter(value)
    return tuple(parsed) if parsed else tuple(default)


def parse_int_csv(value: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if not value:
        return tuple(default)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def fetch_json(path: str) -> Any:
    with urlopen(
        f"{REST_BASE}{path}", timeout=30
    ) as response:  # noqa: S310 - public API archive
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("success") is True:
        return payload.get("data")
    return payload


def fetch_live_symbols() -> list[str]:
    data = fetch_json("/info")
    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected /info response shape; expected list of market objects"
        )
    symbols = sorted(
        {
            str(row["symbol"])
            for row in data
            if isinstance(row, dict) and row.get("symbol")
        }
    )
    if not symbols:
        raise RuntimeError("No symbols returned by /info")
    return symbols


def build_subscriptions(
    symbols: Sequence[str],
    *,
    intervals: Sequence[str] = DEFAULT_INTERVALS,
    agg_levels: Sequence[int] = DEFAULT_AGG_LEVELS,
    include_prices: bool = True,
) -> list[JsonObject]:
    subscriptions: list[JsonObject] = []
    if include_prices:
        subscriptions.append({"method": "subscribe", "params": {"source": "prices"}})

    for symbol in symbols:
        subscriptions.append(
            {"method": "subscribe", "params": {"source": "trades", "symbol": symbol}}
        )
        subscriptions.append(
            {"method": "subscribe", "params": {"source": "bbo", "symbol": symbol}}
        )
        for agg_level in agg_levels:
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "source": "book",
                        "symbol": symbol,
                        "agg_level": int(agg_level),
                    },
                }
            )
        for interval in intervals:
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "source": "candle",
                        "symbol": symbol,
                        "interval": interval,
                    },
                }
            )
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "source": "mark_price_candle",
                        "symbol": symbol,
                        "interval": interval,
                    },
                }
            )
    return subscriptions


def _event_time_ms(channel: str, data: Any, recv_ms: int) -> int:
    if isinstance(data, dict):
        for key in ("timestamp", "t", "T"):
            if key in data and data[key] is not None:
                try:
                    return int(data[key])
                except (TypeError, ValueError):
                    pass
    return recv_ms


def _event_symbol(channel: str, data: Any) -> str:
    if isinstance(data, dict):
        for key in ("symbol", "s"):
            if data.get(key):
                return str(data[key])
    if channel == "prices":
        return "ALL"
    return "UNKNOWN"


def event_rows_from_message(
    message: JsonObject, *, recv_ms: int, raw_text: str
) -> list[JsonObject]:
    channel = str(message.get("channel", "unknown"))
    data = message.get("data")
    events = data if isinstance(data, list) else [data]
    rows: list[JsonObject] = []
    for item in events:
        event_ts_ms = _event_time_ms(channel, item, recv_ms)
        rows.append(
            {
                "recv_ms": recv_ms,
                "event_ts_ms": event_ts_ms,
                "channel": channel,
                "symbol": _event_symbol(channel, item),
                "data": item,
                "raw_message": message,
                "raw_text": raw_text,
            }
        )
    return rows


def channel_symbol_date(record: JsonObject) -> tuple[str, str, str]:
    channel = str(record.get("channel") or "unknown")
    data = record.get("data")
    symbol = str(record.get("symbol") or _event_symbol(channel, data))
    ts_ms = int(
        record.get("event_ts_ms")
        or _event_time_ms(channel, data, int(record["recv_ms"]))
    )
    date = utc_dt_from_ms(ts_ms).date().isoformat()
    return channel, symbol, date


def _hour_from_record(record: JsonObject) -> str:
    ts_ms = int(record.get("event_ts_ms") or record.get("recv_ms") or utc_ms())
    return utc_dt_from_ms(ts_ms).strftime("%H")


def write_jsonl_records(
    root: Path, records: Iterable[JsonObject], *, run_id: str
) -> list[Path]:
    grouped: dict[Path, list[JsonObject]] = {}
    for record in records:
        channel, symbol, date = channel_symbol_date(record)
        path = (
            root
            / f"channel={channel}"
            / f"symbol={symbol}"
            / f"date={date}"
            / f"{run_id}.jsonl.gz"
        )
        grouped.setdefault(path, []).append(record)

    written: list[Path] = []
    for path, rows in grouped.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
                fh.write("\n")
        written.append(path)
    return sorted(written)


def write_rest_snapshot(
    root: Path, endpoint: str, payload: Any, *, run_id: str, recv_ms: int
) -> Path:
    safe_endpoint = endpoint.strip("/").replace("/", "_") or "root"
    date = utc_dt_from_ms(recv_ms).date().isoformat()
    hour = utc_dt_from_ms(recv_ms).strftime("%H")
    path = (
        root
        / "rest"
        / f"endpoint={safe_endpoint}"
        / f"date={date}"
        / f"hour={hour}"
        / f"{run_id}.jsonl.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"recv_ms": recv_ms, "endpoint": endpoint, "payload": payload}
    with gzip.open(path, "at", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
        fh.write("\n")
    return path


async def subscribe_all(
    ws: Any, subscriptions: Sequence[JsonObject], *, batch_size: int, delay_s: float
) -> None:
    for index in range(0, len(subscriptions), batch_size):
        batch = subscriptions[index : index + batch_size]
        for sub in batch:
            await ws.send(json.dumps(sub, separators=(",", ":")))
        if delay_s > 0:
            await asyncio.sleep(delay_s)


async def heartbeat_loop(
    ws: Any, stop: asyncio.Event, *, interval_s: float = 30.0
) -> None:
    while not stop.is_set():
        await asyncio.sleep(interval_s)
        if stop.is_set():
            return
        await ws.send(json.dumps({"method": "ping"}))


async def rest_snapshot_loop(
    root: Path, run_id: str, stop: asyncio.Event, *, interval_s: float
) -> None:
    if interval_s <= 0:
        return
    while not stop.is_set():
        recv_ms = utc_ms()
        for endpoint in PUBLIC_REST_ENDPOINTS:
            try:
                write_rest_snapshot(
                    root, endpoint, fetch_json(endpoint), run_id=run_id, recv_ms=recv_ms
                )
            except Exception as exc:  # pragma: no cover - long-running operational path
                print(f"REST snapshot failed for {endpoint}: {exc}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass


async def collect_ws(
    *,
    ws_url: str,
    out_dir: Path,
    run_id: str,
    subscriptions: Sequence[JsonObject],
    stop: asyncio.Event,
    batch_size: int,
    subscribe_delay_s: float,
) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing dependency: install with `uv add websockets` or run `uv sync`."
        ) from exc

    while not stop.is_set():
        try:
            async with websockets.connect(
                ws_url, ping_interval=None, close_timeout=10
            ) as ws:
                await subscribe_all(
                    ws, subscriptions, batch_size=batch_size, delay_s=subscribe_delay_s
                )
                heartbeat = asyncio.create_task(heartbeat_loop(ws, stop))
                try:
                    while not stop.is_set():
                        raw_text = await ws.recv()
                        recv_ms = utc_ms()
                        if not isinstance(raw_text, str):
                            raw_text = raw_text.decode("utf-8")
                        try:
                            message = json.loads(raw_text)
                        except json.JSONDecodeError:
                            message = {"channel": "unparseable", "data": raw_text}
                        rows = event_rows_from_message(
                            message, recv_ms=recv_ms, raw_text=raw_text
                        )
                        write_jsonl_records(out_dir, rows, run_id=run_id)
                finally:
                    heartbeat.cancel()
        except Exception as exc:  # pragma: no cover - long-running operational path
            if stop.is_set():
                return
            print(
                f"websocket collection error; reconnecting in 5s: {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(5)


async def amain(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    symbols = parse_symbol_filter(args.symbols) or fetch_live_symbols()
    intervals = parse_csv_values(args.intervals, DEFAULT_INTERVALS)
    agg_levels = parse_int_csv(args.agg_levels, DEFAULT_AGG_LEVELS)
    subscriptions = build_subscriptions(
        symbols,
        intervals=intervals,
        agg_levels=agg_levels,
        include_prices=not args.no_prices,
    )
    run_id = args.run_id or datetime.now(tz=UTC).strftime("run-%Y%m%dT%H%M%SZ")

    if args.print_plan or args.dry_run:
        print(
            json.dumps(
                {"run_id": run_id, "symbols": symbols, "subscriptions": subscriptions},
                indent=2,
            )
        )
        if args.dry_run:
            return

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - platform dependent
            pass

    tasks = [
        asyncio.create_task(
            collect_ws(
                ws_url=args.ws_url,
                out_dir=out_dir,
                run_id=run_id,
                subscriptions=subscriptions,
                stop=stop,
                batch_size=args.subscription_batch_size,
                subscribe_delay_s=args.subscription_delay_s,
            )
        )
    ]
    if args.rest_snapshot_interval_s > 0:
        tasks.append(
            asyncio.create_task(
                rest_snapshot_loop(
                    out_dir, run_id, stop, interval_s=args.rest_snapshot_interval_s
                )
            )
        )

    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--symbols", help="Comma-separated symbols. Defaults to live /info symbols."
    )
    parser.add_argument(
        "--intervals",
        help="Comma-separated candle intervals. Defaults to all documented intervals.",
    )
    parser.add_argument(
        "--agg-levels",
        default="1",
        help="Comma-separated book aggregation levels. Default: 1",
    )
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help="Do not subscribe to global prices stream.",
    )
    parser.add_argument("--rest-snapshot-interval-s", type=float, default=60.0)
    parser.add_argument("--subscription-batch-size", type=int, default=50)
    parser.add_argument("--subscription-delay-s", type=float, default=0.25)
    parser.add_argument("--run-id")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

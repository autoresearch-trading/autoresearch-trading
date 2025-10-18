from __future__ import annotations

import asyncio
from collections import defaultdict
from threading import Lock
from typing import Awaitable, Callable, List, MutableMapping

import structlog

from signals.base import Signal

log = structlog.get_logger(__name__)

SignalCallback = Callable[[Signal], None] | Callable[[Signal], Awaitable[None]]


class SignalRouter:
    """Lightweight in-memory fan-out for real-time signals."""

    _subscribers: MutableMapping[str, List[SignalCallback]] = defaultdict(list)
    _global_subscribers: List[SignalCallback] = []
    _signal_queue: asyncio.Queue[Signal] | None = None
    _queue_lock: Lock = Lock()
    _max_queue_size: int = 10_000

    @classmethod
    def initialize(cls) -> None:
        """Ensure the internal queue exists before routing starts."""
        cls._ensure_queue()

    @classmethod
    def subscribe(cls, symbol: str, callback: SignalCallback) -> None:
        """Register a callback invoked for the given symbol."""
        symbol = symbol.upper()
        cls._subscribers[symbol].append(callback)
        log.debug("signal_router_subscribed", symbol=symbol, total=len(cls._subscribers[symbol]))

    @classmethod
    def subscribe_all(cls, callback: SignalCallback) -> None:
        """Register a callback invoked for every signal regardless of symbol."""
        cls._global_subscribers.append(callback)
        log.debug("signal_router_subscribed_all", total=len(cls._global_subscribers))

    @classmethod
    def unsubscribe(cls, symbol: str, callback: SignalCallback) -> None:
        """Remove a previously registered callback."""
        symbol = symbol.upper()
        callbacks = cls._subscribers.get(symbol)
        if not callbacks:
            return
        try:
            callbacks.remove(callback)
            log.debug("signal_router_unsubscribed", symbol=symbol, remaining=len(callbacks))
        except ValueError:
            pass

    @classmethod
    def unsubscribe_all(cls, callback: SignalCallback) -> None:
        """Remove a callback from the global subscriber list."""
        try:
            cls._global_subscribers.remove(callback)
            log.debug("signal_router_unsubscribed_all", remaining=len(cls._global_subscribers))
        except ValueError:
            pass

    @classmethod
    def has_subscribers(cls) -> bool:
        """Return True if any consumer is registered."""
        return bool(cls._global_subscribers or any(cls._subscribers.values()))

    @classmethod
    def route_signal(cls, signal: Signal) -> None:
        """Enqueue a signal for asynchronous dispatch."""
        queue = cls._signal_queue
        if queue is None:
            log.warning("signal_router_queue_uninitialized", symbol=signal.symbol)
            return

        try:
            queue.put_nowait(signal)
        except asyncio.QueueFull:
            log.error("signal_router_queue_full", symbol=signal.symbol)

    @classmethod
    async def dispatch_loop(cls) -> None:
        """Continuously dispatch signals to subscribers."""
        queue = cls._ensure_queue()
        while True:
            signal = await queue.get()
            await cls._dispatch_signal(signal)
            queue.task_done()

    @classmethod
    async def _dispatch_signal(cls, signal: Signal) -> None:
        callbacks = list(cls._global_subscribers)
        callbacks.extend(cls._subscribers.get(signal.symbol.upper(), []))

        if not callbacks:
            return

        for callback in callbacks:
            try:
                result = callback(signal)
                if asyncio.iscoroutine(result):
                    await result  # type: ignore[func-returns-value]
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error(
                    "signal_router_callback_failed",
                    symbol=signal.symbol,
                    callback=getattr(callback, "__qualname__", repr(callback)),
                    error=str(exc),
                )

    @classmethod
    def _ensure_queue(cls) -> asyncio.Queue[Signal]:
        queue = cls._signal_queue
        if queue is not None:
            return queue

        with cls._queue_lock:
            if cls._signal_queue is None:
                cls._signal_queue = asyncio.Queue(maxsize=cls._max_queue_size)
            queue = cls._signal_queue

        return queue  # type: ignore[return-value]

    @classmethod
    async def drain(cls) -> None:
        """Drain pending signals. Primarily used in tests."""
        queue = cls._signal_queue
        if queue is None:
            return
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    @classmethod
    def reset(cls) -> None:
        """Reset router state. Intended for testing."""
        cls._subscribers.clear()
        cls._global_subscribers.clear()
        queue = cls._signal_queue
        cls._signal_queue = None
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()

from __future__ import annotations

import asyncio
from queue import Empty, Full, Queue
from typing import Any, List, Tuple

import structlog
from config import Settings
from db.questdb import QuestDBClient
from signals.base import PaperTrade, Signal

log = structlog.get_logger(__name__)

_QUEUE_MAXSIZE = 100_000
_BATCH_SIZE = 1_000
_QUEUE: Queue[Tuple[str, Any]] = Queue(maxsize=_QUEUE_MAXSIZE)


def enqueue_signal(step_id: str, signal: Signal) -> None:
    """Enqueue a signal for asynchronous persistence (bytewax inspect callback)."""
    _enqueue(("signal", signal))


def enqueue_trade(trade: PaperTrade) -> None:
    """Enqueue a paper trade for asynchronous persistence."""
    _enqueue(("trade", trade))


def _enqueue(item: Tuple[str, Any]) -> None:
    try:
        _QUEUE.put_nowait(item)
    except Full:
        log.error("async_writer_queue_full", item_type=item[0])


async def writer_loop(settings: Settings) -> None:
    """Background coroutine that flushes queued items to QuestDB."""
    client = QuestDBClient(
        host=settings.questdb_host,
        port=settings.questdb_port,
        user=settings.questdb_user,
        password=settings.questdb_password,
    )

    while True:
        try:
            batch = await _drain_batch(timeout=1.0)
            if not batch:
                continue

            signals: List[Signal] = []
            trades: List[PaperTrade] = []
            for item_type, payload in batch:
                if item_type == "signal":
                    signals.append(payload)
                elif item_type == "trade":
                    trades.append(payload)

            if signals:
                await asyncio.to_thread(client.write_signals_batch, signals)
                log.debug("async_writer_signals_written", count=len(signals))

            for trade in trades:
                await asyncio.to_thread(client.write_paper_trade, trade)
                log.debug("async_writer_trade_written", trade_id=trade.trade_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("async_writer_error", error=str(exc))
            await asyncio.sleep(1.0)


async def _drain_batch(timeout: float) -> List[Tuple[str, Any]]:
    batch: List[Tuple[str, Any]] = []
    try:
        item = await asyncio.to_thread(_QUEUE.get, True, timeout)
        _QUEUE.task_done()
        batch.append(item)
    except Empty:
        return batch

    while len(batch) < _BATCH_SIZE:
        try:
            item = _QUEUE.get_nowait()
        except Empty:
            break
        else:
            _QUEUE.task_done()
            batch.append(item)
    return batch

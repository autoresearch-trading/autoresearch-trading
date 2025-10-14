from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

import psycopg
from psycopg.rows import dict_row

from bytewax.outputs import DynamicSink, StatelessSinkPartition

from signals.base import MarketRegime, Signal


class QuestDBClient:
    """Thin QuestDB wrapper tailored for signal storage."""

    def __init__(self, host: str, port: int, user: str, password: str) -> None:
        self.conn_string = (
            f"host={host} port={port} user={user} password={password} dbname=qdb"
        )

    def write_signal(self, signal: Signal) -> None:
        """Insert a single signal row."""
        with psycopg.connect(self.conn_string) as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    ts, recv_ts, symbol, signal_type, value, confidence, direction,
                    price, spread_bps, bid_depth, ask_depth, metadata
                )
                VALUES (%(ts)s, %(recv_ts)s, %(symbol)s, %(signal_type)s, %(value)s,
                        %(confidence)s, %(direction)s, %(price)s, %(spread_bps)s,
                        %(bid_depth)s, %(ask_depth)s, %(metadata)s)
                """,
                {
                    "ts": signal.ts,
                    "recv_ts": signal.recv_ts,
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type.value,
                    "value": signal.value,
                    "confidence": signal.confidence,
                    "direction": signal.direction.value,
                    "price": signal.price,
                    "spread_bps": signal.spread_bps,
                    "bid_depth": signal.bid_depth,
                    "ask_depth": signal.ask_depth,
                    "metadata": json.dumps(signal.metadata),
                },
            )

    def write_signals_batch(self, signals: Sequence[Signal]) -> None:
        """Bulk insert signals using COPY for throughput."""
        if not signals:
            return

        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                with cur.copy(
                    """
                    COPY signals (
                        ts, recv_ts, symbol, signal_type, value, confidence, direction,
                        price, spread_bps, bid_depth, ask_depth, metadata
                    )
                    FROM STDIN
                    """
                ) as copy:
                    for signal in signals:
                        copy.write_row(
                            [
                                signal.ts,
                                signal.recv_ts,
                                signal.symbol,
                                signal.signal_type.value,
                                signal.value,
                                signal.confidence,
                                signal.direction.value,
                                signal.price,
                                signal.spread_bps,
                                signal.bid_depth,
                                signal.ask_depth,
                                json.dumps(signal.metadata),
                            ]
                        )

    def query_signals(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
        signal_types: Iterable[str] | None = None,
    ) -> list[Signal]:
        """Fetch signals for analysis or backtesting."""
        query = [
            "SELECT * FROM signals WHERE symbol = %(symbol)s",
            "AND ts BETWEEN %(start_ts)s AND %(end_ts)s",
        ]
        params: dict[str, object] = {
            "symbol": symbol,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

        if signal_types:
            query.append("AND signal_type = ANY(%(signal_types)s)")
            params["signal_types"] = list(signal_types)

        sql = " ".join(query) + " ORDER BY ts ASC"

        with psycopg.connect(self.conn_string, row_factory=dict_row) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: list[Signal] = []
        for row in rows:
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    row["metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    row["metadata"] = {}
            result.append(Signal(**row))
        return result

    def write_regime(self, regime: MarketRegime) -> None:
        """Persist regime classification."""
        self.write_regimes_batch([regime])

    def write_regimes_batch(self, regimes: Sequence[MarketRegime]) -> None:
        """Bulk insert regimes using COPY."""
        if not regimes:
            return

        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                with cur.copy(
                    """
                    COPY regime_log (ts, symbol, regime, atr, spread_bps, funding_rate)
                    FROM STDIN
                    """
                ) as copy:
                    for regime in regimes:
                        copy.write_row(
                            [
                                regime.ts,
                                regime.symbol,
                                regime.regime.value,
                                regime.atr,
                                regime.spread_bps,
                                regime.funding_rate,
                        ]
                    )

    def query_regimes(
        self,
        *,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketRegime]:
        """Fetch regime classifications for backtesting or monitoring."""
        sql = """
            SELECT ts, symbol, regime, atr, spread_bps, funding_rate
            FROM regime_log
            WHERE symbol = %(symbol)s
              AND ts BETWEEN %(start_ts)s AND %(end_ts)s
            ORDER BY ts ASC
        """
        params = {
            "symbol": symbol,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

        with psycopg.connect(self.conn_string, row_factory=dict_row) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [MarketRegime(**row) for row in rows]


class QuestDBSink(DynamicSink[Any]):
    """Bytewax sink that writes batches to QuestDB via QuestDBClient."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        writer: Callable[[QuestDBClient, Sequence[Any]], None],
    ) -> None:
        self._conn_kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
        }
        self._writer = writer

    @classmethod
    def for_signals(cls, *, host: str, port: int, user: str, password: str) -> "QuestDBSink":
        def writer(client: QuestDBClient, batch: Sequence[Any]) -> None:
            signals = [item for item in batch if isinstance(item, Signal)]
            if len(signals) != len(batch):
                raise TypeError("QuestDB signal sink received non-Signal payload")
            client.write_signals_batch(signals)

        return cls(host=host, port=port, user=user, password=password, writer=writer)

    @classmethod
    def for_regimes(cls, *, host: str, port: int, user: str, password: str) -> "QuestDBSink":
        def writer(client: QuestDBClient, batch: Sequence[Any]) -> None:
            regimes = [item for item in batch if isinstance(item, MarketRegime)]
            if len(regimes) != len(batch):
                raise TypeError("QuestDB regime sink received non-MarketRegime payload")
            client.write_regimes_batch(regimes)

        return cls(host=host, port=port, user=user, password=password, writer=writer)

    def build(
        self, step_id: str, worker_index: int, worker_count: int
    ) -> StatelessSinkPartition[Any]:
        client = QuestDBClient(**self._conn_kwargs)
        return _QuestDBSinkPartition(client, self._writer)


class _QuestDBSinkPartition(StatelessSinkPartition[Any]):
    def __init__(
        self,
        client: QuestDBClient,
        writer: Callable[[QuestDBClient, Sequence[Any]], None],
    ) -> None:
        self._client = client
        self._writer = writer

    def write_batch(self, items: Iterable[Any]) -> None:
        batch = [item for item in items if item is not None]
        if not batch:
            return
        self._writer(self._client, batch)

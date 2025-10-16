from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

import psycopg
from psycopg import ProgrammingError
from psycopg.rows import dict_row

from bytewax.outputs import DynamicSink, StatelessSinkPartition

from signals.base import MarketRegime, Signal, Trade


class QuestDBClient:
    """Thin QuestDB wrapper tailored for signal storage."""

    def __init__(self, host: str, port: int, user: str, password: str) -> None:
        self.conn_string = (
            f"host={host} port={port} user={user} password={password} dbname=qdb"
        )

    def list_symbols(self) -> list[str]:
        """Return distinct symbols that have signals stored in QuestDB."""
        sql = "SELECT DISTINCT symbol FROM signals"
        with psycopg.connect(self.conn_string, row_factory=dict_row) as conn:
            rows = conn.execute(sql).fetchall()

        symbols: set[str] = set()
        for row in rows:
            value: object | None
            if isinstance(row, dict):
                value = row.get("symbol")
            else:
                value = row[0] if row else None
            if not value:
                continue
            symbol = str(value).strip().upper()
            if symbol:
                symbols.add(symbol)

        return sorted(symbols)

    def _copy_with_fallback(
        self,
        copy_sql: str,
        rows: Sequence[Sequence[Any]],
        insert_sql: str,
    ) -> None:
        if not rows:
            return

        with psycopg.connect(self.conn_string) as conn:
            try:
                with conn.cursor() as cur:
                    with cur.copy(copy_sql) as copy:
                        for row in rows:
                            copy.write_row(row)
            except Exception as exc:  # COPY can be disabled or unsupported
                error_msg = str(exc).lower()
                is_copy_failure = (
                    isinstance(exc, ProgrammingError)
                    or "copy" in error_msg
                    or "command_ok" in error_msg
                )

                if not is_copy_failure:
                    raise

                conn.rollback()
                try:
                    with conn.cursor() as cur:
                        cur.executemany(insert_sql, rows)
                except Exception as insert_exc:
                    raise exc from insert_exc

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

        rows: list[list[Any]] = []
        for signal in signals:
            signal_type = getattr(signal.signal_type, "value", signal.signal_type)
            direction = getattr(signal.direction, "value", signal.direction)
            rows.append(
                [
                    signal.ts,
                    signal.recv_ts,
                    signal.symbol,
                    signal_type,
                    signal.value,
                    signal.confidence,
                    direction,
                    signal.price,
                    signal.spread_bps,
                    signal.bid_depth,
                    signal.ask_depth,
                    json.dumps(signal.metadata),
                ]
            )

        copy_sql = """
            COPY signals (
                ts, recv_ts, symbol, signal_type, value, confidence, direction,
                price, spread_bps, bid_depth, ask_depth, metadata
            )
            FROM STDIN
        """
        insert_sql = """
            INSERT INTO signals (
                ts, recv_ts, symbol, signal_type, value, confidence, direction,
                price, spread_bps, bid_depth, ask_depth, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        self._copy_with_fallback(copy_sql, rows, insert_sql)

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

        rows: list[list[Any]] = []
        for regime in regimes:
            regime_value = getattr(regime.regime, "value", regime.regime)
            rows.append(
                [
                    regime.ts,
                    regime.symbol,
                    regime_value,
                    regime.atr,
                    regime.spread_bps,
                    regime.funding_rate,
                    regime.should_trade,
                ]
            )

        copy_sql = """
            COPY regime_log (
                ts, symbol, regime, atr, spread_bps, funding_rate, should_trade
            )
            FROM STDIN
        """
        insert_sql = """
            INSERT INTO regime_log (
                ts, symbol, regime, atr, spread_bps, funding_rate, should_trade
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._copy_with_fallback(copy_sql, rows, insert_sql)

    def query_regimes(
        self,
        *,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketRegime]:
        """Fetch regime classifications for backtesting or monitoring."""
        sql = """
            SELECT ts, symbol, regime, atr, spread_bps, funding_rate, should_trade
            FROM regime_log
            WHERE symbol = %(symbol)s
              AND ts BETWEEN %(start_ts)s AND %(end_ts)s
              AND atr IS NOT NULL
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

    def write_trades_batch(self, trades: Sequence[Trade]) -> None:
        """Bulk insert processed trades."""
        if not trades:
            return

        rows: list[list[Any]] = []
        for trade in trades:
            rows.append(
                [
                    trade.ts,
                    trade.symbol,
                    trade.trade_id,
                    trade.side,
                    trade.price,
                    trade.qty,
                    trade.is_large,
                ]
            )

        copy_sql = """
            COPY trades_processed (ts, symbol, trade_id, side, price, qty, is_large)
            FROM STDIN
        """
        insert_sql = """
            INSERT INTO trades_processed (
                ts, symbol, trade_id, side, price, qty, is_large
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._copy_with_fallback(copy_sql, rows, insert_sql)

    def query_trades(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[Trade]:
        """Fetch trades for backtesting price data."""
        sql = """
            SELECT ts, symbol, trade_id, side, price, qty, is_large
            FROM trades_processed
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

        return [Trade(**row) for row in rows]

    def get_price_map(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> dict[datetime, float]:
        """Build a timestamp -> price mapping from trades for backtesting."""
        trades = self.query_trades(symbol, start_ts, end_ts)
        return {trade.ts: trade.price for trade in trades}


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

    @classmethod
    def for_trades(cls, *, host: str, port: int, user: str, password: str) -> "QuestDBSink":
        def writer(client: QuestDBClient, batch: Sequence[Any]) -> None:
            trades = [item for item in batch if isinstance(item, Trade)]
            if len(trades) != len(batch):
                raise TypeError("QuestDB trade sink received non-Trade payload")
            client.write_trades_batch(trades)

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

from __future__ import annotations

from typing import cast

import bytewax.operators as op
from bytewax.dataflow import Dataflow
from config import Settings
from signals.base import Signal, Trade
from signals.cvd import CVDCalculator
from signals.ofi import OFICalculator
from signals.tfi import TFICalculator
from stream.live_sources import LiveOrderbookStream, LiveTradeStream
from stream.signal_router import SignalRouter


def build_realtime_dataflow(settings: Settings) -> Dataflow:
    """Create a Bytewax dataflow that processes live trades into actionable signals."""

    flow = Dataflow("realtime_signal_processor")

    trades_stream = op.input("live_trades", flow, LiveTradeStream(settings))
    keyed_trades = op.key_on(
        "trades_by_symbol", trades_stream, lambda trade: trade.symbol
    )

    signal_streams = [
        _build_cvd_branch(flow, keyed_trades, settings),
        _build_tfi_branch(flow, keyed_trades, settings),
    ]

    orderbook_stream = op.input(
        "live_orderbook",
        flow,
        LiveOrderbookStream(settings),
    )
    keyed_orderbook = op.key_on(
        "orderbooks_by_symbol",
        orderbook_stream,
        lambda snapshot: snapshot.symbol,
    )
    signal_streams.append(_build_ofi_branch(flow, keyed_orderbook, settings))

    if len(signal_streams) == 1:
        all_signals = signal_streams[0]
    else:
        all_signals = op.merge("merge_signals", *signal_streams)

    op.inspect("route_signals", all_signals, SignalRouter.route_signal)

    try:
        from persistence.async_writer import enqueue_signal
    except ImportError:  # pragma: no cover - defensive fallback
        enqueue_signal = None

    if enqueue_signal is not None:
        op.inspect("log_signals", all_signals, enqueue_signal)

    return flow


def _build_cvd_branch(flow: Dataflow, keyed_trades, settings: Settings):
    def mapper(state: CVDCalculator | None, trade: Trade):
        calculator = state or CVDCalculator(
            symbol=trade.symbol, **settings.cvd_config()
        )
        signal = calculator.process_trade(trade)
        return calculator, signal

    stateful = op.stateful_map("cvd_calculator", keyed_trades, mapper)
    non_null = op.filter("cvd_non_null", stateful, lambda item: item[1] is not None)
    return op.map("cvd_extract", non_null, lambda item: cast(Signal, item[1]))


def _build_tfi_branch(flow: Dataflow, keyed_trades, settings: Settings):
    def mapper(state: TFICalculator | None, trade: Trade):
        calculator = state or TFICalculator(
            symbol=trade.symbol, **settings.tfi_config()
        )
        signal = calculator.process_trade(trade)
        return calculator, signal

    stateful = op.stateful_map("tfi_calculator", keyed_trades, mapper)
    non_null = op.filter("tfi_non_null", stateful, lambda item: item[1] is not None)
    return op.map("tfi_extract", non_null, lambda item: cast(Signal, item[1]))


def _build_ofi_branch(flow: Dataflow, keyed_orderbook, settings: Settings):
    def mapper(state: OFICalculator | None, snapshot):
        calculator = state or OFICalculator(
            symbol=snapshot.symbol, **settings.ofi_config()
        )
        signal = calculator.process_snapshot(snapshot)
        return calculator, signal

    stateful = op.stateful_map("ofi_calculator", keyed_orderbook, mapper)
    non_null = op.filter("ofi_non_null", stateful, lambda item: item[1] is not None)
    return op.map("ofi_extract", non_null, lambda item: cast(Signal, item[1]))

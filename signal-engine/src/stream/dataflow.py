from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import bytewax.operators as op
from bytewax.dataflow import Dataflow, Stream
from bytewax.inputs import Source
from bytewax.outputs import DynamicSink

from regime.detectors import ATRRegimeDetector
from signals.base import MarketRegime, OrderbookSnapshot, Signal, Trade
from signals.cvd import CVDCalculator
from signals.ofi import OFICalculator
from signals.tfi import TFICalculator


@dataclass
class CandleEvent:
    symbol: str
    ts: datetime
    high: float
    low: float
    close: float


@dataclass
class OrderbookContextEvent:
    symbol: str
    ts: datetime
    spread_bps: int
    bid_depth: float
    ask_depth: float


@dataclass
class FundingRateEvent:
    symbol: str
    ts: datetime
    rate: float


_RegimeEvent = Union[CandleEvent, OrderbookContextEvent, FundingRateEvent]


@dataclass
class _MinuteState:
    minute: datetime
    high: float
    low: float
    close: float


@dataclass
class _RegimeState:
    detector: ATRRegimeDetector


def build_signal_dataflow(
    *,
    trades_source: Source[Trade],
    signal_sink: DynamicSink[Signal],
    orderbook_source: Source[OrderbookSnapshot] | None = None,
    regime_sink: DynamicSink[MarketRegime] | None = None,
    funding_source: Source[FundingRateEvent] | None = None,
    trade_sink: DynamicSink[Trade] | None = None,
    cvd_config: Optional[Dict[str, Any]] = None,
    tfi_config: Optional[Dict[str, Any]] = None,
    ofi_config: Optional[Dict[str, Any]] = None,
    atr_config: Optional[Dict[str, Any]] = None,
) -> Dataflow:
    """Construct the Bytewax dataflow for real-time signal computation."""

    cvd_config = cvd_config or {}
    tfi_config = tfi_config or {}
    ofi_config = ofi_config or {}
    atr_config = atr_config or {}

    flow = Dataflow("signal_processor")

    trades_stream = op.input("trades", flow, trades_source)

    if trade_sink is not None:
        op.output("write_trades", trades_stream, trade_sink)

    keyed_trades = op.key_on("trades_by_symbol", trades_stream, lambda trade: trade.symbol)

    signal_streams: List[Stream[Signal]] = []

    # --- CVD ---
    def cvd_mapper(
        state: Optional[CVDCalculator], trade: Trade
    ) -> tuple[CVDCalculator, Optional[Signal]]:
        calculator = state or CVDCalculator(symbol=trade.symbol, **cvd_config)
        signal = calculator.process_trade(trade)
        return calculator, signal

    cvd_stateful = op.stateful_map("cvd_calculator", keyed_trades, cvd_mapper)
    cvd_signals = op.filter(
        "cvd_non_null", cvd_stateful, lambda item: item[1] is not None
    )
    cvd_signals = op.map("cvd_drop_key", cvd_signals, lambda item: cast(Signal, item[1]))
    signal_streams.append(cvd_signals)

    # --- TFI ---
    def tfi_mapper(
        state: Optional[TFICalculator], trade: Trade
    ) -> tuple[TFICalculator, Optional[Signal]]:
        calculator = state or TFICalculator(symbol=trade.symbol, **tfi_config)
        signal = calculator.process_trade(trade)
        return calculator, signal

    tfi_stateful = op.stateful_map("tfi_calculator", keyed_trades, tfi_mapper)
    tfi_signals = op.filter(
        "tfi_non_null", tfi_stateful, lambda item: item[1] is not None
    )
    tfi_signals = op.map("tfi_drop_key", tfi_signals, lambda item: cast(Signal, item[1]))
    signal_streams.append(tfi_signals)

    # --- Minute candles for regime detection ---
    def minute_mapper(
        state: Optional[_MinuteState], trade: Trade
    ) -> tuple[_MinuteState, Optional[CandleEvent]]:
        minute_bucket = trade.ts.replace(second=0, microsecond=0)
        if state is None:
            state = _MinuteState(
                minute=minute_bucket, high=trade.price, low=trade.price, close=trade.price
            )
            return state, None

        if minute_bucket == state.minute:
            state.high = max(state.high, trade.price)
            state.low = min(state.low, trade.price)
            state.close = trade.price
            return state, None

        candle = CandleEvent(
            symbol=trade.symbol,
            ts=state.minute,
            high=state.high,
            low=state.low,
            close=state.close,
        )

        new_state = _MinuteState(
            minute=minute_bucket, high=trade.price, low=trade.price, close=trade.price
        )
        return new_state, candle

    candle_stateful = op.stateful_map("minute_candles", keyed_trades, minute_mapper)
    candle_events = op.filter(
        "candle_non_null", candle_stateful, lambda item: item[1] is not None
    )
    candle_events = op.map(
        "candle_drop_key", candle_events, lambda item: cast(CandleEvent, item[1])
    )

    regime_event_streams: List[Stream[_RegimeEvent]] = [candle_events]

    # --- Orderbook branch ---
    if orderbook_source is not None:
        orderbook_stream = op.input("orderbooks", flow, orderbook_source)
        keyed_orderbooks = op.key_on(
            "orderbooks_by_symbol", orderbook_stream, lambda ob: ob.symbol
        )

        def ofi_mapper(
            state: Optional[OFICalculator], snapshot: OrderbookSnapshot
        ) -> tuple[OFICalculator, Optional[Signal]]:
            calculator = state or OFICalculator(symbol=snapshot.symbol, **ofi_config)
            signal = calculator.process_snapshot(snapshot)
            return calculator, signal

        ofi_stateful = op.stateful_map("ofi_calculator", keyed_orderbooks, ofi_mapper)
        ofi_signals = op.filter(
            "ofi_non_null", ofi_stateful, lambda item: item[1] is not None
        )
        ofi_signals = op.map(
            "ofi_drop_key", ofi_signals, lambda item: cast(Signal, item[1])
        )
        signal_streams.append(ofi_signals)

        orderbook_ctx_events = op.map(
            "orderbook_ctx_event",
            orderbook_stream,
            _snapshot_to_context_event,
        )
        regime_event_streams.append(orderbook_ctx_events)

    # --- Funding branch ---
    if funding_source is not None:
        funding_stream = op.input("funding_rates", flow, funding_source)
        regime_event_streams.append(funding_stream)

    # --- Merge and detect regimes ---
    if regime_event_streams:
        if len(regime_event_streams) == 1:
            regime_events = regime_event_streams[0]
        else:
            regime_events = op.merge("merge_regime_events", *regime_event_streams)

        keyed_regime_events = op.key_on(
            "regime_events_by_symbol", regime_events, lambda event: event.symbol
        )

        def regime_mapper(
            state: Optional[_RegimeState], event: _RegimeEvent
        ) -> tuple[_RegimeState, Optional[MarketRegime]]:
            detector = state.detector if state else None
            if detector is None:
                detector = ATRRegimeDetector(symbol=event.symbol, **atr_config)
                state = _RegimeState(detector=detector)
            else:
                state = state

            if isinstance(event, CandleEvent):
                detector.update_price(event.ts, event.high, event.low, event.close)
                regime = detector.detect_regime(event.ts)
                return state, regime

            if isinstance(event, OrderbookContextEvent):
                detector.update_orderbook_context(
                    event.spread_bps, event.bid_depth, event.ask_depth
                )
                return state, None

            if isinstance(event, FundingRateEvent):
                detector.update_funding_rate(event.rate)
                return state, None

            return state, None

        regime_stateful = op.stateful_map("regime_detector", keyed_regime_events, regime_mapper)
        regime_updates = op.filter(
            "regime_non_null", regime_stateful, lambda item: item[1] is not None
        )
        regime_updates = op.map(
            "regime_drop_key", regime_updates, lambda item: cast(MarketRegime, item[1])
        )

        if regime_sink is not None:
            op.output("write_regimes", regime_updates, regime_sink)

    # --- Combine and output signals ---
    if not signal_streams:
        raise ValueError("At least one signal calculator must be configured.")

    if len(signal_streams) == 1:
        all_signals = signal_streams[0]
    else:
        all_signals = op.merge("merge_signals", *signal_streams)

    op.output("write_signals", all_signals, signal_sink)

    return flow


def _snapshot_to_context_event(snapshot: OrderbookSnapshot) -> OrderbookContextEvent:
    bid_depth = sum(qty for _, qty in snapshot.bids)
    ask_depth = sum(qty for _, qty in snapshot.asks)
    return OrderbookContextEvent(
        symbol=snapshot.symbol,
        ts=snapshot.ts,
        spread_bps=snapshot.spread_bps,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
    )

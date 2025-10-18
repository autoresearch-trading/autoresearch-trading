from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bytewax.testing import TestingSink, TestingSource, run_main
from signals.base import Trade
from stream.dataflow import build_signal_dataflow


def _trade(ts: datetime, side: str, price: float, qty: float) -> Trade:
    return Trade(
        ts=ts,
        recv_ts=ts,
        symbol="BTC",
        trade_id=f"{side}-{int(ts.timestamp())}",
        side=side,
        price=price,
        qty=qty,
    )


def test_dataflow_emits_tfi_signal():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    trades = [
        _trade(start + timedelta(seconds=0), "buy", 100.0, 2.0),
        _trade(start + timedelta(seconds=5), "buy", 100.2, 2.5),
        _trade(start + timedelta(seconds=10), "sell", 100.1, 0.3),
    ]

    trade_source = TestingSource(trades)
    signals_out: list = []
    signal_sink = TestingSink(signals_out)

    flow = build_signal_dataflow(
        trades_source=trade_source,
        signal_sink=signal_sink,
        cvd_config={"lookback_periods": 2, "divergence_threshold": 0.05},
        tfi_config={"window_seconds": 30, "signal_threshold": 0.1},
    )

    run_main(flow)

    assert signals_out, "Expected at least one signal from the pipeline"
    assert any(signal.signal_type == "tfi" for signal in signals_out)
    assert all(signal.symbol == "BTC" for signal in signals_out)

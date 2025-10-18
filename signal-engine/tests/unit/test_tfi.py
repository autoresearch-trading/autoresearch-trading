from __future__ import annotations

from datetime import datetime, timezone

from signals.base import SignalDirection, SignalType
from signals.tfi import TFICalculator

from tests.fixtures.sample_data import generate_trades


def test_tfi_no_signal_when_balanced_flow():
    calc = TFICalculator(symbol="BTC", window_seconds=60, signal_threshold=0.9)
    trades = generate_trades(
        prices=[100.0, 100.1, 100.2, 100.3],
        sides=["buy", "sell", "buy", "sell"],
        qtys=[1.0, 1.0, 2.0, 2.0],
    )

    outputs = [calc.process_trade(trade) for trade in trades]
    assert outputs[-1] is None


def test_tfi_emits_bullish_signal_on_strong_imbalance():
    calc = TFICalculator(symbol="BTC", window_seconds=60, signal_threshold=0.3)
    trades = generate_trades(
        prices=[100.0, 100.1, 100.2],
        sides=["buy", "buy", "sell"],
        qtys=[2.0, 2.0, 1.0],
    )

    signal = None
    for trade in trades:
        signal = calc.process_trade(trade)

    assert signal is not None
    assert signal.signal_type == SignalType.TFI
    assert signal.direction == SignalDirection.BULLISH
    assert signal.confidence == 1.0  # imbalance is 0.5 scaled to 1.0 cap
    assert signal.metadata["trade_count"] == len(trades)


def test_tfi_emits_bearish_signal_on_sell_pressure():
    calc = TFICalculator(symbol="BTC", window_seconds=60, signal_threshold=0.3)
    trades = generate_trades(
        prices=[100.0, 99.9, 99.8],
        sides=["sell", "sell", "buy"],
        qtys=[3.0, 2.0, 1.0],
    )

    signal = None
    for trade in trades:
        signal = calc.process_trade(trade)

    assert signal is not None
    assert signal.direction == SignalDirection.BEARISH

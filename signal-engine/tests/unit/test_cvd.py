from __future__ import annotations

from datetime import datetime, timezone

import pytest
from signals.base import SignalDirection, SignalType
from signals.cvd import CVDCalculator

from tests.fixtures.sample_data import generate_trades


def test_cvd_requires_full_lookback():
    calc = CVDCalculator(symbol="BTC", lookback_periods=4, divergence_threshold=0.1)
    trades = generate_trades(
        prices=[100.0, 100.5, 101.0],
        sides=["buy", "sell", "buy"],
        qtys=[1.0, 0.5, 1.0],
    )

    outputs = [calc.process_trade(trade) for trade in trades]
    assert all(
        signal is None for signal in outputs
    ), "Signal emitted before lookback filled"


def test_cvd_detects_bullish_divergence():
    calc = CVDCalculator(symbol="BTC", lookback_periods=4, divergence_threshold=0.1)
    trades = generate_trades(
        prices=[100.0, 99.0, 98.0, 97.0],
        sides=["sell", "sell", "buy", "buy"],
        qtys=[1.0, 1.0, 1.0, 1.0],
    )

    signal = None
    for trade in trades:
        signal = calc.process_trade(trade)

    assert signal is not None, "Expected bullish divergence signal"
    assert signal.signal_type == SignalType.CVD
    assert signal.direction == SignalDirection.BULLISH
    assert signal.confidence > 0.0
    assert signal.metadata["volume_delta"] == pytest.approx(1.0)


def test_cvd_detects_bearish_divergence():
    calc = CVDCalculator(symbol="BTC", lookback_periods=4, divergence_threshold=0.1)
    trades = generate_trades(
        prices=[100.0, 101.0, 102.5, 103.0],
        sides=["buy", "buy", "sell", "sell"],
        qtys=[1.0, 1.0, 0.5, 0.5],
    )

    signal = None
    for trade in trades:
        signal = calc.process_trade(trade)

    assert signal is not None, "Expected bearish divergence signal"
    assert signal.signal_type == SignalType.CVD
    assert signal.direction == SignalDirection.BEARISH
    assert signal.confidence > 0.0

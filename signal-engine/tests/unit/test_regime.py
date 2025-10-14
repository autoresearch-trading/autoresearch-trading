from __future__ import annotations

from datetime import datetime, timedelta, timezone

from regime.detectors import ATRRegimeDetector
from signals.base import RegimeState


def _seed_prices(detector: ATRRegimeDetector, start: datetime, count: int = 5) -> None:
    ts = start
    for i in range(count):
        high = 100 + i * 0.5
        low = 99 + i * 0.5
        close = 99.5 + i * 0.5
        detector.update_price(ts, high=high, low=low, close=close)
        ts += timedelta(minutes=1)


def test_regime_requires_history_before_trading():
    detector = ATRRegimeDetector(symbol="BTC", atr_period=5)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    detector.update_orderbook_context(spread_bps=5, bid_depth=20, ask_depth=20)
    detector.update_funding_rate(0.0001)

    regime = detector.detect_regime(ts=now)
    assert regime.regime == RegimeState.HIGH_VOL
    assert not regime.should_trade


def test_regime_identifies_low_vol_trending():
    detector = ATRRegimeDetector(
        symbol="BTC",
        atr_period=5,
        atr_threshold_multiplier=2.0,
        spread_threshold_bps=20,
        min_depth_threshold=5.0,
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    _seed_prices(detector, start)

    detector.update_orderbook_context(spread_bps=10, bid_depth=20, ask_depth=20)
    detector.update_funding_rate(0.0001)

    regime = detector.detect_regime(ts=start + timedelta(minutes=5))
    assert regime.regime == RegimeState.LOW_VOL_TRENDING
    assert regime.should_trade


def test_regime_flags_low_liquidity():
    detector = ATRRegimeDetector(
        symbol="BTC",
        atr_period=5,
        min_depth_threshold=25.0,
        spread_threshold_bps=20,
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    _seed_prices(detector, start)

    detector.update_orderbook_context(spread_bps=10, bid_depth=10, ask_depth=10)
    detector.update_funding_rate(0.0)

    regime = detector.detect_regime(ts=start + timedelta(minutes=5))
    assert regime.regime == RegimeState.LOW_LIQUIDITY
    assert not regime.should_trade


def test_regime_flags_extreme_funding():
    detector = ATRRegimeDetector(symbol="BTC", atr_period=5, extreme_funding_threshold=0.0005)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    _seed_prices(detector, start)
    detector.update_orderbook_context(spread_bps=5, bid_depth=20, ask_depth=20)
    detector.update_funding_rate(0.001)

    regime = detector.detect_regime(ts=start + timedelta(minutes=5))
    assert regime.regime == RegimeState.RISK_OFF
    assert not regime.should_trade

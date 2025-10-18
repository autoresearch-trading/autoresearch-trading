from __future__ import annotations

from datetime import datetime, timezone

from paper_trading.position_tracker import PositionTracker
from paper_trading.risk_manager import RiskManager
from paper_trading.trade_executor import TradeExecutor
from signals.base import Signal, SignalDirection, SignalType


class DummyDB:
    def __init__(self) -> None:
        self.trades = []

    def write_paper_trade(self, trade) -> None:
        self.trades.append(trade)


def _make_signal(signal_type: SignalType, *, direction: SignalDirection) -> Signal:
    now = datetime.now(timezone.utc)
    return Signal(
        ts=now,
        recv_ts=now,
        symbol="BTC",
        signal_type=signal_type,
        value=1.0,
        confidence=0.9,
        direction=direction,
        price=100.0,
        spread_bps=10,
        bid_depth=100.0,
        ask_depth=100.0,
        metadata={},
    )


def test_trade_executor_opens_and_closes_position() -> None:
    tracker = PositionTracker(initial_capital=10_000.0)
    risk_manager = RiskManager(
        {
            "initial_capital": 10_000.0,
            "max_daily_loss_pct": 0.10,
            "max_daily_trades": 100,
            "max_consecutive_losses": 10,
            "max_position_size_pct": 0.20,
            "max_total_exposure_pct": 1.0,
            "max_concentration_pct": 1.0,
        }
    )
    db = DummyDB()
    executor = TradeExecutor(
        position_tracker=tracker,
        risk_manager=risk_manager,
        db_client=db,
        config={
            "min_confidence": 0.5,
            "min_signals_agree": 2,
            "require_cvd": True,
            "require_tfi": True,
            "require_ofi": False,
            "position_size_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.03,
            "max_hold_seconds": 300,
        },
    )

    signals = [
        _make_signal(SignalType.CVD, direction=SignalDirection.BULLISH),
        _make_signal(SignalType.TFI, direction=SignalDirection.BULLISH),
    ]

    executor.evaluate_entry(
        symbol="BTC",
        signals=signals,
        current_price=100.0,
        regime=None,
        now=datetime.now(timezone.utc),
    )

    position = tracker.get_position("BTC")
    assert position is not None
    assert position.qty > 0

    executor.close_position(
        position=position, exit_price=110.0, exit_reason="take_profit"
    )

    assert tracker.get_position("BTC") is None
    assert risk_manager.metrics.daily_trade_count == 1
    assert len(db.trades) == 1
    trade = db.trades[0]
    assert trade.pnl and trade.pnl > 0

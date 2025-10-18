from __future__ import annotations

from datetime import datetime, timezone

from paper_trading.models import PaperPosition
from paper_trading.risk_manager import RiskManager
from signals.base import PaperTrade


def make_position(notional: float, symbol: str = "BTC") -> PaperPosition:
    qty = notional / 100.0
    now = datetime.now(timezone.utc)
    return PaperPosition(
        position_id="test",
        symbol=symbol,
        side="long",
        entry_time=now,
        entry_price=100.0,
        qty=qty,
        stop_loss=95.0,
        take_profit=105.0,
        cvd_value=0.0,
        tfi_value=0.0,
        ofi_value=None,
        regime="trend",
        current_price=100.0,
        unrealized_pnl=0.0,
        last_updated=now,
    )


def test_can_open_position_blocks_daily_loss() -> None:
    config = {
        "initial_capital": 10_000.0,
        "max_daily_loss_pct": 0.05,
        "max_daily_trades": 100,
        "max_consecutive_losses": 5,
        "max_position_size_pct": 0.20,
        "max_total_exposure_pct": 1.0,
        "max_concentration_pct": 1.0,
    }
    manager = RiskManager(config)
    manager.metrics.daily_pnl = -600.0

    allowed, reason = manager.can_open_position(
        symbol="BTC",
        side="long",
        qty=1.0,
        price=100.0,
        current_capital=9_400.0,
        existing_positions=[],
    )

    assert allowed is False
    assert reason == "daily_loss_limit_hit"


def test_can_open_position_limits_exposure() -> None:
    config = {
        "initial_capital": 10_000.0,
        "max_total_exposure_pct": 0.50,
        "max_concentration_pct": 0.30,
        "max_position_size_pct": 0.50,
    }
    manager = RiskManager(config)

    existing = [
        make_position(notional=2_000.0, symbol="BTC"),
        make_position(notional=1_600.0, symbol="ETH"),
    ]

    # Try to open SOL position (different symbol) - should hit exposure limit
    allowed, reason = manager.can_open_position(
        symbol="SOL",
        side="long",
        qty=10.0,
        price=100.0,
        current_capital=9_000.0,
        existing_positions=existing,
    )

    assert allowed is False
    assert reason in {"total_exposure_limit_hit", "concentration_limit_hit"}


def test_can_open_position_blocks_duplicate_symbol() -> None:
    config = {"initial_capital": 10_000.0}
    manager = RiskManager(config)

    existing = [make_position(notional=1_000.0, symbol="BTC")]

    # Try to open another BTC position - should be rejected
    allowed, reason = manager.can_open_position(
        symbol="BTC",
        side="long",
        qty=1.0,
        price=100.0,
        current_capital=9_000.0,
        existing_positions=existing,
    )

    assert allowed is False
    assert reason == "position_already_open"


def test_record_trade_updates_loss_streak() -> None:
    config = {"initial_capital": 5_000.0}
    manager = RiskManager(config)

    losing_trade = PaperTrade(
        ts=datetime.now(timezone.utc),
        symbol="BTC",
        trade_id="t1",
        side="long",
        entry_price=100.0,
        qty=1.0,
        stop_loss=95.0,
        take_profit=110.0,
        exit_price=95.0,
        exit_ts=datetime.now(timezone.utc),
        pnl=-5.0,
        pnl_pct=-0.05,
        cvd_value=0.0,
        tfi_value=0.0,
        ofi_value=None,
        regime="trend",
    )

    manager.record_trade(losing_trade, current_capital=4_900.0)
    assert manager.metrics.consecutive_losses == 1

    winning_trade = losing_trade.model_copy(
        update={"exit_price": 110.0, "pnl": 10.0, "pnl_pct": 0.10}
    )
    manager.record_trade(winning_trade, current_capital=5_100.0)
    assert manager.metrics.consecutive_losses == 0


def test_reset_daily_limits() -> None:
    manager = RiskManager({"initial_capital": 1_000.0})
    manager.metrics.daily_pnl = -100.0
    manager.metrics.daily_trade_count = 3
    manager.metrics.last_reset = datetime(2024, 5, 1, tzinfo=timezone.utc)

    now = datetime(2024, 5, 2, tzinfo=timezone.utc)
    assert manager.should_reset_daily(now) is True

    manager.reset_daily_limits()
    assert manager.metrics.daily_pnl == 0.0
    assert manager.metrics.daily_trade_count == 0

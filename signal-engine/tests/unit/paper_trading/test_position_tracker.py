from __future__ import annotations

from paper_trading.position_tracker import PositionTracker


def test_open_and_close_position_updates_capital() -> None:
    tracker = PositionTracker(initial_capital=10_000.0)
    position = tracker.open_position(
        symbol="BTC",
        side="long",
        entry_price=100.0,
        qty=10.0,
        stop_loss=95.0,
        take_profit=110.0,
        cvd_value=0.0,
        tfi_value=0.0,
        ofi_value=None,
        regime="trend",
    )

    assert tracker.capital == 9_000.0
    assert tracker.get_position("BTC") is not None

    tracker.close_position("BTC", exit_price=110.0)
    assert tracker.capital == 10_100.0
    assert tracker.get_position("BTC") is None


def test_update_position_price_tracks_unrealized_pnl() -> None:
    tracker = PositionTracker(initial_capital=1_000.0)
    position = tracker.open_position(
        symbol="ETH",
        side="short",
        entry_price=200.0,
        qty=2.0,
        stop_loss=210.0,
        take_profit=180.0,
        cvd_value=0.0,
        tfi_value=0.0,
        ofi_value=None,
        regime="range",
    )

    tracker.update_position_price("ETH", current_price=190.0)
    updated = tracker.get_position("ETH")
    assert updated is not None
    assert updated.unrealized_pnl == 20.0
    assert tracker.get_equity() == tracker.capital + 20.0

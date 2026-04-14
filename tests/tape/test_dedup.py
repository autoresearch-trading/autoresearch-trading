import pandas as pd
import pytest

from tape.dedup import dedup_trades_pre_april, filter_trades_april


def test_pre_april_dedup_drops_buyer_seller_pair_by_ts_qty_price_only():
    # Gotcha #19: pre-April, two rows share (ts, qty, price) but differ on side.
    # Dedup by (ts, qty, price) → one row.
    df = pd.DataFrame(
        {
            "ts_ms": [1000, 1000],
            "symbol": ["BTC", "BTC"],
            "trade_id": [1, 2],
            "side": ["open_long", "close_short"],
            "qty": [0.5, 0.5],
            "price": [50000.0, 50000.0],
        }
    )
    out = dedup_trades_pre_april(df)
    assert len(out) == 1
    # Gotcha #3: `side` must NOT be in the dedup key
    assert "side" in out.columns  # preserved in output, just not in key


def test_pre_april_dedup_preserves_genuinely_distinct_fills():
    df = pd.DataFrame(
        {
            "ts_ms": [1000, 1000],
            "symbol": ["BTC", "BTC"],
            "trade_id": [1, 2],
            "side": ["open_long", "open_long"],
            "qty": [0.5, 0.3],  # different qty → different fills
            "price": [50000.0, 50000.0],
        }
    )
    out = dedup_trades_pre_april(df)
    assert len(out) == 2


def test_april_filter_keeps_only_fulfill_taker():
    # Gotcha #3: April+ uses event_type == 'fulfill_taker'
    df = pd.DataFrame(
        {
            "ts_ms": [1, 2, 3],
            "event_type": ["fulfill_taker", "fulfill_maker", "fulfill_taker"],
            "qty": [0.1, 0.1, 0.1],
            "price": [1.0, 1.0, 1.0],
            "side": ["open_long", "close_short", "open_long"],
            "symbol": ["BTC"] * 3,
            "trade_id": [1, 2, 3],
        }
    )
    out = filter_trades_april(df)
    assert len(out) == 2
    assert (out["event_type"] == "fulfill_taker").all()


def test_april_filter_raises_if_event_type_missing():
    df = pd.DataFrame({"ts_ms": [1], "qty": [0.1], "price": [1.0]})
    with pytest.raises(ValueError, match="event_type"):
        filter_trades_april(df)

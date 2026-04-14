import numpy as np
import pandas as pd

from tape.events import group_to_events


def test_same_ts_trades_become_one_event():
    df = pd.DataFrame(
        {
            "ts_ms": [1000, 1000, 2000],
            "qty": [0.5, 0.3, 0.4],
            "price": [100.0, 101.0, 102.0],
            "side": ["open_long", "open_long", "open_short"],
        }
    )
    ev = group_to_events(df)
    assert len(ev) == 2
    # First event: 2 fills, total_qty 0.8, vwap = (0.5*100 + 0.3*101)/0.8
    ev0 = ev.iloc[0]
    assert ev0["n_fills"] == 2
    assert np.isclose(ev0["total_qty"], 0.8)
    assert np.isclose(ev0["vwap"], (0.5 * 100.0 + 0.3 * 101.0) / 0.8)
    assert ev0["is_open_frac"] == 1.0  # both opens
    assert np.isclose(ev0["book_walk_abs"], 1.0)  # |101-100|
    # Second event: 1 fill, vwap=102, is_open_frac=1.0 (open_short is an open)
    ev1 = ev.iloc[1]
    assert ev1["is_open_frac"] == 1.0
    assert ev1["n_fills"] == 1


def test_is_open_frac_mixed_side():
    df = pd.DataFrame(
        {
            "ts_ms": [1, 1, 1, 1],
            "qty": [0.25, 0.25, 0.25, 0.25],
            "price": [10.0, 10.0, 10.0, 10.0],
            "side": ["open_long", "close_long", "open_short", "close_short"],
        }
    )
    ev = group_to_events(df)
    # is_open_frac = fraction of fills that are opens (open_long or open_short)
    # Here 2 of 4 are opens → 0.5
    assert len(ev) == 1
    assert np.isclose(ev["is_open_frac"].iloc[0], 0.5)

from __future__ import annotations

from collector.transform import (to_candle_rows, to_funding_rows,
                                 to_orderbook_rows, to_price_rows,
                                 to_trade_rows)


def test_to_price_rows_filters_symbols() -> None:
    payload = {
        "data": {
            "BTC": {"price": "50000", "ts_ms": 1710000000000},
            "ETH": {"price": "3500"},
        }
    }
    rows = to_price_rows(payload, recv_ms=1710000005000, filter_symbols={"BTC"})
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC"
    assert row["price"] == 50000.0
    assert row["ts_ms"] == 1710000000000


def test_to_trade_rows_normalizes_payload() -> None:
    payload = {
        "data": [
            {
                "symbol": "BTC",
                "ts_ms": 1710001000000,
                "id": "1",
                "side": "buy",
                "qty": "0.5",
                "price": "45000",
            }
        ]
    }
    rows = to_trade_rows(payload, recv_ms=1710001005000)
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == "1"
    assert row["qty"] == 0.5
    assert row["side"] == "buy"


def test_to_orderbook_rows_truncates_depth() -> None:
    payload = {
        "data": {
            "ts_ms": 1710002000000,
            "bids": [["45000", "1.2"], ["44900", "0.8"]],
            "asks": [["45100", "1.0"], ["45200", "0.7"]],
        }
    }
    rows = to_orderbook_rows(
        payload, symbol="BTC", recv_ms=1710002001000, depth=1, agg_level=None
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC"
    assert len(row["bids"]) == 1
    assert len(row["asks"]) == 1
    assert row["bids"][0]["price"] == 45000.0


def test_to_orderbook_rows_handles_pacifica_shape() -> None:
    payload = {
        "data": {
            "s": "BTC",
            "l": [
                [
                    {"p": "121921", "a": "0.46321", "n": 2},
                    {"p": "121920", "a": "3.17697", "n": 2},
                ],
                [
                    {"p": "121923", "a": "0.03322", "n": 1},
                    {"p": "121924", "a": "0.001", "n": 1},
                ],
            ],
            "t": 1759877555458,
        }
    }
    rows = to_orderbook_rows(
        payload, symbol="BTC", recv_ms=1759877556000, depth=1, agg_level=None
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC"
    assert row["ts_ms"] == 1759877555458
    assert len(row["bids"]) == 1
    assert len(row["asks"]) == 1
    assert row["bids"][0]["price"] == 121921.0
    assert row["bids"][0]["qty"] == 0.46321


def test_to_funding_rows_casts_fields() -> None:
    payload = {
        "data": [
            {
                "symbol": "BTC",
                "timestamp": 1710003000000,
                "rate": "0.0001",
                "interval_sec": 28800,
            },
            {"symbol": "ETH", "ts_ms": 1710003005000, "rate": 0.0002, "interval": 3600},
        ]
    }
    rows = to_funding_rows(payload, recv_ms=1710003009000)
    assert len(rows) == 2
    assert rows[0]["rate"] == 0.0001
    assert rows[1]["interval_sec"] == 3600


def test_to_candle_rows_handles_dict_payload() -> None:
    payload = {
        "data": [
            {
                "symbol": "btc",
                "start_time": 1710000000000,
                "end_time": 1710000060000,
                "open": "100",
                "high": "110",
                "low": "95",
                "close": 105,
                "volume": "12.5",
            }
        ]
    }
    rows = to_candle_rows(
        payload, recv_ms=1710000065000, symbol="BTC", interval="1m", interval_ms=60_000
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC"
    assert row["interval"] == "1m"
    assert row["open"] == 100.0
    assert row["end_ms"] == 1710000060000


def test_to_candle_rows_handles_list_payload() -> None:
    payload = {
        "data": [[1710000000000, "100", "110", "95", "105", "12.5", 1710000060000]]
    }
    rows = to_candle_rows(
        payload, recv_ms=1710000065000, symbol="ETH", interval="1m", interval_ms=60_000
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "ETH"
    assert row["volume"] == 12.5


def test_to_candle_rows_accepts_zero_start_time() -> None:
    payload = {
        "data": [
            {
                "start_time": 0,
                "end_time": 60_000,
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "volume": 1,
            }
        ]
    }
    rows = to_candle_rows(
        payload, recv_ms=100_000, symbol="btc", interval="1m", interval_ms=60_000
    )
    assert rows[0]["ts_ms"] == 0

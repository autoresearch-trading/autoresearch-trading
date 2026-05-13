import json
from pathlib import Path

import pandas as pd

from scripts.pacifica_r2_inventory import (
    iter_rclone_lsf_inventory_rows,
    rclone_lsf_to_inventory,
    rclone_lsjson_to_inventory,
    write_inventory_csv,
    write_inventory_csv_from_lsf,
    write_inventory_csv_from_lsf_stream,
)


def test_rclone_lsjson_to_inventory_keeps_key_size_and_mod_time():
    payload = [
        {
            "Path": "raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=00/run.jsonl.gz",
            "Size": 123,
            "ModTime": "2026-05-01T00:05:00Z",
        },
        {
            "Path": "raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=00/run.jsonl.gz.sha256",
            "Size": 64,
            "ModTime": "2026-05-01T00:06:00Z",
        },
        {"Path": "raw/pacifica/full_fidelity/channel=book", "IsDir": True},
    ]

    inventory = rclone_lsjson_to_inventory(payload)

    assert inventory.to_dict("records") == [
        {
            "key": "raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=00/run.jsonl.gz",
            "size_bytes": 123,
            "mod_time": "2026-05-01T00:05:00Z",
        },
        {
            "key": "raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-01/hour=00/run.jsonl.gz.sha256",
            "size_bytes": 64,
            "mod_time": "2026-05-01T00:06:00Z",
        },
    ]


def test_write_inventory_csv_writes_sorted_csv(tmp_path):
    lsjson = tmp_path / "lsjson.json"
    out = tmp_path / "inventory.csv"
    lsjson.write_text(
        json.dumps(
            [
                {"Path": "b.jsonl.gz", "Size": 2, "ModTime": "2026-05-01T00:00:02Z"},
                {"Path": "a.jsonl.gz", "Size": 1, "ModTime": "2026-05-01T00:00:01Z"},
            ]
        )
    )

    result = write_inventory_csv(lsjson, out)

    assert result == out
    df = pd.read_csv(out)
    assert df["key"].tolist() == ["a.jsonl.gz", "b.jsonl.gz"]


def test_rclone_lsf_to_inventory_parses_line_oriented_listing():
    listing = "b.jsonl.gz;2;2026-05-01 00:00:02\r\na.jsonl.gz;1;2026-05-01 00:00:01\n"

    inventory = rclone_lsf_to_inventory(listing)

    assert inventory.to_dict("records") == [
        {"key": "a.jsonl.gz", "size_bytes": 1, "mod_time": "2026-05-01 00:00:01"},
        {"key": "b.jsonl.gz", "size_bytes": 2, "mod_time": "2026-05-01 00:00:02"},
    ]


def test_write_inventory_csv_from_lsf_writes_lf_csv(tmp_path):
    lsf = tmp_path / "inventory.lsf"
    out = tmp_path / "inventory.csv"
    lsf.write_text(
        "b.jsonl.gz;2;2026-05-01 00:00:02\r\na.jsonl.gz;1;2026-05-01 00:00:01\n"
    )

    result = write_inventory_csv_from_lsf(lsf, out)

    assert result == out
    raw = out.read_bytes()
    assert b"\r" not in raw
    df = pd.read_csv(out)
    assert df["key"].tolist() == ["a.jsonl.gz", "b.jsonl.gz"]


def test_iter_rclone_lsf_inventory_rows_prefixes_keys_line_by_line():
    lines = iter(
        [
            "channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run.jsonl.gz;10;2026-05-13 12:00:00\r\n",
            "channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run.jsonl.gz.sha256;64;2026-05-13 12:00:01\n",
        ]
    )

    rows = list(
        iter_rclone_lsf_inventory_rows(lines, key_prefix="raw/pacifica/full_fidelity/")
    )

    assert rows == [
        {
            "key": "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run.jsonl.gz",
            "size_bytes": 10,
            "mod_time": "2026-05-13 12:00:00",
        },
        {
            "key": "raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run.jsonl.gz.sha256",
            "size_bytes": 64,
            "mod_time": "2026-05-13 12:00:01",
        },
    ]


def test_write_inventory_csv_from_lsf_stream_writes_prefixed_lf_csv(tmp_path):
    lsf = tmp_path / "inventory.lsf"
    out = tmp_path / "inventory.csv"
    lsf.write_text(
        "channel=book/symbol=ETH/date=2026-05-13/hour=11/run.jsonl.gz;20;2026-05-13 11:59:00\r\n"
    )

    result = write_inventory_csv_from_lsf_stream(
        lsf, out, key_prefix="raw/pacifica/full_fidelity"
    )

    assert result == out
    raw = out.read_bytes()
    assert b"\r" not in raw
    df = pd.read_csv(out)
    assert df.to_dict("records") == [
        {
            "key": "raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=11/run.jsonl.gz",
            "size_bytes": 20,
            "mod_time": "2026-05-13 11:59:00",
        }
    ]

import json
from pathlib import Path

import pandas as pd

from scripts.pacifica_r2_inventory import (
    rclone_lsjson_to_inventory,
    write_inventory_csv,
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

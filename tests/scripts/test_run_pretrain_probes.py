# tests/scripts/test_run_pretrain_probes.py
import json
from pathlib import Path

import pytest


def _has_april_shards() -> bool:
    cache = Path("data/cache")
    if not cache.exists():
        return False
    return any(cache.glob("BTC__2026-04-0*.npz"))


@pytest.mark.skipif(
    not _has_april_shards(), reason="April probe shards not materialized"
)
def test_run_pretrain_probes_smoke(tmp_path):
    from scripts.run_pretrain_probes import run_probes

    # Use a freshly-initialized encoder so this works without an actual checkpoint
    out = tmp_path / "probes"
    res = run_probes(
        checkpoint=None,
        cache_dir=Path("data/cache"),
        symbols=["BTC", "ETH"],
        out_path=out,
    )
    assert (out.with_suffix(".json")).exists()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert "direction_h100_per_symbol" in payload
    assert "symbol_identity_acc" in payload
    assert "hour_of_day_acc" in payload

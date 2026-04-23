# tests/scripts/test_run_pretrain.py
import json
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("data/cache").exists(), reason="local cache not materialized"
)
def test_run_pretrain_smoke(tmp_path):
    """End-to-end smoke: 2 epochs, batch=8, channel-mult=0.5, 2 symbols, asserts checkpoint and log written."""
    from scripts.run_pretrain import run_pretrain

    out = tmp_path / "smoke"
    res = run_pretrain(
        cache_dir=Path("data/cache"),
        symbols=["BTC", "ETH"],
        epochs=2,
        batch_size=8,
        channel_mult=0.5,
        out_dir=out,
        max_hours=0.5,
        seed=0,
    )
    ckpt = out / "encoder.pt"
    log = out / "training-log.jsonl"
    assert ckpt.exists()
    assert log.exists()
    # Final-line summary contains MEM + contrastive losses
    last = log.read_text().strip().splitlines()[-1]
    payload = json.loads(last)
    assert payload["epoch"] == 2
    assert "mem_loss" in payload
    assert "contrastive_loss" in payload

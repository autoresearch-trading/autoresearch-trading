# tests/scripts/test_export_checkpoint.py
from pathlib import Path

import torch

from scripts.export_checkpoint import export_for_gate1
from tape.model import EncoderConfig, TapeEncoder


def test_export_strips_optimizer_and_keeps_encoder(tmp_path):
    enc = TapeEncoder(EncoderConfig(channel_mult=1.0))
    src = tmp_path / "raw.pt"
    torch.save(
        {
            "encoder_state_dict": enc.state_dict(),
            "encoder_config": EncoderConfig().__dict__,
            "optimizer_state_dict": {"junk": 1},
            "n_epochs_run": 30,
            "elapsed_seconds": 12345.6,
            "seed": 0,
        },
        src,
    )
    dst = tmp_path / "encoder-final.pt"
    info = export_for_gate1(src, dst, git_sha="abcd1234", spec_sha="ef567890")
    payload = torch.load(dst, map_location="cpu")
    assert "encoder_state_dict" in payload
    assert "encoder_config" in payload
    assert "optimizer_state_dict" not in payload
    assert payload["git_sha"] == "abcd1234"
    assert payload["spec_sha"] == "ef567890"
    assert info["n_params"] > 0

# tests/tape/test_pretrain.py
import pytest
import torch

from tape.model import EncoderConfig
from tape.pretrain import PretrainConfig, pretrain_step


def make_pretrain_modules(cfg):
    from tape.pretrain import build_pretrain_modules

    return build_pretrain_modules(cfg)


def test_pretrain_step_returns_loss_components():
    cfg = PretrainConfig(encoder=EncoderConfig(channel_mult=0.7), total_steps=10)
    # Synthetic batch: 8 windows, 200 events, 17 features
    batch = torch.randn(8, 200, 17)
    metadata = {
        "symbols": ["BTC"] * 8,
        "dates": ["2026-02-01"] * 8,
        "hours": [10] * 8,
        "eligible": [True] * 8,
    }
    enc, mem_dec, proj, opt, sched = make_pretrain_modules(cfg)
    losses = pretrain_step(
        enc,
        mem_dec,
        proj,
        opt,
        sched,
        batch,
        metadata,
        cfg=cfg,
        current_epoch=0,
    )
    assert "mem" in losses
    assert "contrastive" in losses
    assert "total" in losses
    assert "tau" in losses
    assert "mem_weight" in losses
    assert "contrastive_weight" in losses
    assert "embedding_std" in losses
    assert "effective_rank" in losses
    assert torch.isfinite(torch.tensor(losses["mem"]))
    assert torch.isfinite(torch.tensor(losses["contrastive"]))


def test_pretrain_schedules_warm_to_cool():
    """tau anneals 0.5 -> 0.3 by epoch 10; MEM weight 0.90 -> 0.60; contrastive 0.10 -> 0.40."""
    from tape.pretrain import (
        schedule_contrastive_weight,
        schedule_mem_weight,
        schedule_tau,
    )

    assert schedule_tau(epoch=0) == 0.5
    assert schedule_tau(epoch=5) == pytest.approx(0.4, abs=1e-6)
    assert schedule_tau(epoch=10) == pytest.approx(0.3, abs=1e-6)
    assert schedule_tau(epoch=25) == 0.3  # held constant after epoch 10

    assert schedule_mem_weight(epoch=0, total_anneal_epochs=20) == pytest.approx(
        0.90, abs=1e-6
    )
    assert schedule_mem_weight(epoch=20, total_anneal_epochs=20) == pytest.approx(
        0.60, abs=1e-6
    )
    assert schedule_mem_weight(epoch=30, total_anneal_epochs=20) == pytest.approx(
        0.60, abs=1e-6
    )

    assert schedule_contrastive_weight(
        epoch=0, total_anneal_epochs=20
    ) == pytest.approx(0.10, abs=1e-6)
    assert schedule_contrastive_weight(
        epoch=20, total_anneal_epochs=20
    ) == pytest.approx(0.40, abs=1e-6)


def test_embedding_collapse_detector_flags_constant_embeddings():
    from tape.pretrain import detect_embedding_collapse

    z = torch.zeros(64, 256)  # all identical -> std=0
    assert detect_embedding_collapse(z)  # default threshold 0.05

    z = torch.randn(64, 256)  # spread out -> std large
    assert not detect_embedding_collapse(z)


def test_effective_rank_monitor():
    """Effective rank counts singular values above 1% of max."""
    from tape.pretrain import effective_rank

    # Full-rank random matrix: rank close to min(B, D)
    z = torch.randn(64, 256)
    assert effective_rank(z) > 20

    # Collapsed: all embeddings identical -> rank 1
    z = torch.ones(64, 256)
    assert effective_rank(z) == 1


def test_mem_uses_masked_input_not_unmasked():
    """Regression test for the 'encoder sees unmasked input' bug.

    If MEM trains on unmasked input, it learns trivially (decoder copies) and
    MEM loss drops to ~0 in 1 step. Correctly-masked input keeps MEM loss
    meaningful across steps.
    """
    cfg = PretrainConfig(encoder=EncoderConfig(channel_mult=0.7), total_steps=5)
    enc, mem_dec, proj, opt, sched = make_pretrain_modules(cfg)
    batch = torch.randn(8, 200, 17)
    metadata = {
        "symbols": ["BTC"] * 8,
        "dates": ["2026-02-01"] * 8,
        "hours": [10] * 8,
        "eligible": [True] * 8,
    }
    losses_before = pretrain_step(
        enc, mem_dec, proj, opt, sched, batch, metadata, cfg=cfg, current_epoch=0
    )
    # If the encoder correctly sees MASKED input, loss is non-trivial (> 0.01
    # in BN-normalized space on random data).
    assert (
        losses_before["mem"] > 0.01
    ), "MEM loss near zero suggests encoder sees unmasked input (trivial copy)"

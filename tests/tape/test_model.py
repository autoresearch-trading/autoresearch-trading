# tests/tape/test_model.py
import torch

from tape.model import EncoderConfig, MEMDecoder, ProjectionHead, TapeEncoder


def test_encoder_default_shapes_and_param_count():
    cfg = EncoderConfig()  # channel_mult=1.0, default ~400K params
    enc = TapeEncoder(cfg)
    x = torch.randn(4, 200, 17)  # (B, T, C)
    per_pos, global_emb = enc(x)
    assert per_pos.shape == (4, 200, 128)
    assert global_emb.shape == (4, 256)
    n_params = sum(p.numel() for p in enc.parameters())
    # Spec hard-cap is 500K. Default config sits ~400K.
    assert 350_000 <= n_params <= 460_000, n_params


def test_encoder_channel_mult_scales_param_count():
    small = sum(
        p.numel() for p in TapeEncoder(EncoderConfig(channel_mult=0.7)).parameters()
    )
    base = sum(
        p.numel() for p in TapeEncoder(EncoderConfig(channel_mult=1.0)).parameters()
    )
    assert small < base
    assert small <= 250_000


def test_mem_decoder_per_position_output():
    dec = MEMDecoder(per_position_dim=128, n_features=17)
    h = torch.randn(2, 200, 128)
    out = dec(h)
    assert out.shape == (2, 200, 17)


def test_projection_head_l2_normalized():
    head = ProjectionHead(in_dim=256, hidden=256, out=128)
    z = head(torch.randn(8, 256))
    assert z.shape == (8, 128)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_eval_mode_is_deterministic():
    cfg = EncoderConfig()
    enc = TapeEncoder(cfg).eval()
    x = torch.randn(2, 200, 17)
    a, _ = enc(x)
    b, _ = enc(x)
    assert torch.allclose(a, b)

# tests/tape/test_augment.py
import numpy as np
import torch

from tape.augment import (
    AugmentConfig,
    apply_augment_pipeline,
    make_views_from_context,
)
from tape.constants import FEATURE_NAMES


def _ctx(seed: int = 0) -> torch.Tensor:
    """Synthesize a (T_ctx, 17) context window."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((400, 17)).astype(np.float32))


def test_views_have_correct_shape():
    cfg = AugmentConfig()
    ctx = _ctx(0)
    rng = np.random.default_rng(0)
    v1, v2 = make_views_from_context(ctx, center=200, window_len=200, cfg=cfg, rng=rng)
    assert v1.shape == (200, 17)
    assert v2.shape == (200, 17)


def test_jitter_uses_context_not_pad():
    """With ±25 jitter, both views must be drawn entirely from inside ctx; no
    zero-padding should appear at the edges."""
    cfg = AugmentConfig(
        jitter=25,
        gauss_sigma=0.0,
        timing_sigma=0.0,
        dropout_p=0.0,
        time_dilation_range=(1.0, 1.0),
    )
    ctx = _ctx(0) + 100.0  # shift so any zero-fill would be conspicuous
    rng = np.random.default_rng(0)
    v1, v2 = make_views_from_context(ctx, center=200, window_len=200, cfg=cfg, rng=rng)
    # No element should be near zero — original ctx mean is ~100.
    assert (v1.abs() > 50).all()
    assert (v2.abs() > 50).all()


def test_timing_noise_only_perturbs_time_features():
    cfg = AugmentConfig(
        jitter=0,
        gauss_sigma=0.0,
        timing_sigma=0.10,
        dropout_p=0.0,
        time_dilation_range=(1.0, 1.0),
    )
    ctx = _ctx(0)
    rng = np.random.default_rng(42)
    base = ctx[100:300].clone()
    out = apply_augment_pipeline(base.clone(), cfg=cfg, rng=rng)

    time_idx = FEATURE_NAMES.index("time_delta")
    span_idx = FEATURE_NAMES.index("prev_seq_time_span")
    diff = (out - base).abs()

    # Time channels should have non-zero perturbation
    assert diff[:, time_idx].sum() > 0
    assert diff[:, span_idx].sum() > 0
    # All other channels should be unchanged
    other_mask = torch.ones(17, dtype=torch.bool)
    other_mask[time_idx] = False
    other_mask[span_idx] = False
    assert torch.allclose(diff[:, other_mask], torch.zeros_like(diff[:, other_mask]))


def test_gaussian_noise_scales_with_per_channel_std():
    cfg = AugmentConfig(
        jitter=0,
        gauss_sigma=0.02,
        timing_sigma=0.0,
        dropout_p=0.0,
        time_dilation_range=(1.0, 1.0),
    )
    rng = np.random.default_rng(0)
    base = torch.zeros(200, 17)
    base[:, 0] = torch.randn(200) * 5.0  # channel 0 has std 5
    base[:, 1] = torch.randn(200) * 0.1  # channel 1 has std 0.1
    out = apply_augment_pipeline(base.clone(), cfg=cfg, rng=rng)
    diff = (out - base).std(dim=0)
    # Noise on channel 0 should be roughly 50x larger than on channel 1.
    assert diff[0] > 5 * diff[1]


def test_feature_dropout_zeroes_some_positions_per_feature():
    cfg = AugmentConfig(
        jitter=0,
        gauss_sigma=0.0,
        timing_sigma=0.0,
        dropout_p=0.50,
        time_dilation_range=(1.0, 1.0),
    )
    rng = np.random.default_rng(0)
    base = torch.ones(200, 17)
    out = apply_augment_pipeline(base.clone(), cfg=cfg, rng=rng)
    # With p=0.5, ~half of the (position, feature) cells should be zero.
    rate = (out == 0).float().mean().item()
    assert 0.40 <= rate <= 0.60

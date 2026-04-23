# tape/augment.py
"""SimCLR view generation for tape pretraining.

All augmentations are spec §Training §Pretraining Objective compliant. The
strengthened recipe (council round 6, 2026-04-15):
    - Window jitter ±25 events (was ±10) — must come from CONTEXT (no pad).
    - Timing-feature noise σ=0.10 on `time_delta` and `prev_seq_time_span`.
The other defaults are the round-5 baseline.

Augmentations that DESTROY meaning are NOT implemented:
    - Time reversal (breaks causality)
    - Event shuffling (destroys order)
    - Large noise > 0.1 std
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tape.constants import FEATURE_NAMES

_TIME_DELTA_IDX = FEATURE_NAMES.index("time_delta")
_PREV_SEQ_TIME_SPAN_IDX = FEATURE_NAMES.index("prev_seq_time_span")
_TIMING_FEATURE_INDICES = (_TIME_DELTA_IDX, _PREV_SEQ_TIME_SPAN_IDX)


@dataclass(frozen=True)
class AugmentConfig:
    jitter: int = 25  # ±25 events around the center
    gauss_sigma: float = 0.02  # multiplied by per-channel std
    timing_sigma: float = 0.10  # σ on time_delta + prev_seq_time_span
    dropout_p: float = 0.05  # per-(pos,feat) zero-out probability
    time_dilation_range: tuple[float, float] = (0.8, 1.2)


def make_views_from_context(
    context: torch.Tensor,
    *,
    center: int,
    window_len: int,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two augmented views from a wider context window.

    The caller is responsible for slicing a context that is at least
    ``window_len + 2*cfg.jitter`` events wide so jittered windows never need
    zero-padding.

    Parameters
    ----------
    context : torch.Tensor
        (T_ctx, 17) where T_ctx >= window_len + 2*cfg.jitter
    center : int
        Index in `context` corresponding to the canonical window start.
        Valid range: [cfg.jitter, T_ctx - window_len - cfg.jitter].
    window_len : int
        Output window length (200 in production).
    """
    T = context.shape[0]
    lo = max(0, center - cfg.jitter)
    hi = min(center + cfg.jitter, T - window_len)
    if lo > hi:
        raise ValueError(
            f"context too narrow for jitter ±{cfg.jitter}: T={T}, center={center}, "
            f"window_len={window_len}"
        )

    s1 = int(rng.integers(lo, hi + 1))
    s2 = int(rng.integers(lo, hi + 1))  # noqa: E501 (same clamp)
    v1 = context[s1 : s1 + window_len].clone()
    v2 = context[s2 : s2 + window_len].clone()
    return apply_augment_pipeline(v1, cfg=cfg, rng=rng), apply_augment_pipeline(
        v2, cfg=cfg, rng=rng
    )


def apply_augment_pipeline(
    window: torch.Tensor,
    *,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply (in order): timing noise, gaussian noise, time dilation, feature dropout.

    Window is modified in place and returned.
    """
    T, C = window.shape

    # 1. Timing-feature noise (σ=0.10 on the two timing channels)
    if cfg.timing_sigma > 0:
        for idx in _TIMING_FEATURE_INDICES:
            noise = torch.from_numpy(
                rng.normal(0.0, cfg.timing_sigma, size=T).astype(np.float32)
            )
            window[:, idx] = window[:, idx] + noise

    # 2. Per-channel Gaussian noise (sigma * per-channel std)
    if cfg.gauss_sigma > 0:
        per_channel_std = window.std(dim=0, keepdim=True)  # (1, C)
        noise = torch.from_numpy(rng.standard_normal((T, C)).astype(np.float32))
        window = window + noise * (cfg.gauss_sigma * per_channel_std)

    # 3. Time scale dilation: multiply time_delta by a factor in [a, b]
    a, b = cfg.time_dilation_range
    if not (a == 1.0 and b == 1.0):
        factor = float(rng.uniform(a, b))
        window[:, _TIME_DELTA_IDX] = window[:, _TIME_DELTA_IDX] * factor

    # 4. Per-(pos, feat) dropout to zero (BatchNorm post-input absorbs the bias)
    if cfg.dropout_p > 0:
        keep = rng.random((T, C)) >= cfg.dropout_p
        keep_t = torch.from_numpy(keep)
        window = window * keep_t

    return window

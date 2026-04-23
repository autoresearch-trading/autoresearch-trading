# tape/losses.py
"""Pretraining losses: block-mask MEM MSE + NT-Xent contrastive.

MEM loss is computed in BatchNorm-normalized space (gotcha #23) — the caller
applies the encoder's input BN to BOTH the prediction and the raw feature
target before invoking this function, so we receive matched normalized tensors.

NT-Xent supports OPTIONAL soft positive pairs for cross-symbol contrastive
(spec §Training: same-date same-hour windows from BTC/ETH/SOL/BNB/LINK/LTC,
weight 0.5).  AVAX is filtered out at the dataset level — never appears here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mem_loss(
    pred: torch.Tensor,  # (B, T, F) decoder output in BN-normalized space
    target: torch.Tensor,  # (B, T, F) BN-normalized inputs
    position_mask: torch.Tensor,  # (B, T) bool — True where MEM-masked
    feature_mask: torch.Tensor,  # (F,) bool — True where feature is a target
) -> torch.Tensor:
    """Mean squared error over masked (position, target-feature) cells."""
    if not position_mask.any():
        return pred.new_tensor(0.0)

    # Combine masks: (B, T, F) -> bool
    combined = position_mask.unsqueeze(-1) & feature_mask.view(1, 1, -1)
    diff = (pred - target) ** 2
    selected = diff[combined]
    if selected.numel() == 0:
        return pred.new_tensor(0.0)
    return selected.mean()


def nt_xent_loss(
    z1: torch.Tensor,  # (N, D) L2-normalized
    z2: torch.Tensor,  # (N, D) L2-normalized
    *,
    temperature: float = 0.1,
    soft_positive_pairs: (
        torch.Tensor | None
    ) = None,  # (N, N) — z1[i] / z2[j] soft pair weight
    soft_weight: float = 0.5,
) -> torch.Tensor:
    """SimCLR NT-Xent over 2N samples.  Diagonal positives + optional soft pairs."""
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = (z @ z.T) / temperature  # (2N, 2N)
    # Mask out self-similarity
    eye = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    # Positive index for each row: row i in z1 (0..N) pairs with i + N in z2; vice versa
    pos = torch.arange(2 * N, device=z.device)
    pos = (pos + N) % (2 * N)
    log_softmax = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    primary = -log_softmax[torch.arange(2 * N, device=z.device), pos].mean()

    if soft_positive_pairs is None or soft_weight == 0.0:
        return primary

    # Soft positives: (N, N) → expand into the full 2N×2N similarity grid at
    # blocks (z1, z2) and (z2, z1).
    soft_full = torch.zeros(2 * N, 2 * N, device=z.device)
    soft_full[:N, N:] = soft_positive_pairs
    soft_full[N:, :N] = soft_positive_pairs.T

    # Avoid dividing by zero rows
    row_sum = soft_full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    soft_targets = soft_full / row_sum
    # Guard 0 * -inf = NaN: only multiply where soft_targets is non-zero.
    safe_product = torch.where(
        soft_targets > 0, soft_targets * log_softmax, torch.zeros_like(soft_targets)
    )
    soft_loss = -safe_product.sum(dim=-1)
    # Only rows with any soft positives contribute
    has_soft = soft_full.sum(dim=-1) > 0
    if has_soft.any():
        # Average over ALL 2N rows (zero for non-soft rows) to keep the
        # contribution bounded relative to batch size. Averaging only over
        # has_soft rows inflates the term when few pairs exist.
        soft_term = soft_loss.mean()
        return primary + soft_weight * soft_term
    return primary

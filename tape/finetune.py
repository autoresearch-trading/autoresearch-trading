# tape/finetune.py
"""Step 4 fine-tuning module: DirectionHead + FineTunedModel + weighted-BCE + CKA.

Implements the architecture pre-committed in the ratified Step 4 plan
(`docs/superpowers/plans/2026-04-24-step4-fine-tuning.md`):

    Encoder (frozen → unfrozen)  →  256-dim global embedding
                                            ↓
                                    Linear(256 → 64) + ReLU       ← shared trunk
                                            ↓
                                    4 × Linear(64 → 1)            ← per-horizon heads
                                            ↓
                                    sigmoid + weighted BCE
                                            ↓
                                    weights 0.10 / 0.20 / 0.20 / 0.50

Loss-weight schedule is FIXED at 0.10 / 0.20 / 0.20 / 0.50 for H10 / H50 / H100 / H500
(plan §"Pre-flight: Gate 2 spec drift") — H500 is primary per the 2026-04-24 amendment.
Label smoothing ε is per-horizon (0.10 / 0.08 / 0.05 / 0.05) — less smoothing on the
horizon we trust most (council-6 Q6 #4).

Public surface:
  - DirectionHead             — heads-only nn.Module
  - FineTunedModel            — encoder + heads wrapper
  - weighted_bce_loss(...)    — masked + label-smoothed BCE with per-horizon weights
  - cka_torch(X, Y)           — linear CKA between two (B, D) embedding matrices
  - HORIZON_WEIGHTS, LABEL_SMOOTHING_EPS, HORIZONS, abort-criteria flags
"""

from __future__ import annotations

import torch
from torch import nn

from tape.model import TapeEncoder

# ---------------------------------------------------------------------------
# Module constants (pre-committed in the ratified Step 4 plan)
# ---------------------------------------------------------------------------

HORIZONS: tuple[int, int, int, int] = (10, 50, 100, 500)
HORIZON_WEIGHTS: tuple[float, float, float, float] = (0.10, 0.20, 0.20, 0.50)
LABEL_SMOOTHING_EPS: tuple[float, float, float, float] = (0.10, 0.08, 0.05, 0.05)

HEAD_TRUNK_DIM: int = 64
EMBED_DIM: int = 256

# Numeric abort criteria (plan §"Numeric abort criteria"). Flags here are
# documentation-only — `scripts/run_finetune.py` consults them at runtime.
ABORT_EPOCH_3_H500_VAL_BCE_GT_INIT: bool = True
ABORT_EPOCH_5_H500_VAL_BCE_GT_0_95X_INIT: bool = True
ABORT_EMBED_STD_LT_0_05: bool = True
ABORT_CKA_LT_0_3_AFTER_EPOCH_8: bool = True
ABORT_H100_VAL_BAL_ACC_LT_0_50_AFTER_EPOCH_8: bool = True
ABORT_HOUR_PROBE_GT_0_12_AT_5EPOCH_CHECKPOINT: bool = True


# ---------------------------------------------------------------------------
# DirectionHead
# ---------------------------------------------------------------------------


class DirectionHead(nn.Module):
    """Shared trunk Linear(256→64) + ReLU + 4 separate Linear(64→1) per-horizon heads.

    Forward returns a (B, 4) Tensor of LOGITS for H10/H50/H100/H500 in that order.
    Logits are returned NOT sigmoid-applied — caller does sigmoid+BCE.

    Init policy:
      - Trunk: He/Kaiming uniform with ReLU nonlinearity (matches PyTorch default
        for Linear, but explicit so it does not silently change).
      - Per-horizon heads: small Gaussian σ=0.02 on weights, zero bias. Encourages
        the early sigmoid output to sit near 0.5 (the label-prior under balanced
        sampling) so initial BCE matches the random-init baseline used by the
        epoch-3 / epoch-5 abort criteria.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        trunk_dim: int = HEAD_TRUNK_DIM,
        n_horizons: int = len(HORIZONS),
    ) -> None:
        super().__init__()
        self.trunk = nn.Linear(embed_dim, trunk_dim)
        self.relu = nn.ReLU(inplace=True)
        self.heads = nn.ModuleList([nn.Linear(trunk_dim, 1) for _ in range(n_horizons)])
        self._init_weights()

    def _init_weights(self) -> None:
        # He/Kaiming uniform on the shared trunk (ReLU follows it).
        nn.init.kaiming_uniform_(self.trunk.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(self.trunk.bias)
        # Small Gaussian on per-horizon heads → output near 0.5 at init.
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, embed_dim)
        h = self.relu(self.trunk(embeddings))  # (B, trunk_dim)
        # 4 separate heads → stack to (B, 4)
        per_head = [head(h) for head in self.heads]  # each (B, 1)
        logits = torch.cat(per_head, dim=-1)  # (B, 4)
        return logits


# ---------------------------------------------------------------------------
# FineTunedModel
# ---------------------------------------------------------------------------


class FineTunedModel(nn.Module):
    """Wraps TapeEncoder + DirectionHead. forward((B, 200, 17)) -> (B, 4) logits.

    Provides freeze_encoder() / unfreeze_encoder() utilities — heads always remain
    trainable. Toggling requires_grad on encoder params is the cheap way to
    implement Phase A (frozen warmup) → Phase B (joint fine-tune).
    """

    def __init__(self, encoder: TapeEncoder, head: DirectionHead | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        if head is None:
            head = DirectionHead(embed_dim=encoder.global_dim)
        self.head = head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 200, 17) → 256-dim global embedding → (B, 4) logits
        _per_pos, global_emb = self.encoder(features)
        return self.head(global_emb)

    def freeze_encoder(self) -> None:
        """Set requires_grad=False on all encoder params; heads untouched."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Set requires_grad=True on all encoder params; heads untouched."""
        for p in self.encoder.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Weighted BCE loss (masked + per-horizon label smoothing + per-horizon weight)
# ---------------------------------------------------------------------------


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    label_smoothing_eps: tuple[float, ...] = LABEL_SMOOTHING_EPS,
    horizon_weights: tuple[float, ...] = HORIZON_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-horizon masked BCE with label smoothing, weighted sum across horizons.

    Parameters
    ----------
    logits : (B, H) — H heads, ORDERED H10, H50, H100, H500 (or as specified).
    labels : (B, H) of {0, 1} ints. Values where mask==0 are ignored.
    masks  : (B, H) of {0, 1} ints (or bools). 1 = valid label, 0 = invalid (do
             not contribute to loss).
    label_smoothing_eps : per-horizon ε. Smoothed target = label*(1-ε) + 0.5*ε.
    horizon_weights : per-horizon scalar weights applied to per-horizon mean loss.
                      NOT normalized — weights are gradient-scale, not probability.

    Returns
    -------
    (total_loss_scalar, per_horizon_loss_tensor of shape (H,))

    Notes
    -----
    Per-horizon mean is taken over VALID labels only (mask=1), independently per
    horizon. This is NOT a per-row mean — a row with only H500 valid still
    contributes to H500 loss correctly, with no dilution from invalid horizons.
    If a horizon has zero valid labels in the batch, its per-horizon loss is 0.0
    (it does not contribute to the gradient).
    """
    if logits.shape != labels.shape or logits.shape != masks.shape:
        raise ValueError(
            f"shape mismatch: logits={logits.shape} labels={labels.shape} "
            f"masks={masks.shape}"
        )
    H = logits.shape[1]
    if len(label_smoothing_eps) != H or len(horizon_weights) != H:
        raise ValueError(
            f"per-horizon arity mismatch: H={H}, eps={len(label_smoothing_eps)}, "
            f"weights={len(horizon_weights)}"
        )

    # Promote integers to float for arithmetic; keep on the same device as logits.
    labels_f = labels.to(dtype=logits.dtype, device=logits.device)
    masks_f = masks.to(dtype=logits.dtype, device=logits.device)

    eps_t = torch.tensor(
        label_smoothing_eps, dtype=logits.dtype, device=logits.device
    )  # (H,)
    weights_t = torch.tensor(
        horizon_weights, dtype=logits.dtype, device=logits.device
    )  # (H,)

    # Smooth labels per horizon: smooth = label*(1-ε) + 0.5*ε.
    # Broadcast eps across batch dim: (1, H) → (B, H).
    smoothed = labels_f * (1.0 - eps_t.unsqueeze(0)) + 0.5 * eps_t.unsqueeze(0)

    # Per-element BCE (no reduction). Use the with-logits formulation for
    # numerical stability (log(sigmoid) overflows for large negative logits).
    bce_per_elem = nn.functional.binary_cross_entropy_with_logits(
        logits, smoothed, reduction="none"
    )  # (B, H)

    # Mask out invalid labels and reduce per horizon.
    masked_bce = bce_per_elem * masks_f  # (B, H)
    # Per-horizon sum of valid contributions / count of valid labels.
    valid_count_per_h = masks_f.sum(dim=0)  # (H,)
    bce_sum_per_h = masked_bce.sum(dim=0)  # (H,)
    # Avoid division by zero for empty-mask horizons → set per-horizon loss to 0.
    safe_count = torch.where(
        valid_count_per_h > 0,
        valid_count_per_h,
        torch.ones_like(valid_count_per_h),
    )
    per_horizon = torch.where(
        valid_count_per_h > 0,
        bce_sum_per_h / safe_count,
        torch.zeros_like(bce_sum_per_h),
    )  # (H,)

    total = (per_horizon * weights_t).sum()
    return total, per_horizon


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------


def cka_torch(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (B, D) matrices on the same B samples.

    Centered Frobenius-norm formulation (Kornblith et al., 2019, Eq. 6):

        CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    where X and Y have been centered per-feature (column means subtracted).

    Returns a float in [0, 1] (sign-invariant, scale-invariant). For X==Y or
    X==-Y on centered features the result is 1.0; for unrelated random
    matrices the expected value approaches 0 as B grows.

    The computation runs on whatever device X/Y are on; CPU is fine — this is
    called once per epoch on at most ~1024 windows.
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"expected (B, D) matrices, got X={X.shape} Y={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row count mismatch: X={X.shape[0]} Y={Y.shape[0]}")

    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # ||X^T Y||_F^2 = sum_{i,j} (X^T Y)_{ij}^2
    xty = Xc.T @ Yc
    num = (xty * xty).sum()

    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    denom = torch.sqrt((xtx * xtx).sum()) * torch.sqrt((yty * yty).sum())

    if float(denom) == 0.0:
        return 0.0
    return float(num / denom)

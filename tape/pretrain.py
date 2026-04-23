# tape/pretrain.py
"""Pretraining loop: MEM (block-mask MSE) + SimCLR (NT-Xent + soft cross-symbol).

Applies the council-6 correction plan (2026-04-23):
  - MASK-FIRST-THEN-ENCODE flow (critical): zero-fill masked positions in
    BN-normalized space BEFORE the encoder sees them. Encoding unmasked input
    defeats MEM — the decoder trivially copies (knowledge/concepts/mem-pretraining.md).
  - Block size 20, fraction 0.20 (knowledge/decisions/mem-block-size-20.md).
  - NT-Xent tau annealed 0.5 -> 0.3 over epochs 1..10 (knowledge/decisions/ntxent-temperature.md).
  - MEM weight annealed 0.90 -> 0.60 over 20 epochs; contrastive 0.10 -> 0.40.
  - Gradient clipping max_norm=1.0 (knowledge/concepts/contrastive-learning.md).
  - bf16 autocast + torch.compile(encoder, mode="reduce-overhead").
  - Embedding collapse threshold 0.05 (NOT 1e-4) + effective-rank monitor.

One step:
    1. Build two augmented views per window from the cached context.
    2. BatchNorm BOTH views on the FULL input (clean running stats).
    3. Build per-window block mask on view 1.
    4. Zero masked positions in BN-normalized view 1 (= training mean).
    5. Encode the MASKED view 1 + unmasked view 2.
    6. MEM decoder on encoded-masked-view-1 -> MSE vs BN-normalized
       original at masked positions, over the 14 target features.
    7. Projection head on both global embeddings -> NT-Xent (with optional
       cross-symbol soft positives), tau from schedule.
    8. Total = mem_weight(epoch) * mem_loss + contrastive_weight(epoch) * nt_xent.
    9. Backward -> clip_grad_norm_ -> opt.step() -> sched.step().

Per-epoch hooks (handled by run_pretrain.py, NOT here):
    - sampler.set_epoch(e), dataset.set_epoch(e)
    - embedding-collapse detector on the most recent batch
    - every 5 epochs: probe trio via tape/probes.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import nn

from tape.augment import AugmentConfig, apply_augment_pipeline
from tape.contrastive_batch import (  # noqa: F401
    build_soft_positive_matrix,
    hour_bucket_from_ms,
)
from tape.losses import mem_loss, nt_xent_loss
from tape.masking import block_mask, build_mem_target_mask
from tape.model import EncoderConfig, MEMDecoder, ProjectionHead, TapeEncoder

# Use non-deprecated LRScheduler if available, fall back to _LRScheduler.
# torch 2.0+ exposes torch.optim.lr_scheduler.LRScheduler (no underscore).
try:
    _LRSchedulerType = torch.optim.lr_scheduler.LRScheduler  # type: ignore[attr-defined]
except AttributeError:
    _LRSchedulerType = torch.optim.lr_scheduler._LRScheduler  # type: ignore[attr-defined]


@dataclass
class PretrainConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    # Loss-weight annealing (knowledge/concepts/mem-pretraining.md)
    mem_weight_start: float = 0.90
    mem_weight_end: float = 0.60
    contrastive_weight_start: float = 0.10
    contrastive_weight_end: float = 0.40
    anneal_epochs: int = 20  # both MEM and contrastive anneal over this horizon
    # NT-Xent soft-positive weight for cross-symbol same-hour pairs
    contrastive_soft_weight: float = 0.5
    # NT-Xent temperature schedule (knowledge/decisions/ntxent-temperature.md)
    nt_xent_tau_start: float = 0.5
    nt_xent_tau_end: float = 0.3
    nt_xent_tau_anneal_epochs: int = 10  # then held constant at _end
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    onecycle_pct_start: float = 0.20
    total_steps: int = 0  # set at runtime from dataset size * epochs
    grad_clip_max_norm: float = 1.0
    # Runtime
    seed: int = 0
    use_bf16: bool = True  # torch.autocast(dtype=torch.bfloat16)
    use_torch_compile: bool = True  # torch.compile(encoder, mode="reduce-overhead")
    # Diagnostics
    embedding_collapse_threshold: float = (
        0.05  # NOT 1e-4 — at 256 dims 1e-3 is collapsed
    )
    effective_rank_sv_floor: float = 0.01  # singular values above 1% of max count


def schedule_tau(
    epoch: int,
    *,
    tau_start: float = 0.5,
    tau_end: float = 0.3,
    anneal_epochs: int = 10,
) -> float:
    """Linear anneal of NT-Xent temperature from tau_start to tau_end over
    `anneal_epochs`, then constant at tau_end.

    Formula from knowledge/decisions/ntxent-temperature.md:
        tau = max(tau_end, tau_start - epoch * (tau_start - tau_end) / anneal_epochs)
    """
    if anneal_epochs <= 0:
        return tau_end
    step = (tau_start - tau_end) / anneal_epochs
    return max(tau_end, tau_start - epoch * step)


def schedule_mem_weight(
    epoch: int,
    *,
    start: float = 0.90,
    end: float = 0.60,
    total_anneal_epochs: int = 20,
) -> float:
    """Linear anneal of MEM weight from start to end over total_anneal_epochs."""
    if total_anneal_epochs <= 0:
        return end
    frac = min(1.0, max(0.0, epoch / total_anneal_epochs))
    return start + (end - start) * frac


def schedule_contrastive_weight(
    epoch: int,
    *,
    start: float = 0.10,
    end: float = 0.40,
    total_anneal_epochs: int = 20,
) -> float:
    """Linear anneal of contrastive weight from start to end over total_anneal_epochs."""
    if total_anneal_epochs <= 0:
        return end
    frac = min(1.0, max(0.0, epoch / total_anneal_epochs))
    return start + (end - start) * frac


def detect_embedding_collapse(
    global_emb: torch.Tensor, *, threshold: float = 0.05
) -> bool:
    """True iff mean per-feature std across the batch is below threshold (collapsed).

    Default threshold 0.05 NOT 1e-4: at 256 dims, std 1e-3 is already functionally
    collapsed for a 128-class probe (knowledge/concepts/contrastive-learning.md).
    """
    if global_emb.dim() != 2:
        raise ValueError("expected (B, D) tensor")
    return float(global_emb.std(dim=0).mean()) < threshold


def effective_rank(global_emb: torch.Tensor, *, sv_floor: float = 0.01) -> int:
    """Count singular values above sv_floor * max(singular_values).

    Collapse early-warning:
      - < 20 at epoch 5
      - < 30 at epoch 10
    (knowledge/concepts/contrastive-learning.md "Collapse Prevention")
    """
    if global_emb.dim() != 2:
        raise ValueError("expected (B, D) tensor")
    # Center to avoid rank-1 inflation from a non-zero mean.
    centered = global_emb - global_emb.mean(dim=0, keepdim=True)
    # svdvals is not implemented on MPS in PyTorch 2.10 — fall back to CPU for
    # this diagnostic (called once per step at most; overhead is negligible).
    centered_for_svd = (
        centered.detach().float().cpu()
        if centered.device.type == "mps"
        else centered.float()
    )
    s = torch.linalg.svdvals(centered_for_svd)
    if s.numel() == 0:
        return 0
    max_sv = s.max().item()
    if max_sv == 0.0:
        # All rows identical after centering → rank 1 (one unique value).
        return 1
    thresh = max_sv * sv_floor
    return int((s > thresh).sum().item())


def build_pretrain_modules(cfg: PretrainConfig):  # type: ignore[return]
    """Construct encoder, decoder, projection head, optimizer, and scheduler.

    Returns: (enc, mem_dec, proj, opt, sched)
    """
    enc = TapeEncoder(cfg.encoder)
    mem_dec = MEMDecoder(per_position_dim=enc.per_position_dim, n_features=17)
    proj = ProjectionHead(in_dim=enc.global_dim, hidden=256, out=128)

    # Collect parameters from the UNCOMPILED modules — torch.compile wraps the
    # module in a FunctionType which pyright cannot introspect.
    params = (
        list(enc.parameters()) + list(mem_dec.parameters()) + list(proj.parameters())
    )

    # Optionally compile the encoder for kernel-fusion speedups (dilated CNN
    # with static shapes benefits most). Do NOT compile the MEM decoder —
    # its input shape varies with masked-position count.
    compiled_enc: TapeEncoder = enc  # may be replaced below
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        try:
            compiled_enc = torch.compile(enc, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            # Fall back silently if torch.compile is unavailable on this platform.
            pass

    opt: torch.optim.Optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # OneCycleLR requires total_steps large enough that each phase has at least
    # 1 step (pct_start * total_steps >= 1). With pct_start=0.20, minimum is 5,
    # but ZeroDivision can appear at the boundary — use 10 as safe floor.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.learning_rate,
        total_steps=max(10, cfg.total_steps),
        pct_start=cfg.onecycle_pct_start,
    )
    return compiled_enc, mem_dec, proj, opt, sched


def _apply_block_mask_to_bn_normalized(
    x_bn: torch.Tensor,  # (B, T, F) BN-normalized
    pos_mask: torch.Tensor,  # (B, T) bool — True where MASKED
) -> torch.Tensor:
    """Return a copy of x_bn with masked positions zeroed across all features.

    Zero = training mean in BN-normalized space (BN centers on the running
    mean so zero is the canonical "no information" value).
    """
    x_masked = x_bn.clone()
    # Expand (B, T) -> (B, T, 1) to broadcast across feature dim.
    x_masked[pos_mask] = 0.0
    return x_masked


def pretrain_step(
    enc: TapeEncoder,
    mem_dec: MEMDecoder,
    proj: ProjectionHead,
    opt: torch.optim.Optimizer,
    sched: object,  # LRScheduler — typed as object to avoid deprecated _LRScheduler warning
    batch: torch.Tensor,  # (B, T, F) — already-cropped windows OR contexts
    metadata: dict,  # symbols/dates/hours/eligible (each length B)
    *,
    cfg: PretrainConfig,
    current_epoch: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one optimization step. Returns scalar loss components + schedule + diagnostics.

    Implements the mask-first-then-encode MEM flow (critical — see module docstring).
    """
    if device is not None:
        enc = enc.to(device)  # type: ignore[assignment]
        mem_dec = mem_dec.to(device)
        proj = proj.to(device)
        batch = batch.to(device)

    enc.train()
    mem_dec.train()
    proj.train()

    # Schedules for this epoch
    tau = schedule_tau(
        current_epoch,
        tau_start=cfg.nt_xent_tau_start,
        tau_end=cfg.nt_xent_tau_end,
        anneal_epochs=cfg.nt_xent_tau_anneal_epochs,
    )
    mem_w = schedule_mem_weight(
        current_epoch,
        start=cfg.mem_weight_start,
        end=cfg.mem_weight_end,
        total_anneal_epochs=cfg.anneal_epochs,
    )
    con_w = schedule_contrastive_weight(
        current_epoch,
        start=cfg.contrastive_weight_start,
        end=cfg.contrastive_weight_end,
        total_anneal_epochs=cfg.anneal_epochs,
    )

    # Step counter stored in a dedicated key on the optimizer's state dict.
    # opt.state is keyed by Parameter tensors for gradient state, but the
    # top-level dict also accepts string keys for auxiliary scalars.
    step_idx = cast(int, opt.state.get("_step", 0))  # type: ignore[call-overload]
    rng = np.random.default_rng(cfg.seed + step_idx)

    # Build two augmented views per window (in-place augment of independent clones)
    v1 = torch.stack(
        [apply_augment_pipeline(b.clone(), cfg=cfg.augment, rng=rng) for b in batch]
    )
    v2 = torch.stack(
        [apply_augment_pipeline(b.clone(), cfg=cfg.augment, rng=rng) for b in batch]
    )

    B, T, _Fdim = v1.shape

    # ---- MEM: MASK-FIRST-THEN-ENCODE ----
    # Step 1: BN the FULL, UNMASKED v1 to get clean running stats + target space.
    # We manually apply input_bn then build the mask; the encoder below will receive
    # already-BN-normalized, then-masked input. To avoid double-BN inside the encoder,
    # we call enc.blocks directly after input_bn + masking.
    with torch.no_grad():
        # Use the encoder's own input_bn for consistency (gotcha #23 — loss space
        # matches the encoder's internal normalization).
        v1_bn = enc.input_bn(v1.transpose(1, 2)).transpose(1, 2)  # (B, T, F)

    # Step 2: per-window block mask (block_len=20, fraction=0.20 by default).
    pos_mask_np = np.stack([block_mask(window_len=T, rng=rng) for _ in range(B)])
    pos_mask = torch.from_numpy(pos_mask_np).to(v1.device)

    # Step 3: zero masked positions in BN-normalized space.
    v1_masked_bn = _apply_block_mask_to_bn_normalized(v1_bn, pos_mask)

    # bf16 autocast block for the forward + loss (encoder + heads).
    # Guard: only enable on CUDA — macOS CPU bf16 autocast is inconsistent.
    autocast_ctx = torch.autocast(
        device_type=v1.device.type if v1.is_cuda else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.use_bf16 and v1.is_cuda,
    )
    with autocast_ctx:
        # Step 4: encode the MASKED + BN-normalized view. We bypass input_bn by
        # calling enc.blocks directly — the encoder's forward applies input_bn
        # then blocks, so to match we re-implement the tail here.
        h = v1_masked_bn.transpose(1, 2)  # (B, F, T)
        h = enc.blocks(h)  # (B, C', T)
        per_pos1 = h.transpose(1, 2)  # (B, T, C')
        global1 = torch.cat([per_pos1.mean(dim=1), per_pos1[:, -1, :]], dim=-1)

        # Encode view 2 normally (unmasked, for contrastive).
        per_pos2, global2 = enc(v2)  # noqa: F841

        # MEM loss at masked positions, 14-feature target.
        pred = mem_dec(per_pos1)  # (B, T, 17)
        feat_mask = build_mem_target_mask().to(pred.device)
        L_mem = mem_loss(pred, v1_bn, pos_mask, feat_mask)

        # Contrastive: project + NT-Xent + soft cross-symbol positives
        z1 = proj(global1)
        z2 = proj(global2)
        soft: torch.Tensor | None = None
        if metadata is not None:
            symbols = np.array(metadata["symbols"])
            dates = np.array(metadata["dates"])
            hours = np.array(metadata["hours"], dtype=np.int64)
            eligible = np.array(metadata["eligible"], dtype=bool)
            soft_np = build_soft_positive_matrix(symbols, dates, hours, eligible)
            if soft_np.sum() > 0:
                soft = torch.from_numpy(soft_np).to(z1.device)
        L_con = nt_xent_loss(
            z1,
            z2,
            temperature=tau,
            soft_positive_pairs=soft,
            soft_weight=cfg.contrastive_soft_weight,
        )

        L_total = mem_w * L_mem + con_w * L_con

    opt.zero_grad(set_to_none=True)
    L_total.backward()

    # Gradient clipping (primary anti-collapse mechanism for the projection head)
    all_params = [p for group in opt.param_groups for p in group["params"]]
    nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip_max_norm)

    opt.step()
    cast(object, sched).step()  # type: ignore[union-attr]
    opt.state["_step"] = step_idx + 1  # type: ignore[index]

    return {
        "mem": float(L_mem.detach()),
        "contrastive": float(L_con.detach()),
        "total": float(L_total.detach()),
        "tau": float(tau),
        "mem_weight": float(mem_w),
        "contrastive_weight": float(con_w),
        "embedding_std": float(global1.detach().std(dim=0).mean()),
        "effective_rank": float(
            effective_rank(global1.detach(), sv_floor=cfg.effective_rank_sv_floor)
        ),
    }

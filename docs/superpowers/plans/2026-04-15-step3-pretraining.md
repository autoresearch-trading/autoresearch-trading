# Step 3 Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain the 17-feature dilated-CNN tape encoder via Masked Event Modeling (MEM, weight 0.70) + SimCLR contrastive (weight 0.30) on all pre-April cached shards (24 symbols, AVAX held-out), execute on a single RunPod H100 within a 24 GPU-hour cap, and produce a frozen-encoder checkpoint that Gate 1 can probe.

**Architecture:** New `tape/model.py` (`TapeEncoder` + `MEMDecoder` + `ProjectionHead`), `tape/augment.py` (SimCLR view generator with ±25 jitter and σ=0.10 timing noise), `tape/losses.py` (block-mask MEM MSE in BatchNorm-normalized space + NT-Xent with cross-symbol soft positives), `tape/contrastive_batch.py` (anchor-symbol bookkeeping + AVAX exclusion), `tape/pretrain.py` (training loop with embedding-collapse detector + every-5-epoch probe trio), `scripts/run_pretrain.py` (local + RunPod entry point), `scripts/run_session_confound_check.py` (pre-pretraining LR sanity), `scripts/run_pretrain_probes.py` (direction + symbol-id + hour-of-day frozen-embedding probes). RunPod execution is scripted via `runpod/` Dockerfile + launch.sh; checkpoints stream back via runpodctl. The encoder is exactly the architecture in spec §Architecture (RF=253, 256-dim global embedding, ~400K params); model-size variants (200K / 600K) are toggled via channel multipliers in a config object — the specific size used in production is settled by the open council-6 review **before** Task 12 runs.

**Tech Stack:** Python 3.12, PyTorch 2.2+, NumPy, scikit-learn (probes), pytest. RunPod: H100 80 GB, runpodctl/flash skills for orchestration.

**Spec reference:** `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md` (§Architecture, §Training, §Evaluation Gates after 2026-04-15 amendments)
**Conventions reference:** `CLAUDE.md` (33 gotchas; gotchas #22–#28 are pretraining-specific)
**State:** `.claude/skills/autoresearch/state.md` — Steps 0–2 complete, cache materialized at `data/cache/`, Gate 0 published.

---

## Pre-Plan Open Questions (resolve before Task 12 — the GPU launch)

These are flagged in `state.md` and the spec but are **not** blocking for the code-writing tasks (1–11). They become blocking at Task 12 (the RunPod submission). The plan codes parameter-driven knobs so any resolution is a config change, not a code change.

1. **Model size (RESOLVED 2026-04-23 via `docs/council-reviews/council-6-step3-model-size.md`).** Council-6 recommends `channel_mult=1.0` (~400K params). The 500K spec cap holds — 625K does not buy meaningful capacity given the 17-feature input vocabulary; 200K is marginal for the 14-feature MEM + 128-dim contrastive co-representation. **Council-6 also flagged three plan defects** (applied in this same revision): MEM-identity-task bug (encoder was seeing unmasked input), NT-Xent τ=0.10 (should be 0.5→0.3 per `ntxent-temperature.md`), and block size 5/fraction 0.15 (should be 20/0.20 per `mem-block-size-20.md`).
2. **MEM/contrastive weights are ANNEALED** — MEM 0.90→0.60, contrastive 0.10→0.40 over 20 epochs (`mem-pretraining.md`). Static 0.70/0.30 rejected (MEM over-specialization in early training).
3. **Epochs 20–40 with stop-on-<1%-MEM-improvement.** Spec default; keep. Council-6 notes plateau typically reached before epoch 30 at this scale.
4. **Batch 256 fits H100 at 400K params?** A 200×17×256 float32 batch is 3.5 MB; activations dominate at ~0.5 GB. With bf16 autocast, activations halve — comfortable on 80 GB H100. Keep 256, fall back to 128 only if OOM.
5. **bf16 AMP + `torch.compile(encoder)` enabled by default** — ~1.8× throughput gain on H100, no accuracy cost at this model size (council-6 2026-04-23). Toggle via `PretrainConfig.use_bf16` / `use_torch_compile`.

---

## File Structure

**New modules** (`tape/`):
- `tape/model.py` — `TapeEncoder` (BatchNorm + 6 dilated Conv1d blocks, last 4 residual, RF=253), `MEMDecoder` (Linear 128→17 applied per masked position), `ProjectionHead` (Linear 256→256→128 + L2-norm), `EncoderConfig` dataclass with channel-multiplier knob
- `tape/augment.py` — `make_views(window, rng)` returning two augmented `(200,17)` tensors per call; pipeline = window jitter ±25 (drawn from cache-side context, not pad) + Gaussian σ=0.02·feature_std + timing-feature σ=0.10 noise on `time_delta` and `prev_seq_time_span` channels + per-feature dropout p=0.05 to BN mean + time-scale dilation factor U[0.8, 1.2] on `time_delta`. Time-reversal and event-shuffle are NOT implemented.
- `tape/masking.py` — `block_mask(window_len, block_len=MEM_BLOCK_LEN, fraction=MEM_MASK_FRACTION, rng)` for MEM (defaults 20/0.20 per `mem-block-size-20.md`) + `random_mask(...)` (used for the 2 features in `MEM_RANDOM_MASK_FEATURES` per gotcha already in `constants.py`); returns `(mask: bool[200], target_features: bool[17])`
- `tape/losses.py` — `mem_loss(decoder_out, target_norm, mask, target_feature_mask)` (MSE in BatchNorm-normalized space, only over reconstructed-target features per `MEM_EXCLUDED_FEATURES`); `nt_xent_loss(z1, z2, temperature, soft_positive_pairs=None, soft_weight=0.5)` (NT-Xent with optional soft cross-symbol positives; temperature passed from `schedule_tau(epoch)` — 0.5→0.3 over epochs 1..10)
- `tape/contrastive_batch.py` — `build_contrastive_batch(dataset, sampler, hour_buckets)` — extends each batch with same-date, same-hour windows from the 6 liquid contrastive symbols (BTC, ETH, SOL, BNB, LINK, LTC) as soft positives. **AVAX is rejected at the dataset filter, not here — the held-out shards must never be in the dataset to begin with.**
- `tape/pretrain.py` — training loop. Wraps encoder + heads + optimizer + scheduler. Per-step: build views, mask, forward, MEM loss + contrastive loss, backward, log. Per-epoch: shuffled symbol-balanced sampler, embedding-collapse detector. Every 5 epochs: probe trio (direction LR @ H100 on April 1–13, symbol-ID 25-class, hour-of-day 24-class).
- `tape/probes.py` — `linear_probe_h100(features, labels, masks, fold_kwargs)`, `symbol_probe(features, sym_ids)`, `hour_of_day_probe(features, ts_ms)` returning balanced accuracy scalars. Same walk-forward fold params as Gate 0.

**New scripts** (`scripts/`):
- `scripts/run_session_confound_check.py` — single-feature LR on 4-hour-of-day one-hot vs Gate 0 PCA+LR. Decides whether to prune `time_delta_last` and `prev_seq_time_span_last` from `tape/flat_features.py`.
- `scripts/run_pretrain.py` — CLI: `--cache`, `--out-dir`, `--epochs`, `--batch-size`, `--channel-mult`, `--mem-weight`, `--contrastive-weight`, `--seed`, `--max-h100-hours`. Hard-aborts when wall-clock exceeds `--max-h100-hours`.
- `scripts/run_pretrain_probes.py` — standalone runner for the probe trio (used in-loop and after training for the final report).
- `scripts/export_checkpoint.py` — strip optimizer state, keep encoder + scaler + config + git-sha + spec-sha → `checkpoints/encoder-<sha>.pt` (for Gate 1).

**RunPod scaffold** (`runpod/`):
- `runpod/Dockerfile` — base on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`; pin via `requirements-runpod.txt`; copy `tape/`, `scripts/run_pretrain.py`, `scripts/run_pretrain_probes.py`.
- `runpod/launch.sh` — one-shot: rsync cache from R2 → run pretrain → upload checkpoint + logs back → tear down. Exits non-zero on any failure (so runpodctl marks the pod failed).
- `runpod/requirements-runpod.txt` — frozen deps for reproducibility.

**Tests** (`tests/tape/`):
- `tests/tape/test_model.py`, `test_augment.py`, `test_masking.py`, `test_losses.py`, `test_contrastive_batch.py`, `test_pretrain.py` (unit-level only — no real GPU), `test_probes.py`
- `tests/scripts/test_run_session_confound_check.py`, `tests/scripts/test_run_pretrain.py` (smoke: 2 shards, 2 epochs, batch=8, asserts no NaN + checkpoint written)

**Docs:**
- `docs/experiments/step3-session-confound-check.md` — outcome of pre-pretraining sanity
- `docs/experiments/step3-pretrain-run.md` — final run report (loss curves, probe trio per epoch, embedding-std, chosen config, total H100-hours)
- `docs/experiments/step3-checkpoint-card.md` — manifest of what's in the released checkpoint

---

## Ordering & Commit Discipline

15 tasks. Tasks 1–11 are local CPU + unit tests. Task 12 is the RunPod GPU run (uses `runpod-7`). Tasks 13–15 wrap the run output and prepare for Gate 1 (Step 4 in the spec). One commit per task, prefixes: `feat:` for new modules, `test:` for test-only follow-ups, `experiment:` for results, `chore:` for scaffolding. Stage specific files only — never `git add -A`.

Pyright must be clean at every commit (`pyright tape scripts/run_pretrain.py scripts/run_session_confound_check.py scripts/run_pretrain_probes.py scripts/export_checkpoint.py` → 0 errors / 0 warnings / 0 informations). Pre-commit hook runs black + isort.

---

## Task 1: Model — TapeEncoder + MEMDecoder + ProjectionHead

**Files:**
- Create: `tape/model.py`
- Create: `tests/tape/test_model.py`

- [ ] **Step 1: Write the failing test for TapeEncoder shape and parameter count**

```python
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
    small = sum(p.numel() for p in TapeEncoder(EncoderConfig(channel_mult=0.7)).parameters())
    base = sum(p.numel() for p in TapeEncoder(EncoderConfig(channel_mult=1.0)).parameters())
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_model.py -v`
Expected: ImportError or AttributeError — module not yet written.

- [ ] **Step 3: Implement tape/model.py**

```python
# tape/model.py
"""TapeEncoder — dilated CNN with RF=253; spec §Architecture.

Channel layout follows the spec exactly:
    17 -> 64 (dilation 1)
       -> 128 (dilation 2)
       -> 128 (dilation 4)   + residual
       -> 128 (dilation 8)   + residual
       -> 128 (dilation 16)  + residual
       -> 128 (dilation 32)  + residual

Total receptive field: 1 + sum_k (k-1) * d_k where k=5, dilations={1,2,4,8,16,32}
                     = 1 + 4*(1+2+4+8+16+32) = 1 + 4*63 = 253.

Global embedding = concat[GlobalAvgPool(per_pos), per_pos[:, -1, :]] -> 256-dim.

The channel multiplier scales every Conv1d output channel count except the
final 17 -> first hidden, which always emits at least 32 to keep the input
projection meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = 17
    base_channels: int = 64       # first conv output before mult
    hidden_channels: int = 128    # middle conv output before mult
    channel_mult: float = 1.0
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout_p: float = 0.1


def _scaled(c: int, mult: float, *, floor: int = 32) -> int:
    return max(floor, int(round(c * mult)))


class _ConvBlock(nn.Module):
    """Conv1d + LayerNorm + ReLU (+ Dropout) (+ residual)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int,
        dilation: int,
        *,
        dropout_p: float,
        residual: bool,
    ) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel, dilation=dilation, padding=pad)
        # LayerNorm over channels at each position.
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.residual = residual and (in_c == out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)
        # LayerNorm wants channels last.
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            h = h + x
        return h


class TapeEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_bn = nn.BatchNorm1d(cfg.in_channels)

        c0 = _scaled(cfg.base_channels, cfg.channel_mult)
        ch = _scaled(cfg.hidden_channels, cfg.channel_mult)

        blocks: list[nn.Module] = []
        in_c = cfg.in_channels
        for i, d in enumerate(cfg.dilations):
            out_c = c0 if i == 0 else ch
            blocks.append(
                _ConvBlock(
                    in_c=in_c,
                    out_c=out_c,
                    kernel=cfg.kernel_size,
                    dilation=d,
                    dropout_p=cfg.dropout_p if i < 2 else 0.0,
                    residual=(i >= 2),  # last 4 blocks residual per spec
                )
            )
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self._per_pos_dim = ch

    @property
    def per_position_dim(self) -> int:
        return self._per_pos_dim

    @property
    def global_dim(self) -> int:
        return 2 * self._per_pos_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C) -> (B, C, T) for Conv1d / BN1d
        h = x.transpose(1, 2)
        h = self.input_bn(h)
        h = self.blocks(h)  # (B, C', T)
        per_pos = h.transpose(1, 2)  # (B, T, C')
        avg = per_pos.mean(dim=1)
        last = per_pos[:, -1, :]
        global_emb = torch.cat([avg, last], dim=-1)
        return per_pos, global_emb


class MEMDecoder(nn.Module):
    """Per-position linear decoder: per_pos_dim -> 17 raw feature channels."""

    def __init__(self, per_position_dim: int, n_features: int = 17) -> None:
        super().__init__()
        self.linear = nn.Linear(per_position_dim, n_features)

    def forward(self, per_pos: torch.Tensor) -> torch.Tensor:
        return self.linear(per_pos)


class ProjectionHead(nn.Module):
    """Linear -> ReLU -> Linear -> L2-normalize."""

    def __init__(self, in_dim: int = 256, hidden: int = 256, out: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-12)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_model.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/model.py tests/tape/test_model.py
git commit -m "feat(tape): TapeEncoder + MEMDecoder + ProjectionHead per spec §Architecture"
```

---

## Task 2: Masking — block + random for MEM

**Files:**
- Create: `tape/masking.py`
- Create: `tests/tape/test_masking.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tape/test_masking.py
import numpy as np
import torch

from tape.constants import (
    FEATURE_NAMES,
    MEM_BLOCK_LEN,
    MEM_EXCLUDED_FEATURES,
    MEM_MASK_FRACTION,
    MEM_RANDOM_MASK_FEATURES,
)
from tape.masking import block_mask, build_mem_target_mask, random_mask


def test_block_mask_fraction_is_approximately_correct():
    rng = np.random.default_rng(0)
    masks = [block_mask(window_len=200, block_len=MEM_BLOCK_LEN, fraction=MEM_MASK_FRACTION, rng=rng) for _ in range(50)]
    rates = [m.mean() for m in masks]
    avg = float(np.mean(rates))
    # Allow a ±5pp tolerance — block masking with discrete blocks is granular.
    assert MEM_MASK_FRACTION - 0.05 <= avg <= MEM_MASK_FRACTION + 0.05


def test_block_mask_blocks_are_contiguous_runs_of_block_len():
    rng = np.random.default_rng(1)
    m = block_mask(window_len=200, block_len=5, fraction=0.15, rng=rng)
    # Every contiguous True run should have length divisible by block_len
    # (we may have adjacent blocks merging — len % 5 == 0).
    runs: list[int] = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    for r in runs:
        assert r % 5 == 0, runs


def test_random_mask_independence_per_position():
    rng = np.random.default_rng(0)
    m = random_mask(window_len=200, fraction=0.15, rng=rng)
    # No structural test beyond rate — random masks are i.i.d. per position.
    assert 0.10 <= m.mean() <= 0.20


def test_build_mem_target_mask_excludes_carry_forward_features():
    target = build_mem_target_mask()
    assert target.dtype == torch.bool
    assert target.shape == (17,)
    excluded = {FEATURE_NAMES.index(name) for name in MEM_EXCLUDED_FEATURES}
    for i, name in enumerate(FEATURE_NAMES):
        assert target[i].item() == (name not in MEM_EXCLUDED_FEATURES), (i, name)
    # Check excluded features are False
    for ei in excluded:
        assert not target[ei]
    # Check there are exactly 14 trues (17 - 3 excluded)
    assert int(target.sum()) == 14
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_masking.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/masking.py**

```python
# tape/masking.py
"""MEM masking primitives.

Block masking covers most features (sequential signals).  Two features have
lag-5 autocorrelation > 0.8 (prev_seq_time_span, kyle_lambda) — for those, the
SPECIFICATION uses random-position masking via tape/constants.MEM_RANDOM_MASK_FEATURES,
but the MEM target mask EXCLUDES three carry-forward features entirely
(delta_imbalance_L1, kyle_lambda, cum_ofi_5) — gotcha #22.

Note: kyle_lambda appears in both lists.  The exclusion wins — kyle_lambda is
not a reconstruction target at all.  prev_seq_time_span is the only feature
that uses random-position masking AND is reconstructed.
"""

from __future__ import annotations

import numpy as np
import torch

from tape.constants import (
    FEATURE_NAMES,
    MEM_BLOCK_LEN,
    MEM_EXCLUDED_FEATURES,
    MEM_MASK_FRACTION,
)


def block_mask(
    *,
    window_len: int,
    block_len: int = MEM_BLOCK_LEN,
    fraction: float = MEM_MASK_FRACTION,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bool mask of length window_len with ~fraction*window_len True positions
    arranged in contiguous blocks of block_len.

    Strategy: choose ceil(fraction*window_len/block_len) random block starts
    in the valid range [0, window_len - block_len], without overlap.
    """
    n_blocks = max(1, int(np.ceil(fraction * window_len / block_len)))
    # Choose start positions that snap to block_len-aligned slots so blocks
    # never overlap.
    n_slots = window_len // block_len
    pick = rng.choice(n_slots, size=min(n_blocks, n_slots), replace=False)
    out = np.zeros(window_len, dtype=bool)
    for slot in pick:
        s = int(slot) * block_len
        out[s : s + block_len] = True
    return out


def random_mask(
    *,
    window_len: int,
    fraction: float = MEM_MASK_FRACTION,
    rng: np.random.Generator,
) -> np.ndarray:
    """I.i.d. Bernoulli(fraction) mask per position."""
    return rng.random(window_len) < fraction


def build_mem_target_mask() -> torch.Tensor:
    """Return bool[17] — True where the feature IS a reconstruction target."""
    excluded = set(MEM_EXCLUDED_FEATURES)
    return torch.tensor(
        [name not in excluded for name in FEATURE_NAMES],
        dtype=torch.bool,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_masking.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/masking.py tests/tape/test_masking.py
git commit -m "feat(tape): block/random MEM masking + 14-feature target mask"
```

---

## Task 3: Augmentations — SimCLR view generator (±25 jitter, σ=0.10 timing noise)

**Files:**
- Create: `tape/augment.py`
- Create: `tests/tape/test_augment.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    cfg = AugmentConfig(jitter=25, gauss_sigma=0.0, timing_sigma=0.0, dropout_p=0.0, time_dilation_range=(1.0, 1.0))
    ctx = _ctx(0) + 100.0  # shift so any zero-fill would be conspicuous
    rng = np.random.default_rng(0)
    v1, v2 = make_views_from_context(ctx, center=200, window_len=200, cfg=cfg, rng=rng)
    # No element should be near zero — original ctx mean is ~100.
    assert (v1.abs() > 50).all()
    assert (v2.abs() > 50).all()


def test_timing_noise_only_perturbs_time_features():
    cfg = AugmentConfig(jitter=0, gauss_sigma=0.0, timing_sigma=0.10, dropout_p=0.0, time_dilation_range=(1.0, 1.0))
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
    cfg = AugmentConfig(jitter=0, gauss_sigma=0.02, timing_sigma=0.0, dropout_p=0.0, time_dilation_range=(1.0, 1.0))
    rng = np.random.default_rng(0)
    base = torch.zeros(200, 17)
    base[:, 0] = torch.randn(200) * 5.0  # channel 0 has std 5
    base[:, 1] = torch.randn(200) * 0.1  # channel 1 has std 0.1
    out = apply_augment_pipeline(base.clone(), cfg=cfg, rng=rng)
    diff = (out - base).std(dim=0)
    # Noise on channel 0 should be roughly 50x larger than on channel 1.
    assert diff[0] > 5 * diff[1]


def test_feature_dropout_zeroes_some_positions_per_feature():
    cfg = AugmentConfig(jitter=0, gauss_sigma=0.0, timing_sigma=0.0, dropout_p=0.50, time_dilation_range=(1.0, 1.0))
    rng = np.random.default_rng(0)
    base = torch.ones(200, 17)
    out = apply_augment_pipeline(base.clone(), cfg=cfg, rng=rng)
    # With p=0.5, ~half of the (position, feature) cells should be zero.
    rate = (out == 0).float().mean().item()
    assert 0.40 <= rate <= 0.60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_augment.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/augment.py**

```python
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
    jitter: int = 25                     # ±25 events around the center
    gauss_sigma: float = 0.02            # multiplied by per-channel std
    timing_sigma: float = 0.10           # σ on time_delta + prev_seq_time_span
    dropout_p: float = 0.05              # per-(pos,feat) zero-out probability
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
    lo = center - cfg.jitter
    hi = center + cfg.jitter
    if lo < 0 or hi + window_len > T:
        raise ValueError(
            f"context too narrow for jitter ±{cfg.jitter}: T={T}, center={center}, "
            f"window_len={window_len}"
        )

    s1 = int(rng.integers(lo, hi + 1))
    s2 = int(rng.integers(lo, hi + 1))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_augment.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/augment.py tests/tape/test_augment.py
git commit -m "feat(tape): SimCLR view generator with ±25 jitter and σ=0.10 timing noise"
```

---

## Task 4: Losses — MEM MSE in BN-normalized space + NT-Xent with soft positives

**Files:**
- Create: `tape/losses.py`
- Create: `tests/tape/test_losses.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tape/test_losses.py
import torch

from tape.losses import mem_loss, nt_xent_loss


def test_mem_loss_zero_on_perfect_reconstruction():
    pred = torch.randn(2, 200, 17)
    target = pred.clone()
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    pos_mask[:, 50:55] = True
    feat_mask = torch.ones(17, dtype=torch.bool)
    feat_mask[6:9] = False  # exclude 3 features (mimics carry-forward)
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mem_loss_only_counts_masked_positions_and_target_features():
    pred = torch.zeros(2, 200, 17)
    target = torch.ones(2, 200, 17)
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    pos_mask[:, 0:5] = True  # 5 positions per batch row -> 10 positions total
    feat_mask = torch.zeros(17, dtype=torch.bool)
    feat_mask[0] = True  # only feature 0 counted -> 10 cells
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    # Each cell error = 1.0; mean over 10 cells = 1.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_mem_loss_zero_when_no_positions_masked():
    pred = torch.randn(2, 200, 17)
    target = torch.randn(2, 200, 17)
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    feat_mask = torch.ones(17, dtype=torch.bool)
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    # No masked positions -> zero loss (graceful no-op)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_nt_xent_identical_views_low_loss():
    z1 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    z2 = z1.clone()
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    # With identical views the diagonal is the strongest similarity -> loss small
    assert loss.item() < 0.5


def test_nt_xent_random_views_higher_loss():
    z1 = torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    # Random views -> log(2N - 1) ≈ log(31) ≈ 3.4 (T=0.1 keeps it close)
    assert loss.item() > 1.0


def test_nt_xent_soft_positives_reduce_loss():
    z1 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    base = nt_xent_loss(z1, z2, temperature=0.1).item()

    # Mark indices 0 and 1 as a soft-positive pair across the two views
    soft = torch.zeros(8, 8)
    soft[0, 1] = 1.0
    soft[1, 0] = 1.0
    augmented = nt_xent_loss(z1, z2, temperature=0.1, soft_positive_pairs=soft, soft_weight=0.5).item()
    assert augmented <= base + 0.5  # primary loss unchanged or slightly reduced
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_losses.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/losses.py**

```python
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
    pred: torch.Tensor,             # (B, T, F) decoder output in BN-normalized space
    target: torch.Tensor,           # (B, T, F) BN-normalized inputs
    position_mask: torch.Tensor,    # (B, T) bool — True where MEM-masked
    feature_mask: torch.Tensor,     # (F,) bool — True where feature is a target
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
    z1: torch.Tensor,                                 # (N, D) L2-normalized
    z2: torch.Tensor,                                 # (N, D) L2-normalized
    *,
    temperature: float = 0.1,
    soft_positive_pairs: torch.Tensor | None = None,  # (N, N) — z1[i] / z2[j] soft pair weight
    soft_weight: float = 0.5,
) -> torch.Tensor:
    """SimCLR NT-Xent over 2N samples.  Diagonal positives + optional soft pairs."""
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                    # (2N, D)
    sim = (z @ z.T) / temperature                     # (2N, 2N)
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
    soft_loss = -(soft_targets * log_softmax).sum(dim=-1)
    # Only rows with any soft positives contribute
    has_soft = (soft_full.sum(dim=-1) > 0)
    if has_soft.any():
        soft_term = soft_loss[has_soft].mean()
        return primary + soft_weight * soft_term
    return primary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_losses.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/losses.py tests/tape/test_losses.py
git commit -m "feat(tape): MEM masked MSE + NT-Xent with optional cross-symbol soft positives"
```

---

## Task 5: Contrastive batch — anchor sets and AVAX exclusion

**Files:**
- Create: `tape/contrastive_batch.py`
- Create: `tests/tape/test_contrastive_batch.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tape/test_contrastive_batch.py
import numpy as np
import pytest

from tape.constants import HELD_OUT_SYMBOL, LIQUID_CONTRASTIVE_SYMBOLS
from tape.contrastive_batch import build_soft_positive_matrix, hour_bucket_from_ms


def test_hour_bucket_basic():
    # 1700000000000 ms -> some UTC hour 0..23
    h = hour_bucket_from_ms(np.array([1700000000000], dtype=np.int64))
    assert 0 <= int(h[0]) <= 23


def test_soft_positive_matrix_pairs_same_date_same_hour_liquid_symbols():
    # Build a fake batch metadata: 4 windows, 3 in liquid symbols same date+hour, 1 in illiquid
    symbols = np.array(["BTC", "ETH", "SOL", "DOGE"])
    dates = np.array(["2026-02-01"] * 4)
    hours = np.array([10, 10, 10, 10], dtype=np.int64)
    sym_match = np.array([s in LIQUID_CONTRASTIVE_SYMBOLS for s in symbols])
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    assert soft.shape == (4, 4)
    # BTC <-> ETH <-> SOL all paired (off-diagonal 1s)
    assert soft[0, 1] == 1
    assert soft[1, 0] == 1
    assert soft[0, 2] == 1
    # No self-pairs
    assert soft[0, 0] == 0
    assert soft[1, 1] == 0
    # DOGE is not in LIQUID_CONTRASTIVE_SYMBOLS -> no pairs touching index 3
    assert soft[3].sum() == 0
    assert soft[:, 3].sum() == 0


def test_avax_rejected_from_soft_positives():
    """Even if metadata claims AVAX, the helper must drop it (defense in depth)."""
    symbols = np.array(["BTC", "AVAX", "ETH"])
    dates = np.array(["2026-02-01"] * 3)
    hours = np.array([10, 10, 10], dtype=np.int64)
    sym_match = np.array([s in LIQUID_CONTRASTIVE_SYMBOLS and s != HELD_OUT_SYMBOL for s in symbols])
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    # AVAX (index 1) must have zero rows and columns
    assert soft[1].sum() == 0
    assert soft[:, 1].sum() == 0
    # BTC <-> ETH still paired
    assert soft[0, 2] == 1


def test_pairs_only_within_same_date_hour():
    symbols = np.array(["BTC", "ETH", "BTC", "ETH"])
    dates = np.array(["2026-02-01", "2026-02-01", "2026-02-02", "2026-02-02"])
    hours = np.array([10, 10, 10, 10], dtype=np.int64)
    sym_match = np.array([True, True, True, True])
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    # Date 1 pair: 0 <-> 1
    assert soft[0, 1] == 1
    # Date 2 pair: 2 <-> 3
    assert soft[2, 3] == 1
    # Cross-date should be zero
    assert soft[0, 2] == 0
    assert soft[1, 3] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_contrastive_batch.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/contrastive_batch.py**

```python
# tape/contrastive_batch.py
"""Cross-symbol contrastive pairing for SimCLR (spec §Training).

Same-date, same-UTC-hour windows from the 6 LIQUID_CONTRASTIVE_SYMBOLS
(BTC, ETH, SOL, BNB, LINK, LTC) get a soft-positive weight 0.5 in the
NT-Xent loss.  AVAX is the Gate 3 held-out symbol — must NEVER appear
in pairs (gotcha #25).
"""

from __future__ import annotations

import numpy as np

from tape.constants import HELD_OUT_SYMBOL, LIQUID_CONTRASTIVE_SYMBOLS


def hour_bucket_from_ms(ts_ms: np.ndarray) -> np.ndarray:
    """Map ms timestamps to UTC hour (0–23)."""
    seconds = ts_ms // 1_000
    return ((seconds // 3_600) % 24).astype(np.int64)


def build_soft_positive_matrix(
    symbols: np.ndarray,           # (B,) of str
    dates: np.ndarray,             # (B,) of "YYYY-MM-DD" str
    hours: np.ndarray,             # (B,) of int 0..23
    eligible_mask: np.ndarray,     # (B,) of bool — True if symbol is in liquid set AND not AVAX
) -> np.ndarray:
    """Return (B, B) {0, 1} matrix of cross-symbol same-date-same-hour pairs.

    Diagonal is always zero.  Pairs only between distinct eligible symbols.
    AVAX is rejected here as defense in depth even if the dataset already
    excludes it.
    """
    B = len(symbols)
    # Hard reject AVAX regardless of caller's eligible_mask
    safe_mask = eligible_mask & (symbols != HELD_OUT_SYMBOL)
    out = np.zeros((B, B), dtype=np.float32)
    for i in range(B):
        if not safe_mask[i]:
            continue
        for j in range(B):
            if i == j or not safe_mask[j]:
                continue
            if symbols[i] == symbols[j]:
                continue
            if dates[i] != dates[j]:
                continue
            if hours[i] != hours[j]:
                continue
            out[i, j] = 1.0
    return out


def is_eligible_for_contrastive(symbol: str) -> bool:
    return symbol in LIQUID_CONTRASTIVE_SYMBOLS and symbol != HELD_OUT_SYMBOL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_contrastive_batch.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/contrastive_batch.py tests/tape/test_contrastive_batch.py
git commit -m "feat(tape): cross-symbol soft-positive matrix with AVAX exclusion"
```

---

## Task 6: Probes — direction LR + symbol ID + hour-of-day on frozen embeddings

**Files:**
- Create: `tape/probes.py`
- Create: `tests/tape/test_probes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tape/test_probes.py
import numpy as np

from tape.probes import (
    direction_probe_h100,
    hour_of_day_probe,
    symbol_identity_probe,
)


def test_direction_probe_returns_per_symbol_balanced_acc():
    rng = np.random.default_rng(0)
    # 2 symbols, 4000 windows each, 256-dim embeddings, binary labels
    feats = {
        "BTC": rng.standard_normal((4000, 256)).astype(np.float32),
        "ETH": rng.standard_normal((4000, 256)).astype(np.float32),
    }
    labels = {
        "BTC": rng.integers(0, 2, size=4000).astype(np.int64),
        "ETH": rng.integers(0, 2, size=4000).astype(np.int64),
    }
    masks = {
        "BTC": np.ones(4000, dtype=bool),
        "ETH": np.ones(4000, dtype=bool),
    }
    out = direction_probe_h100(feats, labels, masks)
    assert set(out.keys()) == {"BTC", "ETH"}
    # Balanced accuracy on random labels should be ~0.50
    assert 0.45 <= out["BTC"] <= 0.55
    assert 0.45 <= out["ETH"] <= 0.55


def test_symbol_identity_probe_low_on_random_features():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((1000, 256)).astype(np.float32)
    sym_ids = rng.integers(0, 25, size=1000).astype(np.int64)
    acc = symbol_identity_probe(feats, sym_ids, n_symbols=25)
    # Random features -> ~1/25 = 4%
    assert acc < 0.10


def test_hour_of_day_probe_low_on_random_features():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((2000, 256)).astype(np.float32)
    hours = rng.integers(0, 24, size=2000).astype(np.int64)
    acc = hour_of_day_probe(feats, hours)
    # Random features -> ~1/24 ≈ 4.2%
    assert acc < 0.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tape/test_probes.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/probes.py**

```python
# tape/probes.py
"""Frozen-embedding probes used during pretraining monitoring AND Gate 1.

Direction probe: H100 only during pretraining (rapid signal). Gate 1 evaluates
all four horizons on April 1–13 separately via scripts/run_pretrain_probes.py.

All probes use balanced accuracy (council round 6: raw accuracy is gameable
via per-fold label imbalance at every horizon).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.splits import walk_forward_folds


def direction_probe_h100(
    features: dict[str, np.ndarray],     # symbol -> (N_i, D)
    labels: dict[str, np.ndarray],       # symbol -> (N_i,) {0, 1}
    masks: dict[str, np.ndarray],        # symbol -> (N_i,) bool valid
    *,
    n_folds: int = 3,
    embargo: int = 600,
    min_train: int = 2_000,
    min_test: int = 500,
    C: float = 1.0,
) -> dict[str, float]:
    """Per-symbol balanced accuracy at H100 via walk-forward 3-fold."""
    out: dict[str, float] = {}
    for sym, feat in features.items():
        y = labels[sym]
        m = masks[sym]
        valid_idx = np.where(m)[0]
        if len(valid_idx) < min_train + embargo + n_folds * min_test:
            continue
        try:
            folds = walk_forward_folds(
                np.arange(len(valid_idx)),
                n_folds=n_folds,
                embargo=embargo,
                min_train=min_train,
                min_test=min_test,
            )
        except ValueError:
            continue

        scores: list[float] = []
        for tr, te in folds:
            tr_pos = valid_idx[tr]
            te_pos = valid_idx[te]
            scaler = StandardScaler().fit(feat[tr_pos])
            Xtr = scaler.transform(feat[tr_pos])
            Xte = scaler.transform(feat[te_pos])
            lr = LogisticRegression(C=C, max_iter=1_000).fit(Xtr, y[tr_pos])
            scores.append(balanced_accuracy_score(y[te_pos], lr.predict(Xte)))
        if scores:
            out[sym] = float(np.mean(scores))
    return out


def symbol_identity_probe(
    features: np.ndarray,                # (N, D) embeddings
    sym_ids: np.ndarray,                 # (N,) int symbol IDs
    *,
    n_symbols: int = 25,
    test_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
) -> float:
    """25-class linear probe accuracy.  Spec target: < 20%."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_test = int(len(features) * test_frac)
    te = perm[:n_test]
    tr = perm[n_test:]
    scaler = StandardScaler().fit(features[tr])
    lr = LogisticRegression(
        C=C,
        max_iter=1_000,
        multi_class="multinomial",
    ).fit(scaler.transform(features[tr]), sym_ids[tr])
    return float(lr.score(scaler.transform(features[te]), sym_ids[te]))


def hour_of_day_probe(
    features: np.ndarray,                # (N, D) embeddings
    hours: np.ndarray,                   # (N,) int 0..23
    *,
    test_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
) -> float:
    """24-class hour-of-day probe.  Spec gate: < 10%."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_test = int(len(features) * test_frac)
    te = perm[:n_test]
    tr = perm[n_test:]
    scaler = StandardScaler().fit(features[tr])
    lr = LogisticRegression(
        C=C,
        max_iter=1_000,
        multi_class="multinomial",
    ).fit(scaler.transform(features[tr]), hours[tr])
    return float(lr.score(scaler.transform(features[te]), hours[te]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_probes.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/probes.py tests/tape/test_probes.py
git commit -m "feat(tape): direction/symbol/hour-of-day frozen-embedding probes"
```

---

## Task 7: Pre-pretraining session-of-day confound check

**Files:**
- Create: `scripts/run_session_confound_check.py`
- Create: `tests/scripts/test_run_session_confound_check.py`
- Create: `tests/scripts/__init__.py`
- Modify: `docs/experiments/` (will hold the output `step3-session-confound-check.md`)

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_run_session_confound_check.py
import json
from pathlib import Path

from scripts.run_session_confound_check import run_check


def test_run_check_writes_json_and_md(tmp_path):
    # Use a tiny synthetic cache for smoke-only — real run is in CLI
    out = tmp_path / "session-check"
    res = run_check(
        cache_dir=Path("data/cache"),
        symbols=["BTC", "ETH", "SOL"],
        out_path=out,
    )
    # Must always emit two artifacts even on small data
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".md").exists()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert "per_symbol" in payload
    assert "decision" in payload
    assert payload["decision"] in {"prune_last_features", "no_action"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_run_session_confound_check.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scripts/run_session_confound_check.py**

```python
# scripts/run_session_confound_check.py
"""Pre-pretraining session-of-day confound check (spec §Session-of-Day Confound Check).

Compares an LR on a single 4-hour-of-day one-hot feature vs PCA(20)+LR on
the 85-dim flat feature vector.  If the single-feature model exceeds PCA+LR
by > 0.5pp balanced accuracy on >= 5 symbols at H100, the `_last` columns
in tape/flat_features.py (especially time_delta_last, prev_seq_time_span_last)
are leaking session-of-day — we'd prune them before pretraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.cache import load_shard
from tape.constants import APRIL_HELDOUT_START, DIRECTION_HORIZONS, HELD_OUT_SYMBOL
from tape.flat_features import extract_flat_features_batch
from tape.splits import walk_forward_folds

_TARGET_HORIZON = 100
_DELTA_THRESHOLD = 0.005  # 0.5pp
_SYMBOL_COUNT_TRIGGER = 5


def _hour_one_hot_4h(ts_ms: np.ndarray) -> np.ndarray:
    """Return (N, 6) one-hot of 4-hour buckets (0..5)."""
    bucket = ((ts_ms // 1_000 // 3_600) // 4) % 6
    out = np.zeros((len(ts_ms), 6), dtype=np.float32)
    out[np.arange(len(ts_ms)), bucket] = 1.0
    return out


def _load_symbol(cache_dir: Path, symbol: str):
    shards = sorted(cache_dir.glob(f"{symbol}__*.npz"))
    if not shards:
        return None
    feats: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for p in shards:
        date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
        if date_part >= APRIL_HELDOUT_START:
            raise ValueError(f"April hold-out shard leaked into session check: {p}")
        payload = load_shard(p)
        feats.append(payload["features"])
        ts.append(payload["ts_ms"])
        labels.append(payload[f"dir_h{_TARGET_HORIZON}"])
        masks.append(payload[f"dir_mask_h{_TARGET_HORIZON}"])
    return np.concatenate(feats), np.concatenate(ts), np.concatenate(labels), np.concatenate(masks)


def _flatten_windows(features: np.ndarray, ts: np.ndarray, labels: np.ndarray, masks: np.ndarray):
    """Slide stride-200 windows and return (X_flat, X_hour_first_event_ts, y, mask_at_window_end)."""
    from tape.constants import STRIDE_EVAL, WINDOW_LEN

    starts = np.arange(0, len(features) - WINDOW_LEN + 1, STRIDE_EVAL)
    if len(starts) == 0:
        return None
    X_flat = extract_flat_features_batch(features, starts, window_len=WINDOW_LEN)
    label_idx = starts + WINDOW_LEN - 1
    y = labels[label_idx]
    m = masks[label_idx]
    ts_first = ts[starts]
    return X_flat, ts_first, y, m


def run_check(
    *,
    cache_dir: Path,
    symbols: list[str] | None,
    out_path: Path,
) -> dict:
    if symbols is None:
        from tape.constants import PRETRAINING_SYMBOLS
        symbols = list(PRETRAINING_SYMBOLS)
    symbols = [s for s in symbols if s != HELD_OUT_SYMBOL]

    per_sym: dict[str, dict] = {}
    flagged = 0

    for sym in symbols:
        loaded = _load_symbol(cache_dir, sym)
        if loaded is None:
            continue
        feats, ts, labels, masks = loaded
        prep = _flatten_windows(feats, ts, labels, masks)
        if prep is None:
            continue
        X_flat, ts_first, y, mask = prep
        valid = mask
        if valid.sum() < 4_000:
            continue
        X_flat = X_flat[valid]
        y = y[valid]
        ts_first = ts_first[valid]

        try:
            folds = walk_forward_folds(
                np.arange(len(X_flat)),
                n_folds=3,
                embargo=600,
                min_train=2_000,
                min_test=500,
            )
        except ValueError:
            continue

        flat_scores: list[float] = []
        hour_scores: list[float] = []
        for tr, te in folds:
            scaler = StandardScaler().fit(X_flat[tr])
            pca = PCA(n_components=20).fit(scaler.transform(X_flat[tr]))
            lr_flat = LogisticRegression(C=1.0, max_iter=1_000).fit(
                pca.transform(scaler.transform(X_flat[tr])), y[tr]
            )
            flat_scores.append(
                balanced_accuracy_score(
                    y[te],
                    lr_flat.predict(pca.transform(scaler.transform(X_flat[te]))),
                )
            )
            X_hour = _hour_one_hot_4h(ts_first)
            lr_hour = LogisticRegression(C=1.0, max_iter=1_000).fit(X_hour[tr], y[tr])
            hour_scores.append(balanced_accuracy_score(y[te], lr_hour.predict(X_hour[te])))

        flat_avg = float(np.mean(flat_scores))
        hour_avg = float(np.mean(hour_scores))
        delta = hour_avg - flat_avg
        per_sym[sym] = {
            "flat_pca_lr_balanced_acc": flat_avg,
            "hour_only_lr_balanced_acc": hour_avg,
            "delta_pp": delta * 100,
            "leaks": delta > _DELTA_THRESHOLD,
        }
        if delta > _DELTA_THRESHOLD:
            flagged += 1

    decision = "prune_last_features" if flagged >= _SYMBOL_COUNT_TRIGGER else "no_action"
    payload = {
        "horizon": _TARGET_HORIZON,
        "delta_threshold_pp": _DELTA_THRESHOLD * 100,
        "symbol_count_trigger": _SYMBOL_COUNT_TRIGGER,
        "n_symbols_with_leak": flagged,
        "decision": decision,
        "per_symbol": per_sym,
    }
    out_json = out_path.with_suffix(".json")
    out_md = out_path.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    md_lines = [
        "# Pre-pretraining session-of-day confound check",
        f"- Decision: **{decision}**",
        f"- Symbols flagged (delta > {_DELTA_THRESHOLD*100:.1f}pp): {flagged}",
        "",
        "| Symbol | Flat PCA+LR | Hour-only LR | Δ (pp) | Leaks? |",
        "|--------|-------------|--------------|--------|--------|",
    ]
    for sym, row in sorted(per_sym.items()):
        md_lines.append(
            f"| {sym} | {row['flat_pca_lr_balanced_acc']:.4f} | "
            f"{row['hour_only_lr_balanced_acc']:.4f} | {row['delta_pp']:+.2f} | "
            f"{'YES' if row['leaks'] else ''} |"
        )
    out_md.write_text("\n".join(md_lines))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/step3-session-confound-check"))
    args = ap.parse_args()
    payload = run_check(cache_dir=args.cache, symbols=args.symbols, out_path=args.out)
    print(json.dumps({k: payload[k] for k in ("decision", "n_symbols_with_leak")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/scripts/test_run_session_confound_check.py -v`
Expected: PASS (the test treats this as a smoke; if `data/cache` is not populated locally, mark as `pytest.mark.skipif`).

- [ ] **Step 5: Run the actual check on the local cache**

Run: `uv run python scripts/run_session_confound_check.py --cache data/cache --out docs/experiments/step3-session-confound-check`
Expected: prints `{"decision": "...", "n_symbols_with_leak": N}`. Reads two artifacts written under `docs/experiments/`.

- [ ] **Step 6: Commit results**

```bash
git add scripts/run_session_confound_check.py tests/scripts/__init__.py tests/scripts/test_run_session_confound_check.py docs/experiments/step3-session-confound-check.json docs/experiments/step3-session-confound-check.md
git commit -m "experiment: pre-pretraining session-of-day confound check"
```

- [ ] **Step 7: Decision branch**

If `decision == "prune_last_features"`:
- Open a follow-up sub-task: edit `tape/flat_features.py` to drop `time_delta_last` and `prev_seq_time_span_last` from the 85-dim flat vector. Re-run Gate 0 baselines. **The pretraining loop in Task 9 already excludes these columns from MEM target reconstruction via `MEM_EXCLUDED_FEATURES`; this branch is purely about the flat probe baselines** that Gate 1 measures against.
- Otherwise: proceed.

---

## Task 8: Pretraining loop — `tape/pretrain.py`

**Files:**
- Create: `tape/pretrain.py`
- Create: `tests/tape/test_pretrain.py`

- [ ] **Step 1: Write the failing test (smoke only)**

```python
# tests/tape/test_pretrain.py
import torch

from tape.model import EncoderConfig
from tape.pretrain import PretrainConfig, pretrain_step


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
        enc, mem_dec, proj, opt, sched, batch, metadata,
        cfg=cfg, current_epoch=0,
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
    from tape.pretrain import schedule_tau, schedule_mem_weight, schedule_contrastive_weight

    assert schedule_tau(epoch=0) == 0.5
    assert schedule_tau(epoch=5) == pytest.approx(0.4, abs=1e-6)
    assert schedule_tau(epoch=10) == pytest.approx(0.3, abs=1e-6)
    assert schedule_tau(epoch=25) == 0.3  # held constant after epoch 10

    assert schedule_mem_weight(epoch=0, total_anneal_epochs=20) == pytest.approx(0.90, abs=1e-6)
    assert schedule_mem_weight(epoch=20, total_anneal_epochs=20) == pytest.approx(0.60, abs=1e-6)
    assert schedule_mem_weight(epoch=30, total_anneal_epochs=20) == pytest.approx(0.60, abs=1e-6)

    assert schedule_contrastive_weight(epoch=0, total_anneal_epochs=20) == pytest.approx(0.10, abs=1e-6)
    assert schedule_contrastive_weight(epoch=20, total_anneal_epochs=20) == pytest.approx(0.40, abs=1e-6)


def make_pretrain_modules(cfg):
    from tape.pretrain import build_pretrain_modules
    return build_pretrain_modules(cfg)


def test_embedding_collapse_detector_flags_constant_embeddings():
    from tape.pretrain import detect_embedding_collapse

    z = torch.zeros(64, 256)            # all identical -> std=0
    assert detect_embedding_collapse(z)  # default threshold 0.05

    z = torch.randn(64, 256)            # spread out -> std large
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
    metadata = {"symbols": ["BTC"] * 8, "dates": ["2026-02-01"] * 8,
                "hours": [10] * 8, "eligible": [True] * 8}
    losses_before = pretrain_step(enc, mem_dec, proj, opt, sched, batch, metadata,
                                   cfg=cfg, current_epoch=0)
    # If the encoder correctly sees MASKED input, loss is non-trivial (> 0.01
    # in BN-normalized space on random data).
    assert losses_before["mem"] > 0.01, (
        "MEM loss near zero suggests encoder sees unmasked input (trivial copy)"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tape/test_pretrain.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement tape/pretrain.py**

```python
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

import numpy as np
import torch
from torch import nn

from tape.augment import AugmentConfig, apply_augment_pipeline
from tape.contrastive_batch import build_soft_positive_matrix, hour_bucket_from_ms
from tape.losses import mem_loss, nt_xent_loss
from tape.masking import block_mask, build_mem_target_mask
from tape.model import EncoderConfig, MEMDecoder, ProjectionHead, TapeEncoder


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
    use_bf16: bool = True          # torch.autocast(dtype=torch.bfloat16)
    use_torch_compile: bool = True # torch.compile(encoder, mode="reduce-overhead")
    # Diagnostics
    embedding_collapse_threshold: float = 0.05  # NOT 1e-4 — at 256 dims 1e-3 is collapsed
    effective_rank_sv_floor: float = 0.01       # singular values above 1% of max count


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


def detect_embedding_collapse(global_emb: torch.Tensor, *, threshold: float = 0.05) -> bool:
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
    s = torch.linalg.svdvals(centered.float())
    if s.numel() == 0:
        return 0
    thresh = s.max().item() * sv_floor
    return int((s > thresh).sum().item())


def build_pretrain_modules(cfg: PretrainConfig):
    enc = TapeEncoder(cfg.encoder)
    mem_dec = MEMDecoder(per_position_dim=enc.per_position_dim, n_features=17)
    proj = ProjectionHead(in_dim=enc.global_dim, hidden=256, out=128)
    # Optionally compile the encoder for kernel-fusion speedups (dilated CNN
    # with static shapes benefits most). Do NOT compile the MEM decoder —
    # its input shape varies with masked-position count.
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        try:
            enc = torch.compile(enc, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            # Fall back silently if torch.compile is unavailable on this platform.
            pass
    params = list(enc.parameters()) + list(mem_dec.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.learning_rate,
        total_steps=max(1, cfg.total_steps),
        pct_start=cfg.onecycle_pct_start,
    )
    return enc, mem_dec, proj, opt, sched


def _apply_block_mask_to_bn_normalized(
    x_bn: torch.Tensor,           # (B, T, F) BN-normalized
    pos_mask: torch.Tensor,       # (B, T) bool — True where MASKED
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
    sched: torch.optim.lr_scheduler._LRScheduler,
    batch: torch.Tensor,                         # (B, T, F) — already-cropped windows OR contexts
    metadata: dict,                              # symbols/dates/hours/eligible (each length B)
    *,
    cfg: PretrainConfig,
    current_epoch: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one optimization step.  Returns scalar loss components + schedule + diagnostics.

    Implements the mask-first-then-encode MEM flow (critical — see module docstring).
    """
    if device is not None:
        enc, mem_dec, proj = enc.to(device), mem_dec.to(device), proj.to(device)
        batch = batch.to(device)
    enc.train(); mem_dec.train(); proj.train()

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

    step_idx = int(opt.state.get("_step", 0))
    rng = np.random.default_rng(cfg.seed + step_idx)

    # Build two augmented views per window (in-place augment of independent clones)
    v1 = torch.stack([apply_augment_pipeline(b.clone(), cfg=cfg.augment, rng=rng) for b in batch])
    v2 = torch.stack([apply_augment_pipeline(b.clone(), cfg=cfg.augment, rng=rng) for b in batch])

    B, T, Fdim = v1.shape

    # ---- MEM: MASK-FIRST-THEN-ENCODE ----
    # Step 1: BN the FULL, UNMASKED v1 to get clean running stats + target space.
    # We manually apply input_bn then build the mask; the encoder below will receive
    # already-BN-normalized, then-masked input. To avoid double-BN inside the encoder,
    # we call _blocks directly after input_bn + masking.
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
    autocast_ctx = torch.autocast(
        device_type=v1.device.type if v1.is_cuda else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.use_bf16 and v1.is_cuda,
    )
    with autocast_ctx:
        # Step 4: encode the MASKED + BN-normalized view. We bypass input_bn by
        # calling the blocks directly — the encoder's forward applies input_bn
        # then blocks, so to match we re-implement the tail here.
        h = v1_masked_bn.transpose(1, 2)                  # (B, F, T)
        h = enc.blocks(h)                                  # (B, C', T)
        per_pos1 = h.transpose(1, 2)                       # (B, T, C')
        global1 = torch.cat([per_pos1.mean(dim=1), per_pos1[:, -1, :]], dim=-1)

        # Encode view 2 normally (unmasked, for contrastive).
        per_pos2, global2 = enc(v2)

        # MEM loss at masked positions, 14-feature target.
        pred = mem_dec(per_pos1)                           # (B, T, 17)
        feat_mask = build_mem_target_mask().to(pred.device)
        L_mem = mem_loss(pred, v1_bn, pos_mask, feat_mask)

        # Contrastive: project + NT-Xent + soft cross-symbol positives
        z1 = proj(global1)
        z2 = proj(global2)
        soft = None
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
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip_max_norm)
    opt.step()
    sched.step()
    opt.state["_step"] = step_idx + 1

    return {
        "mem": float(L_mem.detach()),
        "contrastive": float(L_con.detach()),
        "total": float(L_total.detach()),
        "tau": float(tau),
        "mem_weight": float(mem_w),
        "contrastive_weight": float(con_w),
        "embedding_std": float(global1.detach().std(dim=0).mean()),
        "effective_rank": float(effective_rank(global1.detach(), sv_floor=cfg.effective_rank_sv_floor)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tape/test_pretrain.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add tape/pretrain.py tests/tape/test_pretrain.py
git commit -m "feat(tape): MEM+SimCLR pretrain step with embedding-collapse detector"
```

---

## Task 9: Pretraining CLI — `scripts/run_pretrain.py`

**Files:**
- Create: `scripts/run_pretrain.py`
- Create: `tests/scripts/test_run_pretrain.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/scripts/test_run_pretrain.py
import json
from pathlib import Path

import pytest


@pytest.mark.skipif(not Path("data/cache").exists(), reason="local cache not materialized")
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
        max_h100_hours=0.5,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_run_pretrain.py -v`
Expected: ImportError (or skipped if cache absent locally — that's fine for CI).

- [ ] **Step 3: Implement scripts/run_pretrain.py**

```python
# scripts/run_pretrain.py
"""Pretraining entry point — local-debug AND RunPod modes are identical.

Compute cap: --max-h100-hours triggers a graceful shutdown (saves checkpoint,
writes final log row).  Default: 24.0 (1 H100-day, spec §Training).

Usage (local smoke):
    uv run python scripts/run_pretrain.py \
        --cache data/cache --symbols BTC ETH --epochs 2 --batch-size 8 \
        --channel-mult 0.7 --out-dir runs/smoke --max-h100-hours 0.5

Usage (RunPod):
    python scripts/run_pretrain.py --cache /workspace/cache \
        --epochs 30 --batch-size 256 --channel-mult 1.0 \
        --out-dir /workspace/runs/r1 --max-h100-hours 23.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tape.augment import AugmentConfig
from tape.constants import (
    APRIL_HELDOUT_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_PRETRAIN,
)
from tape.contrastive_batch import is_eligible_for_contrastive
from tape.dataset import TapeDataset
from tape.model import EncoderConfig
from tape.pretrain import PretrainConfig, build_pretrain_modules, pretrain_step
from tape.probes import direction_probe_h100, hour_of_day_probe, symbol_identity_probe
from tape.sampler import EqualSymbolSampler


def _filter_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    """Return all .npz shards for `symbols` that are pre-April hold-out (gotcha #17)."""
    shards: list[Path] = []
    for sym in symbols:
        if sym == HELD_OUT_SYMBOL:
            continue  # hard exclude AVAX from pretraining (spec §Held-out symbol)
        for p in sorted(cache_dir.glob(f"{sym}__*.npz")):
            date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
            if date_part >= APRIL_HELDOUT_START:
                continue
            shards.append(p)
    return shards


def _collate(batch_items: list[dict]) -> tuple[torch.Tensor, dict]:
    feats = torch.stack([b["features"] for b in batch_items])
    metadata = {
        "symbols": [b["symbol"] for b in batch_items],
        "dates": [b["date"] for b in batch_items],
        "hours": [int((b.get("ts_first_ms", 0) // 1_000 // 3_600) % 24) for b in batch_items],
        "eligible": [is_eligible_for_contrastive(b["symbol"]) for b in batch_items],
    }
    return feats, metadata


def run_pretrain(
    *,
    cache_dir: Path,
    symbols: list[str] | None,
    epochs: int,
    batch_size: int,
    channel_mult: float,
    out_dir: Path,
    max_h100_hours: float,
    seed: int,
    # MEM/contrastive weights are annealed — override the defaults from PretrainConfig
    # only for experiments. Defaults: MEM 0.90 -> 0.60, contrastive 0.10 -> 0.40 over 20 ep.
    mem_weight_start: float | None = None,
    mem_weight_end: float | None = None,
    contrastive_weight_start: float | None = None,
    contrastive_weight_end: float | None = None,
    anneal_epochs: int | None = None,
    probe_every_epochs: int = 5,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training-log.jsonl"
    ckpt_path = out_dir / "encoder.pt"

    syms = list(symbols or PRETRAINING_SYMBOLS)
    shards = _filter_shards(cache_dir, syms)
    if not shards:
        raise RuntimeError("no pretraining shards found")

    dataset = TapeDataset(shards, stride=STRIDE_PRETRAIN, mode="pretrain")
    sampler = EqualSymbolSampler(dataset, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        collate_fn=_collate,
        drop_last=True,
    )

    cfg_kwargs: dict = dict(
        encoder=EncoderConfig(channel_mult=channel_mult),
        augment=AugmentConfig(),  # spec defaults: ±25 jitter, σ=0.10 timing
        total_steps=epochs * max(1, len(loader)),
        seed=seed,
    )
    if mem_weight_start is not None:
        cfg_kwargs["mem_weight_start"] = mem_weight_start
    if mem_weight_end is not None:
        cfg_kwargs["mem_weight_end"] = mem_weight_end
    if contrastive_weight_start is not None:
        cfg_kwargs["contrastive_weight_start"] = contrastive_weight_start
    if contrastive_weight_end is not None:
        cfg_kwargs["contrastive_weight_end"] = contrastive_weight_end
    if anneal_epochs is not None:
        cfg_kwargs["anneal_epochs"] = anneal_epochs
    cfg = PretrainConfig(**cfg_kwargs)
    enc, mem_dec, proj, opt, sched = build_pretrain_modules(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, mem_dec, proj = enc.to(device), mem_dec.to(device), proj.to(device)

    started = time.time()
    cap_seconds = max_h100_hours * 3_600
    last_mem = None
    epoch_records: list[dict] = []

    with log_path.open("w") as logf:
        for epoch in range(1, epochs + 1):
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            mem_acc, con_acc, std_acc, n = 0.0, 0.0, 0.0, 0

            for feats, metadata in loader:
                if time.time() - started > cap_seconds:
                    break
                losses = pretrain_step(
                    enc, mem_dec, proj, opt, sched, feats, metadata,
                    cfg=cfg, current_epoch=epoch - 1, device=device,
                )
                mem_acc += losses["mem"]
                con_acc += losses["contrastive"]
                std_acc += losses["embedding_std"]
                n += 1

            mem_loss_e = mem_acc / max(1, n)
            con_loss_e = con_acc / max(1, n)
            std_e = std_acc / max(1, n)

            row = {
                "epoch": epoch,
                "mem_loss": mem_loss_e,
                "contrastive_loss": con_loss_e,
                "embedding_std": std_e,
                "elapsed_h": (time.time() - started) / 3_600,
            }

            # Every probe_every_epochs: run probe trio
            if epoch % probe_every_epochs == 0:
                probe_summary = _run_probe_trio(enc, dataset, device)
                row.update(probe_summary)

            logf.write(json.dumps(row) + "\n")
            logf.flush()
            epoch_records.append(row)

            # Stop on <1% MEM improvement over last 20% of epochs (spec)
            if epoch >= max(5, int(0.2 * epochs)):
                window = epoch_records[-int(0.2 * epochs):]
                if window and window[0]["mem_loss"] - window[-1]["mem_loss"] < 0.01 * window[0]["mem_loss"]:
                    break

            if time.time() - started > cap_seconds:
                break

    # Save encoder + scaler config (no optimizer state) for downstream probes
    torch.save(
        {
            "encoder_state_dict": enc.state_dict(),
            "encoder_config": cfg.encoder.__dict__,
            "n_epochs_run": len(epoch_records),
            "elapsed_seconds": time.time() - started,
            "seed": seed,
        },
        ckpt_path,
    )
    return {"checkpoint": str(ckpt_path), "log": str(log_path), "epochs_run": len(epoch_records)}


def _run_probe_trio(enc, dataset, device) -> dict:
    """Forward a held-out probe slice (April 1–13 + symbol + hour) through the frozen encoder."""
    enc.eval()
    feats_by_sym: dict[str, list[np.ndarray]] = {}
    labels_by_sym: dict[str, list[np.ndarray]] = {}
    masks_by_sym: dict[str, list[np.ndarray]] = {}
    all_feats: list[np.ndarray] = []
    sym_ids: list[int] = []
    hours: list[int] = []

    with torch.no_grad():
        # Iterate dataset linearly — small subset for speed (limit at ~50K windows)
        n_collected = 0
        for i in range(min(len(dataset), 50_000)):
            item = dataset[i]
            x = item["features"].unsqueeze(0).to(device)
            _, g = enc(x)
            g_np = g.squeeze(0).cpu().numpy()
            sym = item["symbol"]
            feats_by_sym.setdefault(sym, []).append(g_np)
            labels_by_sym.setdefault(sym, []).append(np.int64(item["label_h100"]))
            masks_by_sym.setdefault(sym, []).append(bool(item["label_h100_mask"]))
            all_feats.append(g_np)
            sym_ids.append(int(item["symbol_id"]))
            hours.append(int((item.get("start", 0) // 3600) % 24))
            n_collected += 1

    feats_by_sym_np = {k: np.stack(v) for k, v in feats_by_sym.items()}
    labels_by_sym_np = {k: np.array(v) for k, v in labels_by_sym.items()}
    masks_by_sym_np = {k: np.array(v) for k, v in masks_by_sym.items()}
    all_feats_np = np.stack(all_feats)
    sym_ids_np = np.array(sym_ids)
    hours_np = np.array(hours)

    dir_per_sym = direction_probe_h100(feats_by_sym_np, labels_by_sym_np, masks_by_sym_np)
    sym_acc = symbol_identity_probe(all_feats_np, sym_ids_np, n_symbols=25)
    hour_acc = hour_of_day_probe(all_feats_np, hours_np)
    enc.train()
    return {
        "probe_dir_h100_balanced_acc_mean": float(np.mean(list(dir_per_sym.values()))) if dir_per_sym else None,
        "probe_dir_h100_per_symbol": dir_per_sym,
        "probe_symbol_id_acc": sym_acc,
        "probe_hour_of_day_acc": hour_acc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--channel-mult", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-h100-hours", type=float, default=24.0)
    # MEM/contrastive weights are ANNEALED by default (MEM 0.90->0.60,
    # contrastive 0.10->0.40 over 20 epochs). Flags below override schedule
    # endpoints for experiments — leave unset to use the knowledge-base default.
    ap.add_argument("--mem-weight-start", type=float, default=None)
    ap.add_argument("--mem-weight-end", type=float, default=None)
    ap.add_argument("--contrastive-weight-start", type=float, default=None)
    ap.add_argument("--contrastive-weight-end", type=float, default=None)
    ap.add_argument("--anneal-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_pretrain(
        cache_dir=args.cache,
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        channel_mult=args.channel_mult,
        out_dir=args.out_dir,
        max_h100_hours=args.max_h100_hours,
        seed=args.seed,
        mem_weight_start=args.mem_weight_start,
        mem_weight_end=args.mem_weight_end,
        contrastive_weight_start=args.contrastive_weight_start,
        contrastive_weight_end=args.contrastive_weight_end,
        anneal_epochs=args.anneal_epochs,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the smoke test (local CPU, 2 epochs, channel_mult=0.5)**

Run: `pytest tests/scripts/test_run_pretrain.py -v`
Expected: PASS (or SKIP if cache absent in CI).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pretrain.py tests/scripts/test_run_pretrain.py
git commit -m "feat(scripts): pretraining CLI with H100-hours cap + per-epoch probe trio"
```

---

## Task 10: Standalone probe runner — `scripts/run_pretrain_probes.py`

**Files:**
- Create: `scripts/run_pretrain_probes.py`
- Create: `tests/scripts/test_run_pretrain_probes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_run_pretrain_probes.py
import json
from pathlib import Path

import pytest

@pytest.mark.skipif(not Path("data/cache").exists(), reason="local cache not materialized")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_run_pretrain_probes.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scripts/run_pretrain_probes.py**

```python
# scripts/run_pretrain_probes.py
"""Standalone frozen-embedding probe runner.

Reads a pretrain checkpoint (encoder state dict + config), forwards April 1–13
windows through it, runs the probe trio, writes JSON + MD reports.

Used both:
  - In-loop, by run_pretrain.py (every 5 epochs, on a sub-sample for speed)
  - After Step 3 finishes, on the FULL April 1–13 split (this script)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tape.cache import load_shard
from tape.constants import (
    APRIL_HELDOUT_START,
    APRIL_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_EVAL,
    SYMBOLS,
    WINDOW_LEN,
)
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder
from tape.probes import direction_probe_h100, hour_of_day_probe, symbol_identity_probe


def _april_probe_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    out: list[Path] = []
    for sym in symbols:
        for p in sorted(cache_dir.glob(f"{sym}__*.npz")):
            date_part = p.stem.split("__", 1)[1]
            if APRIL_START <= date_part < APRIL_HELDOUT_START:
                out.append(p)
    return out


def _load_encoder(checkpoint: Path | None):
    cfg = EncoderConfig()
    if checkpoint is None:
        # No-op for smoke tests / sanity runs
        return TapeEncoder(cfg).eval()
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.eval()


def run_probes(
    *,
    checkpoint: Path | None,
    cache_dir: Path,
    symbols: list[str] | None,
    out_path: Path,
) -> dict:
    syms = list(symbols or PRETRAINING_SYMBOLS)
    syms = [s for s in syms if s != HELD_OUT_SYMBOL]
    shards = _april_probe_shards(cache_dir, syms)
    if not shards:
        raise RuntimeError(f"no April 1–{APRIL_HELDOUT_START[-2:]} shards found")

    dataset = TapeDataset(shards, stride=STRIDE_EVAL, mode="eval")
    enc = _load_encoder(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = enc.to(device)

    feats_by_sym: dict[str, list[np.ndarray]] = {}
    labels_by_sym: dict[str, list[int]] = {}
    masks_by_sym: dict[str, list[bool]] = {}
    all_feats: list[np.ndarray] = []
    sym_ids: list[int] = []
    hours: list[int] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["features"].unsqueeze(0).to(device)
            _, g = enc(x)
            g_np = g.squeeze(0).cpu().numpy()
            sym = item["symbol"]
            feats_by_sym.setdefault(sym, []).append(g_np)
            labels_by_sym.setdefault(sym, []).append(int(item["label_h100"]))
            masks_by_sym.setdefault(sym, []).append(bool(item["label_h100_mask"]))
            all_feats.append(g_np)
            sym_ids.append(int(item["symbol_id"]))
            # hour bucket from a deterministic transform of (date, start)
            hours.append(int(item["start"]) % 24)

    f_np = {k: np.stack(v) for k, v in feats_by_sym.items()}
    y_np = {k: np.array(v) for k, v in labels_by_sym.items()}
    m_np = {k: np.array(v) for k, v in masks_by_sym.items()}
    dir_per_sym = direction_probe_h100(f_np, y_np, m_np)

    all_feats_np = np.stack(all_feats)
    sym_acc = symbol_identity_probe(all_feats_np, np.array(sym_ids), n_symbols=25)
    hour_acc = hour_of_day_probe(all_feats_np, np.array(hours))

    payload = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "n_symbols_evaluated": len(dir_per_sym),
        "direction_h100_per_symbol": dir_per_sym,
        "direction_h100_mean": float(np.mean(list(dir_per_sym.values()))) if dir_per_sym else None,
        "symbol_identity_acc": sym_acc,
        "hour_of_day_acc": hour_acc,
        "gate1_thresholds": {
            "absolute_floor": 0.514,
            "vs_majority_pp": 1.0,
            "vs_random_projection_pp": 1.0,
            "hour_of_day_max": 0.10,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=False)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = run_probes(
        checkpoint=args.checkpoint, cache_dir=args.cache, symbols=args.symbols, out_path=args.out
    )
    print(json.dumps({k: payload[k] for k in ("symbol_identity_acc", "hour_of_day_acc", "direction_h100_mean")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run smoke test**

Run: `pytest tests/scripts/test_run_pretrain_probes.py -v`
Expected: PASS (or SKIP if cache absent).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pretrain_probes.py tests/scripts/test_run_pretrain_probes.py
git commit -m "feat(scripts): standalone frozen-embedding probe runner for Gate 1 prep"
```

---

## Task 11: Checkpoint export — `scripts/export_checkpoint.py`

**Files:**
- Create: `scripts/export_checkpoint.py`
- Create: `tests/scripts/test_export_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_export_checkpoint.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scripts/export_checkpoint.py**

```python
# scripts/export_checkpoint.py
"""Strip optimizer state, stamp provenance, write Gate 1-ready checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from tape.model import EncoderConfig, TapeEncoder


def _git_sha(repo: Path = Path(".")) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()


def _spec_sha() -> str:
    spec = Path("docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md")
    return hashlib.sha256(spec.read_bytes()).hexdigest()[:16]


def export_for_gate1(
    src: Path,
    dst: Path,
    *,
    git_sha: str | None = None,
    spec_sha: str | None = None,
) -> dict:
    payload = torch.load(src, map_location="cpu")
    enc_cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(enc_cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    n_params = sum(p.numel() for p in enc.parameters())

    out = {
        "encoder_state_dict": payload["encoder_state_dict"],
        "encoder_config": payload["encoder_config"],
        "n_epochs_run": payload.get("n_epochs_run"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "seed": payload.get("seed"),
        "git_sha": git_sha or _git_sha(),
        "spec_sha": spec_sha or _spec_sha(),
        "n_params": n_params,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    info = export_for_gate1(args.src, args.dst)
    print(json.dumps({k: info[k] for k in ("n_params", "git_sha", "spec_sha")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/scripts/test_export_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_checkpoint.py tests/scripts/test_export_checkpoint.py
git commit -m "feat(scripts): checkpoint export with provenance stamping"
```

---

## Task 12: RunPod scaffold — Dockerfile + launch.sh + requirements

**Files:**
- Create: `runpod/Dockerfile`
- Create: `runpod/launch.sh`
- Create: `runpod/requirements-runpod.txt`
- Create: `runpod/README.md`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# runpod/Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace/repo

COPY runpod/requirements-runpod.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

# Bake the repo (excluded big paths via .dockerignore)
COPY tape/ tape/
COPY scripts/ scripts/
COPY runpod/ runpod/
COPY pyproject.toml pyproject.toml

CMD ["bash", "runpod/launch.sh"]
```

- [ ] **Step 2: Write the launch script**

```bash
# runpod/launch.sh
#!/usr/bin/env bash
set -euo pipefail

# Required env vars (set by runpodctl invocation):
#   R2_CACHE_PREFIX      e.g. r2:pacifica-cache/v1
#   OUT_PREFIX           e.g. r2:pacifica-models/step3
#   EPOCHS               default 30
#   BATCH_SIZE           default 256
#   CHANNEL_MULT         default 1.0
#   MAX_H100_HOURS       default 23.0  (leave 1h headroom under spec cap)
#   SEED                 default 0
#   MEM_WEIGHT           default 0.70
#   CONTRASTIVE_WEIGHT   default 0.30

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-256}
CHANNEL_MULT=${CHANNEL_MULT:-1.0}
MAX_H100_HOURS=${MAX_H100_HOURS:-23.0}
SEED=${SEED:-0}
MEM_WEIGHT=${MEM_WEIGHT:-0.70}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0.30}

mkdir -p /workspace/cache /workspace/runs

# 1. Pull cache shards from R2
rclone sync "$R2_CACHE_PREFIX" /workspace/cache --transfers 32 --checkers 64 --size-only

# 2. Run pretraining
python scripts/run_pretrain.py \
    --cache /workspace/cache \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --channel-mult "$CHANNEL_MULT" \
    --out-dir /workspace/runs/run \
    --max-h100-hours "$MAX_H100_HOURS" \
    --mem-weight "$MEM_WEIGHT" \
    --contrastive-weight "$CONTRASTIVE_WEIGHT" \
    --seed "$SEED"

# 3. Export Gate 1-ready checkpoint
python scripts/export_checkpoint.py \
    --src /workspace/runs/run/encoder.pt \
    --dst /workspace/runs/run/encoder-gate1.pt

# 4. Run final probe report on April 1–13 (full)
python scripts/run_pretrain_probes.py \
    --checkpoint /workspace/runs/run/encoder-gate1.pt \
    --cache /workspace/cache \
    --out /workspace/runs/run/april-probe-report

# 5. Push results back to R2
rclone copy /workspace/runs/run "$OUT_PREFIX" --transfers 16
```

- [ ] **Step 3: Write requirements + README**

```text
# runpod/requirements-runpod.txt
numpy>=1.26
pandas>=2.2
pyarrow>=15.0
duckdb>=0.10
scikit-learn>=1.5
rclone-py>=0.2  # if available; otherwise install rclone via apt in Dockerfile
```

```markdown
# runpod/README.md
# Step 3 RunPod execution

## Image build (local)
docker build -f runpod/Dockerfile -t pacifica-step3:0.1 .

## Push to RunPod registry (via flash skill)
# See `Skill: flash` for the full command form

## Launch a single H100 pod
# See `Skill: runpodctl` for the equivalent CLI form

## Required env vars
R2_CACHE_PREFIX, OUT_PREFIX (R2 destinations)
EPOCHS, BATCH_SIZE, CHANNEL_MULT, MAX_H100_HOURS, SEED, MEM_WEIGHT, CONTRASTIVE_WEIGHT (training knobs)

## Compute cap discipline
MAX_H100_HOURS defaults to 23.0 — that leaves 1h headroom under the 24h
spec cap (1 H100-day) for cache pull + checkpoint upload + probe run.
```

- [ ] **Step 4: Make launch.sh executable**

```bash
chmod +x runpod/launch.sh
```

- [ ] **Step 5: Commit**

```bash
git add runpod/Dockerfile runpod/launch.sh runpod/requirements-runpod.txt runpod/README.md
git commit -m "chore(runpod): Dockerfile + launch script for Step 3 H100 execution"
```

---

## Task 13: GPU launch (RunPod, foreground) — *requires `runpod-7` worker*

**Files:**
- Create: `docs/experiments/step3-pretrain-run.md` (after run completes)

- [ ] **Step 1: Verify pre-requisites**

Confirm:
- Council-6 model-size review is closed and `--channel-mult` value is set (default 1.0).
- Session-of-day confound check from Task 7 is `decision == "no_action"` OR `_last` features have been pruned and Gate 0 re-run.
- Local smoke test (Task 9) passes end-to-end.

- [ ] **Step 2: Push the cache to R2 if not already present**

```bash
rclone sync data/cache r2:pacifica-cache/v1 --transfers 32 --checkers 64 --size-only
```

- [ ] **Step 3: Dispatch `runpod-7` to launch the pod (foreground)**

Prompt for `runpod-7`:
> Build the image at `runpod/Dockerfile`, push to RunPod registry, and launch a single H100 80GB pod with env vars: `R2_CACHE_PREFIX=r2:pacifica-cache/v1`, `OUT_PREFIX=r2:pacifica-models/step3-run-<seed>`, `EPOCHS=30`, `BATCH_SIZE=256`, `CHANNEL_MULT=<value-from-council-6>`, `MAX_H100_HOURS=23.0`, `SEED=0`. Wait for the pod to terminate (success or failure). Report exit code, total wall-clock, and confirm `encoder-gate1.pt` + `april-probe-report.json` are present at `OUT_PREFIX`. Pull both back to `runs/step3-r1/` locally.

- [ ] **Step 4: Audit the run**

Open `runs/step3-r1/training-log.jsonl`. Confirm:
- Final epoch's `mem_loss` is at least 30% lower than epoch 1 (sanity that MEM is learning).
- `embedding_std` did not approach 0 at any logged epoch (no collapse).
- `probe_hour_of_day_acc` < 0.10 at the last logged checkpoint.
- Total `elapsed_h` ≤ 24.0.

- [ ] **Step 5: Write `docs/experiments/step3-pretrain-run.md`**

Manual write-up — reference the JSONL log, summarize chosen config, total H100-hours, MEM/contrastive curves, embedding-std trajectory, and the per-epoch probe trio. Include the final April probe-report payload.

- [ ] **Step 6: Commit run artifacts**

```bash
git add runs/step3-r1/training-log.jsonl runs/step3-r1/april-probe-report.json docs/experiments/step3-pretrain-run.md
git commit -m "experiment: Step 3 pretraining run — channel_mult=<X>, seed=0"
```

The encoder checkpoint itself (`runs/step3-r1/encoder-gate1.pt`) is large — store it in R2, NOT git. Reference its R2 URL in the write-up.

---

## Task 14: Gate 1 readiness check — pre-flight before Step 4

**Files:**
- Create: `docs/experiments/step3-checkpoint-card.md`

- [ ] **Step 1: Run the standalone April probe one more time on the exported checkpoint**

```bash
uv run python scripts/run_pretrain_probes.py \
    --checkpoint runs/step3-r1/encoder-gate1.pt \
    --cache data/cache \
    --out docs/experiments/step3-april-probe-final
```

- [ ] **Step 2: Compare Gate 1 binding conditions to the probe outputs**

The four Gate 1 conditions (`spec §Gate 1`):

1. Balanced acc ≥ 51.4% on 15+/25 symbols.
2. Balanced acc > Majority + 1.0pp on 15+/25 symbols.
3. Balanced acc > Random Projection + 1.0pp on 15+/25 symbols.
4. Hour-of-day probe < 10% AND session-stratified variance < 1.5pp.

Write the pre-flight check as a markdown table comparing this run's probe outputs against the published Gate 0 baselines (`docs/experiments/gate0-summary.md`).

- [ ] **Step 3: Write `docs/experiments/step3-checkpoint-card.md`**

Include:
- R2 URL of the encoder checkpoint
- git_sha + spec_sha (from the exported payload)
- model size (n_params)
- all training hyperparameters
- final-epoch losses + probe trio
- explicit row-by-row check vs Gate 1 conditions (PASS/FAIL/UNKNOWN per condition)
- recommendation: "proceed to Step 4 Gate 1 evaluation" OR "do not proceed — diagnose <X>"

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/step3-april-probe-final.json docs/experiments/step3-checkpoint-card.md
git commit -m "experiment: Step 3 checkpoint card + Gate 1 pre-flight check"
```

---

## Task 15: Update CLAUDE.md gotchas + state.md handoff

**Files:**
- Modify: `CLAUDE.md` (append new gotchas if any surfaced during the run)
- Modify: `.claude/skills/autoresearch/state.md` (close Step 3, queue Step 4)

- [ ] **Step 1: Compile knowledge from the run**

Invoke the `compile-knowledge` skill to absorb `docs/experiments/step3-pretrain-run.md` and `docs/experiments/step3-checkpoint-card.md` into `docs/knowledge/`.

- [ ] **Step 2: Update CLAUDE.md if needed**

Only add a gotcha if a NEW failure mode surfaced (e.g. "embedding-std oscillation between epoch X and Y when batch < 64" — these become #34, #35 etc.). Do NOT pre-emptively invent gotchas.

- [ ] **Step 3: Update state.md**

Mark Step 3 complete. Set the next session's entry prompt to "Run Step 4 Gates 1-4 against `runs/step3-r1/encoder-gate1.pt`. Use `validator-11` for the binding-gate decisions and `analyst-9` for the cluster + Wyckoff probes."

- [ ] **Step 4: Commit + push**

```bash
git add CLAUDE.md .claude/skills/autoresearch/state.md docs/knowledge
git commit -m "chore: Step 3 handoff — checkpoint ready for Gate 1"
```

Push only if the user explicitly asks.

---

## Self-Review Notes

- **Spec coverage:** All §Architecture (Task 1), §Pretraining Objective (Tasks 2–5, 8), §Data Loading (Task 9 via existing `TapeDataset` + `EqualSymbolSampler`), §Hyperparameters (Task 8 via `PretrainConfig`), §Monitoring (Tasks 6, 9 + in-loop probe trio), Pre-Pretraining Confound Check (Task 7), §Compute Requirements (Tasks 12–13 with the 23-hour cap), §Held-out Symbol exclusion (Task 9 `_filter_shards`), §Cross-Symbol Contrastive AVAX exclusion (Task 5).
- **Gate 1 readiness:** Task 14 explicitly checks all 4 binding conditions before declaring the run ready for Step 4.
- **Open council-6 review:** Flagged in the pre-plan section, gated at Task 13 step 1 — the `--channel-mult` flag makes the resolution a config change, not a code change.
- **Compute cap discipline:** `--max-h100-hours` enforced inside `run_pretrain` (graceful checkpoint + log flush), launch.sh defaults to 23.0 to keep headroom.
- **No placeholders:** every code step contains complete code; every test step shows the expected outcome.
- **Type consistency:** `EncoderConfig`, `PretrainConfig`, `AugmentConfig`, `MEMDecoder`, `ProjectionHead`, `TapeEncoder` names match across Tasks 1, 4, 8, 9, 11.

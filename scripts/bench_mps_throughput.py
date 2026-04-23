# scripts/bench_mps_throughput.py
"""Measure MPS training throughput for the Step 3 pretraining loop.

Runs a short: build → warmup → measured loop → report. Uses the REAL data
pipeline (TapeDataset + EqualSymbolSampler + DataLoader) and the REAL
pretrain_step so numbers reflect production conditions.

Usage:
    uv run python scripts/bench_mps_throughput.py \
        --cache data/cache --batch-size 256 --warmup 5 --measured 30
"""

from __future__ import annotations

import argparse
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
from tape.sampler import EqualSymbolSampler


def _filter_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    shards: list[Path] = []
    for sym in symbols:
        if sym == HELD_OUT_SYMBOL:
            continue
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
        "hours": [
            int((b.get("ts_first_ms", 0) // 1_000 // 3_600) % 24) for b in batch_items
        ],
        "eligible": [is_eligible_for_contrastive(b["symbol"]) for b in batch_items],
    }
    return feats, metadata


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--channel-mult", type=float, default=1.0)
    ap.add_argument(
        "--warmup", type=int, default=5, help="Warmup steps (discarded in timing)"
    )
    ap.add_argument(
        "--measured", type=int, default=30, help="Measured steps for steps/sec"
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs-target", type=int, default=30)
    args = ap.parse_args()

    device = _pick_device()
    print(f"[bench] device = {device}")
    if device.type == "mps":
        # Emit warning if any op silently falls back to CPU.
        import os

        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    shards = _filter_shards(args.cache, list(PRETRAINING_SYMBOLS))
    print(f"[bench] shards = {len(shards)}")
    dataset = TapeDataset(shards, stride=STRIDE_PRETRAIN, mode="pretrain")
    print(f"[bench] dataset windows = {len(dataset):,}")

    sampler = EqualSymbolSampler(dataset, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    print(f"[bench] steps_per_epoch = {steps_per_epoch:,}")

    cfg = PretrainConfig(
        encoder=EncoderConfig(channel_mult=args.channel_mult),
        augment=AugmentConfig(),
        total_steps=args.epochs_target * max(1, steps_per_epoch),
        seed=0,
    )
    if device.type != "cuda":
        cfg.use_torch_compile = False

    enc, mem_dec, proj, opt, sched = build_pretrain_modules(cfg)
    enc, mem_dec, proj = enc.to(device), mem_dec.to(device), proj.to(device)

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[bench] encoder params = {n_params:,}")

    dataset.set_epoch(1)
    sampler.set_epoch(1)

    loader_iter = iter(loader)
    losses_log: list[dict] = []

    # Warmup
    print(f"[bench] --- warmup ({args.warmup} steps) ---")
    t0 = time.time()
    for i in range(args.warmup):
        feats, metadata = next(loader_iter)
        losses = pretrain_step(
            enc,
            mem_dec,
            proj,
            opt,
            sched,
            feats,
            metadata,
            cfg=cfg,
            current_epoch=0,
            device=device,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        print(
            f"  warmup step {i+1}: mem={losses['mem']:.3f} con={losses['contrastive']:.3f} std={losses['embedding_std']:.3f}"
        )
    print(f"[bench] warmup elapsed = {time.time() - t0:.2f}s")

    # Measured loop
    print(f"[bench] --- measured ({args.measured} steps) ---")
    t_start = time.time()
    step_times: list[float] = []
    for i in range(args.measured):
        t_step = time.time()
        feats, metadata = next(loader_iter)
        losses = pretrain_step(
            enc,
            mem_dec,
            proj,
            opt,
            sched,
            feats,
            metadata,
            cfg=cfg,
            current_epoch=0,
            device=device,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        step_times.append(time.time() - t_step)
        losses_log.append(losses)
    t_elapsed = time.time() - t_start

    step_times_np = np.array(step_times)
    steps_per_sec = args.measured / t_elapsed
    sec_per_epoch = steps_per_epoch / steps_per_sec
    total_hours = (steps_per_epoch * args.epochs_target) / steps_per_sec / 3600

    print("\n=== BENCHMARK REPORT ===")
    print(f"device                  = {device}")
    print(f"batch_size              = {args.batch_size}")
    print(f"channel_mult            = {args.channel_mult}")
    print(f"encoder params          = {n_params:,}")
    print(f"dataset windows         = {len(dataset):,}")
    print(f"steps per epoch         = {steps_per_epoch:,}")
    print(f"measured steps          = {args.measured}")
    print(f"elapsed (measured)      = {t_elapsed:.2f}s")
    print(f"steps/sec               = {steps_per_sec:.2f}")
    print(f"sec per step (mean)     = {step_times_np.mean()*1000:.1f}ms")
    print(f"sec per step (p50)      = {np.percentile(step_times_np, 50)*1000:.1f}ms")
    print(f"sec per step (p95)      = {np.percentile(step_times_np, 95)*1000:.1f}ms")
    print(
        f"sec per epoch (est)     = {sec_per_epoch:.1f}s  ({sec_per_epoch/60:.1f} min)"
    )
    print(f"hours for {args.epochs_target} epochs    = {total_hours:.2f}h")

    # Sanity: MEM loss should be non-trivial and descending
    mem_first = losses_log[0]["mem"]
    mem_last = losses_log[-1]["mem"]
    print(
        f"MEM loss: first={mem_first:.4f}  last={mem_last:.4f}  delta={mem_last - mem_first:+.4f}"
    )

    emb_stds = [d["embedding_std"] for d in losses_log]
    print(
        f"embedding_std: min={min(emb_stds):.4f} mean={np.mean(emb_stds):.4f} max={max(emb_stds):.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

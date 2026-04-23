# scripts/run_pretrain.py
"""Pretraining entry point — local (CPU/MPS) AND RunPod (CUDA) modes are identical.

Compute cap: --max-hours triggers a graceful shutdown (saves checkpoint,
writes final log row). Default: 24.0 (hardware-agnostic wall-clock).

The legacy --max-h100-hours flag is accepted as a deprecated alias.

Usage (local smoke):
    uv run python scripts/run_pretrain.py \
        --cache data/cache --symbols BTC ETH --epochs 2 --batch-size 8 \
        --channel-mult 0.7 --out-dir runs/smoke --max-hours 0.5

Usage (RunPod H100):
    python scripts/run_pretrain.py --cache /workspace/cache \
        --epochs 30 --batch-size 256 --channel-mult 1.0 \
        --out-dir /workspace/runs/r1 --max-hours 23.0

Usage (local MPS, M-series Mac):
    caffeinate -i uv run python scripts/run_pretrain.py \
        --cache data/cache --epochs 30 --batch-size 256 --channel-mult 1.0 \
        --out-dir runs/step3-r1 --max-hours 10.0
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
        "hours": [
            int((b.get("ts_first_ms", 0) // 1_000 // 3_600) % 24) for b in batch_items
        ],
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
    max_hours: float,
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

    # Device selection: prefer CUDA, then MPS (Apple Silicon), else CPU.
    # torch.compile(mode="reduce-overhead") triggers CUDA graphs, so disable on
    # non-CUDA. bf16 autocast in tape/pretrain.py is already CUDA-guarded — no
    # change needed there.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        cfg.use_torch_compile = False
    else:
        device = torch.device("cpu")
        cfg.use_torch_compile = False

    enc, mem_dec, proj, opt, sched = build_pretrain_modules(cfg)
    enc, mem_dec, proj = enc.to(device), mem_dec.to(device), proj.to(device)

    started = time.time()
    cap_seconds = max_hours * 3_600
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
                    enc,
                    mem_dec,
                    proj,
                    opt,
                    sched,
                    feats,
                    metadata,
                    cfg=cfg,
                    current_epoch=epoch - 1,
                    device=device,
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
                window = epoch_records[-int(0.2 * epochs) :]
                if (
                    window
                    and window[0]["mem_loss"] - window[-1]["mem_loss"]
                    < 0.01 * window[0]["mem_loss"]
                ):
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
    return {
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "epochs_run": len(epoch_records),
    }


def _run_probe_trio(enc, dataset: TapeDataset, device: torch.device) -> dict:
    """Forward a held-out probe slice (April 1–13 + symbol + hour) through the frozen encoder."""
    enc.eval()
    feats_by_sym: dict[str, list[np.ndarray]] = {}
    labels_by_sym: dict[str, list[int]] = {}
    masks_by_sym: dict[str, list[bool]] = {}
    all_feats: list[np.ndarray] = []
    sym_ids: list[int] = []
    hours: list[int] = []

    with torch.no_grad():
        # Iterate dataset linearly — small subset for speed (limit at ~50K windows)
        for i in range(min(len(dataset), 50_000)):
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
            hours.append(int((item.get("start", 0) // 3600) % 24))

    feats_by_sym_np = {k: np.stack(v) for k, v in feats_by_sym.items()}
    labels_by_sym_np = {k: np.array(v) for k, v in labels_by_sym.items()}
    masks_by_sym_np = {k: np.array(v) for k, v in masks_by_sym.items()}
    all_feats_np = np.stack(all_feats)
    sym_ids_np = np.array(sym_ids)
    hours_np = np.array(hours)

    dir_per_sym = direction_probe_h100(
        feats_by_sym_np, labels_by_sym_np, masks_by_sym_np
    )
    sym_acc = symbol_identity_probe(all_feats_np, sym_ids_np, n_symbols=25)
    hour_acc = hour_of_day_probe(all_feats_np, hours_np)
    enc.train()
    return {
        "probe_dir_h100_balanced_acc_mean": (
            float(np.mean(list(dir_per_sym.values()))) if dir_per_sym else None
        ),
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
    # Primary wall-clock cap (hardware-agnostic). --max-h100-hours is a legacy
    # alias kept for backward compatibility with runpod/launch.sh.
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument(
        "--max-h100-hours",
        type=float,
        default=None,
        help="DEPRECATED alias for --max-hours (retained for RunPod launch.sh back-compat)",
    )
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

    # Resolve wall-clock cap. Prefer --max-hours; fall back to deprecated
    # --max-h100-hours with a warning; default 24.0 if neither is set.
    if args.max_hours is not None and args.max_h100_hours is not None:
        ap.error("pass --max-hours OR --max-h100-hours, not both")
    if args.max_hours is not None:
        max_hours = args.max_hours
    elif args.max_h100_hours is not None:
        import warnings

        warnings.warn(
            "--max-h100-hours is deprecated; use --max-hours (hardware-agnostic)",
            DeprecationWarning,
            stacklevel=2,
        )
        max_hours = args.max_h100_hours
    else:
        max_hours = 24.0

    res = run_pretrain(
        cache_dir=args.cache,
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        channel_mult=args.channel_mult,
        out_dir=args.out_dir,
        max_hours=max_hours,
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

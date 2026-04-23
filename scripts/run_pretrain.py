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


def _filter_shards(
    cache_dir: Path,
    symbols: list[str],
    *,
    train_end_date: str = APRIL_HELDOUT_START,
) -> list[Path]:
    """Return all .npz shards for `symbols` with date strictly less than train_end_date.

    train_end_date defaults to APRIL_HELDOUT_START (2026-04-14) — matches the
    original spec's train/test split. Pass an earlier ISO date to carve out a
    held-out period from the tail of the pre-April data (e.g. "2026-02-01"
    holds out Feb-Mar for evaluation after council 2026-04-23 diagnostics
    showed April is terminally underpowered at stride=200 eval).
    """
    shards: list[Path] = []
    for sym in symbols:
        if sym == HELD_OUT_SYMBOL:
            continue  # hard exclude AVAX from pretraining (spec §Held-out symbol)
        for p in sorted(cache_dir.glob(f"{sym}__*.npz")):
            date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
            if date_part >= train_end_date:
                continue
            # Also enforce the hard April hold-out guard regardless of cutoff.
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
    train_end_date: str = APRIL_HELDOUT_START,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training-log.jsonl"
    ckpt_path = out_dir / "encoder.pt"

    syms = list(symbols or PRETRAINING_SYMBOLS)
    shards = _filter_shards(cache_dir, syms, train_end_date=train_end_date)
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
    best_mem_loss = float("inf")
    best_epoch = 0
    ckpt_best_path = out_dir / "encoder-best.pt"

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

            # Best-val checkpoint: save encoder-best.pt on MEM-loss improvement.
            # Run-0 (2026-04-23) stopped early at epoch 8 with MEM=0.74; the
            # epoch-5 MEM-minimum (0.51) encoder was never saved and was lost.
            improved = mem_loss_e < best_mem_loss
            if improved:
                best_mem_loss = mem_loss_e
                best_epoch = epoch
                torch.save(
                    {
                        "encoder_state_dict": enc.state_dict(),
                        "encoder_config": cfg.encoder.__dict__,
                        "epoch": epoch,
                        "mem_loss": mem_loss_e,
                        "seed": seed,
                    },
                    ckpt_best_path,
                )

            row = {
                "epoch": epoch,
                "mem_loss": mem_loss_e,
                "contrastive_loss": con_loss_e,
                "embedding_std": std_e,
                "elapsed_h": (time.time() - started) / 3_600,
                "best_mem_loss": best_mem_loss,
                "best_epoch": best_epoch,
                "saved_best": improved,
            }

            # Every probe_every_epochs: run probe trio
            if epoch % probe_every_epochs == 0:
                probe_summary = _run_probe_trio(enc, dataset, device)
                row.update(probe_summary)

            # Early-stop plateau warning — LOGGED ONLY, no break (council-5
            # diagnosis 2026-04-23: run-0 killed at epoch 8 of 30 because MEM
            # natural mid-training regression around LR peak triggered the
            # <1% improvement rule. Encoder was still learning real signal —
            # preserving this as a diagnostic flag, not a stop condition).
            if epoch >= max(5, int(0.2 * epochs)):
                window = epoch_records[-int(0.2 * epochs) :]
                if (
                    window
                    and window[0]["mem_loss"] - window[-1]["mem_loss"]
                    < 0.01 * window[0]["mem_loss"]
                ):
                    row["plateau_warning"] = True

            logf.write(json.dumps(row) + "\n")
            logf.flush()
            epoch_records.append(row)

            if time.time() - started > cap_seconds:
                break

    # Save encoder + scaler config (no optimizer state) — LAST checkpoint.
    # encoder-best.pt already carries the MEM-min checkpoint.
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
        "checkpoint_best": str(ckpt_best_path) if ckpt_best_path.exists() else None,
        "best_epoch": best_epoch,
        "best_mem_loss": best_mem_loss,
        "log": str(log_path),
        "epochs_run": len(epoch_records),
    }


def _run_probe_trio(
    enc,
    dataset: TapeDataset,
    device: torch.device,
    *,
    per_symbol: int = 1500,
    batch_size: int = 64,
    seed: int = 0,
) -> dict:
    """Forward a stratified per-symbol probe sample through the frozen encoder.

    Council-5 bugs patched 2026-04-23 (see commit history + council-5 review):
    - Stratified-per-symbol sampling (was: first 50K linear → only 3 alphabetical symbols).
    - Real UTC hour from `item["ts_first_ms"]` (was: event-index `start // 3600`).
    """
    from collections import defaultdict

    enc.eval()

    # Build per-symbol index pool and stratified sample.
    by_symbol: dict[str, list[int]] = defaultdict(list)
    for idx, ref in enumerate(dataset._refs):  # noqa: SLF001 — dataset is ours
        by_symbol[ref.symbol].append(idx)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for sym in sorted(by_symbol.keys()):
        pool = by_symbol[sym]
        if len(pool) <= per_symbol:
            chosen.extend(pool)
        else:
            chosen.extend(rng.choice(pool, size=per_symbol, replace=False).tolist())
    rng.shuffle(chosen)

    embeddings: list[np.ndarray] = []
    symbols: list[str] = []
    sym_ids: list[int] = []
    hours: list[int] = []
    labels_h100: list[int] = []
    masks_h100: list[bool] = []

    with torch.no_grad():
        for bstart in range(0, len(chosen), batch_size):
            bend = min(bstart + batch_size, len(chosen))
            batch_items = [dataset[int(chosen[i])] for i in range(bstart, bend)]
            feats = torch.stack([b["features"] for b in batch_items]).to(device)
            _, g = enc(feats)  # (B, 256)
            embeddings.append(g.cpu().numpy())
            for b in batch_items:
                symbols.append(b["symbol"])
                sym_ids.append(int(b["symbol_id"]))
                # CORRECTED: UTC hour from ms-epoch (was event-index // 3600).
                hours.append(int((b["ts_first_ms"] // 1_000 // 3_600) % 24))
                labels_h100.append(int(b["label_h100"]))
                masks_h100.append(bool(b["label_h100_mask"]))

    all_feats_np = np.concatenate(embeddings, axis=0)
    sym_ids_np = np.array(sym_ids, dtype=np.int64)
    hours_np = np.array(hours, dtype=np.int64)

    # Regroup by symbol for direction probe.
    feats_by_sym_np: dict[str, np.ndarray] = {}
    labels_by_sym_np: dict[str, np.ndarray] = {}
    masks_by_sym_np: dict[str, np.ndarray] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, sym in enumerate(symbols):
        grouped[sym].append(i)
    for sym, idxs in grouped.items():
        feats_by_sym_np[sym] = all_feats_np[idxs]
        labels_by_sym_np[sym] = np.array([labels_h100[i] for i in idxs])
        masks_by_sym_np[sym] = np.array([masks_h100[i] for i in idxs])

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
    ap.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help=(
            "ISO date (YYYY-MM-DD). Only shards with date < this are used "
            "for training. Defaults to APRIL_HELDOUT_START. Use an earlier "
            "date (e.g. 2026-02-01) to carve out a held-out period."
        ),
    )
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
        train_end_date=args.train_end_date or APRIL_HELDOUT_START,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

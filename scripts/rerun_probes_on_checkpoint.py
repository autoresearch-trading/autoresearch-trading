# scripts/rerun_probes_on_checkpoint.py
"""Phase-0 diagnostic — re-run the probe trio on an existing pretraining
checkpoint with the corrected stratified-sampling + UTC-hour logic.

Why this exists (council-5 diagnosis, 2026-04-23):
- scripts/run_pretrain.py::_run_probe_trio (and run_pretrain_probes.py) use
  item["start"] as if it were a timestamp; it is an event index within the
  shard. Hour-of-day probe was measuring event-index-bucket predictability,
  not UTC hour. `probe_hour_of_day_acc = 0.587` at epoch 5 of run-0 is
  therefore uninterpretable.
- The same loop iterates dataset[:50_000] linearly; the shards are sorted
  alphabetically, so only the first 3 symbols (2Z, AAVE, ASTER) get
  covered. `probe_dir_h100_balanced_acc_mean = 0.499` was a 3-illiquid-alts
  average, not a 25-symbol signal.

This script fixes both: (a) stratified per-symbol sampling, (b) real UTC
hour from ts_first_ms (now emitted by TapeDataset.__getitem__).

Output: `runs/step3-r1/corrected-probes.json` + stdout summary.

Usage:
    uv run python scripts/rerun_probes_on_checkpoint.py \
        --checkpoint runs/step3-r1/encoder.pt \
        --cache data/cache \
        --out runs/step3-r1/corrected-probes.json \
        --per-symbol 5000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tape.constants import (
    APRIL_HELDOUT_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_PRETRAIN,
)
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder
from tape.probes import direction_probe_h100, hour_of_day_probe, symbol_identity_probe


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _filter_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    """Pre-April shards only, AVAX excluded (match training filter)."""
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


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["encoder_config"]
    cfg = EncoderConfig(**cfg_dict)
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    enc = enc.to(device)
    enc.eval()
    return enc


def _stratified_sample_indices(
    dataset: TapeDataset, per_symbol: int, seed: int = 0
) -> np.ndarray:
    """Return `per_symbol` window indices per symbol (or all, if fewer exist)."""
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
    return np.array(chosen, dtype=np.int64)


def _forward_batch(
    enc: TapeEncoder,
    dataset: TapeDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Forward `indices` through encoder in batches; collect embeddings + metadata."""
    embeddings: list[np.ndarray] = []
    symbols: list[str] = []
    sym_ids: list[int] = []
    hours: list[int] = []
    labels_h100: list[int] = []
    masks_h100: list[bool] = []

    n = len(indices)
    t_start = time.time()
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch_items = [dataset[int(indices[i])] for i in range(bstart, bend)]
            feats = torch.stack([b["features"] for b in batch_items]).to(device)
            _, g = enc(feats)  # (B, 256)
            embeddings.append(g.cpu().numpy())
            for b in batch_items:
                symbols.append(b["symbol"])
                sym_ids.append(int(b["symbol_id"]))
                # CORRECTED: UTC hour from ms-epoch, not event index.
                hours.append(int((b["ts_first_ms"] // 1_000 // 3_600) % 24))
                labels_h100.append(int(b["label_h100"]))
                masks_h100.append(bool(b["label_h100_mask"]))
            if bstart % (batch_size * 20) == 0:
                elapsed = time.time() - t_start
                done = bend
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(
                    f"  forward: {done:,}/{n:,} ({100 * done / n:.1f}%)  "
                    f"{rate:.1f} items/s  ETA {eta:.0f}s"
                )

    emb = np.concatenate(embeddings, axis=0)
    return {
        "embeddings": emb,
        "symbols": np.array(symbols),
        "sym_ids": np.array(sym_ids, dtype=np.int64),
        "hours": np.array(hours, dtype=np.int64),
        "labels_h100": np.array(labels_h100, dtype=np.int64),
        "masks_h100": np.array(masks_h100, dtype=bool),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-symbol", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = _pick_device()
    print(f"[probes] device = {device}")

    symbols = list(PRETRAINING_SYMBOLS)
    shards = _filter_shards(args.cache, symbols)
    print(f"[probes] shards = {len(shards)}")
    dataset = TapeDataset(shards, stride=STRIDE_PRETRAIN, mode="pretrain")
    print(f"[probes] total windows = {len(dataset):,}")

    indices = _stratified_sample_indices(dataset, args.per_symbol, seed=args.seed)
    print(f"[probes] stratified sample = {len(indices):,} windows")

    enc = _load_encoder(args.checkpoint, device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[probes] encoder params = {n_params:,}")

    t0 = time.time()
    collected = _forward_batch(
        enc, dataset, indices, device, batch_size=args.batch_size
    )
    print(f"[probes] forward elapsed = {time.time() - t0:.1f}s")

    # Regroup by symbol for direction probe
    feats_by_sym: dict[str, list[np.ndarray]] = defaultdict(list)
    labels_by_sym: dict[str, list[int]] = defaultdict(list)
    masks_by_sym: dict[str, list[bool]] = defaultdict(list)
    for i, sym in enumerate(collected["symbols"]):
        feats_by_sym[sym].append(collected["embeddings"][i])
        labels_by_sym[sym].append(int(collected["labels_h100"][i]))
        masks_by_sym[sym].append(bool(collected["masks_h100"][i]))

    feats_np = {k: np.stack(v) for k, v in feats_by_sym.items()}
    labels_np = {k: np.array(v) for k, v in labels_by_sym.items()}
    masks_np = {k: np.array(v) for k, v in masks_by_sym.items()}

    print("[probes] running direction probe (per-symbol, walk-forward)...")
    t0 = time.time()
    dir_per_sym = direction_probe_h100(feats_np, labels_np, masks_np)
    print(
        f"[probes]   elapsed {time.time() - t0:.1f}s  covered {len(dir_per_sym)} symbols"
    )

    print("[probes] running symbol-id probe (stratified pool)...")
    t0 = time.time()
    sym_id_acc = symbol_identity_probe(
        collected["embeddings"], collected["sym_ids"], n_symbols=25, seed=args.seed
    )
    print(f"[probes]   elapsed {time.time() - t0:.1f}s  acc = {sym_id_acc:.4f}")

    print("[probes] running hour-of-day probe (CORRECTED UTC hour)...")
    t0 = time.time()
    hour_acc = hour_of_day_probe(
        collected["embeddings"], collected["hours"], seed=args.seed
    )
    print(f"[probes]   elapsed {time.time() - t0:.1f}s  acc = {hour_acc:.4f}")

    # Also report hour histogram so we can verify the UTC computation
    hour_hist = np.bincount(collected["hours"], minlength=24).tolist()
    print(f"[probes] hour distribution (0..23): {hour_hist}")

    # Summary statistics for direction
    dir_values = list(dir_per_sym.values())
    dir_mean = float(np.mean(dir_values)) if dir_values else None
    above_514 = sum(1 for v in dir_values if v >= 0.514)
    above_510 = sum(1 for v in dir_values if v >= 0.510)

    report = {
        "checkpoint": str(args.checkpoint),
        "per_symbol_sample": args.per_symbol,
        "stratified_n": int(len(indices)),
        "symbols_covered_by_direction_probe": len(dir_per_sym),
        "direction_h100_per_symbol": dir_per_sym,
        "direction_h100_balanced_acc_mean": dir_mean,
        "direction_h100_symbols_above_0.510": above_510,
        "direction_h100_symbols_above_0.514": above_514,
        "symbol_id_acc": sym_id_acc,
        "hour_of_day_acc": hour_acc,
        "hour_distribution": hour_hist,
        "encoder_params": n_params,
        "device": str(device),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n=== REPORT written to {args.out} ===")
    print(f"direction H100 balanced acc mean = {dir_mean}")
    print(f"  symbols >= 0.510: {above_510}/{len(dir_per_sym)}")
    print(f"  symbols >= 0.514: {above_514}/{len(dir_per_sym)}")
    print(
        f"symbol-id acc   = {sym_id_acc:.4f}  (threshold < 0.20 held-out; ~0.50 training)"
    )
    print(f"hour-of-day acc = {hour_acc:.4f}  (threshold < 0.10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

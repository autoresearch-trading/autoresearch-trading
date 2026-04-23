# scripts/april_gate1_preflight.py
"""Phase-2 diagnostic — runs a direction + hour + symbol-id probe on April
1-13 held-out data using the epoch-8 encoder from run-0.

Differs from scripts/run_pretrain_probes.py in two ways:
1. UTC hour from ts_first_ms (bug B fix, also in main probe runner).
2. Direction probe uses simple per-symbol 80/20 split rather than walk-
   forward 3-fold — April 1-13 only yields ~75-150 windows/symbol at
   stride=200 (roughly half March density due to fulfill_taker dedup),
   well below the 4100 required for walk-forward. 80/20 is honest about
   the data-size constraint and tells us whether the epoch-8 encoder has
   ANY direction signal on held-out April, which is the core Phase 2
   question.

This is NOT a Gate 1 certification — the formal Gate 1 threshold logic
requires walk-forward. It's a Phase 2 pre-flight to decide whether run-1
is worth launching, and to estimate the training-vs-held-out gap.

Usage:
    uv run python scripts/april_gate1_preflight.py \
        --checkpoint runs/step3-r1/encoder.pt \
        --cache data/cache \
        --out runs/step3-r1/april-preflight.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.constants import APRIL_HELDOUT_START, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder
from tape.probes import hour_of_day_probe, symbol_identity_probe


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _april_probe_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    """April 1 through APRIL_HELDOUT_START-exclusive (2026-04-14)."""
    APRIL_START = "2026-04-01"
    shards: list[Path] = []
    for sym in symbols:
        for p in sorted(cache_dir.glob(f"{sym}__2026-04-*.npz")):
            date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
            if APRIL_START <= date_part < APRIL_HELDOUT_START:
                shards.append(p)
    return shards


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    enc = enc.to(device)
    enc.eval()
    return enc


def direction_probe_h100_simple_split(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    *,
    test_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
    min_valid: int = 40,
) -> dict[str, float]:
    """Per-symbol balanced accuracy via a single 80/20 time-ordered split.

    Used for April 1-13 pre-flight — walk-forward 3-fold is infeasible
    given the ~75-150 windows/symbol we get at stride=200.
    """
    out: dict[str, float] = {}
    for sym, feat in features.items():
        y = labels[sym]
        m = masks[sym]
        valid_idx = np.where(m)[0]
        if len(valid_idx) < min_valid:
            continue
        # Time-ordered split: last 20% is the test fold. Respects any intra-
        # April chronology (windows are ordered by shard then start).
        n_valid = len(valid_idx)
        n_test = max(1, int(n_valid * test_frac))
        tr_pos = valid_idx[: n_valid - n_test]
        te_pos = valid_idx[n_valid - n_test :]
        if len(np.unique(y[tr_pos])) < 2 or len(np.unique(y[te_pos])) < 2:
            continue  # degenerate split — both classes needed
        scaler = StandardScaler().fit(feat[tr_pos])
        Xtr = scaler.transform(feat[tr_pos])
        Xte = scaler.transform(feat[te_pos])
        lr = LogisticRegression(C=C, max_iter=1_000).fit(Xtr, y[tr_pos])
        out[sym] = float(balanced_accuracy_score(y[te_pos], lr.predict(Xte)))
    return out


def _forward_all(
    enc: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    horizon: int = 100,
) -> dict:
    """Forward all April windows in batches; collect embeddings + metadata."""
    n = len(dataset)
    embeddings: list[np.ndarray] = []
    symbols: list[str] = []
    sym_ids: list[int] = []
    hours: list[int] = []
    labels_h100: list[int] = []
    masks_h100: list[bool] = []
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch_items = [dataset[i] for i in range(bstart, bend)]
            feats = torch.stack([b["features"] for b in batch_items]).to(device)
            _, g = enc(feats)
            embeddings.append(g.cpu().numpy())
            for b in batch_items:
                symbols.append(b["symbol"])
                sym_ids.append(int(b["symbol_id"]))
                hours.append(int((b["ts_first_ms"] // 1_000 // 3_600) % 24))
                labels_h100.append(int(b[f"label_h{horizon}"]))
                masks_h100.append(bool(b[f"label_h{horizon}_mask"]))
            if bstart % (batch_size * 10) == 0:
                elapsed = time.time() - t0
                done = bend
                rate = done / max(elapsed, 1e-6)
                print(f"  forward: {done:,}/{n:,}  {rate:.1f} items/s")
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--horizon",
        type=int,
        default=100,
        choices=[10, 50, 100, 500],
    )
    args = ap.parse_args()

    device = _pick_device()
    print(f"[preflight] device = {device}")

    syms = list(PRETRAINING_SYMBOLS)  # Gate 1 is over pretraining symbols
    shards = _april_probe_shards(args.cache, syms)
    print(f"[preflight] April shards = {len(shards)}")
    if not shards:
        print("ERROR: no April shards found", flush=True)
        return 2

    dataset = TapeDataset(shards, stride=STRIDE_EVAL, mode="eval")
    print(f"[preflight] April windows (stride={STRIDE_EVAL}): {len(dataset):,}")

    enc = _load_encoder(args.checkpoint, device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[preflight] encoder params = {n_params:,}")

    t0 = time.time()
    collected = _forward_all(
        enc, dataset, device, batch_size=args.batch_size, horizon=args.horizon
    )
    print(f"[preflight] forward elapsed = {time.time() - t0:.1f}s")

    # Regroup for direction probe
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

    print("[preflight] direction probe (per-symbol 80/20, time-ordered)...")
    dir_per_sym = direction_probe_h100_simple_split(
        feats_np, labels_np, masks_np, seed=args.seed
    )

    print("[preflight] symbol-id probe...")
    sym_id_acc = symbol_identity_probe(
        collected["embeddings"], collected["sym_ids"], n_symbols=25, seed=args.seed
    )

    print("[preflight] hour-of-day probe (corrected UTC)...")
    hour_acc = hour_of_day_probe(
        collected["embeddings"], collected["hours"], seed=args.seed
    )

    dir_vals = list(dir_per_sym.values())
    dir_mean = float(np.mean(dir_vals)) if dir_vals else None
    n_covered = len(dir_per_sym)
    above_514 = sum(1 for v in dir_vals if v >= 0.514)
    above_510 = sum(1 for v in dir_vals if v >= 0.510)

    report = {
        "checkpoint": str(args.checkpoint),
        "april_shards": len(shards),
        "april_windows": int(len(dataset)),
        "symbols_covered_direction": n_covered,
        "direction_h100_per_symbol": dir_per_sym,
        "direction_h100_mean": dir_mean,
        "direction_h100_symbols_above_0.510": above_510,
        "direction_h100_symbols_above_0.514": above_514,
        "symbol_id_acc": sym_id_acc,
        "hour_of_day_acc": hour_acc,
        "encoder_params": n_params,
        "device": str(device),
        "probe_methodology": "per-symbol 80/20 time-ordered split (walk-forward infeasible at stride=200, ~75-150 windows/symbol)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n=== APRIL PRE-FLIGHT REPORT ===")
    print(f"  written to: {args.out}")
    print(f"  direction mean = {dir_mean}  |  covered {n_covered}/24 symbols")
    print(f"  symbols >= 0.510 = {above_510}  |  >= 0.514 = {above_514}")
    print(f"  symbol-id acc   = {sym_id_acc:.4f}  (threshold <0.20 held-out)")
    print(f"  hour-of-day acc = {hour_acc:.4f}  (threshold <0.10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

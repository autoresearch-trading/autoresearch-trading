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

from tape.constants import (
    APRIL_HELDOUT_START,
    APRIL_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_EVAL,
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


def _load_encoder(checkpoint: Path | None) -> TapeEncoder:
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
    # Prefer CUDA, then MPS (Apple Silicon), else CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        "direction_h100_mean": (
            float(np.mean(list(dir_per_sym.values()))) if dir_per_sym else None
        ),
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
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        symbols=args.symbols,
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "symbol_identity_acc",
                    "hour_of_day_acc",
                    "direction_h100_mean",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

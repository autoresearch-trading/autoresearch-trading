# tape/probe_utils.py
"""Shared helpers for frozen-encoder evaluation probes.

Used by:
  - scripts/run_gate4.py (Gate 4 temporal stability)
  - scripts/run_condition_c1.py (Wyckoff absorption probe, post-Gate-2 pre-reg)
  - scripts/run_condition_c3.py (ARI cluster–Wyckoff alignment)
  - scripts/run_condition_c4.py (embedding trajectory test)

All helpers operate on the frozen Run-2 encoder checkpoint
(`runs/step3-r2/encoder-best.pt`). No fine-tuning happens here — these
probes are diagnostic-only against `model.eval()`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tape.constants import STRIDE_EVAL
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shards_for_sym_months(
    cache_dir: Path, sym: str, month_prefixes: tuple[str, ...]
) -> list[Path]:
    """Cache shards for one symbol across one or more month prefixes.

    A shard filename has the form `{SYM}__{YYYY-MM-DD}.npz`. This helper
    expands to all dates inside the requested months and returns the sorted
    paths. Missing months silently contribute zero shards.
    """
    shards: list[Path] = []
    for mp in month_prefixes:
        shards.extend(sorted(cache_dir.glob(f"{sym}__{mp}-*.npz")))
    return shards


def load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    """Restore the frozen encoder in `eval()` mode on `device`."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def forward_embeddings(
    enc: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    batch_size: int = 64,
    *,
    return_features: bool = False,
) -> dict:
    """Embed every window in `dataset` with the frozen encoder.

    Returns a dict containing at least:
      - emb : (N, 256) float32 — global embedding, post-encoder
      - sym : list[str] — per-window symbol
      - date: list[str] — per-window date (YYYY-MM-DD)
      - start: (N,) int64 — per-window event-index start within the shard
      - label_h500: (N,) int64 — direction label at H500
      - mask_h500 : (N,) bool — valid-label mask at H500

    If `return_features=True`, also returns:
      - features : (N, 200, 17) float32 — raw (post-FE, pre-BN) feature window

    The features tensor is needed for window-internal Wyckoff labels (C1, C3)
    and for the climax-event seed criterion (C4). It is large
    (~28 MB per 1000 windows × 17 channels × 200 events × 4 bytes), so it is
    only materialized when explicitly requested.
    """
    embs: list[np.ndarray] = []
    syms: list[str] = []
    dates: list[str] = []
    starts: list[int] = []
    labels: list[int] = []
    masks: list[bool] = []
    feats_all: list[np.ndarray] = []

    n = len(dataset)
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch = [dataset[i] for i in range(bstart, bend)]
            feats_t = torch.stack([b["features"] for b in batch]).to(device)
            _, g = enc(feats_t)
            embs.append(g.cpu().numpy().astype(np.float32))
            for b in batch:
                syms.append(str(b["symbol"]))
                dates.append(str(b["date"]))
                starts.append(int(b["start"]))
                labels.append(int(b["label_h500"]))
                masks.append(bool(b["label_h500_mask"]))
            if return_features:
                feats_all.append(feats_t.cpu().numpy().astype(np.float32))

    out: dict = {
        "emb": np.concatenate(embs, axis=0),
        "sym": syms,
        "date": dates,
        "start": np.array(starts, dtype=np.int64),
        "label_h500": np.array(labels, dtype=np.int64),
        "mask_h500": np.array(masks, dtype=bool),
    }
    if return_features:
        out["features"] = np.concatenate(feats_all, axis=0)
    return out


def build_eval_dataset(shards: list[Path], stride: int = STRIDE_EVAL) -> TapeDataset:
    """Construct a TapeDataset for evaluation (no augmentation)."""
    return TapeDataset(shards, stride=stride, mode="pretrain")

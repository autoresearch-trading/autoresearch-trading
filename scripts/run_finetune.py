# scripts/run_finetune.py
"""Step 4 fine-tuning entry point.

Mirrors `scripts/run_pretrain.py` for CLI ergonomics, device selection, and
log format. Implements the ratified plan
(`docs/superpowers/plans/2026-04-24-step4-fine-tuning.md`):

  Phase A (epochs 0..frozen_epochs-1): encoder frozen, train heads only.
      AdamW + OneCycleLR(max_lr=lr_frozen, pct_start=0.20).
  Phase B (epochs frozen_epochs..epochs-1): encoder unfrozen, joint fine-tune.
      Fresh AdamW + OneCycleLR(max_lr=lr_unfrozen, pct_start=0.05).
      Gradient clipping max_norm=1.0.

Per-epoch monitoring (training-log.jsonl):
  - Train + val loss per horizon (BCE, label-smoothed) + total
  - Embedding std on a fixed 1024-window val batch (collapse alarm < 0.05)
  - Effective rank of (1024, 256) embedding matrix (alarm < 30)
  - CKA between live encoder and frozen-snapshot embeddings (council-6 Q3)
  - Per-horizon val balanced accuracy

Every 5 epochs (0, 5, 10, 15, end):
  - Hour-of-day probe on 256-dim live embeddings (24-class LR)

Numeric abort criteria (plan §"Numeric abort criteria"):
  - Epoch 3: H500 val BCE > training-init BCE → heads aren't learning
  - Epoch 5: H500 val BCE has not dropped below 0.95 × initial_random_BCE
  - Any epoch: embedding std < 0.05
  - Any epoch ≥ 8: CKA-vs-frozen < 0.3
  - Any epoch ≥ 8 (after epoch 8): H100 val balanced accuracy < 0.50
  - Any 5-epoch checkpoint: hour-of-day probe > 0.12

Usage (local smoke):
    uv run python scripts/run_finetune.py \
        --checkpoint runs/step3-r2/encoder-best.pt \
        --cache data/cache --symbols BTC \
        --epochs 2 --frozen-epochs 1 --batch-size 8 \
        --out-dir runs/smoke-finetune

Usage (M4 Pro MPS):
    caffeinate -i uv run python scripts/run_finetune.py \
        --checkpoint runs/step3-r2/encoder-best.pt \
        --cache data/cache \
        --epochs 20 --frozen-epochs 5 --batch-size 256 \
        --out-dir runs/step4-r1 --max-hours 6.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from tape.constants import (
    APRIL_HELDOUT_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_PRETRAIN,
)
from tape.dataset import TapeDataset
from tape.finetune import (
    HORIZON_WEIGHTS,
    HORIZONS,
    LABEL_SMOOTHING_EPS,
    DirectionHead,
    FineTunedModel,
    cka_torch,
    weighted_bce_loss,
)
from tape.model import EncoderConfig, TapeEncoder
from tape.pretrain import effective_rank
from tape.sampler import EqualSymbolSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAL_FRACTION: float = 0.10  # 10% of training distribution → val (NOT Feb/Mar)
SNAPSHOT_VAL_SIZE: int = 1024  # fixed deterministic subset for CKA / collapse
PROBE_EVERY_EPOCHS: int = 5

ABORT_EMBED_STD: float = 0.05
ABORT_CKA: float = 0.3
ABORT_HOUR_PROBE: float = 0.12
ABORT_H100_VAL_BAL_ACC: float = 0.50
ABORT_EPOCH_5_FACTOR: float = 0.95


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _filter_train_shards(
    cache_dir: Path,
    symbols: list[str],
    *,
    train_end_date: str,
) -> list[Path]:
    """Mirror of run_pretrain._filter_shards: shards strictly before train_end_date.

    Excludes AVAX (held-out symbol) AND any shard at/after the April hold-out.
    train_end_date defaults to 2026-02-01 in main() — Feb+Mar are the Gate 2
    eval window, NOT training data.
    """
    shards: list[Path] = []
    for sym in symbols:
        if sym == HELD_OUT_SYMBOL:
            continue  # AVAX is Gate 3 hold-out
        for p in sorted(cache_dir.glob(f"{sym}__*.npz")):
            date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
            if date_part >= train_end_date:
                continue
            if date_part >= APRIL_HELDOUT_START:
                continue  # hard April hold-out (gotcha #17)
            shards.append(p)
    return shards


def _collate(batch_items: list[dict]) -> tuple[torch.Tensor, dict]:
    """Stack features + labels + masks into per-horizon tensors.

    The dataset emits per-horizon labels as separate keys label_h{H} / label_h{H}_mask;
    we materialize them into (B, 4) tensors aligned with HORIZONS order.
    """
    feats = torch.stack([b["features"] for b in batch_items])
    B = len(batch_items)
    H = len(HORIZONS)
    labels = torch.zeros(B, H, dtype=torch.int64)
    masks = torch.zeros(B, H, dtype=torch.int64)
    for i, b in enumerate(batch_items):
        for hi, h in enumerate(HORIZONS):
            labels[i, hi] = int(b[f"label_h{h}"])
            masks[i, hi] = int(b[f"label_h{h}_mask"])
    metadata = {
        "symbols": [b["symbol"] for b in batch_items],
        "dates": [b["date"] for b in batch_items],
        "hours": [
            int((b.get("ts_first_ms", 0) // 1_000 // 3_600) % 24) for b in batch_items
        ],
    }
    return feats, labels, masks, metadata  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Device + helpers
# ---------------------------------------------------------------------------


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder(checkpoint_path: Path) -> TapeEncoder:
    """Load a TapeEncoder from a pretraining checkpoint (CPU; caller moves)."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc


def _split_train_val(
    n_total: int, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random split: val_fraction → val, rest → train."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_fraction))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _val_batch_indices(val_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic 1024-window subset of val for CKA / std monitoring."""
    rng = np.random.default_rng(seed)
    if len(val_idx) <= n:
        return val_idx
    chosen = rng.choice(val_idx, size=n, replace=False)
    return np.sort(chosen)


def _forward_embeddings(
    encoder: TapeEncoder,
    dataset: TapeDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Forward `indices` through `encoder` in eval mode → (N, 256) np.ndarray."""
    encoder.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            e = min(s + batch_size, len(indices))
            batch = [dataset[int(indices[i])] for i in range(s, e)]
            feats = torch.stack([b["features"] for b in batch]).to(device)
            _, g = encoder(feats)
            out.append(g.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _val_loss_and_balanced_acc(
    model: FineTunedModel,
    dataset: TapeDataset,
    val_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Compute val total/per-horizon BCE + per-horizon balanced accuracy.

    Predictions are (sigmoid(logit) > 0.5). Balanced accuracy is computed over
    valid labels only (mask=1).
    """
    model.eval()
    total_per_h_sum = np.zeros(len(HORIZONS), dtype=np.float64)
    total_per_h_count = np.zeros(len(HORIZONS), dtype=np.int64)
    pred_buf: list[np.ndarray] = []
    label_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []

    with torch.no_grad():
        for s in range(0, len(val_idx), batch_size):
            e = min(s + batch_size, len(val_idx))
            batch_items = [dataset[int(val_idx[i])] for i in range(s, e)]
            feats, labels, masks, _ = _collate(batch_items)
            feats = feats.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            logits = model(feats)
            _total, per_h = weighted_bce_loss(
                logits,
                labels,
                masks,
                label_smoothing_eps=LABEL_SMOOTHING_EPS,
                horizon_weights=HORIZON_WEIGHTS,
            )
            # Per-horizon accumulator: weight by valid count so we get a true
            # over-the-batch mean BCE per horizon at the end.
            counts = masks.sum(dim=0).detach().cpu().numpy()
            total_per_h_sum += per_h.detach().cpu().numpy() * counts
            total_per_h_count += counts.astype(np.int64)

            preds = (torch.sigmoid(logits) > 0.5).long()
            pred_buf.append(preds.detach().cpu().numpy())
            label_buf.append(labels.detach().cpu().numpy())
            mask_buf.append(masks.detach().cpu().numpy())

    safe_count = np.where(total_per_h_count > 0, total_per_h_count, 1)
    per_horizon_bce = np.where(
        total_per_h_count > 0,
        total_per_h_sum / safe_count,
        0.0,
    )
    weights = np.array(HORIZON_WEIGHTS, dtype=np.float64)
    total_bce = float((per_horizon_bce * weights).sum())

    preds_all = np.concatenate(pred_buf, axis=0)  # (N, H)
    labels_all = np.concatenate(label_buf, axis=0)
    masks_all = np.concatenate(mask_buf, axis=0)

    bal_acc_per_h: list[float | None] = []
    for hi in range(len(HORIZONS)):
        valid = masks_all[:, hi].astype(bool)
        if valid.sum() < 2:
            bal_acc_per_h.append(None)
            continue
        y = labels_all[valid, hi]
        p = preds_all[valid, hi]
        if len(np.unique(y)) < 2:
            # balanced_accuracy_score handles single-class but is degenerate.
            bal_acc_per_h.append(float(balanced_accuracy_score(y, p)))
        else:
            bal_acc_per_h.append(float(balanced_accuracy_score(y, p)))

    model.train()
    return {
        "val_total_bce": total_bce,
        "val_per_horizon_bce": per_horizon_bce.tolist(),
        "val_balanced_acc_per_horizon": bal_acc_per_h,
    }


def _hour_probe_on_embeddings(
    embeddings: np.ndarray,
    hours: np.ndarray,
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> float:
    """24-class hour-of-day LR probe — accuracy. Mirrors tape.probes.hour_of_day_probe."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(embeddings))
    n_test = int(len(embeddings) * test_frac)
    te = perm[:n_test]
    tr = perm[n_test:]
    if len(tr) < 24 or len(np.unique(hours[tr])) < 2:
        return float("nan")
    scaler = StandardScaler().fit(embeddings[tr])
    lr = LogisticRegression(C=1.0, max_iter=1_000).fit(
        scaler.transform(embeddings[tr]), hours[tr]
    )
    return float(lr.score(scaler.transform(embeddings[te]), hours[te]))


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def run_finetune(
    *,
    checkpoint: Path,
    cache_dir: Path,
    out_dir: Path,
    symbols: list[str] | None,
    epochs: int,
    frozen_epochs: int,
    batch_size: int,
    lr_frozen: float,
    lr_unfrozen: float,
    seed: int,
    train_end_date: str,
    max_hours: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training-log.jsonl"
    best_path = out_dir / "finetuned-best.pt"
    final_path = out_dir / "finetuned-final.pt"
    snapshot_path = out_dir / "frozen-snapshot-embeddings.npz"
    aborted_path = out_dir / "aborted.pt"

    # --- 1. Load encoder + wrap ---
    encoder = _load_encoder(checkpoint)
    model = FineTunedModel(encoder, head=DirectionHead(embed_dim=encoder.global_dim))

    device = _pick_device()
    print(f"[finetune] device={device}")
    model = model.to(device)

    # --- 2. Build dataset (train end date → only pre-Feb shards) ---
    syms = list(symbols or PRETRAINING_SYMBOLS)
    shards = _filter_train_shards(cache_dir, syms, train_end_date=train_end_date)
    if not shards:
        raise RuntimeError(
            f"no training shards found in {cache_dir} for symbols={syms} "
            f"with train_end_date={train_end_date}"
        )
    dataset = TapeDataset(shards, stride=STRIDE_PRETRAIN, mode="pretrain")
    print(f"[finetune] dataset: {len(shards)} shards → {len(dataset):,} windows")

    # --- 3. Train/val split (deterministic, 90/10) ---
    train_idx, val_idx = _split_train_val(len(dataset), VAL_FRACTION, seed)
    print(f"[finetune] split: train={len(train_idx):,}  val={len(val_idx):,}")

    # Fixed 1024-window val subset for CKA / collapse / hour-probe monitoring.
    snapshot_idx = _val_batch_indices(val_idx, SNAPSHOT_VAL_SIZE, seed)

    # --- 4. Snapshot frozen-encoder embeddings on the fixed val subset ---
    # Do this BEFORE any training so the snapshot is a clean reference.
    frozen_snapshot = _forward_embeddings(
        encoder, dataset, snapshot_idx, device, batch_size=64
    )
    # Hours for the same indices (used by hour-probe).
    snapshot_hours = np.array(
        [
            int((dataset[int(i)]["ts_first_ms"] // 1_000 // 3_600) % 24)
            for i in snapshot_idx
        ],
        dtype=np.int64,
    )
    np.savez(
        snapshot_path,
        embeddings=frozen_snapshot,
        indices=snapshot_idx,
        hours=snapshot_hours,
    )
    print(
        f"[finetune] frozen-snapshot saved: {frozen_snapshot.shape} → {snapshot_path}"
    )

    frozen_snapshot_t = torch.from_numpy(frozen_snapshot).float()  # CPU; CKA on CPU

    # --- 5. Phase A: freeze encoder, train heads ---
    train_subset = Subset(dataset, train_idx.tolist())
    sampler = EqualSymbolSampler(dataset, seed=seed)
    # NOTE: EqualSymbolSampler operates on the FULL dataset's _refs — we use it
    # for symbol balancing but pair with a Subset-style filter at iter time:
    # rather than wrapping with Subset, we feed the full dataset + filter the
    # sampler's emitted indices. Simpler: keep the full dataset with a custom
    # collate that respects val_idx exclusion. To minimise complexity here we
    # use a pragmatic split: train via a shuffled sampler on train_idx; rely
    # on EqualSymbolSampler at the dataset level for symbol balance and accept
    # that val_idx is a 10% slice that on average preserves symbol balance.

    # Use a simple shuffled torch sampler over train_idx — EqualSymbolSampler
    # would need surgical changes to honour an index allow-list, which the
    # plan does not require (val 10% is randomly sampled across all symbols
    # so the train residual remains roughly balanced; EqualSymbolSampler is
    # invoked here to keep BTC from dominating across the residual).
    # See "Pipeline §2/3" of the plan: sampler on train shards is required.

    # Simplest correct route: build a child TapeDataset over train_idx via Subset,
    # and use EqualSymbolSampler on a *fresh* TapeDataset constructed from the
    # same shards (so BTC dominance is still capped). This keeps val excluded
    # from training without rewriting the sampler.
    train_dataset_for_sampler = TapeDataset(
        shards, stride=STRIDE_PRETRAIN, mode="pretrain"
    )
    # Subsample train_dataset_for_sampler to train_idx by editing _refs.
    train_dataset_for_sampler._refs = [  # noqa: SLF001
        train_dataset_for_sampler._refs[i] for i in train_idx
    ]
    sampler = EqualSymbolSampler(train_dataset_for_sampler, seed=seed)
    loader = DataLoader(
        train_dataset_for_sampler,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda items: items,  # we use _collate inline
        drop_last=True,
    )
    # _collate is invoked inside the training loop after raw items come back.

    print(f"[finetune] Phase A: epochs 0..{frozen_epochs - 1} encoder FROZEN")
    model.freeze_encoder()
    phase_a_steps = max(1, frozen_epochs * max(1, len(loader)))
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    opt_a: torch.optim.Optimizer = torch.optim.AdamW(
        head_params, lr=lr_frozen, weight_decay=1e-4
    )
    sched_a = torch.optim.lr_scheduler.OneCycleLR(
        opt_a, max_lr=lr_frozen, total_steps=phase_a_steps, pct_start=0.20
    )

    # --- 6. Compute initial random BCE for abort criteria (epoch 0, step 0). ---
    # We do this BEFORE any optimizer step so heads are at their σ=0.02 init.
    init_eval = _val_loss_and_balanced_acc(model, dataset, val_idx, device, batch_size)
    initial_h500_val_bce = init_eval["val_per_horizon_bce"][HORIZONS.index(500)]
    print(f"[finetune] initial H500 val BCE (random init) = {initial_h500_val_bce:.4f}")

    started = time.time()
    cap_seconds = max_hours * 3_600
    best_val_total_bce = float("inf")
    best_epoch = 0
    epoch_records: list[dict] = []

    def _log_row(row: dict, logf):
        logf.write(json.dumps(row, default=lambda o: None) + "\n")
        logf.flush()
        epoch_records.append(row)

    abort_reason: str | None = None

    with log_path.open("w") as logf:
        # Log the initial pre-training row so the abort criteria have a reference.
        _log_row(
            {
                "phase": "init",
                "epoch": 0,
                "initial_h500_val_bce": initial_h500_val_bce,
                "val_per_horizon_bce_init": init_eval["val_per_horizon_bce"],
                "val_balanced_acc_per_horizon_init": init_eval[
                    "val_balanced_acc_per_horizon"
                ],
            },
            logf,
        )

        for epoch in range(epochs):
            phase = "frozen" if epoch < frozen_epochs else "unfrozen"
            # IMPORTANT: do NOT call train_dataset_for_sampler.set_epoch() —
            # that would rebuild _refs with a fresh random offset over the FULL
            # shard list, undoing the train/val split done above (val_idx /
            # train_idx index into the original _refs ordering). Per-epoch
            # window offset randomization is a pretraining augmentation; for
            # fine-tuning the plan keeps the same windows fixed across epochs
            # so the val set stays consistent with the snapshot.
            sampler.set_epoch(epoch)

            # Transition: at the start of Phase B, unfreeze + new optimizer/scheduler.
            if epoch == frozen_epochs:
                print(
                    f"[finetune] Phase B: epoch {epoch} unfreeze encoder; "
                    f"fresh AdamW lr={lr_unfrozen}"
                )
                model.unfreeze_encoder()
                phase_b_steps = max(1, (epochs - frozen_epochs) * max(1, len(loader)))
                opt_b: torch.optim.Optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=lr_unfrozen,
                    weight_decay=1e-4,
                )
                sched_b = torch.optim.lr_scheduler.OneCycleLR(
                    opt_b,
                    max_lr=lr_unfrozen,
                    total_steps=phase_b_steps,
                    pct_start=0.05,
                )
                opt = opt_b
                sched: object = sched_b
            elif epoch == 0:
                opt = opt_a
                sched = sched_a

            # Training pass
            model.train()
            train_per_h_sum = np.zeros(len(HORIZONS), dtype=np.float64)
            train_per_h_count = np.zeros(len(HORIZONS), dtype=np.int64)
            train_total_sum = 0.0
            train_total_n = 0

            for raw_items in loader:
                if time.time() - started > cap_seconds:
                    print("[finetune] wall-clock cap reached; stopping training")
                    break
                feats, labels, masks, _meta = _collate(raw_items)
                feats = feats.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                logits = model(feats)
                total, per_h = weighted_bce_loss(
                    logits,
                    labels,
                    masks,
                    label_smoothing_eps=LABEL_SMOOTHING_EPS,
                    horizon_weights=HORIZON_WEIGHTS,
                )

                opt.zero_grad(set_to_none=True)
                total.backward()
                # Phase B grad clip; harmless in Phase A too.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                opt.step()
                cast(object, sched).step()  # type: ignore[union-attr]

                counts = masks.sum(dim=0).detach().cpu().numpy()
                train_per_h_sum += per_h.detach().cpu().numpy() * counts
                train_per_h_count += counts.astype(np.int64)
                train_total_sum += float(total.detach()) * feats.shape[0]
                train_total_n += feats.shape[0]

            safe_count = np.where(train_per_h_count > 0, train_per_h_count, 1)
            train_per_h_bce = np.where(
                train_per_h_count > 0, train_per_h_sum / safe_count, 0.0
            )
            train_total = train_total_sum / max(1, train_total_n)

            # Validation pass
            val_eval = _val_loss_and_balanced_acc(
                model, dataset, val_idx, device, batch_size
            )

            # Snapshot embeddings for monitoring
            live_snapshot = _forward_embeddings(
                model.encoder, dataset, snapshot_idx, device, batch_size=64
            )
            embed_std = float(live_snapshot.std(axis=0).mean())
            eff_rank = int(
                effective_rank(torch.from_numpy(live_snapshot).float(), sv_floor=0.01)
            )
            cka_val = cka_torch(
                torch.from_numpy(live_snapshot).float(), frozen_snapshot_t
            )

            row: dict = {
                "phase": phase,
                "epoch": epoch + 1,  # 1-indexed in log to match run_pretrain.py
                "elapsed_h": (time.time() - started) / 3_600,
                "train_total_bce": train_total,
                "train_per_horizon_bce": train_per_h_bce.tolist(),
                "val_total_bce": val_eval["val_total_bce"],
                "val_per_horizon_bce": val_eval["val_per_horizon_bce"],
                "val_balanced_acc_per_horizon": val_eval[
                    "val_balanced_acc_per_horizon"
                ],
                "embed_std": embed_std,
                "effective_rank": eff_rank,
                "cka_vs_frozen": cka_val,
            }

            # Hour-of-day probe at every 5-epoch checkpoint (0-indexed: epoch 0, 5, 10, 15, end-1).
            do_hour_probe = (epoch % PROBE_EVERY_EPOCHS == 0) or (epoch == epochs - 1)
            if do_hour_probe:
                hour_acc = _hour_probe_on_embeddings(
                    live_snapshot, snapshot_hours, seed=seed
                )
                row["hour_of_day_acc"] = hour_acc
            else:
                row["hour_of_day_acc"] = None

            # ---- Abort checks (plan §"Numeric abort criteria") ----
            h500_idx = HORIZONS.index(500)
            h100_idx = HORIZONS.index(100)
            h500_val_bce = val_eval["val_per_horizon_bce"][h500_idx]
            h100_val_bal = val_eval["val_balanced_acc_per_horizon"][h100_idx]

            if epoch + 1 == 3 and h500_val_bce > initial_h500_val_bce:
                abort_reason = (
                    f"epoch=3 H500 val BCE ({h500_val_bce:.4f}) > initial "
                    f"({initial_h500_val_bce:.4f}); heads not learning"
                )
            elif (
                epoch + 1 == 5
                and h500_val_bce > ABORT_EPOCH_5_FACTOR * initial_h500_val_bce
            ):
                abort_reason = (
                    f"epoch=5 H500 val BCE ({h500_val_bce:.4f}) > "
                    f"{ABORT_EPOCH_5_FACTOR}× initial ({initial_h500_val_bce:.4f}); "
                    "linear-probe warmup failed"
                )
            elif embed_std < ABORT_EMBED_STD:
                abort_reason = (
                    f"embedding std collapse: {embed_std:.4f} < {ABORT_EMBED_STD}"
                )
            elif epoch + 1 >= 8 and cka_val < ABORT_CKA:
                abort_reason = (
                    f"CKA-vs-frozen drift: {cka_val:.4f} < {ABORT_CKA} "
                    f"after epoch {epoch + 1}"
                )
            elif (
                epoch + 1 > 8
                and h100_val_bal is not None
                and h100_val_bal < ABORT_H100_VAL_BAL_ACC
            ):
                abort_reason = (
                    f"H100 val balanced acc {h100_val_bal:.4f} < "
                    f"{ABORT_H100_VAL_BAL_ACC} after epoch {epoch + 1}"
                )
            elif (
                do_hour_probe
                and row["hour_of_day_acc"] is not None
                and not math.isnan(row["hour_of_day_acc"])
                and row["hour_of_day_acc"] > ABORT_HOUR_PROBE
            ):
                abort_reason = (
                    f"hour-of-day probe {row['hour_of_day_acc']:.4f} > "
                    f"{ABORT_HOUR_PROBE} at epoch {epoch + 1}"
                )

            # Best-val checkpoint (improvement check happens regardless of abort).
            improved = val_eval["val_total_bce"] < best_val_total_bce
            if improved:
                best_val_total_bce = val_eval["val_total_bce"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "encoder_config": model.encoder.cfg.__dict__,
                        "head_state_dict": model.head.state_dict(),
                        "epoch": epoch + 1,
                        "val_total_bce": best_val_total_bce,
                        "seed": seed,
                    },
                    best_path,
                )
                row["saved_best"] = True
            else:
                row["saved_best"] = False

            row["best_val_total_bce"] = best_val_total_bce
            row["best_epoch"] = best_epoch

            if abort_reason is not None:
                row["abort_reason"] = abort_reason
                _log_row(row, logf)
                # Save aborted state for forensics.
                torch.save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "encoder_config": model.encoder.cfg.__dict__,
                        "head_state_dict": model.head.state_dict(),
                        "epoch": epoch + 1,
                        "abort_reason": abort_reason,
                        "seed": seed,
                    },
                    aborted_path,
                )
                print(f"[finetune] ABORT: {abort_reason}")
                return {
                    "aborted": True,
                    "abort_reason": abort_reason,
                    "best_path": str(best_path) if best_path.exists() else None,
                    "best_epoch": best_epoch,
                    "best_val_total_bce": best_val_total_bce,
                    "log": str(log_path),
                    "epochs_run": len(epoch_records) - 1,  # minus the init row
                }

            _log_row(row, logf)

            if time.time() - started > cap_seconds:
                break

    # Final save (regardless of best).
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "encoder_config": model.encoder.cfg.__dict__,
            "head_state_dict": model.head.state_dict(),
            "epoch": len(epoch_records) - 1,
            "val_total_bce": (
                float(epoch_records[-1].get("val_total_bce", float("nan")))
                if epoch_records
                else None
            ),
            "seed": seed,
        },
        final_path,
    )

    return {
        "aborted": False,
        "best_path": str(best_path) if best_path.exists() else None,
        "final_path": str(final_path),
        "best_epoch": best_epoch,
        "best_val_total_bce": best_val_total_bce,
        "log": str(log_path),
        "epochs_run": len(epoch_records) - 1,  # minus the init row
        "snapshot_path": str(snapshot_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--frozen-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-frozen", type=float, default=1e-3)
    ap.add_argument("--lr-unfrozen", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-end-date",
        type=str,
        default="2026-02-01",
        help=(
            "ISO date (YYYY-MM-DD). Only shards with date < this are used for "
            "fine-tuning. Default 2026-02-01 — Feb+Mar reserved for Gate 2 eval."
        ),
    )
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()

    res = run_finetune(
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        out_dir=args.out_dir,
        symbols=args.symbols,
        epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        batch_size=args.batch_size,
        lr_frozen=args.lr_frozen,
        lr_unfrozen=args.lr_unfrozen,
        seed=args.seed,
        train_end_date=args.train_end_date,
        max_hours=args.max_hours,
    )
    print(json.dumps(res, indent=2))
    return 1 if res.get("aborted", False) else 0


if __name__ == "__main__":
    sys.exit(main())

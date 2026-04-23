# scripts/temporal_stability.py
"""Angle 1 — probe the existing encoder on monthly slices of training data.

Tests "is the H500 signal stable over time?" by running the head-to-head
predictor comparison on separate months WITHIN the encoder's training
distribution. All data is pre-April; the encoder was trained on all of it.
Drift across months signals fragile representation; stability signals
generalization potential.

For each month in {2025-11, 2025-12, 2026-01, 2026-02, 2026-03}:
  - Filter cache shards to that month
  - Stratified sample (per_symbol default 2000, smaller than full-training
    head-to-head to keep runtime manageable across 5 months)
  - Forward through frozen encoder + extract flat features
  - Run 5 predictors: encoder_lr, pca_lr, rp_lr, majority, shuffled_pca_lr
  - Per-symbol 80/20 time-ordered split

Output: runs/step3-r1/temporal-stability.json + pretty per-month table.

Usage:
    uv run python scripts/temporal_stability.py \
        --checkpoint runs/step3-r1/encoder.pt \
        --cache data/cache \
        --out runs/step3-r1/temporal-stability.json \
        --per-symbol 2000 --horizon 500
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from tape.constants import (
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_EVAL,
    STRIDE_PRETRAIN,
)
from tape.dataset import TapeDataset
from tape.flat_features import window_to_flat
from tape.model import EncoderConfig, TapeEncoder


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEFAULT_MONTHS = [
    ("2025-11", "November"),
    ("2025-12", "December"),
    ("2026-01", "January"),
    ("2026-02", "February"),
    ("2026-03", "March"),
]

_MONTH_NAMES = {
    "2025-10": "October",
    "2025-11": "November",
    "2025-12": "December",
    "2026-01": "January",
    "2026-02": "February",
    "2026-03": "March",
    "2026-04": "April",
}


def _months_from_arg(arg: list[str] | None) -> list[tuple[str, str]]:
    if not arg:
        return DEFAULT_MONTHS
    out = []
    for prefix in arg:
        out.append((prefix, _MONTH_NAMES.get(prefix, prefix)))
    return out


def _month_shards(cache_dir: Path, month_prefix: str) -> list[Path]:
    shards = []
    for sym in PRETRAINING_SYMBOLS:
        if sym == HELD_OUT_SYMBOL:
            continue
        for p in sorted(cache_dir.glob(f"{sym}__{month_prefix}-*.npz")):
            shards.append(p)
    return shards


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _stratified_sample(dataset: TapeDataset, per_symbol: int, seed: int) -> np.ndarray:
    by_symbol: dict[str, list[int]] = defaultdict(list)
    for idx, ref in enumerate(dataset._refs):  # noqa: SLF001
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


def _forward_and_extract(
    enc: TapeEncoder,
    dataset: TapeDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    horizon: int,
) -> dict:
    enc_embs: list[np.ndarray] = []
    flat_feats: list[np.ndarray] = []
    symbols: list[str] = []
    labels: list[int] = []
    masks: list[bool] = []
    with torch.no_grad():
        for bstart in range(0, len(indices), batch_size):
            bend = min(bstart + batch_size, len(indices))
            batch = [dataset[int(indices[i])] for i in range(bstart, bend)]
            feats_t = torch.stack([b["features"] for b in batch]).to(device)
            _, g = enc(feats_t)
            enc_embs.append(g.cpu().numpy())
            for b in batch:
                w = b["features"].numpy()
                flat_feats.append(window_to_flat(w))
                symbols.append(b["symbol"])
                labels.append(int(b[f"label_h{horizon}"]))
                masks.append(bool(b[f"label_h{horizon}_mask"]))
    return {
        "enc": np.concatenate(enc_embs, axis=0),
        "flat": np.stack(flat_feats),
        "symbols": np.array(symbols),
        "labels": np.array(labels, dtype=np.int64),
        "masks": np.array(masks, dtype=bool),
    }


def _per_symbol_probe(
    X: np.ndarray,
    symbols: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    *,
    predictor: str,
    seed: int = 0,
    test_frac: float = 0.2,
    min_valid: int = 200,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym in sorted(set(symbols.tolist())):
        sym_mask = symbols == sym
        Xs = X[sym_mask]
        ys = y[sym_mask]
        ms = masks[sym_mask]
        valid_idx = np.where(ms)[0]
        if len(valid_idx) < min_valid:
            continue
        n_valid = len(valid_idx)
        n_test = max(50, int(n_valid * test_frac))
        tr = valid_idx[: n_valid - n_test]
        te = valid_idx[n_valid - n_test :]
        if len(np.unique(ys[tr])) < 2 or len(np.unique(ys[te])) < 2:
            if predictor == "majority":
                majority = int(np.bincount(ys[tr]).argmax())
                pred = np.full(len(te), majority)
                out[sym] = float(balanced_accuracy_score(ys[te], pred))
            continue
        Xtr, Xte = Xs[tr], Xs[te]
        ytr, yte = ys[tr], ys[te]
        rng = np.random.default_rng(seed)

        if predictor == "encoder_lr":
            scaler = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=1.0, max_iter=1_000).fit(
                scaler.transform(Xtr), ytr
            )
            pred = lr.predict(scaler.transform(Xte))
        elif predictor == "pca_lr":
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
            pca = PCA(n_components=min(20, Xtr_s.shape[1])).fit(Xtr_s)
            lr = LogisticRegression(C=1.0, max_iter=1_000).fit(
                pca.transform(Xtr_s), ytr
            )
            pred = lr.predict(pca.transform(Xte_s))
        elif predictor == "rp_lr":
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
            rp = GaussianRandomProjection(
                n_components=min(20, Xtr_s.shape[1]), random_state=seed
            ).fit(Xtr_s)
            lr = LogisticRegression(C=1.0, max_iter=1_000).fit(rp.transform(Xtr_s), ytr)
            pred = lr.predict(rp.transform(Xte_s))
        elif predictor == "majority":
            majority = int(np.bincount(ytr).argmax())
            pred = np.full(len(te), majority)
        elif predictor == "shuffled_pca_lr":
            ytr_shuf = rng.permutation(ytr)
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
            pca = PCA(n_components=min(20, Xtr_s.shape[1])).fit(Xtr_s)
            lr = LogisticRegression(C=1.0, max_iter=1_000).fit(
                pca.transform(Xtr_s), ytr_shuf
            )
            pred = lr.predict(pca.transform(Xte_s))
        else:
            raise ValueError(predictor)
        out[sym] = float(balanced_accuracy_score(yte, pred))
    return out


def _stats(d: dict[str, float]) -> dict:
    vals = list(d.values())
    if not vals:
        return {"n": 0, "mean": None, "above_0.510": 0, "above_0.514": 0}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "above_0.510": sum(1 for v in vals if v >= 0.510),
        "above_0.514": sum(1 for v in vals if v >= 0.514),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-symbol", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=500, choices=[10, 50, 100, 500])
    ap.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="ISO YYYY-MM prefixes to evaluate (e.g. 2026-02 2026-03 2026-04)",
    )
    ap.add_argument(
        "--mode",
        choices=["pretrain", "eval"],
        default="pretrain",
        help="Stride: pretrain=50 (dense), eval=200 (no window overlap)",
    )
    args = ap.parse_args()
    MONTHS = _months_from_arg(args.months)
    stride = STRIDE_PRETRAIN if args.mode == "pretrain" else STRIDE_EVAL

    device = _pick_device()
    print(f"[temporal] device={device} horizon=H{args.horizon}")

    enc = _load_encoder(args.checkpoint, device)
    print(f"[temporal] encoder params = {sum(p.numel() for p in enc.parameters()):,}")

    per_month_results: dict[str, dict] = {}
    for month_prefix, month_name in MONTHS:
        shards = _month_shards(args.cache, month_prefix)
        if not shards:
            print(f"[temporal] {month_name} — no shards, skipping")
            continue
        dataset = TapeDataset(shards, stride=stride, mode=args.mode)
        indices = _stratified_sample(dataset, args.per_symbol, args.seed)
        print(
            f"[temporal] {month_name} ({month_prefix}): {len(shards)} shards, "
            f"{len(dataset):,} windows, sample {len(indices):,}"
        )

        t0 = time.time()
        data = _forward_and_extract(
            enc, dataset, indices, device, args.batch_size, args.horizon
        )
        elapsed = time.time() - t0
        print(f"  forward+extract elapsed {elapsed:.1f}s")

        month_summary: dict[str, dict] = {}
        month_per_symbol: dict[str, dict[str, float]] = {}
        for pname in [
            "encoder_lr",
            "pca_lr",
            "rp_lr",
            "majority",
            "shuffled_pca_lr",
        ]:
            X = data["enc"] if pname == "encoder_lr" else data["flat"]
            probe = _per_symbol_probe(
                X,
                data["symbols"],
                data["labels"],
                data["masks"],
                predictor=pname,
                seed=args.seed,
            )
            month_per_symbol[pname] = probe
            month_summary[pname] = _stats(probe)
        per_month_results[month_prefix] = {
            "summary": month_summary,
            "per_symbol": month_per_symbol,
            "windows": int(len(dataset)),
            "sample": int(len(indices)),
        }

        print(f"  {'predictor':<20s} {'mean':>8s} {'≥0.510':>7s} {'≥0.514':>7s}")
        for pname, s in month_summary.items():
            mean = f"{s['mean']:.4f}" if s["mean"] is not None else "N/A"
            print(
                f"  {pname:<20s} {mean:>8s} "
                f"{s['above_0.510']:>7d} {s['above_0.514']:>7d}"
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "horizon": args.horizon,
        "per_symbol": args.per_symbol,
        "months": per_month_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Cross-month summary
    print("\n=== CROSS-MONTH SUMMARY (H" + str(args.horizon) + ") ===")
    months_found = list(per_month_results.keys())
    print(f"{'predictor':<20s}  " + "  ".join(f"{m:>10s}" for m in months_found))
    for pname in [
        "encoder_lr",
        "pca_lr",
        "rp_lr",
        "majority",
        "shuffled_pca_lr",
    ]:
        row = f"{pname:<20s}  "
        for m in months_found:
            s = per_month_results[m]["summary"][pname]
            above = s["above_0.514"]
            n = s["n"]
            row += (
                f"{above:>3d}/{n:<2d}(mn{s['mean']:.3f})  "
                if s["mean"]
                else f"{'N/A':>10s}  "
            )
        print(row)
    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

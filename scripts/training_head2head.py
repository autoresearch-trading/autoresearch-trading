# scripts/training_head2head.py
"""Head-to-head on TRAINING data: encoder vs flat-feature baselines.

Decisive test for whether the SSL encoder has learned anything useful over
PCA/RP on simple summary statistics. Per-symbol 80/20 time-ordered split,
matched across all predictors, at a sample size large enough to discriminate.

Rationale: the April pre-flight was too underpowered (15-20 test windows
per symbol, shuffled-labels noise floor ~10/24 above 0.514) to tell
encoder from baselines. On pre-April training data with 5000 windows per
symbol (× 24 symbols = 120K total), the 80/20 split gives ~1000 test
windows per symbol — binomial SE ~1.6pp — sharply discriminating.

Predictors (all evaluated on the same windows, same split):
- Encoder-256 + LR: the learned 256-dim embedding + logistic regression.
- PCA(20) + LR on flat-83: the Gate 0 baseline.
- RandomProjection(20) + LR on flat-83: adaptive-structure control.
- Majority-class: true noise floor.
- Shuffled-labels PCA+LR: null-hypothesis pipeline check.

Decision rule: if encoder beats PCA+LR AND RP+LR by ≥1pp balanced accuracy
on 15+/24 symbols, SSL learned something incremental → run-1 is justified.
If not, SSL has no incremental signal over flat stats → pivot.

Usage:
    uv run python scripts/training_head2head.py \
        --checkpoint runs/step3-r1/encoder.pt \
        --cache data/cache \
        --out runs/step3-r1/training-head2head.json \
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from tape.constants import (
    APRIL_HELDOUT_START,
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
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


def _pretrain_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    """Pre-April shards for pretraining symbols (AVAX excluded)."""
    shards = []
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
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    enc = enc.to(device)
    enc.eval()
    return enc


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
    batch_size: int = 64,
    horizon: int = 100,
) -> dict:
    """Forward all indices through encoder; also extract 83-dim flat features.

    Returns (encoder_embeddings, flat_features, symbols, labels, masks) aligned.
    """
    n = len(indices)
    enc_embs: list[np.ndarray] = []
    flat_feats: list[np.ndarray] = []
    symbols: list[str] = []
    labels: list[int] = []
    masks: list[bool] = []
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch_items = [dataset[int(indices[i])] for i in range(bstart, bend)]
            feats_t = torch.stack([b["features"] for b in batch_items]).to(device)
            _, g = enc(feats_t)
            enc_embs.append(g.cpu().numpy())
            for b in batch_items:
                w = b["features"].numpy()  # (200, 17)
                flat_feats.append(window_to_flat(w))
                symbols.append(b["symbol"])
                labels.append(int(b[f"label_h{horizon}"]))
                masks.append(bool(b[f"label_h{horizon}_mask"]))
            if bstart % (batch_size * 40) == 0:
                elapsed = time.time() - t0
                done = bend
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"  {done:,}/{n:,}  {rate:.0f} items/s  ETA {eta:.0f}s")
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
    """Per-symbol balanced accuracy via time-ordered 80/20 split."""
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
            raise ValueError(f"unknown predictor {predictor!r}")
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
    ap.add_argument("--per-symbol", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--horizon",
        type=int,
        default=100,
        choices=[10, 50, 100, 500],
        help="Direction horizon: label_h{H}",
    )
    args = ap.parse_args()

    device = _pick_device()
    syms = list(PRETRAINING_SYMBOLS)
    shards = _pretrain_shards(args.cache, syms)
    print(f"[h2h] device={device} shards={len(shards)}")
    dataset = TapeDataset(shards, stride=STRIDE_PRETRAIN, mode="pretrain")
    print(f"[h2h] dataset windows = {len(dataset):,}")

    indices = _stratified_sample(dataset, args.per_symbol, args.seed)
    print(f"[h2h] stratified sample = {len(indices):,}")

    enc = _load_encoder(args.checkpoint, device)
    print(f"[h2h] encoder params = {sum(p.numel() for p in enc.parameters()):,}")

    print("[h2h] forward + flat feature extraction...")
    t0 = time.time()
    data = _forward_and_extract(
        enc, dataset, indices, device, args.batch_size, horizon=args.horizon
    )
    print(f"[h2h] extraction elapsed = {time.time() - t0:.1f}s")

    print("[h2h] running predictors (80/20 per-symbol)...")
    results: dict[str, dict[str, float]] = {}
    for name, X, key in [
        ("encoder_lr", data["enc"], "encoder_lr"),
        ("pca_lr", data["flat"], "pca_lr"),
        ("rp_lr", data["flat"], "rp_lr"),
        ("majority", data["flat"], "majority"),  # X unused for majority
        ("shuffled_pca_lr", data["flat"], "shuffled_pca_lr"),
    ]:
        t0 = time.time()
        results[name] = _per_symbol_probe(
            X,
            data["symbols"],
            data["labels"],
            data["masks"],
            predictor=key,
            seed=args.seed,
        )
        print(f"  {name:<18s} elapsed {time.time() - t0:.1f}s")

    summary = {name: _stats(d) for name, d in results.items()}

    report = {
        "checkpoint": str(args.checkpoint),
        "per_symbol_sample": args.per_symbol,
        "total_windows": int(len(indices)),
        "results": results,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Head-to-head decision table
    print("\n=== HEAD-TO-HEAD ON TRAINING DATA ===")
    print(f"{'predictor':<20s} {'n':>4s} {'mean':>8s} {'≥0.510':>7s} {'≥0.514':>7s}")
    for name in ["encoder_lr", "pca_lr", "rp_lr", "majority", "shuffled_pca_lr"]:
        s = summary[name]
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "N/A"
        print(
            f"  {name:<18s} {s['n']:>4d} {mean:>8s} "
            f"{s['above_0.510']:>7d} {s['above_0.514']:>7d}"
        )

    # Decision rule
    enc_stats = summary["encoder_lr"]
    pca_stats = summary["pca_lr"]
    rp_stats = summary["rp_lr"]
    if (
        enc_stats["mean"] is None
        or pca_stats["mean"] is None
        or rp_stats["mean"] is None
    ):
        print("\n[DECISION] UNDETERMINED — some predictor returned no symbols.")
        return 0

    enc_vs_pca = enc_stats["mean"] - pca_stats["mean"]
    enc_vs_rp = enc_stats["mean"] - rp_stats["mean"]
    print(f"\nEncoder - PCA+LR mean = {enc_vs_pca:+.4f}")
    print(f"Encoder - RP+LR  mean = {enc_vs_rp:+.4f}")

    # Per-symbol wins for encoder vs each flat baseline
    enc_per_sym = results["encoder_lr"]
    pca_per_sym = results["pca_lr"]
    rp_per_sym = results["rp_lr"]
    wins_vs_pca = sum(
        1
        for s in enc_per_sym
        if s in pca_per_sym and enc_per_sym[s] >= pca_per_sym[s] + 0.01
    )
    wins_vs_rp = sum(
        1
        for s in enc_per_sym
        if s in rp_per_sym and enc_per_sym[s] >= rp_per_sym[s] + 0.01
    )
    print(f"Encoder beats PCA+LR by ≥1pp on: {wins_vs_pca}/{len(enc_per_sym)} symbols")
    print(f"Encoder beats RP+LR  by ≥1pp on: {wins_vs_rp}/{len(enc_per_sym)} symbols")

    if wins_vs_pca >= 15 and wins_vs_rp >= 15:
        verdict = "GO — encoder has incremental signal, run-1 justified"
    elif wins_vs_pca >= 10 or wins_vs_rp >= 10:
        verdict = "MARGINAL — encoder occasionally wins; Gate 1 unlikely without recipe change"
    else:
        verdict = "NO-GO — SSL encoder not beating flat baselines; pivot"
    print(f"\n[DECISION] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# scripts/april_bootstrap.py
"""Angle 3 — bootstrap the April H100/H500 per-symbol probe to get CIs.

Reuses the encoder forward pass on April from scripts/april_gate1_preflight.py
and scripts/april_diagnostics.py, but computes N=100 independent seeds for the
80/20 time-ordered split (randomizing the seed in the probe's LR estimator
and in the random projection). Gives a confidence interval on
"# symbols ≥ 0.514" for each predictor.

The goal: is the 11/24 encoder result distinguishable from the ~9/24 shuffled
floor at current April sample size? If the 95% CIs overlap, April evaluation
is terminally underpowered for our sample size.

Usage:
    uv run python scripts/april_bootstrap.py \
        --checkpoint runs/step3-r1/encoder.pt \
        --cache data/cache \
        --out runs/step3-r1/april-bootstrap.json \
        --horizon 100 --n-seeds 100
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

from tape.constants import APRIL_HELDOUT_START, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.dataset import TapeDataset
from tape.flat_features import window_to_flat
from tape.model import EncoderConfig, TapeEncoder


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _april_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    APRIL_START = "2026-04-01"
    out = []
    for sym in symbols:
        for p in sorted(cache_dir.glob(f"{sym}__2026-04-*.npz")):
            date_part = p.stem.split("__", 1)[1]
            if APRIL_START <= date_part < APRIL_HELDOUT_START:
                out.append(p)
    return out


def _load_encoder(ckpt: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _forward(
    enc: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    batch_size: int,
    horizon: int,
) -> dict:
    enc_embs: list[np.ndarray] = []
    flat_feats: list[np.ndarray] = []
    symbols: list[str] = []
    labels: list[int] = []
    masks: list[bool] = []
    n = len(dataset)
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch = [dataset[i] for i in range(bstart, bend)]
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


def _probe_one_seed(
    data: dict,
    *,
    seed: int,
    test_frac: float = 0.2,
    min_valid: int = 40,
) -> dict[str, dict[str, float]]:
    """Return per-symbol balanced acc for each predictor for this seed."""
    out: dict[str, dict[str, float]] = {
        "encoder_lr": {},
        "pca_lr": {},
        "rp_lr": {},
        "majority": {},
        "shuffled_pca_lr": {},
    }
    rng = np.random.default_rng(seed)
    symbols = data["symbols"]
    for sym in sorted(set(symbols.tolist())):
        sym_mask = symbols == sym
        X_enc = data["enc"][sym_mask]
        X_flat = data["flat"][sym_mask]
        ys = data["labels"][sym_mask]
        ms = data["masks"][sym_mask]
        valid_idx = np.where(ms)[0]
        if len(valid_idx) < min_valid:
            continue
        # RANDOM 80/20 this time (for bootstrap over split seeds)
        perm = rng.permutation(len(valid_idx))
        n_test = max(1, int(len(valid_idx) * test_frac))
        te = valid_idx[perm[:n_test]]
        tr = valid_idx[perm[n_test:]]
        if len(np.unique(ys[tr])) < 2:
            continue
        if len(np.unique(ys[te])) < 2:
            continue

        ytr, yte = ys[tr], ys[te]

        # Encoder + LR
        Xtr_e = X_enc[tr]
        Xte_e = X_enc[te]
        sc = StandardScaler().fit(Xtr_e)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(sc.transform(Xtr_e), ytr)
        out["encoder_lr"][sym] = float(
            balanced_accuracy_score(yte, lr.predict(sc.transform(Xte_e)))
        )

        # Flat-83 based
        Xtr_f = X_flat[tr]
        Xte_f = X_flat[te]
        sc_f = StandardScaler().fit(Xtr_f)
        Xtr_fs = sc_f.transform(Xtr_f)
        Xte_fs = sc_f.transform(Xte_f)

        # PCA + LR
        pca = PCA(n_components=min(20, Xtr_fs.shape[1])).fit(Xtr_fs)
        lr_p = LogisticRegression(C=1.0, max_iter=1_000).fit(pca.transform(Xtr_fs), ytr)
        out["pca_lr"][sym] = float(
            balanced_accuracy_score(yte, lr_p.predict(pca.transform(Xte_fs)))
        )

        # RP + LR (seed depends on outer seed for bootstrap)
        rp = GaussianRandomProjection(
            n_components=min(20, Xtr_fs.shape[1]), random_state=seed
        ).fit(Xtr_fs)
        lr_r = LogisticRegression(C=1.0, max_iter=1_000).fit(rp.transform(Xtr_fs), ytr)
        out["rp_lr"][sym] = float(
            balanced_accuracy_score(yte, lr_r.predict(rp.transform(Xte_fs)))
        )

        # Majority
        maj = int(np.bincount(ytr).argmax())
        out["majority"][sym] = float(
            balanced_accuracy_score(yte, np.full(len(te), maj))
        )

        # Shuffled PCA+LR
        ytr_shuf = rng.permutation(ytr)
        lr_s = LogisticRegression(C=1.0, max_iter=1_000).fit(
            pca.transform(Xtr_fs), ytr_shuf
        )
        out["shuffled_pca_lr"][sym] = float(
            balanced_accuracy_score(yte, lr_s.predict(pca.transform(Xte_fs)))
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=100, choices=[10, 50, 100, 500])
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = _pick_device()
    print(f"[bootstrap] device={device} horizon=H{args.horizon} n_seeds={args.n_seeds}")

    syms = list(PRETRAINING_SYMBOLS)
    shards = _april_shards(args.cache, syms)
    print(f"[bootstrap] April shards = {len(shards)}")
    dataset = TapeDataset(shards, stride=STRIDE_EVAL, mode="eval")
    print(f"[bootstrap] April windows (stride={STRIDE_EVAL}): {len(dataset):,}")

    enc = _load_encoder(args.checkpoint, device)
    print(f"[bootstrap] forward all windows...")
    t0 = time.time()
    data = _forward(enc, dataset, device, args.batch_size, args.horizon)
    print(f"[bootstrap] forward elapsed = {time.time() - t0:.1f}s")

    # Run N seeds, collect # symbols ≥ 0.514 per predictor per seed
    print(f"[bootstrap] running {args.n_seeds} bootstrap seeds...")
    predictors = ["encoder_lr", "pca_lr", "rp_lr", "majority", "shuffled_pca_lr"]
    counts_514: dict[str, list[int]] = {p: [] for p in predictors}
    counts_510: dict[str, list[int]] = {p: [] for p in predictors}
    means: dict[str, list[float]] = {p: [] for p in predictors}

    t0 = time.time()
    for seed in range(args.n_seeds):
        probe = _probe_one_seed(data, seed=seed)
        for p in predictors:
            vals = list(probe[p].values())
            if not vals:
                continue
            counts_514[p].append(sum(1 for v in vals if v >= 0.514))
            counts_510[p].append(sum(1 for v in vals if v >= 0.510))
            means[p].append(float(np.mean(vals)))
        if (seed + 1) % 20 == 0:
            print(f"  {seed + 1}/{args.n_seeds}  elapsed {time.time() - t0:.1f}s")

    # Compute percentiles for 95% CI
    summary: dict[str, dict] = {}
    for p in predictors:
        if not counts_514[p]:
            continue
        ct = np.array(counts_514[p])
        ct10 = np.array(counts_510[p])
        mn = np.array(means[p])
        summary[p] = {
            "n_seeds": len(ct),
            "above_0.514": {
                "mean": float(ct.mean()),
                "median": float(np.median(ct)),
                "ci_95_low": float(np.percentile(ct, 2.5)),
                "ci_95_high": float(np.percentile(ct, 97.5)),
                "min": int(ct.min()),
                "max": int(ct.max()),
            },
            "above_0.510": {
                "mean": float(ct10.mean()),
                "ci_95_low": float(np.percentile(ct10, 2.5)),
                "ci_95_high": float(np.percentile(ct10, 97.5)),
            },
            "direction_mean": {
                "mean": float(mn.mean()),
                "ci_95_low": float(np.percentile(mn, 2.5)),
                "ci_95_high": float(np.percentile(mn, 97.5)),
            },
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "horizon": args.horizon,
        "n_seeds": args.n_seeds,
        "n_windows_april": int(len(dataset)),
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Pretty print
    print(f"\n=== APRIL BOOTSTRAP (H{args.horizon}, n_seeds={args.n_seeds}) ===")
    print(
        f"{'predictor':<20s} {'mean ≥.514':>12s} {'95% CI ≥.514':>18s} "
        f"{'mean bal-acc':>14s} {'95% CI bal-acc':>20s}"
    )
    for p in predictors:
        if p not in summary:
            continue
        s = summary[p]
        c = s["above_0.514"]
        dm = s["direction_mean"]
        ci_514 = f"[{c['ci_95_low']:.1f}, {c['ci_95_high']:.1f}]"
        ci_mean = f"[{dm['ci_95_low']:.3f}, {dm['ci_95_high']:.3f}]"
        print(
            f"  {p:<18s} {c['mean']:>12.2f} {ci_514:>18s}  "
            f"{dm['mean']:>13.4f} {ci_mean:>20s}"
        )

    # Decisive CI overlap check: encoder vs shuffled
    enc_ci = summary["encoder_lr"]["above_0.514"]
    shuf_ci = summary["shuffled_pca_lr"]["above_0.514"]
    pca_ci = summary["pca_lr"]["above_0.514"]
    print(
        f"\nEncoder 95% CI: [{enc_ci['ci_95_low']:.1f}, {enc_ci['ci_95_high']:.1f}]  "
        f"vs Shuffled CI: [{shuf_ci['ci_95_low']:.1f}, {shuf_ci['ci_95_high']:.1f}]"
    )
    enc_low = enc_ci["ci_95_low"]
    shuf_high = shuf_ci["ci_95_high"]
    pca_high = pca_ci["ci_95_high"]
    if enc_low > shuf_high:
        print("✓ Encoder CI is ABOVE shuffled CI — statistically distinguishable")
    else:
        print("✗ Encoder CI OVERLAPS with shuffled — April is terminally underpowered")
    if enc_low > pca_high:
        print("✓ Encoder CI is ABOVE PCA CI — encoder adds real signal")
    else:
        print("✗ Encoder CI OVERLAPS with PCA — encoder does NOT beat flat baselines")

    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

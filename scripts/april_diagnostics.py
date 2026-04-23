# scripts/april_diagnostics.py
"""Three diagnostics to calibrate our April pre-flight result.

Motivated by the concerns raised after the Phase-2 pre-flight (2026-04-23):
- The per-symbol direction-probe distribution on April is bimodal (0.28 to
  0.72, mean 0.489) — could be small-sample noise, could be regime divergence.
- We don't know the actual Gate 1 bar on April (conditions 2-3 require
  beating Majority+1pp and RP+1pp).
- Training events/day (~20K on BTC) vs April events/day (~4K) is a 5x density
  shift that the encoder never saw.

Produces a single JSON report with:

  A. Gate 0-style baselines on April, matched protocol to the Phase-2
     encoder probe (per-symbol 80/20 time-ordered split):
       - Majority-class predictor
       - Random projection (83->20) + LR on flat 83-dim features
       - PCA (20) + LR on flat 83-dim features
       - Shuffled-labels sanity (PCA+LR on shuffled y)
     All measured as balanced accuracy per symbol.

  B. Label imbalance per symbol in April: fraction of label_h100 == 1 among
     valid (mask=True) windows.

  C. Event density per symbol, March vs April: median events/day.

Output: runs/step3-r1/april-diagnostics.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from tape.constants import APRIL_HELDOUT_START, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.dataset import TapeDataset
from tape.flat_features import window_to_flat


def _iso_shards(cache_dir: Path, pattern_suffix: str) -> list[Path]:
    """Sorted .npz paths where filename ends with pattern_suffix."""
    return sorted(cache_dir.glob(f"*__{pattern_suffix}"))


def _april_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    APRIL_START = "2026-04-01"
    shards = []
    for sym in symbols:
        for p in sorted(cache_dir.glob(f"{sym}__2026-04-*.npz")):
            date_part = p.stem.split("__", 1)[1]
            if APRIL_START <= date_part < APRIL_HELDOUT_START:
                shards.append(p)
    return shards


def _march_shards(cache_dir: Path, symbols: list[str]) -> list[Path]:
    shards = []
    for sym in symbols:
        for p in sorted(cache_dir.glob(f"{sym}__2026-03-*.npz")):
            shards.append(p)
    return shards


def _extract_flat_features(dataset: TapeDataset, horizon: int = 100) -> dict:
    """Forward all windows; extract 83-dim flat features + labels + symbol."""
    feats: list[np.ndarray] = []
    symbols: list[str] = []
    labels_h100: list[int] = []
    masks_h100: list[bool] = []
    lkey = f"label_h{horizon}"
    mkey = f"label_h{horizon}_mask"
    for i in range(len(dataset)):
        item = dataset[i]
        w = item["features"].numpy()  # (200, 17)
        feats.append(window_to_flat(w))
        symbols.append(item["symbol"])
        labels_h100.append(int(item[lkey]))
        masks_h100.append(bool(item[mkey]))
    return {
        "feats": np.stack(feats),
        "symbols": np.array(symbols),
        "labels": np.array(labels_h100, dtype=np.int64),
        "masks": np.array(masks_h100, dtype=bool),
    }


def _per_symbol_split_balacc(
    feats: np.ndarray,
    symbols: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    *,
    predictor: str,
    seed: int = 0,
    test_frac: float = 0.2,
    min_valid: int = 40,
) -> dict[str, float]:
    """Per-symbol balanced accuracy with a single 80/20 time-ordered split.

    predictor ∈ {"pca_lr", "rp_lr", "majority", "shuffled_pca_lr"}
    """
    out: dict[str, float] = {}
    for sym in sorted(set(symbols.tolist())):
        sym_mask = symbols == sym
        sym_feats = feats[sym_mask]
        sym_y = labels[sym_mask]
        sym_m = masks[sym_mask]
        valid_idx = np.where(sym_m)[0]
        if len(valid_idx) < min_valid:
            continue
        n_valid = len(valid_idx)
        n_test = max(1, int(n_valid * test_frac))
        tr_idx = valid_idx[: n_valid - n_test]
        te_idx = valid_idx[n_valid - n_test :]
        if len(np.unique(sym_y[tr_idx])) < 2 or len(np.unique(sym_y[te_idx])) < 2:
            # Degenerate split. Skip Majority also falls through — instead
            # use majority of train on test.
            if predictor == "majority":
                majority_class = int(np.bincount(sym_y[tr_idx]).argmax())
                pred = np.full(len(te_idx), majority_class)
                out[sym] = float(balanced_accuracy_score(sym_y[te_idx], pred))
            continue

        Xtr = sym_feats[tr_idx]
        Xte = sym_feats[te_idx]
        ytr = sym_y[tr_idx]
        yte = sym_y[te_idx]

        rng = np.random.default_rng(seed)
        if predictor == "pca_lr":
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
            majority_class = int(np.bincount(ytr).argmax())
            pred = np.full(len(yte), majority_class)
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


def _label_imbalance(
    labels: np.ndarray, masks: np.ndarray, symbols: np.ndarray
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sym in sorted(set(symbols.tolist())):
        m = (symbols == sym) & masks
        y = labels[m]
        if len(y) == 0:
            continue
        p_up = float(y.mean())
        out[sym] = {
            "n_valid": int(len(y)),
            "p_up": p_up,
            "skew": float(abs(p_up - 0.5)),
        }
    return out


def _event_density(shards: list[Path]) -> dict[str, float]:
    """Median events per shard-day, per symbol."""
    by_sym: dict[str, list[int]] = defaultdict(list)
    for p in shards:
        with np.load(p, allow_pickle=False) as d:
            if "features" in d.files:
                n = int(d["features"].shape[0])
                by_sym[p.stem.split("__")[0]].append(n)
    return {sym: float(np.median(ns)) for sym, ns in by_sym.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--horizon",
        type=int,
        default=100,
        choices=[10, 50, 100, 500],
        help="Direction horizon to evaluate (default 100)",
    )
    args = ap.parse_args()

    syms = list(PRETRAINING_SYMBOLS)
    april_shards = _april_shards(args.cache, syms)
    march_shards = _march_shards(args.cache, syms)
    print(f"[diag] April shards: {len(april_shards)}")
    print(f"[diag] March shards: {len(march_shards)}")

    # --- C. Event density ---
    print("[diag] computing event density (C)...")
    april_density = _event_density(april_shards)
    march_density = _event_density(march_shards)

    # --- Flat features for April ---
    print("[diag] extracting flat 83-dim features on April...")
    april_ds = TapeDataset(april_shards, stride=STRIDE_EVAL, mode="eval")
    data = _extract_flat_features(april_ds, horizon=args.horizon)

    # --- B. Label imbalance on April ---
    print("[diag] computing label imbalance (B)...")
    imbalance = _label_imbalance(data["labels"], data["masks"], data["symbols"])

    # --- A. Baselines on April ---
    print("[diag] running PCA+LR baseline (A)...")
    pca_lr = _per_symbol_split_balacc(
        data["feats"],
        data["symbols"],
        data["labels"],
        data["masks"],
        predictor="pca_lr",
        seed=args.seed,
    )
    print("[diag] running RP+LR baseline (A)...")
    rp_lr = _per_symbol_split_balacc(
        data["feats"],
        data["symbols"],
        data["labels"],
        data["masks"],
        predictor="rp_lr",
        seed=args.seed,
    )
    print("[diag] running Majority-class baseline (A)...")
    majority = _per_symbol_split_balacc(
        data["feats"],
        data["symbols"],
        data["labels"],
        data["masks"],
        predictor="majority",
        seed=args.seed,
    )
    print("[diag] running Shuffled-labels sanity (A)...")
    shuffled = _per_symbol_split_balacc(
        data["feats"],
        data["symbols"],
        data["labels"],
        data["masks"],
        predictor="shuffled_pca_lr",
        seed=args.seed,
    )

    # Load the Phase-2 preflight result for direct comparison
    preflight_path = Path("runs/step3-r1/april-preflight.json")
    encoder_per_sym: dict[str, float] = {}
    if preflight_path.exists():
        encoder_per_sym = json.loads(preflight_path.read_text())[
            "direction_h100_per_symbol"
        ]

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

    report = {
        "event_density_median_per_day": {
            "march": march_density,
            "april": april_density,
            "density_ratio_march_over_april": {
                sym: (
                    (march_density[sym] / april_density[sym])
                    if sym in march_density
                    and sym in april_density
                    and april_density[sym] > 0
                    else None
                )
                for sym in sorted(set(march_density) | set(april_density))
            },
        },
        "label_imbalance_april": imbalance,
        "april_gate0_per_symbol": {
            "encoder_phase2_preflight": encoder_per_sym,
            "pca_lr_flat83": pca_lr,
            "rp_lr_flat83": rp_lr,
            "majority_class": majority,
            "shuffled_labels_pca_lr": shuffled,
        },
        "april_gate0_summary": {
            "encoder_phase2_preflight": _stats(encoder_per_sym),
            "pca_lr_flat83": _stats(pca_lr),
            "rp_lr_flat83": _stats(rp_lr),
            "majority_class": _stats(majority),
            "shuffled_labels_pca_lr": _stats(shuffled),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Pretty print the summary to stdout
    print("\n=== DIAGNOSTICS SUMMARY ===")
    print("\n--- (A) Gate 0 on April — per-symbol, 80/20 time-ordered ---")
    print(f"{'predictor':<28s} {'n':>4s} {'mean':>8s} {'>=.510':>8s} {'>=.514':>8s}")
    for name in [
        "encoder_phase2_preflight",
        "pca_lr_flat83",
        "rp_lr_flat83",
        "majority_class",
        "shuffled_labels_pca_lr",
    ]:
        s = report["april_gate0_summary"][name]
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "N/A"
        print(
            f"  {name:<28s} {s['n']:>4d} {mean:>8s} "
            f"{s['above_0.510']:>8d} {s['above_0.514']:>8d}"
        )

    print("\n--- (B) Label imbalance — April H100, per-symbol |p_up - 0.5| ---")
    print("(sorted by skew, descending)")
    items = sorted(imbalance.items(), key=lambda kv: -kv[1]["skew"])
    for sym, d in items:
        flag = "🚨" if d["skew"] > 0.15 else ("≈" if d["skew"] > 0.10 else " ")
        print(
            f"  {flag} {sym:<10s} n={d['n_valid']:<5d} p_up={d['p_up']:.3f} skew={d['skew']:+.3f}"
        )

    print("\n--- (C) Event density — median events/day, March -> April ---")
    print("(sorted by March-to-April ratio, descending)")
    ratios = report["event_density_median_per_day"]["density_ratio_march_over_april"]
    items2 = sorted(
        [(s, r) for s, r in ratios.items() if r is not None], key=lambda kv: -kv[1]
    )
    for sym, ratio in items2:
        m = march_density.get(sym, 0)
        a = april_density.get(sym, 0)
        print(f"  {sym:<10s} march={m:>8.0f}  april={a:>8.0f}  ratio={ratio:.2f}x")

    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

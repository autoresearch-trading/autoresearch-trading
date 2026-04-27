# scripts/run_condition_c3.py
"""Condition 3 — ARI cluster–Wyckoff alignment (post-Gate-2 pre-registration).

Per the ratified pre-registration (commit `c28bc17`,
`docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md`):

    Condition 3 — Cluster–Wyckoff alignment (amended 2026-04-26 PM).
    - k-means with k=16 on frozen encoder embeddings of held-out
      Feb+Mar windows.
    - For each of {is_absorption, is_buying_climax, is_selling_climax,
      is_stressed}:
        Compute ARI (adjusted Rand index) between the binary Wyckoff labels
        and the 16 cluster assignments (treating clusters as a 16-class
        partition, Wyckoff labels as a 2-class partition).
    - PASS: at least 2 of 4 Wyckoff labels have ARI >= 0.05.

Implementation notes
--------------------
- All Feb+Mar windows from the 24 non-AVAX symbols are pooled into a single
  embedding matrix. k-means is fit globally (k=16). This is the spec's
  "global geometry" reading — clusters should reflect tape state regardless
  of the symbol that generated the window.
- ARI is computed per Wyckoff label across the entire pooled set. We also
  publish per-symbol per-label rates and a permutation null (1000 shuffles
  of the cluster labels) for each ARI to give context, but the TEST is the
  point ARI against 0.05 per the pre-reg.
- Underpowered guard: skip Wyckoff labels with overall positive rate < 1%
  (insufficient signal for any clustering to align with).

Output: full JSON record at --out, including:
  - per-Wyckoff ARI (point estimate)
  - per-Wyckoff permutation-null mean and 99-th percentile (information only)
  - cluster size distribution
  - per-label, per-cluster contingency table for the 4 Wyckoff variants
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from tape.constants import HELD_OUT_SYMBOL, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.probe_utils import (
    build_eval_dataset,
    forward_embeddings,
    load_encoder,
    pick_device,
    shards_for_sym_months,
)
from tape.wyckoff_labels import (
    is_absorption_window,
    is_buying_climax_window,
    is_selling_climax_window,
    is_stressed_window,
)

# Pre-registered constants (binding via commit c28bc17).
TEST_FEB_MAR: tuple[str, str] = ("2026-02", "2026-03")
K_CLUSTERS: int = 16
ARI_THRESHOLD: float = 0.05
PASS_COUNT_REQUIRED: int = 2  # ≥2 of 4 Wyckoff labels
N_PERMUTATIONS: int = 1000
MIN_POS_RATE: float = 0.01  # underpowered if <1% positive rate

WYCKOFF_LABELERS = {
    "is_absorption": is_absorption_window,
    "is_buying_climax": is_buying_climax_window,
    "is_selling_climax": is_selling_climax_window,
    "is_stressed": is_stressed_window,
}


def _permutation_null_ari(
    cluster_assignments: np.ndarray, y: np.ndarray, *, n: int, seed: int
) -> dict:
    """Compute permutation-null distribution of ARI under random label shuffle."""
    rng = np.random.default_rng(seed)
    aris = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        aris[i] = float(adjusted_rand_score(y_perm, cluster_assignments))
    return {
        "mean": float(np.mean(aris)),
        "p99": float(np.percentile(aris, 99)),
        "p95": float(np.percentile(aris, 95)),
        "max": float(np.max(aris)),
    }


def _contingency_table(
    cluster_assignments: np.ndarray, y: np.ndarray, k: int
) -> list[list[int]]:
    """Per-cluster, per-label counts (k × 2)."""
    table = np.zeros((k, 2), dtype=np.int64)
    for c in range(k):
        mask = cluster_assignments == c
        table[c, 0] = int(((y[mask] == 0)).sum())
        table[c, 1] = int(((y[mask] == 1)).sum())
    return table.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[c3] device={device}")
    enc = load_encoder(args.checkpoint, device)
    print(f"[c3] loaded encoder from {args.checkpoint}")
    print(f"[c3] test={TEST_FEB_MAR} k={K_CLUSTERS} ari_threshold={ARI_THRESHOLD}")
    print(f"[c3] PASS_COUNT_REQUIRED={PASS_COUNT_REQUIRED}/4 Wyckoff labels")

    target_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]
    print(f"[c3] pooling embeddings across {len(target_symbols)} symbols")

    # ---- 1. Embed all Feb+Mar windows across the 24 non-AVAX symbols. ----
    started = time.time()
    embs_all: list[np.ndarray] = []
    feats_all: list[np.ndarray] = []
    sym_all: list[str] = []
    for i, sym in enumerate(target_symbols, 1):
        shards = shards_for_sym_months(args.cache, sym, TEST_FEB_MAR)
        if not shards:
            print(f"[c3] [{i:2d}/{len(target_symbols)}] {sym:8s}: no shards, skip")
            continue
        ds = build_eval_dataset(shards, stride=STRIDE_EVAL)
        if len(ds) < 50:
            print(
                f"[c3] [{i:2d}/{len(target_symbols)}] {sym:8s}: "
                f"underpowered (n={len(ds)}), skip"
            )
            continue
        bundle = forward_embeddings(
            enc, ds, device, batch_size=args.batch_size, return_features=True
        )
        embs_all.append(bundle["emb"])
        feats_all.append(bundle["features"])
        sym_all.extend([sym] * len(bundle["emb"]))
        elapsed = time.time() - started
        print(
            f"[c3] [{i:2d}/{len(target_symbols)}] {sym:8s}: "
            f"n={len(bundle['emb'])} elapsed={elapsed:.0f}s"
        )

    if not embs_all:
        print("[c3] no embeddings collected — abort")
        return 1

    X = np.concatenate(embs_all, axis=0)
    feats_pooled = np.concatenate(feats_all, axis=0)
    syms_pooled = np.array(sym_all)
    print(f"[c3] pooled embedding matrix: {X.shape} (N windows × 256 dim)")

    # ---- 2. Compute per-window Wyckoff labels (4 binary vectors). ----
    print("[c3] computing per-window Wyckoff labels…")
    wyckoff_labels: dict[str, np.ndarray] = {}
    for name, fn in WYCKOFF_LABELERS.items():
        y = np.fromiter(
            (fn(feats_pooled[i]) for i in range(len(feats_pooled))), dtype=bool
        )
        wyckoff_labels[name] = y.astype(np.int64)
        print(
            f"[c3]   {name}: positive rate "
            f"{float(y.mean()):.4f} ({int(y.sum())}/{len(y)})"
        )

    # Free the features — no longer needed past this point.
    del feats_pooled, feats_all

    # ---- 3. k-means on standardized embeddings (k=16). ----
    print(f"[c3] fitting k-means (k={K_CLUSTERS}) on {X.shape[0]} windows…")
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    km = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=10,
        random_state=args.seed,
        max_iter=300,
    ).fit(X_s)
    cluster_assignments = km.labels_.astype(np.int64)
    cluster_sizes = np.bincount(cluster_assignments, minlength=K_CLUSTERS).tolist()
    print(f"[c3] cluster sizes: min={min(cluster_sizes)} max={max(cluster_sizes)}")

    # ---- 4. Per-Wyckoff ARI + permutation null. ----
    per_wyckoff: list[dict] = []
    for name, y in wyckoff_labels.items():
        rec: dict = {"label": name}
        rec["pos_rate"] = float(y.mean())
        rec["n_pos"] = int(y.sum())
        rec["n_neg"] = int(len(y) - y.sum())
        if rec["pos_rate"] < MIN_POS_RATE or rec["pos_rate"] > 1.0 - MIN_POS_RATE:
            rec["status"] = (
                f"underpowered_pos_rate (pos_rate={rec['pos_rate']:.4f}, "
                f"need >{MIN_POS_RATE})"
            )
            rec["pass_label"] = False
            rec["ari"] = None
            per_wyckoff.append(rec)
            print(
                f"[c3]   {name}: SKIP underpowered " f"(pos_rate={rec['pos_rate']:.4f})"
            )
            continue
        ari = float(adjusted_rand_score(y, cluster_assignments))
        null = _permutation_null_ari(
            cluster_assignments, y, n=N_PERMUTATIONS, seed=args.seed
        )
        rec["status"] = "ok"
        rec["ari"] = ari
        rec["null_mean"] = null["mean"]
        rec["null_p95"] = null["p95"]
        rec["null_p99"] = null["p99"]
        rec["null_max"] = null["max"]
        rec["pass_label"] = bool(ari >= ARI_THRESHOLD)
        rec["contingency_per_cluster"] = _contingency_table(
            cluster_assignments, y, K_CLUSTERS
        )
        per_wyckoff.append(rec)
        print(
            f"[c3]   {name}: ari={ari:+.4f} "
            f"null_p95={null['p95']:+.4f} pass={rec['pass_label']}"
        )

    # ---- 5. Aggregate. ----
    pass_count = sum(1 for r in per_wyckoff if r.get("pass_label"))
    c3_pass = pass_count >= PASS_COUNT_REQUIRED

    # Per-symbol cluster mass distribution (information only).
    per_symbol_clusters: dict[str, list[int]] = {}
    for sym in np.unique(syms_pooled):
        mask = syms_pooled == sym
        sym_clusters = cluster_assignments[mask]
        per_symbol_clusters[str(sym)] = np.bincount(
            sym_clusters, minlength=K_CLUSTERS
        ).tolist()

    summary = {
        "condition": "C3_cluster_wyckoff_ari_alignment",
        "test_window": list(TEST_FEB_MAR),
        "k_clusters": K_CLUSTERS,
        "ari_threshold": ARI_THRESHOLD,
        "pass_count_required": PASS_COUNT_REQUIRED,
        "pass_count": pass_count,
        "c3_pass": c3_pass,
        "n_windows_total": int(X.shape[0]),
        "cluster_sizes": cluster_sizes,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "n_permutations": N_PERMUTATIONS,
        "per_wyckoff": per_wyckoff,
        "per_symbol_cluster_counts": per_symbol_clusters,
    }

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== CONDITION 3 VERDICT ===")
    print(f"  pass_count = {pass_count}/4 (threshold {PASS_COUNT_REQUIRED}/4)")
    for r in per_wyckoff:
        if r.get("status") == "ok":
            ari = r["ari"]
            ari_str = f"{ari:+.4f}" if ari is not None else "N/A"
            print(
                f"    {r['label']}: ari={ari_str} "
                f"(pos_rate={r['pos_rate']:.3f}, pass={r['pass_label']})"
            )
        else:
            print(f"    {r['label']}: {r['status']}")
    print(f"  CONDITION 3 = {'PASS' if c3_pass else 'FAIL'}")
    print(f"  Report: {args.out}")
    return 0 if c3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

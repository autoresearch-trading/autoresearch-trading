# scripts/run_rankme.py
"""RankMe (Garrido et al. 2023) — descriptive SSL representation-quality scalar.

Per the post-Gate-2 pre-publication strengthening review (council-6,
`docs/council-reviews/2026-04-26-prepublication-strengthening-c6.md`),
RankMe is added as a zero-amendment-budget descriptive scalar alongside
embed_std and effective_rank in the program end-state writeup.

RankMe(Z) = exp( H(p) )
    where p_k = σ_k(Z) / Σ_j σ_j(Z) and σ_k are the singular values.

Higher RankMe = the encoder uses more of its 256-d output space; it is
the "perplexity" of the singular-value spectrum. Range: [1, min(N, d)].

This is a property of the embedding matrix, not a hypothesis test
against a pre-registered threshold. We publish it as descriptive
context for the program end-state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tape.constants import HELD_OUT_SYMBOL, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.probe_utils import (
    build_eval_dataset,
    forward_embeddings,
    load_encoder,
    pick_device,
    shards_for_sym_months,
)


def rankme(emb: np.ndarray, eps: float = 1e-12) -> float:
    """Compute RankMe per Garrido et al. 2023.

    emb: (N, d) embedding matrix (float).
    Returns exp(entropy of normalized singular value spectrum).
    """
    if emb.shape[0] < 2:
        return 1.0
    sv = np.linalg.svd(emb.astype(np.float64), compute_uv=False)
    p = sv / (sv.sum() + eps)
    p = p[p > eps]
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--months",
        nargs="+",
        default=["2026-02", "2026-03"],
        help="ISO YYYY-MM prefixes to evaluate",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[rankme] device={device}")
    enc = load_encoder(args.checkpoint, device)
    print(f"[rankme] loaded encoder from {args.checkpoint}")
    print(f"[rankme] months={args.months}")

    target_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]
    print(f"[rankme] pooling embeddings across {len(target_symbols)} symbols")

    embs_all: list[np.ndarray] = []
    per_symbol: dict[str, dict] = {}
    for sym in target_symbols:
        shards = shards_for_sym_months(args.cache, sym, tuple(args.months))
        if not shards:
            continue
        ds = build_eval_dataset(shards, stride=STRIDE_EVAL)
        if len(ds) < 50:
            continue
        bundle = forward_embeddings(enc, ds, device, batch_size=args.batch_size)
        emb = bundle["emb"]
        embs_all.append(emb)
        per_symbol[sym] = {
            "n_windows": int(len(emb)),
            "rankme": rankme(emb),
            "embed_std": float(emb.std()),
        }
        print(
            f"[rankme] {sym:8s}: n={len(emb):5d} "
            f"rankme={per_symbol[sym]['rankme']:.2f} "
            f"std={per_symbol[sym]['embed_std']:.3f}"
        )

    if not embs_all:
        print("[rankme] no embeddings collected — abort")
        return 1

    pooled = np.concatenate(embs_all, axis=0)
    pooled_rankme = rankme(pooled)
    pooled_std = float(pooled.std())

    summary = {
        "checkpoint": str(args.checkpoint),
        "months": args.months,
        "pooled_n_windows": int(len(pooled)),
        "pooled_embedding_dim": int(pooled.shape[1]),
        "pooled_rankme": pooled_rankme,
        "pooled_embed_std": pooled_std,
        "per_symbol": per_symbol,
        "per_symbol_rankme_mean": float(
            np.mean([v["rankme"] for v in per_symbol.values()])
        ),
        "per_symbol_rankme_median": float(
            np.median([v["rankme"] for v in per_symbol.values()])
        ),
        "per_symbol_rankme_min": float(
            np.min([v["rankme"] for v in per_symbol.values()])
        ),
        "per_symbol_rankme_max": float(
            np.max([v["rankme"] for v in per_symbol.values()])
        ),
    }
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== RankMe (descriptive scalar) ===")
    print(f"  pooled embedding matrix: {pooled.shape}")
    print(f"  pooled RankMe = {pooled_rankme:.2f} (out of {pooled.shape[1]})")
    print(f"  pooled embed_std = {pooled_std:.3f}")
    print(
        f"  per-symbol RankMe: mean={summary['per_symbol_rankme_mean']:.2f} "
        f"median={summary['per_symbol_rankme_median']:.2f} "
        f"min={summary['per_symbol_rankme_min']:.2f} "
        f"max={summary['per_symbol_rankme_max']:.2f}"
    )
    print(f"  Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

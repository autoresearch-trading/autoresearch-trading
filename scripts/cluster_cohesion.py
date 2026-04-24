# scripts/cluster_cohesion.py
"""Cluster-cohesion diagnostic on the 6 LIQUID_CONTRASTIVE_SYMBOLS.

Council-5 Rank 3 (docs/council-reviews/council-5-gate3-avax-falsifiability.md):
the key measurement that decides whether SimCLR actually forced the encoder
to produce similar embeddings for different symbols at the same market
moment, or whether it learned purely symbol-conditional features.

Primary measurement: cosine similarity of encoder embeddings between pairs
of windows from DIFFERENT symbols but the SAME (date, UTC hour). If this is
materially larger than random cross-symbol pairs, SimCLR kicked in. If not,
the universality claim was unearned.

Verdict thresholds (council-5):
  strong_invariance:  cross_symbol_same_hour > 0.6
  some_invariance:    cross_symbol_same_hour > 0.3 AND > cross_symbol_diff_hour + 0.1
  no_invariance:      cross_symbol_same_hour <= cross_symbol_diff_hour + 0.1

Usage:
    uv run python scripts/cluster_cohesion.py \
        --checkpoint runs/step3-r2/encoder-best.pt \
        --cache data/cache \
        --out runs/step3-r2/cluster-cohesion.json \
        --symbols BTC ETH SOL BNB LINK LTC \
        --month 2026-02 --per-bucket 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tape.constants import LIQUID_CONTRASTIVE_SYMBOLS, STRIDE_EVAL
from tape.contrastive_batch import hour_bucket_from_ms
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder

MAX_PAIRS_PER_POPULATION = 50_000


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _month_shards(cache_dir: Path, month_prefix: str, symbols: list[str]) -> list[Path]:
    shards: list[Path] = []
    for sym in symbols:
        shards.extend(sorted(cache_dir.glob(f"{sym}__{month_prefix}-*.npz")))
    return shards


def _subsample_buckets(
    dataset: TapeDataset,
    per_bucket: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[str], list[str], list[int]]:
    """Randomly sample up to `per_bucket` windows per (symbol, date, hour) bucket.

    Returns (indices_into_dataset, symbols, dates, hours). We iterate the
    dataset once to peek at ts_first_ms (cheap — this just touches the
    WindowRef and the shard's event_ts array via the dataset's internal
    cache).
    """
    buckets: dict[tuple[str, str, int], list[int]] = {}
    n = len(dataset)
    for i in range(n):
        ref = dataset._refs[i]  # WindowRef — cheap
        # Use the shard cache to fetch event_ts for this start without doing
        # a full feature forward. __getitem__ materializes torch tensors; we
        # pull the raw dict ourselves to skip that.
        payload = dataset._get_shard(ref.shard_path)
        event_ts = payload.get("event_ts")
        if event_ts is None:
            raise RuntimeError(
                f"Shard {ref.shard_path} is missing 'event_ts'. Cache fix "
                "`117di7d` required — rebuild the cache or check "
                "tape/cache.py output schema."
            )
        ts_first_ms = int(event_ts[ref.start])
        hour = int(hour_bucket_from_ms(np.array([ts_first_ms], dtype=np.int64))[0])
        key = (ref.symbol, ref.date, hour)
        buckets.setdefault(key, []).append(i)

    picked_idx: list[int] = []
    picked_sym: list[str] = []
    picked_date: list[str] = []
    picked_hour: list[int] = []
    for key, idxs in buckets.items():
        sym, date, hour = key
        if len(idxs) > per_bucket:
            chosen = rng.choice(idxs, size=per_bucket, replace=False)
        else:
            chosen = np.array(idxs, dtype=np.int64)
        for ix in chosen:
            picked_idx.append(int(ix))
            picked_sym.append(sym)
            picked_date.append(date)
            picked_hour.append(hour)
    return picked_idx, picked_sym, picked_date, picked_hour


def _forward_embeddings(
    enc: TapeEncoder,
    dataset: TapeDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Forward the selected windows through the encoder; returns L2-normalized (N, 256)."""
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for bstart in range(0, len(indices), batch_size):
            bend = min(bstart + batch_size, len(indices))
            chunk = indices[bstart:bend]
            feats = torch.stack([dataset[i]["features"] for i in chunk]).to(device)
            _, g = enc(feats)
            embs.append(g.cpu().numpy())
    arr = np.concatenate(embs, axis=0)
    # L2-normalize (projection head was normalized during training; we mirror that)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return arr / norms


def _summarize(vals: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "n_pairs": int(len(vals)),
    }


def _sample_pairs(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    *,
    same_symbol: bool,
    same_hour: bool,
    symbols: np.ndarray,
    dates: np.ndarray,
    hours: np.ndarray,
    embeddings: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> np.ndarray:
    """Compute cosines over pairs satisfying the requested (symbol, hour) criteria.

    We vectorize by building the full pairwise cosine matrix on the L2-normalized
    embeddings (N <~ 10k), then masking the upper-triangle region by the criteria.
    If the passing-mask cardinality exceeds `max_pairs`, subsample uniformly.

    mask_i/mask_j are ignored for now — we always use all embeddings; the
    criteria are encoded in same_symbol/same_hour.
    """
    n = len(embeddings)
    # Pairwise cosine: (N, N), but we only look at i < j upper triangle.
    cos = embeddings @ embeddings.T  # (N, N) — already L2-normalized
    i_idx, j_idx = np.triu_indices(n, k=1)
    s_i = symbols[i_idx]
    s_j = symbols[j_idx]
    d_i = dates[i_idx]
    d_j = dates[j_idx]
    h_i = hours[i_idx]
    h_j = hours[j_idx]

    if same_symbol:
        sym_ok = s_i == s_j
    else:
        sym_ok = s_i != s_j
    if same_hour:
        # same (date, hour) bucket
        hour_ok = (d_i == d_j) & (h_i == h_j)
    else:
        # different (date, hour) bucket
        hour_ok = (d_i != d_j) | (h_i != h_j)

    keep = sym_ok & hour_ok
    i_k = i_idx[keep]
    j_k = j_idx[keep]
    if len(i_k) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(i_k) > max_pairs:
        sel = rng.choice(len(i_k), size=max_pairs, replace=False)
        i_k = i_k[sel]
        j_k = j_k[sel]
    return cos[i_k, j_k].astype(np.float64)


def _per_symbol_mean_cos(
    embeddings: np.ndarray, symbols: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    cos = embeddings @ embeddings.T
    n = len(embeddings)
    i_idx, j_idx = np.triu_indices(n, k=1)
    out: dict[str, float] = {}
    for sym in np.unique(symbols):
        keep = (symbols[i_idx] == sym) & (symbols[j_idx] == sym)
        if not keep.any():
            out[str(sym)] = float("nan")
            continue
        vals = cos[i_idx[keep], j_idx[keep]]
        if len(vals) > MAX_PAIRS_PER_POPULATION:
            sel = rng.choice(len(vals), size=MAX_PAIRS_PER_POPULATION, replace=False)
            vals = vals[sel]
        out[str(sym)] = float(np.mean(vals))
    return out


def _symbol_id_probe(embeddings: np.ndarray, symbols: np.ndarray, seed: int) -> float:
    sym_unique = list(np.unique(symbols))
    sym_to_id = {s: i for i, s in enumerate(sym_unique)}
    y = np.array([sym_to_id[s] for s in symbols], dtype=np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        embeddings, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1_000).fit(scaler.transform(Xtr), ytr)
    pred = lr.predict(scaler.transform(Xte))
    return float(balanced_accuracy_score(yte, pred))


def _verdict(cross_same: float, cross_diff: float) -> str:
    if cross_same > 0.6:
        return "strong"
    if cross_same > 0.3 and cross_same > cross_diff + 0.1:
        return "some"
    if cross_same <= cross_diff + 0.1:
        return "none"
    return "some"  # fallback band — >0.3 but not strongly above diff baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--symbols", nargs="+", default=list(LIQUID_CONTRASTIVE_SYMBOLS))
    ap.add_argument("--month", type=str, default="2026-02")
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = _pick_device()
    print(
        f"[cluster-cohesion] device={device} month={args.month} "
        f"symbols={args.symbols} per_bucket={args.per_bucket}"
    )

    enc = _load_encoder(args.checkpoint, device)
    print(
        f"[cluster-cohesion] encoder params = "
        f"{sum(p.numel() for p in enc.parameters()):,}"
    )

    shards = _month_shards(args.cache, args.month, args.symbols)
    if not shards:
        print(f"[cluster-cohesion] no shards for month {args.month} — aborting")
        return 1
    dataset = TapeDataset(shards, stride=STRIDE_EVAL, mode="eval")
    print(
        f"[cluster-cohesion] {len(shards)} shards, {len(dataset):,} windows "
        f"(stride={STRIDE_EVAL}, mode=eval)"
    )

    t0 = time.time()
    indices, sym_list, date_list, hour_list = _subsample_buckets(
        dataset, args.per_bucket, rng
    )
    print(
        f"[cluster-cohesion] sampled {len(indices)} windows from "
        f"{len(set(zip(sym_list, date_list, hour_list)))} buckets "
        f"(bucket-subsample elapsed {time.time() - t0:.1f}s)"
    )

    symbols = np.array(sym_list)
    dates = np.array(date_list)
    hours = np.array(hour_list, dtype=np.int64)

    n_windows_per_sym = {s: int(np.sum(symbols == s)) for s in np.unique(symbols)}
    print(f"[cluster-cohesion] n_windows_per_symbol = {n_windows_per_sym}")

    t0 = time.time()
    embeddings = _forward_embeddings(enc, dataset, indices, device, args.batch_size)
    print(
        f"[cluster-cohesion] embeddings shape = {embeddings.shape} "
        f"(forward elapsed {time.time() - t0:.1f}s)"
    )

    # Four populations
    t0 = time.time()
    populations = {
        "within_symbol": _sample_pairs(
            None,
            None,
            same_symbol=True,
            same_hour=True,
            symbols=symbols,
            dates=dates,
            hours=hours,
            embeddings=embeddings,
            rng=rng,
            max_pairs=MAX_PAIRS_PER_POPULATION,
        ),
        "cross_symbol_same_hour": _sample_pairs(
            None,
            None,
            same_symbol=False,
            same_hour=True,
            symbols=symbols,
            dates=dates,
            hours=hours,
            embeddings=embeddings,
            rng=rng,
            max_pairs=MAX_PAIRS_PER_POPULATION,
        ),
        "cross_symbol_diff_hour": _sample_pairs(
            None,
            None,
            same_symbol=False,
            same_hour=False,
            symbols=symbols,
            dates=dates,
            hours=hours,
            embeddings=embeddings,
            rng=rng,
            max_pairs=MAX_PAIRS_PER_POPULATION,
        ),
        "same_symbol_diff_hour": _sample_pairs(
            None,
            None,
            same_symbol=True,
            same_hour=False,
            symbols=symbols,
            dates=dates,
            hours=hours,
            embeddings=embeddings,
            rng=rng,
            max_pairs=MAX_PAIRS_PER_POPULATION,
        ),
    }
    print(f"[cluster-cohesion] pair extraction elapsed {time.time() - t0:.1f}s")

    pop_summary = {name: _summarize(vals) for name, vals in populations.items()}

    print("\n=== COSINE POPULATIONS ===")
    print(
        f"{'population':<28s} {'mean':>7s} {'std':>7s} "
        f"{'p5':>7s} {'p50':>7s} {'p95':>7s} {'n_pairs':>10s}"
    )
    for name, s in pop_summary.items():
        print(
            f"{name:<28s} {s['mean']:>7.4f} {s['std']:>7.4f} "
            f"{s['p5']:>7.4f} {s['p50']:>7.4f} {s['p95']:>7.4f} "
            f"{s['n_pairs']:>10d}"
        )

    per_symbol_mean = _per_symbol_mean_cos(embeddings, symbols, rng)
    print("\n=== PER-SYMBOL MEAN PAIRWISE COSINE ===")
    for sym, v in sorted(per_symbol_mean.items()):
        print(f"  {sym:<6s} {v:.4f}")

    symbol_id_bal_acc = _symbol_id_probe(embeddings, symbols, args.seed)
    print(f"\n=== SYMBOL-ID PROBE (6-way, LR) ===")
    print(f"  balanced accuracy = {symbol_id_bal_acc:.4f}")
    print("  reference: pretraining-monitoring reported 0.54 → 0.67")

    cross_same = pop_summary["cross_symbol_same_hour"]["mean"]
    cross_diff = pop_summary["cross_symbol_diff_hour"]["mean"]
    verdict = _verdict(cross_same, cross_diff)
    print(
        f"\n=== VERDICT ===\n  cross_symbol_same_hour = {cross_same:.4f}\n"
        f"  cross_symbol_diff_hour = {cross_diff:.4f}\n"
        f"  delta = {cross_same - cross_diff:+.4f}\n"
        f"  verdict = {verdict}"
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "month": args.month,
        "symbols": list(args.symbols),
        "per_bucket": int(args.per_bucket),
        "seed": int(args.seed),
        "stride": STRIDE_EVAL,
        "n_windows_total": int(len(indices)),
        "n_windows": n_windows_per_sym,
        "n_buckets": int(len(set(zip(sym_list, date_list, hour_list)))),
        "populations": pop_summary,
        "per_symbol_mean_pairwise_cos": per_symbol_mean,
        "symbol_id_balanced_acc": symbol_id_bal_acc,
        "interpretation_thresholds": {
            "strong_invariance": "cross_symbol_same_hour > 0.6",
            "some_invariance": (
                "cross_symbol_same_hour > 0.3 AND > cross_symbol_diff_hour + 0.1"
            ),
            "no_invariance": "cross_symbol_same_hour <= cross_symbol_diff_hour + 0.1",
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

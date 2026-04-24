# scripts/avax_gate3_probe.py
"""Gate 3 — AVAX held-out symbol probe.

Tests "do the encoder's representations generalize to a symbol that was
NEVER seen during pretraining?" AVAX was excluded from the pretraining
universe (PRETRAINING_SYMBOLS), from cross-symbol contrastive pairs, and
from every probing pass so far. This script is the first time the encoder
sees AVAX.

Protocol matches the Gate 1 matched-density held-out:
  - Feb + Mar 2026 as primary evaluation months (matched density to training)
  - April 1-13 reported but informational only (underpowered at stride=200)
  - Per-month time-ordered 80/20 split on AVAX windows
  - Five predictors: encoder_lr, pca_lr, rp_lr, majority, shuffled_pca_lr
  - Both H100 (spec horizon) and H500 (Gate-1 empirical horizon)
  - balanced_accuracy_score, min_valid=200

Gate 3 spec threshold: > 51.4% balanced accuracy at H100.
We probe H500 too because Gate 1's 2026-04-23 diagnostics established H100
is at noise floor for all predictors on this data — a Gate 3 verdict should
not be blind to that.

Usage:
    uv run python scripts/avax_gate3_probe.py \\
        --checkpoint runs/step3-r2/encoder-best.pt \\
        --cache data/cache \\
        --out runs/step3-r2/avax-gate3-probe.json \\
        --months 2026-02 2026-03 2026-04
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from tape.constants import HELD_OUT_SYMBOL, STRIDE_EVAL, STRIDE_PRETRAIN
from tape.dataset import TapeDataset
from tape.flat_features import window_to_flat
from tape.model import EncoderConfig, TapeEncoder


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_MONTH_NAMES = {
    "2025-10": "October",
    "2025-11": "November",
    "2025-12": "December",
    "2026-01": "January",
    "2026-02": "February",
    "2026-03": "March",
    "2026-04": "April",
}


def _avax_month_shards(cache_dir: Path, month_prefix: str) -> list[Path]:
    """Shards for the held-out symbol only, within the given month."""
    return sorted(cache_dir.glob(f"{HELD_OUT_SYMBOL}__{month_prefix}-*.npz"))


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _forward_and_extract(
    enc: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    batch_size: int,
    horizons: tuple[int, ...],
) -> dict:
    enc_embs: list[np.ndarray] = []
    flat_feats: list[np.ndarray] = []
    labels: dict[int, list[int]] = {h: [] for h in horizons}
    masks: dict[int, list[bool]] = {h: [] for h in horizons}
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
                for h in horizons:
                    labels[h].append(int(b[f"label_h{h}"]))
                    masks[h].append(bool(b[f"label_h{h}_mask"]))
    return {
        "enc": np.concatenate(enc_embs, axis=0),
        "flat": np.stack(flat_feats),
        "labels": {h: np.array(labels[h], dtype=np.int64) for h in horizons},
        "masks": {h: np.array(masks[h], dtype=bool) for h in horizons},
    }


def _probe_one(
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    *,
    predictor: str,
    seed: int = 0,
    test_frac: float = 0.2,
    min_valid: int = 200,
) -> float | None:
    valid_idx = np.where(masks)[0]
    if len(valid_idx) < min_valid:
        return None
    n_valid = len(valid_idx)
    n_test = max(50, int(n_valid * test_frac))
    tr = valid_idx[: n_valid - n_test]
    te = valid_idx[n_valid - n_test :]
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        if predictor == "majority":
            majority = int(np.bincount(y[tr]).argmax())
            pred = np.full(len(te), majority)
            return float(balanced_accuracy_score(y[te], pred))
        return None
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    rng = np.random.default_rng(seed)

    if predictor == "encoder_lr":
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(scaler.transform(Xtr), ytr)
        pred = lr.predict(scaler.transform(Xte))
    elif predictor == "pca_lr":
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        pca = PCA(n_components=min(20, Xtr_s.shape[1])).fit(Xtr_s)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(pca.transform(Xtr_s), ytr)
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
    return float(balanced_accuracy_score(yte, pred))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[100, 500],
        help="Horizons to probe (default: H100 spec + H500 empirical).",
    )
    ap.add_argument(
        "--months",
        nargs="+",
        default=["2026-02", "2026-03", "2026-04"],
        help="ISO YYYY-MM prefixes to evaluate.",
    )
    ap.add_argument(
        "--mode",
        choices=["pretrain", "eval"],
        default="eval",
        help="Stride: eval=200 (no overlap, recommended), pretrain=50 (4x density)",
    )
    args = ap.parse_args()
    stride = STRIDE_PRETRAIN if args.mode == "pretrain" else STRIDE_EVAL
    horizons = tuple(args.horizons)

    device = _pick_device()
    print(f"[avax-gate3] device={device} horizons={horizons} mode={args.mode}")

    enc = _load_encoder(args.checkpoint, device)
    print(f"[avax-gate3] encoder params = {sum(p.numel() for p in enc.parameters()):,}")
    print(f"[avax-gate3] held-out symbol = {HELD_OUT_SYMBOL}")

    per_month: dict[str, dict] = {}
    predictors = ("encoder_lr", "pca_lr", "rp_lr", "majority", "shuffled_pca_lr")

    for month_prefix in args.months:
        month_name = _MONTH_NAMES.get(month_prefix, month_prefix)
        shards = _avax_month_shards(args.cache, month_prefix)
        if not shards:
            print(f"[avax-gate3] {month_name} — no shards, skipping")
            continue
        dataset = TapeDataset(shards, stride=stride, mode=args.mode)
        print(
            f"[avax-gate3] {month_name} ({month_prefix}): "
            f"{len(shards)} shards, {len(dataset):,} windows"
        )
        if len(dataset) < 200:
            print(
                f"  WARNING: {len(dataset)} windows < min_valid=200 "
                "— underpowered, expect N/A"
            )

        t0 = time.time()
        data = _forward_and_extract(enc, dataset, device, args.batch_size, horizons)
        print(f"  forward+extract elapsed {time.time() - t0:.1f}s")

        month_entry: dict = {"windows": int(len(dataset)), "horizons": {}}
        for h in horizons:
            horizon_results: dict[str, float | None] = {}
            for pname in predictors:
                X = data["enc"] if pname == "encoder_lr" else data["flat"]
                bal = _probe_one(
                    X,
                    data["labels"][h],
                    data["masks"][h],
                    predictor=pname,
                    seed=args.seed,
                )
                horizon_results[pname] = bal
            month_entry["horizons"][f"H{h}"] = horizon_results

            print(f"  H{h}:")
            print(f"    {'predictor':<20s} {'balanced_acc':>14s}")
            for pname in predictors:
                v = horizon_results[pname]
                vs = f"{v:.4f}" if v is not None else "N/A"
                print(f"    {pname:<20s} {vs:>14s}")

        per_month[month_prefix] = month_entry

    # Gate 3 verdict per horizon
    print("\n=== GATE 3 VERDICT (balanced accuracy > 0.514 = pass) ===")
    gate3_verdict: dict[str, dict] = {}
    for h in horizons:
        hkey = f"H{h}"
        print(f"\n{hkey}:")
        print(
            f"{'month':<12s}  {'encoder':>8s}  {'pca':>6s}  {'rp':>6s}  {'verdict':>10s}"
        )
        gate3_verdict[hkey] = {}
        for month_prefix, entry in per_month.items():
            h_res = entry["horizons"].get(hkey, {})
            enc_v = h_res.get("encoder_lr")
            pca_v = h_res.get("pca_lr")
            rp_v = h_res.get("rp_lr")
            passes = enc_v is not None and enc_v >= 0.514
            enc_s = f"{enc_v:.4f}" if enc_v is not None else "N/A"
            pca_s = f"{pca_v:.4f}" if pca_v is not None else "N/A"
            rp_s = f"{rp_v:.4f}" if rp_v is not None else "N/A"
            v_s = "PASS" if passes else "FAIL"
            print(
                f"{month_prefix:<12s}  {enc_s:>8s}  {pca_s:>6s}  {rp_s:>6s}  {v_s:>10s}"
            )
            gate3_verdict[hkey][month_prefix] = {
                "encoder": enc_v,
                "pca": pca_v,
                "rp": rp_v,
                "passes_514": passes,
            }

    report = {
        "checkpoint": str(args.checkpoint),
        "held_out_symbol": HELD_OUT_SYMBOL,
        "horizons": list(horizons),
        "mode": args.mode,
        "stride": stride,
        "months": per_month,
        "gate3_verdict": gate3_verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

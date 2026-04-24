# scripts/avax_gate3_probe.py
"""Gate 3 — held-out symbol probe (generalized beyond AVAX).

Originally written as the AVAX-only Gate 3 probe; now generalized to accept
any target symbol or set of symbols via `--target-symbols`. The script name
is preserved to avoid downstream churn, but it is used for:

  * Gate 3 primary: AVAX only (default) — tests out-of-pretraining-universe
    transfer against the pre-registered 51.4% H100 threshold.
  * In-sample control: e.g. `--target-symbols LINK LTC` — tests the SAME
    methodology (time-ordered 80/20 on 1-month windows, stride=50) on
    symbols that WERE in pretraining. Required by council-5 to distinguish
    "AVAX doesn't transfer" from "any mid-liquid symbol fails under this
    small-n probe".

Protocol (unchanged from original):
  - Time-ordered 80/20 split per month on the union of target-symbol windows
  - Five predictors: encoder_lr, pca_lr, rp_lr, majority, shuffled_pca_lr
  - Balanced accuracy, min_valid=200

Council-5 upgrades (2026-04-24):
  - Bootstrap 95% CIs per cell (1000 resamples of the test set; the
    classifier is refit once on train, then re-evaluated on bootstrapped
    test indices. This estimates sampling variance of balanced_accuracy at
    FIXED MODEL — NOT including model-fit variance.)
  - Shuffled control runs N=50 label permutations per cell; we report
    mean, std, and mean+2σ as the upper band of the null distribution.
  - `class_prior` reports the fraction of class-1 in the test fold for
    every cell (required to interpret balanced accuracy under imbalance).

Gate 3 verdict logic only makes sense when the target set is literally the
pre-registered held-out symbol. For in-sample controls we still print the
verdict but callers should ignore the PASS/FAIL column and read the raw
numbers plus CIs.

Usage:
    # Gate 3 primary (AVAX held-out, matched-density months)
    uv run python scripts/avax_gate3_probe.py \\
        --checkpoint runs/step3-r2/encoder-best.pt \\
        --cache data/cache \\
        --out runs/step3-r2/gate3-avax-bootstrap.json \\
        --months 2026-02 2026-03 --horizons 100 500 \\
        --mode pretrain --seed 0

    # In-sample control (LINK + LTC, same months, same protocol)
    uv run python scripts/avax_gate3_probe.py \\
        --checkpoint runs/step3-r2/encoder-best.pt \\
        --cache data/cache \\
        --out runs/step3-r2/gate3-insample-control.json \\
        --target-symbols LINK LTC \\
        --months 2026-02 2026-03 --horizons 100 500 \\
        --mode pretrain --seed 0
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

# ----- Bootstrap / shuffle settings (council-5) -----
N_BOOTSTRAP = 1000
N_SHUFFLES = 50


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


def _target_month_shards(
    cache_dir: Path, month_prefix: str, target_symbols: list[str]
) -> list[Path]:
    """Shards for the requested target symbols, within the given month.

    The script used to hard-code AVAX via `HELD_OUT_SYMBOL`; now it takes
    an explicit list. Shard names follow the convention
    `{SYMBOL}__{YYYY-MM-DD}.npz` so we glob each symbol separately and
    concatenate, sorted chronologically within each symbol.
    """
    shards: list[Path] = []
    for sym in target_symbols:
        shards.extend(sorted(cache_dir.glob(f"{sym}__{month_prefix}-*.npz")))
    return shards


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


def _bootstrap_ci(
    yte: np.ndarray,
    pred: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on balanced accuracy.

    Resamples the TEST set indices with replacement `n_resamples` times;
    at each resample both `yte` and `pred` are indexed together. The fit
    classifier is held fixed — this estimates the sampling variance of
    balanced_accuracy at a fixed model, which is the right quantity when
    comparing predictors that were all trained on the same train set.
    """
    n = len(yte)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        # Guard: if the bootstrap draw gives a single-class test set, fall
        # back to the original balanced_accuracy for that draw (equivalent
        # to treating that draw as degenerate).
        yb = yte[idx]
        pb = pred[idx]
        if len(np.unique(yb)) < 2:
            # balanced_accuracy_score works on single-class predictions; it
            # collapses to 0 or 1 depending on whether we got it right.
            accs[i] = float(balanced_accuracy_score(yb, pb))
        else:
            accs[i] = float(balanced_accuracy_score(yb, pb))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _fit_predict(
    predictor: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit the given predictor family on train and return TEST predictions."""
    if predictor == "encoder_lr":
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(scaler.transform(Xtr), ytr)
        return lr.predict(scaler.transform(Xte))
    if predictor == "pca_lr":
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        pca = PCA(n_components=min(20, Xtr_s.shape[1])).fit(Xtr_s)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(pca.transform(Xtr_s), ytr)
        return lr.predict(pca.transform(Xte_s))
    if predictor == "rp_lr":
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        rp = GaussianRandomProjection(
            n_components=min(20, Xtr_s.shape[1]), random_state=seed
        ).fit(Xtr_s)
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(rp.transform(Xtr_s), ytr)
        return lr.predict(rp.transform(Xte_s))
    if predictor == "majority":
        majority = int(np.bincount(ytr).argmax())
        return np.full(len(Xte), majority, dtype=np.int64)
    raise ValueError(predictor)


def _probe_one(
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    *,
    predictor: str,
    seed: int = 0,
    test_frac: float = 0.2,
    min_valid: int = 200,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict | None:
    """Fit predictor, return point estimate + bootstrap 95% CI + class prior.

    Returns None if the cell is underpowered; otherwise a dict with
    `balanced_acc`, `ci_lo`, `ci_hi`, `class_prior` (fraction of class 1 in
    the TEST fold), and `n_test`.
    """
    valid_idx = np.where(masks)[0]
    if len(valid_idx) < min_valid:
        return None
    n_valid = len(valid_idx)
    n_test = max(50, int(n_valid * test_frac))
    tr = valid_idx[: n_valid - n_test]
    te = valid_idx[n_valid - n_test :]

    yte_full = y[te]
    class_prior = float(np.mean(yte_full == 1)) if len(yte_full) > 0 else float("nan")

    # Handle degenerate single-class train/test (rare at n_valid >= 200 but
    # possible on illiquid symbols).
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        if predictor == "majority":
            majority = int(np.bincount(y[tr]).argmax())
            pred = np.full(len(te), majority)
            bal = float(balanced_accuracy_score(y[te], pred))
            ci_lo, ci_hi = _bootstrap_ci(
                y[te], pred, n_resamples=n_bootstrap, seed=seed
            )
            return {
                "balanced_acc": bal,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "class_prior": class_prior,
                "n_test": int(len(te)),
            }
        return None

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    if predictor in ("encoder_lr", "pca_lr", "rp_lr", "majority"):
        pred = _fit_predict(predictor, Xtr, ytr, Xte, seed=seed)
    else:
        raise ValueError(
            f"_probe_one does not handle predictor={predictor!r}; use "
            "_probe_shuffled for the shuffled control."
        )

    bal = float(balanced_accuracy_score(yte, pred))
    ci_lo, ci_hi = _bootstrap_ci(yte, pred, n_resamples=n_bootstrap, seed=seed)
    return {
        "balanced_acc": bal,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "class_prior": class_prior,
        "n_test": int(len(te)),
    }


def _probe_shuffled(
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    *,
    seed: int = 0,
    test_frac: float = 0.2,
    min_valid: int = 200,
    n_shuffles: int = N_SHUFFLES,
) -> dict | None:
    """Shuffled PCA+LR control: N permutations of ytr, mean ± σ of bal-acc.

    Council-5 required fix: single-seed shuffled draws are too noisy (the
    original Apr H500 0.700 outlier was a 1-in-50 draw). We run N=50
    permutations and report mean, std, and mean+2σ (the upper band of the
    null distribution). Also returns the class prior on the test fold.
    """
    valid_idx = np.where(masks)[0]
    if len(valid_idx) < min_valid:
        return None
    n_valid = len(valid_idx)
    n_test = max(50, int(n_valid * test_frac))
    tr = valid_idx[: n_valid - n_test]
    te = valid_idx[n_valid - n_test :]
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    class_prior = float(np.mean(yte == 1))

    # Precompute the PCA projection once — it is label-independent, so
    # running N=50 shuffles only re-fits the logistic regression.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    pca = PCA(n_components=min(20, Xtr_s.shape[1])).fit(Xtr_s)
    Xtr_pca = pca.transform(Xtr_s)
    Xte_pca = pca.transform(Xte_s)

    rng = np.random.default_rng(seed)
    accs = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        ytr_shuf = rng.permutation(ytr)
        # Skip draws where the shuffled labels are somehow degenerate
        # (shouldn't happen if ytr has both classes, but guard anyway).
        if len(np.unique(ytr_shuf)) < 2:
            accs[i] = 0.5
            continue
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(Xtr_pca, ytr_shuf)
        pred = lr.predict(Xte_pca)
        accs[i] = float(balanced_accuracy_score(yte, pred))
    mean = float(accs.mean())
    std = float(accs.std(ddof=1)) if n_shuffles > 1 else 0.0
    return {
        "balanced_acc": mean,  # canonical point estimate == mean of shuffles
        "shuffled_mean": mean,
        "shuffled_std": std,
        "shuffled_95ci_hi": mean + 2 * std,
        "class_prior": class_prior,
        "n_test": int(len(te)),
        "n_shuffles": int(n_shuffles),
    }


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
    ap.add_argument(
        "--target-symbols",
        nargs="+",
        default=[HELD_OUT_SYMBOL],
        help=(
            "Symbols whose shards define the probe pool. Default is the "
            "pre-registered Gate 3 held-out symbol (AVAX). Pass e.g. "
            "'LINK LTC' for an in-sample control."
        ),
    )
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Bootstrap resamples per cell (default 1000).",
    )
    ap.add_argument(
        "--n-shuffles",
        type=int,
        default=N_SHUFFLES,
        help="Label-shuffle permutations for the null control (default 50).",
    )
    args = ap.parse_args()
    stride = STRIDE_PRETRAIN if args.mode == "pretrain" else STRIDE_EVAL
    horizons = tuple(args.horizons)
    target_symbols = list(args.target_symbols)
    is_gate3_primary = target_symbols == [HELD_OUT_SYMBOL]

    device = _pick_device()
    print(
        f"[gate3-probe] device={device} horizons={horizons} mode={args.mode} "
        f"targets={target_symbols}"
    )

    enc = _load_encoder(args.checkpoint, device)
    print(
        f"[gate3-probe] encoder params = {sum(p.numel() for p in enc.parameters()):,}"
    )
    print(f"[gate3-probe] target_symbols = {target_symbols}")
    print(
        f"[gate3-probe] bootstrap resamples = {args.n_bootstrap}, "
        f"null shuffles = {args.n_shuffles}"
    )

    per_month: dict[str, dict] = {}
    predictors_fit = ("encoder_lr", "pca_lr", "rp_lr", "majority")
    # shuffled_pca_lr is handled separately because it is N=50 shuffles

    for month_prefix in args.months:
        month_name = _MONTH_NAMES.get(month_prefix, month_prefix)
        shards = _target_month_shards(args.cache, month_prefix, target_symbols)
        if not shards:
            print(f"[gate3-probe] {month_name} — no shards, skipping")
            continue
        dataset = TapeDataset(shards, stride=stride, mode=args.mode)
        print(
            f"[gate3-probe] {month_name} ({month_prefix}): "
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

        month_entry: dict = {
            "windows": int(len(dataset)),
            "shards": len(shards),
            "horizons": {},
        }
        for h in horizons:
            horizon_results: dict[str, dict | None] = {}

            t_cell = time.time()
            for pname in predictors_fit:
                X = data["enc"] if pname == "encoder_lr" else data["flat"]
                res = _probe_one(
                    X,
                    data["labels"][h],
                    data["masks"][h],
                    predictor=pname,
                    seed=args.seed,
                    n_bootstrap=args.n_bootstrap,
                )
                horizon_results[pname] = res

            # Shuffled null control: PCA+LR with N=50 shuffles.
            shuf_res = _probe_shuffled(
                data["flat"],
                data["labels"][h],
                data["masks"][h],
                seed=args.seed,
                n_shuffles=args.n_shuffles,
            )
            horizon_results["shuffled_pca_lr"] = shuf_res

            month_entry["horizons"][f"H{h}"] = horizon_results
            print(f"  H{h}: elapsed {time.time() - t_cell:.1f}s")
            print(
                f"    {'predictor':<20s} {'bal_acc':>8s} {'ci_lo':>8s} "
                f"{'ci_hi':>8s} {'prior1':>8s} {'n_te':>6s}"
            )
            for pname in (
                "encoder_lr",
                "pca_lr",
                "rp_lr",
                "majority",
                "shuffled_pca_lr",
            ):
                r = horizon_results.get(pname)
                if r is None:
                    print(f"    {pname:<20s} {'N/A':>8s}")
                    continue
                bal = r["balanced_acc"]
                ci_lo = r.get("ci_lo")
                ci_hi = r.get("ci_hi")
                # For shuffled we don't emit percentile CI (null mean ± 2σ
                # is reported separately as shuffled_95ci_hi); just print
                # the mean and leave CI blank.
                if pname == "shuffled_pca_lr":
                    up95 = r.get("shuffled_95ci_hi", float("nan"))
                    sd = r.get("shuffled_std", float("nan"))
                    print(
                        f"    {pname:<20s} {bal:>8.4f} "
                        f"{'μ±2σ':>8s} {up95:>8.4f} "
                        f"{r['class_prior']:>8.3f} {r['n_test']:>6d}  "
                        f"(σ={sd:.4f})"
                    )
                else:
                    print(
                        f"    {pname:<20s} {bal:>8.4f} {ci_lo:>8.4f} "
                        f"{ci_hi:>8.4f} {r['class_prior']:>8.3f} "
                        f"{r['n_test']:>6d}"
                    )

        per_month[month_prefix] = month_entry

    # Gate 3 verdict per horizon — only meaningful for the AVAX-only run.
    print("\n=== GATE 3 VERDICT (balanced accuracy > 0.514 = pass) ===")
    if not is_gate3_primary:
        print(
            f"NOTE: target_symbols={target_symbols} is not the pre-registered "
            "Gate 3 held-out set ([AVAX]). The PASS/FAIL column below is "
            "reported for reference only — read raw numbers + CIs instead."
        )
    gate3_verdict: dict[str, dict] = {}
    for h in horizons:
        hkey = f"H{h}"
        print(f"\n{hkey}:")
        print(
            f"{'month':<12s}  {'enc':>8s} {'enc_lo':>8s} {'enc_hi':>8s}  "
            f"{'pca':>8s} {'pca_lo':>8s} {'pca_hi':>8s}  {'verdict':>10s}"
        )
        gate3_verdict[hkey] = {}
        for month_prefix, entry in per_month.items():
            h_res = entry["horizons"].get(hkey, {})
            enc_r = h_res.get("encoder_lr")
            pca_r = h_res.get("pca_lr")
            rp_r = h_res.get("rp_lr")
            enc_v = enc_r["balanced_acc"] if enc_r else None
            pca_v = pca_r["balanced_acc"] if pca_r else None
            rp_v = rp_r["balanced_acc"] if rp_r else None
            passes = enc_v is not None and enc_v >= 0.514
            enc_s = f"{enc_v:.4f}" if enc_v is not None else "N/A"
            enc_lo_s = f"{enc_r['ci_lo']:.4f}" if enc_r else "N/A"
            enc_hi_s = f"{enc_r['ci_hi']:.4f}" if enc_r else "N/A"
            pca_s = f"{pca_v:.4f}" if pca_v is not None else "N/A"
            pca_lo_s = f"{pca_r['ci_lo']:.4f}" if pca_r else "N/A"
            pca_hi_s = f"{pca_r['ci_hi']:.4f}" if pca_r else "N/A"
            v_s = "PASS" if passes else "FAIL"
            print(
                f"{month_prefix:<12s}  {enc_s:>8s} {enc_lo_s:>8s} {enc_hi_s:>8s}  "
                f"{pca_s:>8s} {pca_lo_s:>8s} {pca_hi_s:>8s}  {v_s:>10s}"
            )
            gate3_verdict[hkey][month_prefix] = {
                "encoder": enc_v,
                "encoder_ci_lo": enc_r["ci_lo"] if enc_r else None,
                "encoder_ci_hi": enc_r["ci_hi"] if enc_r else None,
                "pca": pca_v,
                "pca_ci_lo": pca_r["ci_lo"] if pca_r else None,
                "pca_ci_hi": pca_r["ci_hi"] if pca_r else None,
                "rp": rp_v,
                "rp_ci_lo": rp_r["ci_lo"] if rp_r else None,
                "rp_ci_hi": rp_r["ci_hi"] if rp_r else None,
                "passes_514": passes,
            }

    report = {
        "checkpoint": str(args.checkpoint),
        "target_symbols": target_symbols,
        "is_gate3_primary": is_gate3_primary,
        "held_out_symbol": HELD_OUT_SYMBOL,
        "horizons": list(horizons),
        "mode": args.mode,
        "stride": stride,
        "n_bootstrap": args.n_bootstrap,
        "n_shuffles": args.n_shuffles,
        "months": per_month,
        "gate3_verdict": gate3_verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

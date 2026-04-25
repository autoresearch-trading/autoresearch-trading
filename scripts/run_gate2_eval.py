# scripts/run_gate2_eval.py
"""Gate 2 evaluation — three-comparator per-symbol, per-month, per-horizon.

Implements the Gate 2 protocol from the ratified Step 4 plan
(`docs/superpowers/plans/2026-04-24-step4-fine-tuning.md`):

Three comparators, all measured per-symbol per-month with 1000-resample
bootstrap CIs (concepts/bootstrap-methodology.md):

  A. Flat-LR baseline:
     window_to_flat → 83-dim → sklearn LogisticRegression with
     C ∈ {0.001, 0.01, 0.1} cross-validated → predict on test fold.

  B. Frozen-encoder LR (Gate 1 baseline replication):
     Forward held-out windows through the gate1-baseline-checkpoint's encoder
     → 256-dim embeddings → sklearn LR with the same C-grid as A. Identical
     to Gate 1 protocol.

  C. Fine-tuned CNN:
     Forward held-out windows through the fine-tuned model → take H500 head
     logit → sigmoid > 0.5 → balanced accuracy. NOT trained at eval — uses
     finetuned-best.pt directly.

Time-ordered 80/20 split per (symbol, month) cell. min_valid=200. Same
balanced_accuracy + 1000-resample bootstrap CI protocol used by
`scripts/avax_gate3_probe.py`.

Gate 2 verdict (THREE criteria, all must hold on BOTH Feb and Mar):

  1. Fine-tuned CNN ≥ flat-LR + 0.5pp on 15+/24 symbols at H500.
  2. No per-symbol regression vs Gate 1 baseline:
     CNN bal-acc CI lower bound must NOT regress below Gate 1 frozen-encoder
     LR's CI lower bound by more than max(1.0pp, 1× CI half-width) AND
     point estimate must not regress ≥1.0pp.
  3. Fine-tuned CNN ≥ frozen-encoder LR + 0.3pp on 13+/24 symbols.

Usage:
    uv run python scripts/run_gate2_eval.py \\
        --checkpoint runs/step4-r1/finetuned-best.pt \\
        --gate1-baseline-checkpoint runs/step3-r2/encoder-best.pt \\
        --cache data/cache \\
        --out runs/step4-r1/gate2-eval.json \\
        --months 2026-02 2026-03 --horizon 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.constants import HELD_OUT_SYMBOL, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.dataset import TapeDataset
from tape.finetune import HORIZONS, DirectionHead, FineTunedModel
from tape.flat_features import window_to_flat
from tape.model import EncoderConfig, TapeEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BOOTSTRAP_DEFAULT: int = 1000
MIN_VALID: int = 200
TEST_FRAC: float = 0.20

# Comparator-LR C-grid (Gate 1 protocol; council-6 Q6 #5).
LR_C_GRID: tuple[float, ...] = (0.001, 0.01, 0.1)

# Verdict thresholds
CRITERION_1_DELTA: float = 0.005  # +0.5pp vs flat-LR
CRITERION_1_MIN_SYMBOLS: int = 15
CRITERION_2_REGRESSION_PP: float = 0.010  # 1.0pp point-estimate regression cap
CRITERION_3_DELTA: float = 0.003  # +0.3pp vs frozen-encoder LR
CRITERION_3_MIN_SYMBOLS: int = 13

PRIOR_TRIAL_LOG: tuple[str, ...] = (
    "Gate 1 H500 frozen-encoder LR (binding)",
    "Gate 3 informational AVAX bootstrap (4 cells)",
    "Cluster cohesion 6-anchor probe (Feb only)",
    "Surrogate sweep 5 sym × 2 mo × 2 hor (20 cells)",
    "Gate 2 fine-tuned CNN (this evaluation)",
)


# ---------------------------------------------------------------------------
# Device + checkpoint loaders
# ---------------------------------------------------------------------------


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    """Load a TapeEncoder from a pretraining checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _load_finetuned(checkpoint_path: Path, device: torch.device) -> FineTunedModel:
    """Load a FineTunedModel (encoder + head) from a finetune checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    head = DirectionHead(embed_dim=enc.global_dim)
    head.load_state_dict(payload["head_state_dict"])
    model = FineTunedModel(enc, head)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _symbol_month_shards(cache_dir: Path, symbol: str, month_prefix: str) -> list[Path]:
    """All shards for one (symbol, month) cell."""
    return sorted(cache_dir.glob(f"{symbol}__{month_prefix}-*.npz"))


def _forward_and_extract(
    finetuned: FineTunedModel,
    gate1_encoder: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    batch_size: int,
    horizon: int,
) -> dict[str, Any]:
    """Forward `dataset` through both models; return per-window features + labels.

    Returns dict with:
      - flat: (N, 83) flat-feature vectors (window_to_flat)
      - gate1_emb: (N, 256) frozen-encoder embeddings
      - finetuned_logits: (N, 4) fine-tuned model logits at HORIZONS order
      - labels: (N,) int labels at the requested horizon
      - masks: (N,) bool masks at the requested horizon
    """
    h_idx = HORIZONS.index(horizon)
    flat_buf: list[np.ndarray] = []
    gate1_buf: list[np.ndarray] = []
    finetune_buf: list[np.ndarray] = []
    labels_buf: list[int] = []
    masks_buf: list[bool] = []

    n = len(dataset)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            batch = [dataset[i] for i in range(s, e)]
            feats_t = torch.stack([b["features"] for b in batch]).to(device)

            # Frozen encoder pass
            _, g_gate1 = gate1_encoder(feats_t)
            gate1_buf.append(g_gate1.detach().float().cpu().numpy())

            # Fine-tuned model pass (encoder + head → logits)
            logits = finetuned(feats_t)
            finetune_buf.append(logits.detach().float().cpu().numpy())

            for b in batch:
                w = b["features"].numpy()
                flat_buf.append(window_to_flat(w))
                labels_buf.append(int(b[f"label_h{horizon}"]))
                masks_buf.append(bool(b[f"label_h{horizon}_mask"]))

    return {
        "flat": np.stack(flat_buf) if flat_buf else np.zeros((0, 83), dtype=np.float32),
        "gate1_emb": (
            np.concatenate(gate1_buf, axis=0)
            if gate1_buf
            else np.zeros((0, 256), dtype=np.float32)
        ),
        "finetuned_logits": (
            np.concatenate(finetune_buf, axis=0)
            if finetune_buf
            else np.zeros((0, len(HORIZONS)), dtype=np.float32)
        ),
        "labels": np.array(labels_buf, dtype=np.int64),
        "masks": np.array(masks_buf, dtype=bool),
        "horizon_index": h_idx,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI + LR with C-search
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    yte: np.ndarray,
    pred: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on balanced accuracy at fixed model.

    Same formulation as scripts/avax_gate3_probe.py — resamples the test
    indices with replacement, recomputes balanced accuracy on each resample.
    """
    n = len(yte)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(balanced_accuracy_score(yte[idx], pred[idx]))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _fit_lr_with_c_search(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    c_grid: tuple[float, ...] = LR_C_GRID,
    cv_inner: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Fit LR with cross-validated C on `c_grid`, predict on Xte.

    Inner cross-validation is a simple time-ordered split of train into
    cv_inner contiguous chunks; we pick the C with the best mean balanced
    accuracy across inner folds. This is a lightweight stand-in for
    sklearn.model_selection.GridSearchCV that respects time ordering.
    """
    if len(np.unique(ytr)) < 2:
        # Degenerate: predict majority on test.
        majority = int(np.bincount(ytr).argmax())
        return np.full(len(Xte), majority, dtype=np.int64)

    n = len(Xtr)
    if n < cv_inner * 50 or len(c_grid) == 1:
        # Too small to CV meaningfully — just use the middle C.
        chosen_c = c_grid[len(c_grid) // 2]
    else:
        fold_size = n // cv_inner
        scores: dict[float, list[float]] = {c: [] for c in c_grid}
        for c in c_grid:
            for k in range(cv_inner):
                ts = k * fold_size
                te_end = ts + fold_size
                tr_idx = np.concatenate([np.arange(0, ts), np.arange(te_end, n)])
                te_idx = np.arange(ts, te_end)
                if len(np.unique(ytr[tr_idx])) < 2 or len(np.unique(ytr[te_idx])) < 2:
                    continue
                scaler = StandardScaler().fit(Xtr[tr_idx])
                lr = LogisticRegression(C=c, max_iter=1_000).fit(
                    scaler.transform(Xtr[tr_idx]), ytr[tr_idx]
                )
                pred = lr.predict(scaler.transform(Xtr[te_idx]))
                scores[c].append(balanced_accuracy_score(ytr[te_idx], pred))
        c_means = {c: (np.mean(v) if v else 0.0) for c, v in scores.items()}
        chosen_c = max(c_means, key=lambda k: c_means[k])

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=chosen_c, max_iter=1_000).fit(scaler.transform(Xtr), ytr)
    return lr.predict(scaler.transform(Xte))


def _compute_cell(
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    *,
    predictor: str,
    n_bootstrap: int,
    seed: int,
    finetuned_pred: np.ndarray | None = None,
) -> dict | None:
    """Run one comparator on one (symbol, month) cell.

    For `predictor='flat_lr'` and `'frozen_encoder_lr'`: time-ordered 80/20,
    LR with C-grid, predict, balanced accuracy + bootstrap CI.

    For `predictor='finetuned_cnn'`: ignores X/y fitting; uses the supplied
    `finetuned_pred` aligned with the time-ordered TEST fold.
    """
    valid_idx = np.where(masks)[0]
    if len(valid_idx) < MIN_VALID:
        return None

    n_valid = len(valid_idx)
    n_test = max(50, int(n_valid * TEST_FRAC))
    tr = valid_idx[: n_valid - n_test]
    te = valid_idx[n_valid - n_test :]

    yte = y[te]
    class_prior = float(np.mean(yte == 1)) if len(yte) > 0 else float("nan")

    if predictor == "finetuned_cnn":
        if finetuned_pred is None:
            return None
        pred = finetuned_pred[te]
    elif predictor in ("flat_lr", "frozen_encoder_lr"):
        Xtr = X[tr]
        ytr = y[tr]
        Xte = X[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            # Degenerate cell — return majority for predictor parity.
            majority = int(np.bincount(ytr).argmax())
            pred = np.full(len(te), majority, dtype=np.int64)
        else:
            pred = _fit_lr_with_c_search(Xtr, ytr, Xte, seed=seed)
    else:
        raise ValueError(predictor)

    bal = float(balanced_accuracy_score(yte, pred))
    ci_lo, ci_hi = _bootstrap_ci(yte, pred, n_resamples=n_bootstrap, seed=seed)
    return {
        "balanced_acc": bal,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "class_prior": class_prior,
        "n_test": int(len(te)),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _per_month_verdict(month_results: dict) -> dict:
    """Compute the three-criterion verdict for a single month.

    Each entry is per-symbol with `flat_lr`, `frozen_encoder_lr`,
    `finetuned_cnn` results. Returns dict with the three criterion booleans
    plus per-symbol detail.
    """
    n_pass_c1 = 0
    n_pass_c3 = 0
    n_eval = 0
    c2_violations: list[str] = []
    detail: dict[str, dict] = {}

    for sym, syms_data in month_results.items():
        flat = syms_data.get("flat_lr")
        gate1 = syms_data.get("frozen_encoder_lr")
        cnn = syms_data.get("finetuned_cnn")
        if flat is None or gate1 is None or cnn is None:
            detail[sym] = {"underpowered": True}
            continue
        n_eval += 1
        c1 = (cnn["balanced_acc"] - flat["balanced_acc"]) >= CRITERION_1_DELTA
        c3 = (cnn["balanced_acc"] - gate1["balanced_acc"]) >= CRITERION_3_DELTA

        # Criterion 2: per-symbol regression check.
        # Point estimate must not regress ≥1.0pp.
        regress_pp = gate1["balanced_acc"] - cnn["balanced_acc"]
        # CI half-width of Gate 1 baseline.
        gate1_half = (gate1["ci_hi"] - gate1["ci_lo"]) / 2.0
        # CI lower-bound regression: CNN's CI lo vs Gate 1's CI lo.
        ci_lo_regression = gate1["ci_lo"] - cnn["ci_lo"]
        ci_lo_threshold = max(0.010, gate1_half)
        c2_pass = (
            regress_pp < CRITERION_2_REGRESSION_PP
            and ci_lo_regression < ci_lo_threshold
        )
        if not c2_pass:
            c2_violations.append(sym)

        if c1:
            n_pass_c1 += 1
        if c3:
            n_pass_c3 += 1

        detail[sym] = {
            "flat_lr_acc": flat["balanced_acc"],
            "gate1_acc": gate1["balanced_acc"],
            "cnn_acc": cnn["balanced_acc"],
            "delta_vs_flat": cnn["balanced_acc"] - flat["balanced_acc"],
            "delta_vs_gate1": cnn["balanced_acc"] - gate1["balanced_acc"],
            "criterion_1_pass": c1,
            "criterion_2_pass": c2_pass,
            "criterion_3_pass": c3,
        }

    return {
        "n_evaluated": n_eval,
        "criterion_1_n_pass": n_pass_c1,
        "criterion_1_required": CRITERION_1_MIN_SYMBOLS,
        "criterion_1_pass": n_pass_c1 >= CRITERION_1_MIN_SYMBOLS,
        "criterion_2_n_violations": len(c2_violations),
        "criterion_2_violations": c2_violations,
        "criterion_2_pass": len(c2_violations) == 0,
        "criterion_3_n_pass": n_pass_c3,
        "criterion_3_required": CRITERION_3_MIN_SYMBOLS,
        "criterion_3_pass": n_pass_c3 >= CRITERION_3_MIN_SYMBOLS,
        "per_symbol": detail,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned model checkpoint (finetuned-best.pt)",
    )
    ap.add_argument(
        "--gate1-baseline-checkpoint",
        type=Path,
        required=True,
        help="Frozen encoder checkpoint for Criterion 3 (encoder-best.pt)",
    )
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--months", nargs="+", default=["2026-02", "2026-03"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Primary horizon (default 500 per spec amendment).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.horizon not in HORIZONS:
        raise SystemExit(f"--horizon must be in {HORIZONS}, got {args.horizon}")

    device = _pick_device()
    print(
        f"[gate2-eval] device={device}  horizon=H{args.horizon}  months={args.months}"
    )

    finetuned = _load_finetuned(args.checkpoint, device)
    gate1_encoder = _load_encoder(args.gate1_baseline_checkpoint, device)
    print(f"[gate2-eval] fine-tuned ckpt: {args.checkpoint}")
    print(f"[gate2-eval] gate1 baseline ckpt: {args.gate1_baseline_checkpoint}")

    eval_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]
    months_results: dict[str, dict] = {}

    for month_prefix in args.months:
        month_entry: dict[str, dict] = {}
        for sym in eval_symbols:
            shards = _symbol_month_shards(args.cache, sym, month_prefix)
            if not shards:
                month_entry[sym] = {
                    "flat_lr": None,
                    "frozen_encoder_lr": None,
                    "finetuned_cnn": None,
                    "n_test": 0,
                    "class_prior": None,
                    "underpowered": True,
                    "reason": "no shards",
                }
                continue
            dataset = TapeDataset(shards, stride=STRIDE_EVAL, mode="eval")
            if len(dataset) < MIN_VALID:
                month_entry[sym] = {
                    "flat_lr": None,
                    "frozen_encoder_lr": None,
                    "finetuned_cnn": None,
                    "n_test": 0,
                    "class_prior": None,
                    "underpowered": True,
                    "reason": f"only {len(dataset)} windows < min {MIN_VALID}",
                }
                continue

            t0 = time.time()
            data = _forward_and_extract(
                finetuned,
                gate1_encoder,
                dataset,
                device,
                args.batch_size,
                args.horizon,
            )

            # Fine-tuned predictions: sigmoid(logit_at_horizon) > 0.5.
            h_idx = data["horizon_index"]
            ft_logits = data["finetuned_logits"][:, h_idx]
            ft_pred = (1.0 / (1.0 + np.exp(-ft_logits)) > 0.5).astype(np.int64)

            flat = _compute_cell(
                data["flat"],
                data["labels"],
                data["masks"],
                predictor="flat_lr",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            gate1 = _compute_cell(
                data["gate1_emb"],
                data["labels"],
                data["masks"],
                predictor="frozen_encoder_lr",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            cnn = _compute_cell(
                data["flat"],  # X is unused for finetuned_cnn
                data["labels"],
                data["masks"],
                predictor="finetuned_cnn",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                finetuned_pred=ft_pred,
            )
            elapsed = time.time() - t0

            month_entry[sym] = {
                "flat_lr": flat,
                "frozen_encoder_lr": gate1,
                "finetuned_cnn": cnn,
                "n_test": cnn["n_test"] if cnn else 0,
                "class_prior": cnn["class_prior"] if cnn else None,
                "elapsed_s": elapsed,
            }
            if flat is None or gate1 is None or cnn is None:
                print(
                    f"[gate2-eval] {month_prefix} {sym:<8s}: "
                    "underpowered (single-class fold or insufficient windows)"
                )
            else:
                print(
                    f"[gate2-eval] {month_prefix} {sym:<8s}: "
                    f"flat={flat['balanced_acc']:.4f} "
                    f"gate1={gate1['balanced_acc']:.4f} "
                    f"cnn={cnn['balanced_acc']:.4f}  "
                    f"(n_te={cnn['n_test']}, prior={cnn['class_prior']:.3f})"
                )

        months_results[month_prefix] = month_entry

    # ---- Verdict ----
    verdict_per_month: dict[str, dict] = {
        m: _per_month_verdict(month_entry) for m, month_entry in months_results.items()
    }

    # Both months must satisfy ALL THREE criteria.
    overall_pass = all(
        v["criterion_1_pass"] and v["criterion_2_pass"] and v["criterion_3_pass"]
        for v in verdict_per_month.values()
    )

    gate2_verdict_summary = {
        f"criterion_1_flat_lr_+0_5pp_{CRITERION_1_MIN_SYMBOLS}plus_24": {
            m: ("PASS" if v["criterion_1_pass"] else "FAIL")
            for m, v in verdict_per_month.items()
        },
        "criterion_2_no_per_symbol_regression": {
            m: ("PASS" if v["criterion_2_pass"] else "FAIL")
            for m, v in verdict_per_month.items()
        },
        f"criterion_3_frozen_encoder_lr_+0_3pp_{CRITERION_3_MIN_SYMBOLS}plus_24": {
            m: ("PASS" if v["criterion_3_pass"] else "FAIL")
            for m, v in verdict_per_month.items()
        },
        "overall": "PASS" if overall_pass else "FAIL",
    }

    report = {
        "checkpoint_finetuned": str(args.checkpoint),
        "checkpoint_gate1_baseline": str(args.gate1_baseline_checkpoint),
        "horizon": args.horizon,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "months": months_results,
        "verdict_per_month": verdict_per_month,
        "gate2_verdict": gate2_verdict_summary,
        "trial_count_log": list(PRIOR_TRIAL_LOG),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[gate2-eval] Report written to: {args.out}")

    # ---- Verdict table ----
    print("\n=== GATE 2 VERDICT ===")
    print(
        f"{'month':<10s}  {'C1_n':>5s}  {'C1':>4s}   "
        f"{'C2_v':>5s}  {'C2':>4s}   {'C3_n':>5s}  {'C3':>4s}"
    )
    for m, v in verdict_per_month.items():
        c1 = "PASS" if v["criterion_1_pass"] else "FAIL"
        c2 = "PASS" if v["criterion_2_pass"] else "FAIL"
        c3 = "PASS" if v["criterion_3_pass"] else "FAIL"
        print(
            f"{m:<10s}  {v['criterion_1_n_pass']:>5d}  {c1:>4s}   "
            f"{v['criterion_2_n_violations']:>5d}  {c2:>4s}   "
            f"{v['criterion_3_n_pass']:>5d}  {c3:>4s}"
        )
    print(f"OVERALL: {gate2_verdict_summary['overall']}")
    print("\nTrial-count log (anti-amnesia hygiene per plan §" 'Trial-count log"):')
    for t in PRIOR_TRIAL_LOG:
        print(f"  - {t}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

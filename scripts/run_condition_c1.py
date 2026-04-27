# scripts/run_condition_c1.py
"""Condition 1 — Wyckoff absorption probe (post-Gate-2 pre-registration).

Per the ratified pre-registration (commit `c28bc17`,
`docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md`):

    Condition 1 — Wyckoff absorption probe.
    - Construct binary label per window:
        is_absorption = (mean(effort_vs_result[-100:]) > 1.5) AND
                        (std(log_return[-100:]) < 0.5 * rolling_std_log_return) AND
                        (mean(log_total_qty[-100:]) > 0.5)
    - Train logistic regression on frozen 256-dim encoder embeddings.
    - PASS: balanced accuracy > majority+2pp on 12+/24 symbols (point
      estimate, bootstrap CI lower bound published but not the test).
    - Computed on Feb+Mar held-out windows only.

Mechanically (per symbol, AVAX excluded):

  1. Embed all stride=200 windows from Feb+Mar 2026 via the frozen encoder.
  2. Compute is_absorption per window using post-FE features.
  3. Stratified 80/20 split — fit LR on 80%, evaluate on 20%.
  4. PASS_SYM = bal_acc(LR) > base_majority + 0.02
       where base_majority = max(P(label=0), P(label=1)) on the test split.

Underpowered guards (skipped, not failed):
  - <50 absorption-positive windows in either split
  - test-fold majority class > 0.99 (degenerate)
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
from sklearn.preprocessing import StandardScaler

from tape.constants import HELD_OUT_SYMBOL, PRETRAINING_SYMBOLS, STRIDE_EVAL
from tape.probe_utils import (
    build_eval_dataset,
    forward_embeddings,
    load_encoder,
    pick_device,
    shards_for_sym_months,
)
from tape.wyckoff_labels import is_absorption_window

# Pre-registered constants (binding via commit c28bc17).
TEST_FEB_MAR: tuple[str, str] = ("2026-02", "2026-03")
MARGIN_OVER_MAJORITY: float = 0.02  # 2pp
PASS_COUNT_REQUIRED: int = 12  # 12+/24
N_BOOTSTRAP: int = 1000
C_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
TRAIN_FRAC: float = 0.80
MIN_POS_PER_SPLIT: int = 50  # underpowered guard


def _stratified_split(
    y: np.ndarray, train_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split — preserves per-class proportions."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_cut = int(len(pos) * train_frac)
    neg_cut = int(len(neg) * train_frac)
    train = np.concatenate([pos[:pos_cut], neg[:neg_cut]])
    test = np.concatenate([pos[pos_cut:], neg[neg_cut:]])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _fit_lr_with_c_search(
    X_tr: np.ndarray, y_tr: np.ndarray, *, seed: int
) -> tuple[LogisticRegression, StandardScaler, float]:
    """C-search via stratified 80/20 internal split, then refit on full train."""
    train_idx, val_idx = _stratified_split(y_tr, train_frac=0.80, seed=seed)
    if len(np.unique(y_tr[train_idx])) < 2 or len(np.unique(y_tr[val_idx])) < 2:
        # Degenerate internal split — fall back to mid-grid C
        scaler = StandardScaler().fit(X_tr)
        lr = LogisticRegression(C=1e-2, max_iter=1000, random_state=seed).fit(
            scaler.transform(X_tr), y_tr
        )
        return lr, scaler, 1e-2

    inner_scaler = StandardScaler().fit(X_tr[train_idx])
    X_inner_tr = inner_scaler.transform(X_tr[train_idx])
    X_inner_val = inner_scaler.transform(X_tr[val_idx])

    best_C = C_GRID[0]
    best_val = -1.0
    for C in C_GRID:
        lr = LogisticRegression(
            C=C, max_iter=1000, random_state=seed, class_weight="balanced"
        ).fit(X_inner_tr, y_tr[train_idx])
        pred = lr.predict(X_inner_val)
        val = float(balanced_accuracy_score(y_tr[val_idx], pred))
        if val > best_val:
            best_val = val
            best_C = C

    full_scaler = StandardScaler().fit(X_tr)
    final_lr = LogisticRegression(
        C=best_C, max_iter=1000, random_state=seed, class_weight="balanced"
    ).fit(full_scaler.transform(X_tr), y_tr)
    return final_lr, full_scaler, best_C


def _bootstrap_balanced_acc_ci(
    y_te: np.ndarray, pred: np.ndarray, *, n_resamples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_te)
    accs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(balanced_accuracy_score(y_te[idx], pred[idx]))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _evaluate_one_symbol(
    sym: str,
    cache_dir: Path,
    enc,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict:
    out: dict = {"symbol": sym}

    shards = shards_for_sym_months(cache_dir, sym, TEST_FEB_MAR)
    out["n_shards"] = len(shards)
    if not shards:
        out["status"] = "no_shards"
        out["pass_sym"] = False
        return out

    ds = build_eval_dataset(shards, stride=STRIDE_EVAL)
    out["n_windows"] = len(ds)
    if len(ds) < 200:
        out["status"] = f"underpowered_window_count (n={len(ds)})"
        out["pass_sym"] = False
        return out

    bundle = forward_embeddings(
        enc, ds, device, batch_size=batch_size, return_features=True
    )
    X = bundle["emb"]
    feats = bundle["features"]  # (N, 200, 17)

    # Per-window is_absorption.
    y = np.zeros(len(X), dtype=np.int64)
    for i in range(len(X)):
        y[i] = 1 if is_absorption_window(feats[i]) else 0

    pos_rate = float(y.mean())
    out["absorption_rate"] = pos_rate
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int(len(y) - y.sum())

    if y.sum() < MIN_POS_PER_SPLIT * 2 or (len(y) - y.sum()) < MIN_POS_PER_SPLIT * 2:
        out["status"] = (
            f"underpowered_class_count (pos={int(y.sum())}, "
            f"neg={int(len(y) - y.sum())}, need ≥{MIN_POS_PER_SPLIT*2} each)"
        )
        out["pass_sym"] = False
        return out

    train_idx, test_idx = _stratified_split(y, train_frac=TRAIN_FRAC, seed=seed)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    if len(np.unique(y_te)) < 2:
        out["status"] = "single_class_test_fold"
        out["pass_sym"] = False
        return out

    lr, scaler, best_C = _fit_lr_with_c_search(X_tr, y_tr, seed=seed)
    pred = lr.predict(scaler.transform(X_te))
    bal_acc = float(balanced_accuracy_score(y_te, pred))

    test_prior_pos = float(y_te.mean())
    base_majority = max(test_prior_pos, 1.0 - test_prior_pos)
    margin = bal_acc - base_majority
    pass_sym = margin > MARGIN_OVER_MAJORITY

    ci_lo, ci_hi = _bootstrap_balanced_acc_ci(
        y_te, pred, n_resamples=N_BOOTSTRAP, seed=seed
    )

    out["status"] = "ok"
    out["best_C"] = float(best_C)
    out["bal_acc"] = bal_acc
    out["bal_acc_ci"] = [ci_lo, ci_hi]
    out["base_majority"] = base_majority
    out["margin_over_majority"] = float(margin)
    out["pass_sym"] = bool(pass_sym)
    out["test_prior_pos"] = test_prior_pos
    out["n_train"] = int(len(y_tr))
    out["n_test"] = int(len(y_te))
    return out


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
    print(f"[c1] device={device}")
    enc = load_encoder(args.checkpoint, device)
    print(f"[c1] loaded encoder from {args.checkpoint}")
    print(f"[c1] test={TEST_FEB_MAR} margin={MARGIN_OVER_MAJORITY:+.3f}")
    print(f"[c1] PASS_COUNT_REQUIRED={PASS_COUNT_REQUIRED}/24")

    target_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]
    print(f"[c1] evaluating {len(target_symbols)} symbols")

    started = time.time()
    per_symbol: list[dict] = []
    for i, sym in enumerate(target_symbols, 1):
        rec = _evaluate_one_symbol(
            sym=sym,
            cache_dir=args.cache,
            enc=enc,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        per_symbol.append(rec)
        elapsed = time.time() - started
        if rec.get("status") == "ok":
            print(
                f"[c1] [{i:2d}/{len(target_symbols)}] {rec['symbol']:8s}: "
                f"abs_rate={rec['absorption_rate']:.3f} "
                f"bal_acc={rec['bal_acc']:.4f} "
                f"vs_maj={rec['margin_over_majority']:+.4f} "
                f"pass={rec['pass_sym']} "
                f"(n={rec['n_test']}, ci=[{rec['bal_acc_ci'][0]:.3f},{rec['bal_acc_ci'][1]:.3f}]) "
                f"elapsed={elapsed:.0f}s"
            )
        else:
            print(
                f"[c1] [{i:2d}/{len(target_symbols)}] {rec['symbol']:8s}: "
                f"{rec['status']} pass={rec['pass_sym']} elapsed={elapsed:.0f}s"
            )

    n_evaluated = sum(1 for r in per_symbol if r.get("status") == "ok")
    pass_count = sum(1 for r in per_symbol if r.get("pass_sym"))
    c1_pass = pass_count >= PASS_COUNT_REQUIRED

    margins_ok = [
        r["margin_over_majority"] for r in per_symbol if r.get("status") == "ok"
    ]
    bal_ok = [r["bal_acc"] for r in per_symbol if r.get("status") == "ok"]
    abs_rate_ok = [r["absorption_rate"] for r in per_symbol if r.get("status") == "ok"]

    summary = {
        "condition": "C1_wyckoff_absorption_probe",
        "test_window": list(TEST_FEB_MAR),
        "margin_over_majority_threshold": MARGIN_OVER_MAJORITY,
        "pass_count_required": PASS_COUNT_REQUIRED,
        "n_symbols_total": len(target_symbols),
        "n_evaluated_ok": n_evaluated,
        "pass_count": pass_count,
        "c1_pass": c1_pass,
        "mean_bal_acc": float(np.mean(bal_ok)) if bal_ok else None,
        "median_bal_acc": float(np.median(bal_ok)) if bal_ok else None,
        "mean_margin_over_majority": float(np.mean(margins_ok)) if margins_ok else None,
        "median_margin_over_majority": (
            float(np.median(margins_ok)) if margins_ok else None
        ),
        "mean_absorption_rate": float(np.mean(abs_rate_ok)) if abs_rate_ok else None,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "n_bootstrap": N_BOOTSTRAP,
        "per_symbol": per_symbol,
    }

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== CONDITION 1 VERDICT ===")
    print(
        f"  pass_count = {pass_count}/{n_evaluated} "
        f"(threshold {PASS_COUNT_REQUIRED}/24)"
    )
    if margins_ok:
        print(
            f"  margin over majority: "
            f"mean={summary['mean_margin_over_majority']:+.4f} "
            f"median={summary['median_margin_over_majority']:+.4f}"
        )
    if bal_ok:
        print(
            f"  bal_acc: "
            f"mean={summary['mean_bal_acc']:.4f} "
            f"median={summary['median_bal_acc']:.4f}"
        )
    if abs_rate_ok:
        print(
            f"  absorption_rate (mean across symbols): {summary['mean_absorption_rate']:.3f}"
        )
    print(f"  CONDITION 1 = {'PASS' if c1_pass else 'FAIL'}")
    print(f"  Report: {args.out}")
    return 0 if c1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

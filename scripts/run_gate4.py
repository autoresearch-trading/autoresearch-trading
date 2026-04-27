# scripts/run_gate4.py
"""Gate 4 — temporal stability of the frozen encoder.

Per the post-Gate-2 pre-registration ratified at commit `c28bc17`
(`docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md`):

  Gate 4 PASS = balanced accuracy at H500 drops by < 3pp on > 14/24 symbols
                between the Oct-Nov-trained probe and the Dec-Jan-trained
                probe, both evaluated on Feb-Mar 2026 held-out windows.

Mechanically:

  for sym in 24 symbols (excl AVAX held-out):
      train LR_oct_nov on encoder embeddings of Oct-Nov 2025 windows for sym
      train LR_dec_jan on encoder embeddings of Dec-Jan 2026 windows for sym
      eval both on Feb-Mar 2026 windows for sym
      drop = bal_acc(LR_dec_jan, Feb-Mar) - bal_acc(LR_oct_nov, Feb-Mar)
      pass_sym = (drop < 0.03)  # i.e., older probe didn't degrade by more

  aggregate: pass_count = sum(pass_sym for all 24 symbols)
  GATE 4 PASS = pass_count > 14

Interpretation: if the Oct-Nov-trained probe achieves comparable Feb-Mar
balanced accuracy to the Dec-Jan-trained probe, the encoder's representation
geometry is stable across the training period. If the Oct-Nov probe
significantly underperforms, the encoder's features have time-decayed —
implying the +1pp Gate-1 signal was regime-conditional.

Council-6 prediction (from post-Gate-2 review): Gate 4 PASSES with mild
stress. Within-period drift of 1.5-2.5pp on 6-10 symbols is plausible, but
no 3pp+ drop on >10/24 symbols expected.

Bootstrap CIs (1000 resamples, percentile method, fixed-classifier
sampling-variance estimator) are reported per cell for information; the
TEST is the point estimate of the drop against 0.03.

Usage:
    uv run python scripts/run_gate4.py \\
        --checkpoint runs/step3-r2/encoder-best.pt \\
        --cache data/cache \\
        --out runs/step4-r1-gate4/gate4.json \\
        --seed 0
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

from tape.constants import (
    HELD_OUT_SYMBOL,
    PRETRAINING_SYMBOLS,
    STRIDE_EVAL,
)
from tape.dataset import TapeDataset
from tape.model import EncoderConfig, TapeEncoder

# Pre-registered constants (binding via commit c28bc17).
HORIZON: int = 500
DROP_THRESHOLD: float = 0.03  # 3pp
PASS_COUNT_REQUIRED: int = 15  # > 14/24
N_BOOTSTRAP: int = 1000
TRAIN_OCT_NOV: tuple[str, str] = ("2025-10", "2025-11")
TRAIN_DEC_JAN: tuple[str, str] = ("2025-12", "2026-01")
TEST_FEB_MAR: tuple[str, str] = ("2026-02", "2026-03")
C_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1)  # match Gate 1 protocol


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _shards_for_sym_months(
    cache_dir: Path, sym: str, month_prefixes: tuple[str, ...]
) -> list[Path]:
    """Cache shards for one symbol across one or more month prefixes."""
    shards: list[Path] = []
    for mp in month_prefixes:
        shards.extend(sorted(cache_dir.glob(f"{sym}__{mp}-*.npz")))
    return shards


def _load_encoder(checkpoint_path: Path, device: torch.device) -> TapeEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc.to(device).eval()


def _forward_embeddings(
    enc: TapeEncoder,
    dataset: TapeDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings_256, labels_h500, masks_h500)."""
    embs: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[bool] = []
    n = len(dataset)
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch = [dataset[i] for i in range(bstart, bend)]
            feats_t = torch.stack([b["features"] for b in batch]).to(device)
            _, g = enc(feats_t)
            embs.append(g.cpu().numpy())
            for b in batch:
                labels.append(int(b[f"label_h{HORIZON}"]))
                masks.append(bool(b[f"label_h{HORIZON}_mask"]))
    return (
        np.concatenate(embs, axis=0),
        np.array(labels, dtype=np.int64),
        np.array(masks, dtype=bool),
    )


def _fit_lr_with_c_search(
    X_tr: np.ndarray, y_tr: np.ndarray, *, seed: int
) -> tuple[LogisticRegression, StandardScaler, float]:
    """Train an LR with C-search using a stratified 80/20 internal split.

    Mirrors Gate 1 protocol — pick the C that maximizes balanced_acc on the
    internal 20% holdout, then refit on the full train set with that C.
    Returns (fitted_lr, fitted_scaler, best_C).
    """
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr_idx = perm[:cut]
    val_idx = perm[cut:]

    scaler = StandardScaler().fit(X_tr[tr_idx])
    X_tr_s = scaler.transform(X_tr[tr_idx])
    X_val_s = scaler.transform(X_tr[val_idx])

    best_C = C_GRID[0]
    best_val_acc = -1.0
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(
            X_tr_s, y_tr[tr_idx]
        )
        pred_val = lr.predict(X_val_s)
        acc = float(balanced_accuracy_score(y_tr[val_idx], pred_val))
        if acc > best_val_acc:
            best_val_acc = acc
            best_C = C

    # Refit on full train with best C, fitting scaler on the full train.
    full_scaler = StandardScaler().fit(X_tr)
    X_tr_full_s = full_scaler.transform(X_tr)
    final_lr = LogisticRegression(C=best_C, max_iter=1000, random_state=seed).fit(
        X_tr_full_s, y_tr
    )
    return final_lr, full_scaler, best_C


def _bootstrap_balanced_acc_ci(
    y_te: np.ndarray, pred: np.ndarray, *, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile 95% CI on balanced_accuracy via test-resample bootstrap."""
    n = len(y_te)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(balanced_accuracy_score(y_te[idx], pred[idx]))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _evaluate_one_symbol(
    sym: str,
    cache_dir: Path,
    enc: TapeEncoder,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict:
    """Train LR_oct_nov + LR_dec_jan, eval both on Feb-Mar test fold.

    Returns per-symbol record with both probes' balanced accuracies, the
    drop, the pass flag, and bootstrap CIs.
    """
    out: dict = {"symbol": sym}

    # Build the three datasets.
    tr_oct_nov_shards = _shards_for_sym_months(cache_dir, sym, TRAIN_OCT_NOV)
    tr_dec_jan_shards = _shards_for_sym_months(cache_dir, sym, TRAIN_DEC_JAN)
    te_feb_mar_shards = _shards_for_sym_months(cache_dir, sym, TEST_FEB_MAR)

    out["n_shards_oct_nov"] = len(tr_oct_nov_shards)
    out["n_shards_dec_jan"] = len(tr_dec_jan_shards)
    out["n_shards_feb_mar"] = len(te_feb_mar_shards)

    # Underpowered guard: every fold must have at least some shards.
    if not (tr_oct_nov_shards and tr_dec_jan_shards and te_feb_mar_shards):
        out["status"] = "underpowered_no_shards"
        out["pass_sym"] = False
        return out

    ds_oct_nov = TapeDataset(tr_oct_nov_shards, stride=STRIDE_EVAL, mode="pretrain")
    ds_dec_jan = TapeDataset(tr_dec_jan_shards, stride=STRIDE_EVAL, mode="pretrain")
    ds_feb_mar = TapeDataset(te_feb_mar_shards, stride=STRIDE_EVAL, mode="pretrain")

    if len(ds_oct_nov) < 50 or len(ds_dec_jan) < 50 or len(ds_feb_mar) < 50:
        out["status"] = (
            f"underpowered_window_count "
            f"(oct-nov={len(ds_oct_nov)}, dec-jan={len(ds_dec_jan)}, "
            f"feb-mar={len(ds_feb_mar)})"
        )
        out["pass_sym"] = False
        return out

    # Embed all three folds.
    X_oct_nov, y_oct_nov, m_oct_nov = _forward_embeddings(
        enc, ds_oct_nov, device, batch_size
    )
    X_dec_jan, y_dec_jan, m_dec_jan = _forward_embeddings(
        enc, ds_dec_jan, device, batch_size
    )
    X_feb_mar, y_feb_mar, m_feb_mar = _forward_embeddings(
        enc, ds_feb_mar, device, batch_size
    )

    # Filter to valid labels only.
    X_oct_nov, y_oct_nov = X_oct_nov[m_oct_nov], y_oct_nov[m_oct_nov]
    X_dec_jan, y_dec_jan = X_dec_jan[m_dec_jan], y_dec_jan[m_dec_jan]
    X_feb_mar, y_feb_mar = X_feb_mar[m_feb_mar], y_feb_mar[m_feb_mar]

    out["n_train_oct_nov"] = int(len(y_oct_nov))
    out["n_train_dec_jan"] = int(len(y_dec_jan))
    out["n_test_feb_mar"] = int(len(y_feb_mar))
    out["test_class_prior"] = float(y_feb_mar.mean())

    # Skip if any fold is single-class or too small.
    if len(y_feb_mar) < 200 or len(np.unique(y_feb_mar)) < 2:
        out["status"] = (
            f"underpowered_test (n={len(y_feb_mar)}, "
            f"unique_y={len(np.unique(y_feb_mar))})"
        )
        out["pass_sym"] = False
        return out
    if len(np.unique(y_oct_nov)) < 2 or len(np.unique(y_dec_jan)) < 2:
        out["status"] = "single_class_train_fold"
        out["pass_sym"] = False
        return out

    # Train both probes.
    lr_on, sc_on, C_on = _fit_lr_with_c_search(X_oct_nov, y_oct_nov, seed=seed)
    lr_dj, sc_dj, C_dj = _fit_lr_with_c_search(X_dec_jan, y_dec_jan, seed=seed)

    # Predict on Feb-Mar.
    pred_on = lr_on.predict(sc_on.transform(X_feb_mar))
    pred_dj = lr_dj.predict(sc_dj.transform(X_feb_mar))

    bal_on = float(balanced_accuracy_score(y_feb_mar, pred_on))
    bal_dj = float(balanced_accuracy_score(y_feb_mar, pred_dj))
    drop = bal_dj - bal_on  # positive = dec_jan better than oct_nov
    # PASS condition: oct_nov should NOT be much worse than dec_jan.
    # Equivalently: drop = bal_dj - bal_on < 3pp means oct_nov holds up.
    pass_sym = drop < DROP_THRESHOLD

    # Bootstrap CIs for transparency.
    ci_on = _bootstrap_balanced_acc_ci(
        y_feb_mar, pred_on, n_resamples=N_BOOTSTRAP, seed=seed
    )
    ci_dj = _bootstrap_balanced_acc_ci(
        y_feb_mar, pred_dj, n_resamples=N_BOOTSTRAP, seed=seed + 1
    )

    out["status"] = "ok"
    out["bal_acc_oct_nov"] = bal_on
    out["bal_acc_dec_jan"] = bal_dj
    out["drop_dec_jan_minus_oct_nov"] = drop
    out["pass_sym"] = bool(pass_sym)
    out["bal_acc_oct_nov_ci"] = list(ci_on)
    out["bal_acc_dec_jan_ci"] = list(ci_dj)
    out["best_C_oct_nov"] = float(C_on)
    out["best_C_dec_jan"] = float(C_dj)
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

    device = _pick_device()
    print(f"[gate4] device={device}")
    enc = _load_encoder(args.checkpoint, device)
    print(f"[gate4] loaded encoder from {args.checkpoint}")
    print(
        f"[gate4] train_oct_nov={TRAIN_OCT_NOV} "
        f"train_dec_jan={TRAIN_DEC_JAN} test={TEST_FEB_MAR}"
    )
    print(f"[gate4] horizon=H{HORIZON} drop_threshold={DROP_THRESHOLD}")
    print(f"[gate4] PASS_COUNT_REQUIRED={PASS_COUNT_REQUIRED}/24")

    # 24 symbols (exclude AVAX held-out).
    target_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]
    print(f"[gate4] evaluating {len(target_symbols)} symbols")

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
                f"[gate4] [{i:2d}/{len(target_symbols)}] {sym:8s}: "
                f"oct_nov={rec['bal_acc_oct_nov']:.4f} "
                f"dec_jan={rec['bal_acc_dec_jan']:.4f} "
                f"drop={rec['drop_dec_jan_minus_oct_nov']:+.4f} "
                f"pass={rec['pass_sym']} "
                f"(n_te={rec['n_test_feb_mar']}, prior={rec['test_class_prior']:.3f}) "
                f"elapsed={elapsed:.0f}s"
            )
        else:
            print(
                f"[gate4] [{i:2d}/{len(target_symbols)}] {sym:8s}: "
                f"{rec['status']} pass={rec['pass_sym']} elapsed={elapsed:.0f}s"
            )

    # Aggregate
    n_evaluated = sum(1 for r in per_symbol if r.get("status") == "ok")
    pass_count = sum(1 for r in per_symbol if r.get("pass_sym"))
    gate4_pass = pass_count >= PASS_COUNT_REQUIRED

    drops_ok = [
        r["drop_dec_jan_minus_oct_nov"] for r in per_symbol if r.get("status") == "ok"
    ]
    bal_on_ok = [r["bal_acc_oct_nov"] for r in per_symbol if r.get("status") == "ok"]
    bal_dj_ok = [r["bal_acc_dec_jan"] for r in per_symbol if r.get("status") == "ok"]

    summary = {
        "horizon": HORIZON,
        "drop_threshold": DROP_THRESHOLD,
        "pass_count_required": PASS_COUNT_REQUIRED,
        "n_symbols_total": len(target_symbols),
        "n_evaluated_ok": n_evaluated,
        "pass_count": pass_count,
        "gate4_pass": gate4_pass,
        "mean_drop": float(np.mean(drops_ok)) if drops_ok else None,
        "median_drop": float(np.median(drops_ok)) if drops_ok else None,
        "max_drop": float(np.max(drops_ok)) if drops_ok else None,
        "mean_bal_acc_oct_nov": float(np.mean(bal_on_ok)) if bal_on_ok else None,
        "mean_bal_acc_dec_jan": float(np.mean(bal_dj_ok)) if bal_dj_ok else None,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "n_bootstrap": N_BOOTSTRAP,
        "train_oct_nov": list(TRAIN_OCT_NOV),
        "train_dec_jan": list(TRAIN_DEC_JAN),
        "test_feb_mar": list(TEST_FEB_MAR),
        "per_symbol": per_symbol,
    }

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== GATE 4 VERDICT ===")
    print(
        f"  pass_count = {pass_count}/{n_evaluated} "
        f"(threshold {PASS_COUNT_REQUIRED}/24)"
    )
    if drops_ok:
        print(
            f"  drop (dec_jan - oct_nov): "
            f"mean={summary['mean_drop']:+.4f} "
            f"median={summary['median_drop']:+.4f} "
            f"max={summary['max_drop']:+.4f}"
        )
    print(
        f"  mean bal_acc: oct_nov-trained={summary['mean_bal_acc_oct_nov']:.4f}, "
        f"dec_jan-trained={summary['mean_bal_acc_dec_jan']:.4f}"
    )
    print(f"  GATE 4 = {'PASS' if gate4_pass else 'FAIL'}")
    print(f"  Report: {args.out}")
    return 0 if gate4_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

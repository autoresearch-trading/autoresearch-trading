"""Random-projection control baseline for Gate 0.

Same walk-forward pipeline as run_gate0.py but swaps PCA(n=20) for a
fixed-seed random projection of the 85-dim flat features to 20 dimensions.

Purpose: establish a noise floor.  If PCA+LR substantially beats RP+LR, the
85-dim flat features carry real structure.  If they are similar, Gate 0 results
could be base-rate artifacts.

The projection matrix is NON-ADAPTIVE: it is drawn once from a fixed seed and
frozen for ALL folds and ALL symbols.  Only the downstream StandardScaler +
LogisticRegression refit per fold (same as Gate 0).

Usage:
    uv run python scripts/run_random_baseline.py \\
        --cache data/cache \\
        --out docs/experiments/gate0-random-control \\
        --symbols BTC ETH SOL \\
        --seed 42

Output:
    <out>.json  — machine-readable, per-symbol per-horizon scores + summary
    <out>.md    — human-readable table

Critical invariants (same as run_gate0.py):
  - Shards with date >= 2026-04-14 are rejected (gotcha #17, April hold-out).
  - AVAX is included in per-symbol table but flagged as Gate-3 held-out.
  - Direction labels with mask=False (NaN territory) are dropped before
    training/testing — never imputed.
  - StandardScaler fitted on training fold only (no leakage).
  - Projection matrix is FIXED (not fitted) — no leakage possible.
  - Walk-forward 3-fold, embargo=600 events (gotcha #12).
  - H500 primary metric is balanced accuracy (base-rate non-stationary, Gate 4).
  - H10/H50/H100 primary metric is raw accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.cache import load_shard
from tape.constants import (
    APRIL_HELDOUT_START,
    DIRECTION_HORIZONS,
    HELD_OUT_SYMBOL,
    STRIDE_EVAL,
    WINDOW_LEN,
)
from tape.flat_features import extract_flat_features_batch
from tape.splits import walk_forward_folds

# ---------------------------------------------------------------------------
# Constants (mirror Gate 0 defaults for direct comparability)
# ---------------------------------------------------------------------------

_RP_IN_DIM: int = 85
_RP_OUT_DIM: int = 20
_LR_C: float = 1.0
_LR_MAX_ITER: int = 1_000
_K_FOLDS: int = 3
_EMBARGO: int = 600
_MIN_TRAIN: int = 2_000
_MIN_TEST: int = 500
_MIN_LABELED_WINDOWS: int = 50


# ---------------------------------------------------------------------------
# Projection matrix (the ONLY public function added vs. Gate 0)
# ---------------------------------------------------------------------------


def build_projection_matrix(seed: int = 42) -> np.ndarray:
    """Return a fixed random projection matrix of shape (85, 20).

    Columns are L2-normalized so each output dimension has unit variance in
    expectation when input is drawn from N(0, I_{85}).

    The matrix is deterministic given `seed` and must NOT be modified after
    construction.  Freeze it across all folds and all symbols.

    Parameters
    ----------
    seed : int
        RNG seed.  Default 42 (spec requirement).

    Returns
    -------
    np.ndarray of shape (85, 20), dtype float32
    """
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((_RP_IN_DIM, _RP_OUT_DIM)).astype(np.float32)
    # Column-normalize: divide each column by its L2 norm so
    # ||P[:,j]||^2 == 1, meaning var(x @ P[:,j]) == 1 when x ~ N(0, I_{85}).
    col_norms = np.linalg.norm(P, axis=0, keepdims=True)  # (1, 20)
    P = P / col_norms
    return P


# ---------------------------------------------------------------------------
# Internal helpers (mirroring run_gate0.py structure)
# ---------------------------------------------------------------------------


def _load_symbol_shards(
    cache: Path,
    symbol: str,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Load and concatenate all shards for `symbol` from `cache`.

    Returns (features, labels, masks) or None if no shards found.

    Raises ValueError if any shard has date >= APRIL_HELDOUT_START (gotcha #17).
    """
    shards = sorted(cache.glob(f"{symbol}__*.npz"))
    if not shards:
        return None

    feat_parts: list[np.ndarray] = []
    label_parts: dict[int, list[np.ndarray]] = {h: [] for h in DIRECTION_HORIZONS}
    mask_parts: dict[int, list[np.ndarray]] = {h: [] for h in DIRECTION_HORIZONS}

    for shard_path in shards:
        stem = shard_path.stem  # e.g. "BTC__2025-10-16"
        date_part = stem.split("__", 1)[1] if "__" in stem else ""
        if date_part >= APRIL_HELDOUT_START:
            raise ValueError(
                f"Shard {shard_path.name} has date {date_part} >= {APRIL_HELDOUT_START} "
                f"(gotcha #17 — April hold-out must not be loaded for Gate 0 / RP control)"
            )

        payload = load_shard(shard_path)
        feat_parts.append(payload["features"].astype(np.float32))
        for h in DIRECTION_HORIZONS:
            label_parts[h].append(payload[f"dir_h{h}"])
            mask_parts[h].append(payload[f"dir_mask_h{h}"])

    all_feats = np.concatenate(feat_parts, axis=0)
    all_labels = {h: np.concatenate(label_parts[h], axis=0) for h in DIRECTION_HORIZONS}
    all_masks = {h: np.concatenate(mask_parts[h], axis=0) for h in DIRECTION_HORIZONS}
    return all_feats, all_labels, all_masks


def _build_eval_windows(
    n_events: int,
    window_len: int = WINDOW_LEN,
    stride: int = STRIDE_EVAL,
) -> list[int]:
    """Return list of valid window start indices at eval stride."""
    if n_events < window_len:
        return []
    return list(range(0, n_events - window_len + 1, stride))


def _extract_flat_features_for_windows(
    features: np.ndarray,
    starts: list[int],
    window_len: int = WINDOW_LEN,
) -> np.ndarray:
    """Extract 85-dim flat features for each window start.

    Returns float32 array of shape (len(starts), 85).
    """
    if not starts:
        return np.empty((0, 85), dtype=np.float32)
    windows = np.stack([features[s : s + window_len] for s in starts], axis=0)
    return extract_flat_features_batch(windows)


def _fit_fold_rp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    proj: np.ndarray,
    C: float = _LR_C,
    random_state: int = 0,
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit StandardScaler → random-projection → LogisticRegression on training data.

    The projection matrix `proj` is FROZEN (not fitted). Only StandardScaler
    and LogisticRegression are fitted on training data.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 85)
    y_train : np.ndarray, shape (n_train,)
    proj : np.ndarray, shape (85, 20) — fixed projection matrix (not mutated)
    C : float — LR regularization strength
    random_state : int

    Returns
    -------
    (StandardScaler, LogisticRegression) — fitted on training fold only
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_tr_proj = X_tr_scaled @ proj  # (n_train, 20)

    lr = LogisticRegression(C=C, max_iter=_LR_MAX_ITER, random_state=random_state)
    lr.fit(X_tr_proj, y_train)
    return scaler, lr


# ---------------------------------------------------------------------------
# Per-symbol evaluator
# ---------------------------------------------------------------------------


def evaluate_symbol(
    cache: Path,
    symbol: str,
    proj: np.ndarray,
    horizons: tuple[int, ...] = DIRECTION_HORIZONS,
    seed: int = 42,
    *,
    min_labeled_windows: int = _MIN_LABELED_WINDOWS,
    min_train: int = _MIN_TRAIN,
    min_test: int = _MIN_TEST,
    embargo: int = _EMBARGO,
    n_folds: int = _K_FOLDS,
) -> dict[str, Any] | None:
    """Run random-projection + LR evaluation for a single symbol.

    Returns a dict keyed by 'h{horizon}' with accuracy metrics, or None if no
    data found.  Same interface as run_gate0.evaluate_symbol.
    """
    loaded = _load_symbol_shards(cache, symbol)
    if loaded is None:
        return None
    feats, labels, masks = loaded

    starts = _build_eval_windows(len(feats))
    if not starts:
        return None

    X = _extract_flat_features_for_windows(feats, starts)
    ends = np.array([s + WINDOW_LEN - 1 for s in starts], dtype=np.int64)

    is_avax = symbol == HELD_OUT_SYMBOL

    per_h: dict[str, Any] = {}
    for h in horizons:
        lbl = labels[h][ends]
        msk = masks[h][ends]

        valid_idx = np.where(msk)[0]
        if len(valid_idx) < min_labeled_windows:
            per_h[f"h{h}"] = {
                "error": "insufficient_labeled_windows",
                "n": int(len(valid_idx)),
                "avax_gate3_holdout": is_avax,
            }
            continue

        Xv = X[valid_idx]
        yv = lbl[valid_idx].astype(np.int64)

        accs: list[float] = []
        bals: list[float] = []

        try:
            folds = walk_forward_folds(
                np.arange(len(Xv), dtype=np.int64),
                n_folds=n_folds,
                embargo=embargo,
                min_train=min_train,
                min_test=min_test,
            )
        except ValueError:
            folds = []

        for tr_idx, te_idx in folds:
            scaler, lr = _fit_fold_rp(Xv[tr_idx], yv[tr_idx], proj, random_state=seed)
            X_te_proj = scaler.transform(Xv[te_idx]) @ proj
            y_pred = lr.predict(X_te_proj)

            accs.append(float(accuracy_score(yv[te_idx], y_pred)))
            bals.append(float(balanced_accuracy_score(yv[te_idx], y_pred)))

        if not accs:
            per_h[f"h{h}"] = {
                "error": "no_folds_completed",
                "n_windows": int(len(Xv)),
                "avax_gate3_holdout": is_avax,
            }
        else:
            per_h[f"h{h}"] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "balanced_accuracy_mean": float(np.mean(bals)),
                "balanced_accuracy_std": float(np.std(bals)),
                "n_folds": len(accs),
                "n_windows": int(len(Xv)),
                "avax_gate3_holdout": is_avax,
            }

    return per_h


# ---------------------------------------------------------------------------
# Summary aggregation (identical logic to run_gate0._build_summary)
# ---------------------------------------------------------------------------


def _build_summary(
    results: dict[str, Any],
    horizons: tuple[int, ...],
    threshold: float = 0.514,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for h in horizons:
        key = f"h{h}"
        metric = "balanced_accuracy_mean" if h == 500 else "accuracy_mean"
        vals: list[float] = []
        n_successful: int = 0
        n_errors: int = 0
        error_types: dict[str, int] = {}
        symbol_window_counts: dict[str, int] = {}

        for sym, sym_data in results.items():
            h_data = sym_data.get(key, {})
            if "error" in h_data:
                n_errors += 1
                err = h_data["error"]
                error_types[err] = error_types.get(err, 0) + 1
            elif metric in h_data:
                vals.append(h_data[metric])
                n_successful += 1
            if "n_windows" in h_data:
                symbol_window_counts[sym] = int(h_data["n_windows"])

        if vals:
            arr = np.array(vals)
            summary[key] = {
                "mean_accuracy": float(arr.mean()),
                "median_accuracy": float(np.median(arr)),
                "std_accuracy": float(arr.std()),
                "n_symbols": len(vals),
                "n_symbols_above_514": int((arr > threshold).sum()),
                "n_successful": n_successful,
                "n_errors": n_errors,
                "error_types": error_types,
                "symbol_window_counts": symbol_window_counts,
                "primary_metric": metric,
                "threshold": threshold,
            }
        else:
            summary[key] = {
                "n_symbols": 0,
                "n_successful": 0,
                "n_errors": n_errors,
                "error_types": error_types,
                "symbol_window_counts": symbol_window_counts,
                "error": "no_symbols_with_data",
            }
    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_md_report(
    results: dict[str, Any],
    summary: dict[str, Any],
    horizons: tuple[int, ...],
    out_path: Path,
    proj_seed: int,
) -> None:
    """Write human-readable Markdown report."""
    lines: list[str] = [
        "# Gate 0 Random-Projection Control Baseline",
        "",
        "**Method:** 85-dim flat features (mean/std/skew/kurt/last per channel) → "
        "StandardScaler → FixedRandomProjection(seed={seed}, 85→20) → "
        "LogisticRegression(C=1.0).".format(seed=proj_seed),
        f"Walk-forward {_K_FOLDS}-fold, 600-event embargo. Stride=200 for evaluation windows.",
        "",
        "**Key difference from Gate 0 PCA:** The projection matrix is drawn once from "
        "a fixed seed and frozen for ALL folds and ALL symbols.  It is non-adaptive — "
        "it does not fit to training data.  Only StandardScaler and LR refit per fold.",
        "",
        "**Purpose:** Noise floor.  If PCA substantially beats this, the 85-dim features "
        "carry real structure.  If they are similar, Gate 0 may reflect base-rate drift "
        "rather than learned structure.",
        "",
        "**Metric:** H500 uses balanced accuracy. H10/H50/H100 use raw accuracy.",
        "",
    ]

    lines.append("## Per-horizon summary")
    lines.append("")
    header = "| Horizon | Mean acc | Median acc | n successful | n errors | n above 51.4% | Metric |"
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if "error" in s and s.get("n_symbols", 0) == 0:
            n_err = s.get("n_errors", 0)
            lines.append(f"| H{h} | — | — | 0 | {n_err} | — | — |")
        else:
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            n_err = s.get("n_errors", 0)
            lines.append(
                f"| H{h} "
                f"| {s['mean_accuracy']:.4f} "
                f"| {s['median_accuracy']:.4f} "
                f"| {n_succ} "
                f"| {n_err} "
                f"| {s['n_symbols_above_514']} "
                f"| {s['primary_metric']} |"
            )
    lines.append("")

    any_errors = any(summary.get(f"h{h}", {}).get("n_errors", 0) > 0 for h in horizons)
    if any_errors:
        lines.append("## Error breakdown")
        lines.append("")
        for h in horizons:
            key = f"h{h}"
            s = summary.get(key, {})
            etypes = s.get("error_types", {})
            if etypes:
                lines.append(
                    f"**H{h}:** " + ", ".join(f"{k}: {v}" for k, v in etypes.items())
                )
        lines.append("")

    lines.append("## Per-symbol results")
    lines.append("")
    col_headers = ["Symbol"] + [f"H{h}" for h in horizons] + ["n_windows", "Notes"]
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(col_headers)) + "|")

    for sym in sorted(results.keys()):
        sym_data = results[sym]
        row = [sym]
        n_windows_val = "—"
        for h in horizons:
            key = f"h{h}"
            h_data = sym_data.get(key, {})
            if "error" in h_data:
                row.append(f"err:{h_data['error'][:8]}")
            else:
                metric = "balanced_accuracy_mean" if h == 500 else "accuracy_mean"
                val = h_data.get(metric, float("nan"))
                row.append(f"{val:.4f}")
                if n_windows_val == "—" and "n_windows" in h_data:
                    n_windows_val = str(h_data["n_windows"])
        row.append(n_windows_val)
        notes = []
        if sym == HELD_OUT_SYMBOL:
            notes.append("Gate-3 held-out")
        row.append("; ".join(notes) if notes else "")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "_Generated by `scripts/run_random_baseline.py`. "
        "Do not hand-edit — re-run to update._"
    )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Random-projection control baseline (Gate 0 noise floor). "
            "Swaps PCA(n=20) for a fixed-seed random projection."
        )
    )
    parser.add_argument(
        "--cache", required=True, help="Cache directory containing .npz shards."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to evaluate (space-separated). AVAX included but flagged.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DIRECTION_HORIZONS),
        help="Direction horizons (default: 10 50 100 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the fixed random projection matrix AND LogisticRegression (default: 42).",
    )
    parser.add_argument(
        "--out",
        default="docs/experiments/gate0-random-control",
        help="Output path prefix (without extension). Writes <out>.json and <out>.md.",
    )
    parser.add_argument("--n-folds", type=int, default=_K_FOLDS)
    parser.add_argument("--min-train", type=int, default=_MIN_TRAIN)
    parser.add_argument("--min-test", type=int, default=_MIN_TEST)
    parser.add_argument("--min-labeled-windows", type=int, default=_MIN_LABELED_WINDOWS)
    parser.add_argument("--embargo", type=int, default=_EMBARGO)
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    horizons: tuple[int, ...] = tuple(sorted(args.horizons))
    seed: int = args.seed

    if not cache.is_dir():
        print(f"ERROR: cache directory not found: {cache}", file=sys.stderr)
        return 1

    # Build the fixed projection matrix once — frozen for all folds and symbols.
    proj = build_projection_matrix(seed=seed)
    print(f"Random projection matrix shape: {proj.shape}  seed={seed}")
    print(
        f"Column norms (should all be 1.0): {np.linalg.norm(proj, axis=0).round(4)[:5]} ..."
    )

    results: dict[str, Any] = {}
    for sym in args.symbols:
        if sym == HELD_OUT_SYMBOL:
            print(
                f"[{sym}] NOTE: {sym} is the Gate-3 held-out symbol — "
                "included in table with flag, NOT excluded."
            )
        try:
            r = evaluate_symbol(
                cache,
                sym,
                proj=proj,
                horizons=horizons,
                seed=seed,
                min_labeled_windows=args.min_labeled_windows,
                min_train=args.min_train,
                min_test=args.min_test,
                embargo=args.embargo,
                n_folds=args.n_folds,
            )
        except ValueError as exc:
            raise exc  # Hard gate: propagate hold-out date violations

        if r is None:
            print(f"[{sym}] no data — skipping")
            continue
        results[sym] = r

        for h in horizons:
            key = f"h{h}"
            h_data = r.get(key, {})
            if "error" in h_data:
                print(f"[{sym}] H{h}: {h_data['error']}")
            else:
                metric = "balanced_accuracy_mean" if h == 500 else "accuracy_mean"
                val = h_data.get(metric, float("nan"))
                print(
                    f"[{sym}] H{h} {metric}={val:.4f}  (n_folds={h_data.get('n_folds', 0)})"
                )

    summary = _build_summary(results, horizons)
    output: dict[str, Any] = {
        "method": "random_projection_85to20",
        "proj_seed": seed,
        **results,
        "summary": summary,
    }

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(str(out_prefix) + ".json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON report  → {json_path}")

    md_path = Path(str(out_prefix) + ".md")
    _write_md_report(results, summary, horizons, md_path, proj_seed=seed)
    print(f"MD report    → {md_path}")

    print("\n=== Random-Projection Control Summary ===")
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if s.get("n_symbols", 0) == 0 and "error" in s:
            n_err = s.get("n_errors", 0)
            print(f"  H{h}: no data  (errors={n_err})")
        else:
            metric_label = "bal-acc" if h == 500 else "acc"
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            n_err = s.get("n_errors", 0)
            etypes = s.get("error_types", {})
            err_detail = "  errors=" + str(etypes) if etypes else ""
            print(
                f"  H{h}: mean {metric_label}={s['mean_accuracy']:.4f}  "
                f"median={s['median_accuracy']:.4f}  "
                f"n>{int(s['threshold'] * 100)}%={s['n_symbols_above_514']}/{n_succ}"
                f"  completed={n_succ}/25  errors={n_err}{err_detail}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

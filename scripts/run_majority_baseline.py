"""Majority-class baseline for Gate 0.

Per-symbol per-horizon per-fold: predict the training-fold majority class for
every test observation.  This is the noise floor that Gate 0 PCA+LR must beat.

Reuses:
  _load_symbol_shards  from run_gate0.py (via tape.cache)
  _build_eval_windows  from run_gate0.py (via tape.constants)
  walk_forward_folds   from tape/splits.py

Output:
  <out>.json  — machine-readable, per-symbol per-horizon scores + summary
  <out>.md    — human-readable table

Usage:
    uv run python scripts/run_majority_baseline.py \\
        --cache data/cache \\
        --out docs/experiments/gate0-majority-baseline \\
        --symbols BTC ETH SOL

Critical invariants (same as run_gate0.py):
  - Shards with date >= 2026-04-14 are rejected (gotcha #17, April hold-out).
  - AVAX is included in per-symbol table but flagged as Gate-3 held-out.
  - Direction labels with mask=False are dropped before training/testing.
  - Walk-forward 3-fold, embargo=600 events (gotcha #12).
  - H500 primary metric is balanced accuracy; H10/H50/H100 use raw accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from tape.cache import load_shard
from tape.constants import (
    APRIL_HELDOUT_START,
    DIRECTION_HORIZONS,
    HELD_OUT_SYMBOL,
    STRIDE_EVAL,
    WINDOW_LEN,
)
from tape.splits import walk_forward_folds

# ---------------------------------------------------------------------------
# Constants (mirroring Gate 0 defaults)
# ---------------------------------------------------------------------------

_K_FOLDS: int = 3
_EMBARGO: int = 600
_MIN_TRAIN: int = 2_000
_MIN_TEST: int = 500
_MIN_LABELED_WINDOWS: int = 50


# ---------------------------------------------------------------------------
# Internal helpers (shared interface with run_gate0.py)
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
                f"(gotcha #17 — April hold-out must not be loaded for majority baseline)"
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


def _majority_class(y: np.ndarray) -> int:
    """Return the majority class (0 or 1) in y.  Ties broken in favour of 1."""
    counts = np.bincount(y.astype(np.int64), minlength=2)
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Per-symbol evaluator
# ---------------------------------------------------------------------------


def evaluate_symbol(
    cache: Path,
    symbol: str,
    horizons: tuple[int, ...] = DIRECTION_HORIZONS,
    *,
    min_labeled_windows: int = _MIN_LABELED_WINDOWS,
    min_train: int = _MIN_TRAIN,
    min_test: int = _MIN_TEST,
    embargo: int = _EMBARGO,
    n_folds: int = _K_FOLDS,
) -> dict[str, Any] | None:
    """Run majority-class evaluation for a single symbol.

    Returns a dict keyed by 'h{horizon}' with accuracy metrics, or None if no
    data found.
    """
    loaded = _load_symbol_shards(cache, symbol)
    if loaded is None:
        return None
    feats, labels, masks = loaded

    starts = _build_eval_windows(len(feats))
    if not starts:
        return None

    # Label position: end of each window (last event index)
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

        yv = lbl[valid_idx].astype(np.int64)

        accs: list[float] = []
        bals: list[float] = []

        try:
            folds = walk_forward_folds(
                np.arange(len(yv), dtype=np.int64),
                n_folds=n_folds,
                embargo=embargo,
                min_train=min_train,
                min_test=min_test,
            )
        except ValueError:
            folds = []

        for tr_idx, te_idx in folds:
            maj = _majority_class(yv[tr_idx])
            y_pred = np.full(len(te_idx), maj, dtype=np.int64)

            accs.append(float(accuracy_score(yv[te_idx], y_pred)))
            bals.append(float(balanced_accuracy_score(yv[te_idx], y_pred)))

        if not accs:
            per_h[f"h{h}"] = {
                "error": "no_folds_completed",
                "n_windows": int(len(yv)),
                "avax_gate3_holdout": is_avax,
            }
        else:
            per_h[f"h{h}"] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "balanced_accuracy_mean": float(np.mean(bals)),
                "balanced_accuracy_std": float(np.std(bals)),
                "n_folds": len(accs),
                "n_windows": int(len(yv)),
                "avax_gate3_holdout": is_avax,
            }

    return per_h


# ---------------------------------------------------------------------------
# Summary aggregation (same logic as run_gate0._build_summary)
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
) -> None:
    """Write human-readable Markdown report."""
    lines: list[str] = [
        "# Gate 0 Majority-Class Baseline",
        "",
        "**Method:** Per-symbol per-horizon per-fold majority-class predictor. "
        "For each walk-forward fold, predict the training-fold majority class "
        "uniformly for all test observations.",
        f"Walk-forward {_K_FOLDS}-fold, 600-event embargo. Stride=200 for evaluation windows.",
        "",
        "**Purpose:** Noise floor anchored to class imbalance. "
        "Gate 0 PCA+LR must exceed majority-class by >0 on 15+/25 symbols per "
        "council-1 / council-5 review.",
        "",
        "**Metric:** H500 uses balanced accuracy (base-rate non-stationary). "
        "H10/H50/H100 use raw accuracy.",
        "",
        "**Council note (council-5):** Raw accuracy at H10/H50/H100 that matches "
        "majority class is a class-imbalance artifact, not signal. "
        "Balanced accuracy (council-preferred reading) is reported separately.",
        "",
    ]

    # --- Raw-accuracy summary table ---
    lines.append("## Per-horizon summary — Raw Accuracy")
    lines.append("")
    lines.append(
        "| Horizon | Mean raw acc | Median raw acc | n successful | n errors | n above 51.4% | Metric |"
    )
    lines.append("|---|---|---|---|---|---|---|")
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

    # --- Balanced-accuracy summary table (council-preferred) ---
    lines.append(
        "## Per-horizon summary — Balanced Accuracy *(council-preferred reading)*"
    )
    lines.append("")
    lines.append(
        "| Horizon | Mean bal acc | Median bal acc | n successful | n errors | n above 51.4% |"
    )
    lines.append("|---|---|---|---|---|---|")
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if "error" in s and s.get("n_symbols", 0) == 0:
            n_err = s.get("n_errors", 0)
            lines.append(f"| H{h} | — | — | 0 | {n_err} | — |")
        else:
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            n_err = s.get("n_errors", 0)
            bal_mean = s.get("mean_balanced_accuracy", float("nan"))
            bal_med = s.get("median_balanced_accuracy", float("nan"))
            bal_above = s.get("n_symbols_above_514_balanced", "—")
            lines.append(
                f"| H{h} "
                f"| {bal_mean:.4f} "
                f"| {bal_med:.4f} "
                f"| {n_succ} "
                f"| {n_err} "
                f"| {bal_above} |"
            )
    lines.append("")

    # Per-symbol results
    lines.append("## Per-symbol results")
    lines.append("")
    col_headers = (
        ["Symbol"]
        + [f"H{h} raw" for h in horizons]
        + [f"H{h} bal" for h in horizons]
        + ["n_windows", "Notes"]
    )
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
                val = h_data.get("accuracy_mean", float("nan"))
                row.append(f"{val:.4f}")
                if n_windows_val == "—" and "n_windows" in h_data:
                    n_windows_val = str(h_data["n_windows"])
        for h in horizons:
            key = f"h{h}"
            h_data = sym_data.get(key, {})
            if "error" in h_data:
                row.append("—")
            else:
                val = h_data.get("balanced_accuracy_mean", float("nan"))
                row.append(f"{val:.4f}")
        row.append(n_windows_val)
        notes = []
        if sym == HELD_OUT_SYMBOL:
            notes.append("Gate-3 held-out")
        row.append("; ".join(notes) if notes else "")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "_Generated by `scripts/run_majority_baseline.py`. "
        "Do not hand-edit — re-run to update._"
    )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Majority-class baseline (Gate 0 noise floor anchored to class imbalance)."
    )
    parser.add_argument(
        "--cache", required=True, help="Cache directory containing .npz shards."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to evaluate.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DIRECTION_HORIZONS),
        help="Direction horizons (default: 10 50 100 500).",
    )
    parser.add_argument(
        "--out",
        default="docs/experiments/gate0-majority-baseline",
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

    if not cache.is_dir():
        print(f"ERROR: cache directory not found: {cache}", file=sys.stderr)
        return 1

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
                horizons=horizons,
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
                bal = h_data.get("balanced_accuracy_mean", float("nan"))
                print(
                    f"[{sym}] H{h} raw={h_data.get('accuracy_mean', float('nan')):.4f}  "
                    f"bal={bal:.4f}  (n_folds={h_data.get('n_folds', 0)})"
                )

    summary = _build_summary(results, horizons)

    # Augment summary with balanced-accuracy aggregate (council-preferred)
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if "n_symbols" in s and s["n_symbols"] > 0:
            bal_vals: list[float] = []
            for sym, sym_data in results.items():
                h_data = sym_data.get(key, {})
                if "balanced_accuracy_mean" in h_data:
                    bal_vals.append(h_data["balanced_accuracy_mean"])
            if bal_vals:
                bal_arr = np.array(bal_vals)
                threshold = float(s.get("threshold", 0.514))
                s["mean_balanced_accuracy"] = float(bal_arr.mean())
                s["median_balanced_accuracy"] = float(np.median(bal_arr))
                s["std_balanced_accuracy"] = float(bal_arr.std())
                s["n_symbols_above_514_balanced"] = int((bal_arr > threshold).sum())

    output: dict[str, Any] = {
        "method": "majority_class_per_fold",
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
    _write_md_report(results, summary, horizons, md_path)
    print(f"MD report    → {md_path}")

    # stdout summary
    print("\n=== Majority-Class Baseline Summary ===")
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if s.get("n_symbols", 0) == 0 and "error" in s:
            n_err = s.get("n_errors", 0)
            print(f"  H{h}: no data  (errors={n_err})")
        else:
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            n_err = s.get("n_errors", 0)
            raw_mean = s.get("mean_accuracy", float("nan"))
            bal_mean = s.get("mean_balanced_accuracy", float("nan"))
            print(
                f"  H{h}: mean raw={raw_mean:.4f}  "
                f"mean bal={bal_mean:.4f}  "
                f"n>{int(s.get('threshold', 0.514) * 100)}%={s.get('n_symbols_above_514', 0)}/{n_succ}  "
                f"errors={n_err}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

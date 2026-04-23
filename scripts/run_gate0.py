"""Gate 0: PCA + logistic regression baseline on flat features.

Walk-forward with 600-event embargo, per symbol.  Writes:
  <out>.json  — machine-readable, per-symbol per-horizon scores + summary
  <out>.md    — human-readable table

Usage:
    uv run python scripts/run_gate0.py \\
        --cache data/cache \\
        --symbols BTC ETH SOL \\
        --out docs/experiments/gate0-baseline

Optional:
    --horizons 10 50 100 500   (default: all four)
    --seed 42                  (reproducibility)

Critical invariants:
  - Shards with date >= 2026-04-14 are rejected (gotcha #17, April hold-out).
  - AVAX is included in per-symbol table but flagged as Gate-3 held-out.
  - Direction labels with mask=False (NaN territory) are dropped before
    training/testing — never imputed.
  - PCA + StandardScaler fitted on training fold only (no leakage).
  - Walk-forward 5-fold, embargo=600 events (gotcha #12).
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
from sklearn.decomposition import PCA
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
from tape.flat_features import extract_flat_features_batch, window_to_flat
from tape.splits import walk_forward_folds

# ---------------------------------------------------------------------------
# Constants (Task 12: PCA n=20, LR C=1.0)
# ---------------------------------------------------------------------------

_PCA_N_COMPONENTS: int = 20
_LR_C: float = 1.0
_LR_MAX_ITER: int = 1_000
# New defaults sized to fit illiquid symbols (e.g. LDO at H500: 4577 windows).
# needed = min_train + embargo + n_folds * min_test = 2000 + 600 + 3*500 = 4100 <= 4577.
_K_FOLDS: int = 3
_EMBARGO: int = 600
_MIN_TRAIN: int = 2_000
_MIN_TEST: int = 500
_MIN_LABELED_WINDOWS: int = 50  # minimum valid windows to attempt a fold


# ---------------------------------------------------------------------------
# Internal helpers (prefixed with _ for import by tests)
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
        # Extract date from filename: <symbol>__<date>.npz
        stem = shard_path.stem  # e.g. "BTC__2025-10-16"
        date_part = stem.split("__", 1)[1] if "__" in stem else ""
        if date_part >= APRIL_HELDOUT_START:
            raise ValueError(
                f"Shard {shard_path.name} has date {date_part} >= {APRIL_HELDOUT_START} "
                f"(gotcha #17 — April hold-out must not be loaded for Gate 0)"
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
    """Return list of valid window start indices at eval stride.

    Only starts where start + window_len <= n_events are included.
    """
    if n_events < window_len:
        return []
    return list(range(0, n_events - window_len + 1, stride))


def _extract_flat_features_for_windows(
    features: np.ndarray,
    starts: list[int],
    window_len: int = WINDOW_LEN,
) -> np.ndarray:
    """Extract 83-dim flat features for each window start.

    Returns float32 array of shape (len(starts), 83).
    """
    if not starts:
        return np.empty((0, 83), dtype=np.float32)
    windows = np.stack([features[s : s + window_len] for s in starts], axis=0)
    # windows: (N, 200, 17)
    return extract_flat_features_batch(windows)


def _fit_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = _PCA_N_COMPONENTS,
    C: float = _LR_C,
    random_state: int = 0,
) -> tuple[StandardScaler, PCA, LogisticRegression]:
    """Fit StandardScaler → PCA → LogisticRegression on training data.

    Returns the fitted (scaler, pca, lr) triple.
    The caller must apply scaler.transform and pca.transform to test data.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)

    n_comp = min(n_components, X_tr_scaled.shape[1], X_tr_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_tr_pca = pca.fit_transform(X_tr_scaled)

    lr = LogisticRegression(C=C, max_iter=_LR_MAX_ITER, random_state=random_state)
    lr.fit(X_tr_pca, y_train)
    return scaler, pca, lr


# ---------------------------------------------------------------------------
# Per-symbol evaluator
# ---------------------------------------------------------------------------


def evaluate_symbol(
    cache: Path,
    symbol: str,
    horizons: tuple[int, ...] = DIRECTION_HORIZONS,
    seed: int = 42,
    *,
    min_labeled_windows: int = _MIN_LABELED_WINDOWS,
    min_train: int = _MIN_TRAIN,
    min_test: int = _MIN_TEST,
    embargo: int = _EMBARGO,
    n_folds: int = _K_FOLDS,
) -> dict[str, Any] | None:
    """Run Gate 0 PCA+LR evaluation for a single symbol.

    Returns a dict keyed by 'h{horizon}' with accuracy metrics, or None if no
    data found.

    Parameters
    ----------
    cache : Path
        Cache directory containing .npz shards.
    symbol : str
        Symbol name.
    horizons : tuple of int
        Direction horizons to evaluate.
    seed : int
        Random seed for PCA and LR.
    min_labeled_windows : int
        Minimum number of valid (non-masked) windows to attempt evaluation.
        Exposed as a parameter so unit tests can use small synthetic caches.
    min_train : int
        Minimum training windows per fold (passed to walk_forward_splits).
        Exposed for unit tests.
    min_test : int
        Minimum test windows per fold.
    embargo : int
        Gap (in window-index units) between train and test.  Production default
        is 600 (events); expose here for unit tests with small window counts.
    n_folds : int
        Number of walk-forward folds.  Default is _K_FOLDS (3).
        Pass a larger value (e.g. 5) for liquid symbols if desired.
    """
    loaded = _load_symbol_shards(cache, symbol)
    if loaded is None:
        return None
    feats, labels, masks = loaded

    starts = _build_eval_windows(len(feats))
    if not starts:
        return None

    # Flat features for every eval window
    X = _extract_flat_features_for_windows(feats, starts)
    # Label position: end of each window (last event index)
    ends = np.array([s + WINDOW_LEN - 1 for s in starts], dtype=np.int64)

    is_avax = symbol == HELD_OUT_SYMBOL

    per_h: dict[str, Any] = {}
    for h in horizons:
        lbl = labels[h][ends]
        msk = masks[h][ends]

        # Drop masked labels (NaN territory at tail — gotcha: never impute)
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
            # Insufficient data for the requested fold configuration
            folds = []

        for tr_idx, te_idx in folds:
            scaler, pca, lr = _fit_fold(Xv[tr_idx], yv[tr_idx], random_state=seed)
            X_te = pca.transform(scaler.transform(Xv[te_idx]))
            y_pred = lr.predict(X_te)

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
# Summary aggregation
# ---------------------------------------------------------------------------


def _build_summary(
    results: dict[str, Any],
    horizons: tuple[int, ...],
    threshold: float = 0.514,
) -> dict[str, Any]:
    """Aggregate per-symbol per-horizon scores into a summary dict.

    Enriched fields:
      n_successful              — symbols that completed all requested folds
      n_errors                  — symbols that hit an error at this horizon
      error_types               — dict mapping error string → count
      symbol_window_counts      — {symbol: n_windows} for every symbol with data
      mean_balanced_accuracy    — mean balanced accuracy across symbols (all horizons)
      median_balanced_accuracy  — median balanced accuracy (council-preferred)
      n_symbols_above_514_balanced — count with balanced_accuracy_mean > 0.514
    """
    summary: dict[str, Any] = {}
    for h in horizons:
        key = f"h{h}"
        metric = "balanced_accuracy_mean" if h == 500 else "accuracy_mean"
        vals: list[float] = []
        bal_vals: list[float] = []
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
            # Always collect balanced accuracy if present (all horizons)
            if "balanced_accuracy_mean" in h_data:
                bal_vals.append(h_data["balanced_accuracy_mean"])
            # Collect window counts regardless of success/error
            if "n_windows" in h_data:
                symbol_window_counts[sym] = int(h_data["n_windows"])

        if vals:
            arr = np.array(vals)
            entry: dict[str, Any] = {
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
            if bal_vals:
                bal_arr = np.array(bal_vals)
                entry["mean_balanced_accuracy"] = float(bal_arr.mean())
                entry["median_balanced_accuracy"] = float(np.median(bal_arr))
                entry["std_balanced_accuracy"] = float(bal_arr.std())
                entry["n_symbols_above_514_balanced"] = int((bal_arr > threshold).sum())
            summary[key] = entry
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
        "# Gate 0 Results — PCA + Logistic Regression on Flat Features",
        "",
        "**Method:** 83-dim flat features (mean/std/skew/kurt/last per channel, "
        "minus time_delta_last + prev_seq_time_span_last — session-of-day leak pruned 2026-04-23) → "
        "StandardScaler → PCA(n=20) → LogisticRegression(C=1.0).",
        f"Walk-forward {_K_FOLDS}-fold, 600-event embargo. Stride=200 for evaluation windows.",
        "",
        "**Metric:** Two tables reported per council-1 / council-5 review. "
        "Balanced accuracy is the council-preferred reading at ALL horizons (not just H500). "
        "Raw accuracy is preserved for historical comparison.",
        "",
        "**Reference bar:** Gate 1 CNN probe must exceed these numbers by ≥ 0.5pp on "
        "15+/25 symbols at H100.",
        "",
    ]

    # --- Raw-accuracy summary table (historical reference) ---
    lines.append("## Per-horizon summary — Raw Accuracy *(historical reference)*")
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

    # Error breakdown
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

    # Per-symbol window counts table
    lines.append("## Per-symbol window counts")
    lines.append("")
    wc_headers = ["Symbol"] + [f"H{h} windows" for h in horizons]
    lines.append("| " + " | ".join(wc_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(wc_headers)) + "|")
    for sym in sorted(results.keys()):
        row = [sym]
        for h in horizons:
            key = f"h{h}"
            wc_map = summary.get(key, {}).get("symbol_window_counts", {})
            row.append(str(wc_map.get(sym, "—")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-symbol table — both raw and balanced accuracy columns
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
        "_Generated by `scripts/run_gate0.py`. " "Do not hand-edit — re-run to update._"
    )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate 0: PCA + logistic regression baseline on flat tape features."
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to the cache directory containing .npz shards.",
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
        help="Direction horizons to evaluate (default: 10 50 100 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA and LogisticRegression (default: 42).",
    )
    parser.add_argument(
        "--out",
        default="docs/experiments/gate0-baseline",
        help="Output path prefix (without extension). Writes <out>.json and <out>.md.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=_K_FOLDS,
        help=(
            f"Number of walk-forward folds (default: {_K_FOLDS}). "
            "Smaller values allow illiquid symbols to complete evaluation."
        ),
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=_MIN_TRAIN,
        help=(
            f"Minimum training windows per fold (default: {_MIN_TRAIN}). "
            "Reduce to fit illiquid symbols."
        ),
    )
    parser.add_argument(
        "--min-test",
        type=int,
        default=_MIN_TEST,
        help=(
            f"Minimum test windows per fold (default: {_MIN_TEST}). "
            "Reduce to fit illiquid symbols."
        ),
    )
    parser.add_argument(
        "--min-labeled-windows",
        type=int,
        default=_MIN_LABELED_WINDOWS,
        help=argparse.SUPPRESS,  # hidden: only for unit tests on small synthetic caches
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=_EMBARGO,
        help=argparse.SUPPRESS,  # hidden: only for unit tests on small synthetic caches
    )
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    horizons: tuple[int, ...] = tuple(sorted(args.horizons))
    seed: int = args.seed

    if not cache.is_dir():
        print(f"ERROR: cache directory not found: {cache}", file=sys.stderr)
        return 1

    min_train: int = args.min_train
    min_labeled_windows: int = args.min_labeled_windows
    min_test: int = args.min_test
    n_folds: int = args.n_folds

    results: dict[str, Any] = {}
    for sym in args.symbols:
        if sym == HELD_OUT_SYMBOL:
            print(
                f"[{sym}] NOTE: {sym} is the Gate-3 held-out symbol — "
                "included in Gate 0 table with flag, NOT excluded."
            )
        try:
            r = evaluate_symbol(
                cache,
                sym,
                horizons=horizons,
                seed=seed,
                min_labeled_windows=min_labeled_windows,
                min_train=min_train,
                min_test=min_test,
                embargo=args.embargo,
                n_folds=n_folds,
            )
        except ValueError as exc:
            # Hard gate: propagate hold-out date violations
            raise exc

        if r is None:
            print(f"[{sym}] no data — skipping")
            continue
        results[sym] = r

        # Log primary metric per horizon
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
    output: dict[str, Any] = {**results, "summary": summary}

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(str(out_prefix) + ".json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON report  → {json_path}")

    md_path = Path(str(out_prefix) + ".md")
    _write_md_report(results, summary, horizons, md_path)
    print(f"MD report    → {md_path}")

    # Print summary to stdout — both raw and balanced accuracy per council-1/council-5
    print("\n=== Gate 0 Summary ===")
    print("  [Raw accuracy — historical reference]")
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if s.get("n_symbols", 0) == 0 and "error" in s:
            n_err = s.get("n_errors", 0)
            print(f"  H{h}: no data  (errors={n_err})")
        else:
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            n_err = s.get("n_errors", 0)
            etypes = s.get("error_types", {})
            err_detail = "  errors=" + str(etypes) if etypes else ""
            print(
                f"  H{h}: mean raw={s['mean_accuracy']:.4f}  "
                f"median={s['median_accuracy']:.4f}  "
                f"n>{int(s['threshold'] * 100)}%={s['n_symbols_above_514']}/{n_succ}"
                f"  completed={n_succ}/25  errors={n_err}{err_detail}"
            )
    print("  [Balanced accuracy — council-preferred reading (council-1/council-5)]")
    for h in horizons:
        key = f"h{h}"
        s = summary.get(key, {})
        if s.get("n_symbols", 0) == 0 and "error" in s:
            print(f"  H{h}: no data")
        else:
            n_succ = s.get("n_successful", s.get("n_symbols", 0))
            bal_mean = s.get("mean_balanced_accuracy", float("nan"))
            bal_med = s.get("median_balanced_accuracy", float("nan"))
            bal_above = s.get("n_symbols_above_514_balanced", "—")
            print(
                f"  H{h}: mean bal={bal_mean:.4f}  "
                f"median={bal_med:.4f}  "
                f"n>{int(s.get('threshold', 0.514) * 100)}%={bal_above}/{n_succ}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

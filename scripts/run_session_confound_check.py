"""Pre-pretraining session-of-day confound check (spec §Session-of-Day Confound Check).

Compares an LR on a single 4-hour-of-day one-hot feature vs PCA(20)+LR on
the 85-dim flat feature vector.  If the single-feature model exceeds PCA+LR
by > 0.5pp balanced accuracy on >= 5 symbols at H100, the `_last` columns
in tape/flat_features.py (especially time_delta_last, prev_seq_time_span_last)
are leaking session-of-day — we'd prune them before pretraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.cache import load_shard
from tape.constants import APRIL_HELDOUT_START, HELD_OUT_SYMBOL
from tape.flat_features import extract_flat_features_batch
from tape.splits import walk_forward_folds

_TARGET_HORIZON = 100
_DELTA_THRESHOLD = 0.005  # 0.5pp
_SYMBOL_COUNT_TRIGGER = 5


def _hour_one_hot_4h(ts_ms: np.ndarray) -> np.ndarray:
    """Return (N, 6) one-hot of 4-hour buckets (0..5)."""
    bucket = ((ts_ms // 1_000 // 3_600) // 4) % 6
    out = np.zeros((len(ts_ms), 6), dtype=np.float32)
    out[np.arange(len(ts_ms)), bucket] = 1.0
    return out


def _load_symbol(
    cache_dir: Path, symbol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    shards = sorted(cache_dir.glob(f"{symbol}__*.npz"))
    if not shards:
        return None
    feats: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for p in shards:
        # Guard against hold-out data — extract date from filename e.g. BTC__2026-04-14.npz
        date_part = p.stem.split("__", 1)[1] if "__" in p.stem else ""
        if date_part >= APRIL_HELDOUT_START:
            raise ValueError(f"April hold-out shard leaked into session check: {p}")
        payload = load_shard(p)
        feats.append(payload["features"])
        # Shard key is 'event_ts' (int64 ms), not 'ts_ms'
        ts.append(payload["event_ts"])
        labels.append(payload[f"dir_h{_TARGET_HORIZON}"])
        masks.append(payload[f"dir_mask_h{_TARGET_HORIZON}"])
    return (
        np.concatenate(feats),
        np.concatenate(ts),
        np.concatenate(labels),
        np.concatenate(masks),
    )


def _flatten_windows(
    features: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Slide stride-200 windows and return (X_flat, ts_first, y, mask_at_window_end)."""
    from tape.constants import STRIDE_EVAL, WINDOW_LEN

    n_events = len(features)
    starts = np.arange(0, n_events - WINDOW_LEN + 1, STRIDE_EVAL)
    if len(starts) == 0:
        return None

    # Build (N, WINDOW_LEN, 17) windows array for batch extraction
    idx = starts[:, None] + np.arange(WINDOW_LEN)[None, :]  # (N, WINDOW_LEN)
    windows = features[idx]  # (N, WINDOW_LEN, 17)

    X_flat = extract_flat_features_batch(windows)  # (N, 85)

    label_idx = starts + WINDOW_LEN - 1
    y = labels[label_idx]
    m = masks[label_idx]
    ts_first = ts[starts]
    return X_flat, ts_first, y, m


def run_check(
    *,
    cache_dir: Path,
    symbols: list[str] | None,
    out_path: Path,
) -> dict:
    if symbols is None:
        from tape.constants import PRETRAINING_SYMBOLS

        symbols = list(PRETRAINING_SYMBOLS)
    symbols = [s for s in symbols if s != HELD_OUT_SYMBOL]

    per_sym: dict[str, dict] = {}
    flagged = 0

    for sym in symbols:
        loaded = _load_symbol(cache_dir, sym)
        if loaded is None:
            continue
        feats, ts, labels, masks = loaded
        prep = _flatten_windows(feats, ts, labels, masks)
        if prep is None:
            continue
        X_flat, ts_first, y, mask = prep
        valid = mask.astype(bool)
        if valid.sum() < 4_000:
            continue
        X_flat = X_flat[valid]
        y = y[valid]
        ts_first = ts_first[valid]

        try:
            folds = walk_forward_folds(
                np.arange(len(X_flat)),
                n_folds=3,
                embargo=600,
                min_train=2_000,
                min_test=500,
            )
        except ValueError:
            continue

        flat_scores: list[float] = []
        hour_scores: list[float] = []

        # Pre-compute hour features for the full valid set
        X_hour = _hour_one_hot_4h(ts_first)

        for tr, te in folds:
            scaler = StandardScaler().fit(X_flat[tr])
            pca = PCA(n_components=20).fit(scaler.transform(X_flat[tr]))
            lr_flat = LogisticRegression(C=1.0, max_iter=1_000).fit(
                pca.transform(scaler.transform(X_flat[tr])), y[tr]
            )
            flat_scores.append(
                balanced_accuracy_score(
                    y[te],
                    lr_flat.predict(pca.transform(scaler.transform(X_flat[te]))),
                )
            )
            lr_hour = LogisticRegression(C=1.0, max_iter=1_000).fit(X_hour[tr], y[tr])
            hour_scores.append(
                balanced_accuracy_score(y[te], lr_hour.predict(X_hour[te]))
            )

        flat_avg = float(np.mean(flat_scores))
        hour_avg = float(np.mean(hour_scores))
        delta = hour_avg - flat_avg
        per_sym[sym] = {
            "flat_pca_lr_balanced_acc": flat_avg,
            "hour_only_lr_balanced_acc": hour_avg,
            "delta_pp": round(delta * 100, 4),
            "leaks": bool(delta > _DELTA_THRESHOLD),
        }
        if delta > _DELTA_THRESHOLD:
            flagged += 1

    decision = (
        "prune_last_features" if flagged >= _SYMBOL_COUNT_TRIGGER else "no_action"
    )
    payload: dict = {
        "horizon": _TARGET_HORIZON,
        "delta_threshold_pp": _DELTA_THRESHOLD * 100,
        "symbol_count_trigger": _SYMBOL_COUNT_TRIGGER,
        "n_symbols_with_leak": flagged,
        "decision": decision,
        "per_symbol": per_sym,
    }
    out_json = out_path.with_suffix(".json")
    out_md = out_path.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    md_lines = [
        "# Pre-pretraining session-of-day confound check",
        f"- Decision: **{decision}**",
        f"- Symbols flagged (delta > {_DELTA_THRESHOLD * 100:.1f}pp): {flagged}",
        "",
        "| Symbol | Flat PCA+LR | Hour-only LR | Δ (pp) | Leaks? |",
        "|--------|-------------|--------------|--------|--------|",
    ]
    for sym, row in sorted(per_sym.items()):
        md_lines.append(
            f"| {sym} | {row['flat_pca_lr_balanced_acc']:.4f} | "
            f"{row['hour_only_lr_balanced_acc']:.4f} | {row['delta_pp']:+.2f} | "
            f"{'YES' if row['leaks'] else ''} |"
        )
    out_md.write_text("\n".join(md_lines))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/step3-session-confound-check"),
    )
    args = ap.parse_args()
    payload = run_check(cache_dir=args.cache, symbols=args.symbols, out_path=args.out)
    print(
        json.dumps(
            {k: payload[k] for k in ("decision", "n_symbols_with_leak")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

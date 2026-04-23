# tape/probes.py
"""Frozen-embedding probes used during pretraining monitoring AND Gate 1.

Direction probe: H100 only during pretraining (rapid signal). Gate 1 evaluates
all four horizons on April 1–13 separately via scripts/run_pretrain_probes.py.

All probes use balanced accuracy (council round 6: raw accuracy is gameable
via per-fold label imbalance at every horizon).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tape.splits import walk_forward_folds


def direction_probe_h100(
    features: dict[str, np.ndarray],  # symbol -> (N_i, D)
    labels: dict[str, np.ndarray],  # symbol -> (N_i,) {0, 1}
    masks: dict[str, np.ndarray],  # symbol -> (N_i,) bool valid
    *,
    n_folds: int = 3,
    embargo: int = 600,
    min_train: int = 2_000,
    min_test: int = 500,
    C: float = 1.0,
) -> dict[str, float]:
    """Per-symbol balanced accuracy at H100 via walk-forward 3-fold."""
    out: dict[str, float] = {}
    for sym, feat in features.items():
        y = labels[sym]
        m = masks[sym]
        valid_idx = np.where(m)[0]
        if len(valid_idx) < min_train + embargo + n_folds * min_test:
            continue
        try:
            folds = walk_forward_folds(
                np.arange(len(valid_idx)),
                n_folds=n_folds,
                embargo=embargo,
                min_train=min_train,
                min_test=min_test,
            )
        except ValueError:
            continue

        scores: list[float] = []
        for tr, te in folds:
            tr_pos = valid_idx[tr]
            te_pos = valid_idx[te]
            scaler = StandardScaler().fit(feat[tr_pos])
            Xtr = scaler.transform(feat[tr_pos])
            Xte = scaler.transform(feat[te_pos])
            lr = LogisticRegression(C=C, max_iter=1_000).fit(Xtr, y[tr_pos])
            scores.append(balanced_accuracy_score(y[te_pos], lr.predict(Xte)))
        if scores:
            out[sym] = float(np.mean(scores))
    return out


def symbol_identity_probe(
    features: np.ndarray,  # (N, D) embeddings
    sym_ids: np.ndarray,  # (N,) int symbol IDs
    *,
    n_symbols: int = 25,
    test_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
) -> float:
    """25-class linear probe accuracy.  Spec target: < 20%.

    Note: multi_class="multinomial" kwarg dropped — removed in sklearn 1.7+.
    LogisticRegression auto-selects multinomial for multiclass targets.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_test = int(len(features) * test_frac)
    te = perm[:n_test]
    tr = perm[n_test:]
    scaler = StandardScaler().fit(features[tr])
    lr = LogisticRegression(
        C=C,
        max_iter=1_000,
    ).fit(scaler.transform(features[tr]), sym_ids[tr])
    return float(lr.score(scaler.transform(features[te]), sym_ids[te]))


def hour_of_day_probe(
    features: np.ndarray,  # (N, D) embeddings
    hours: np.ndarray,  # (N,) int 0..23
    *,
    test_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
) -> float:
    """24-class hour-of-day probe.  Spec gate: < 10%.

    Note: multi_class="multinomial" kwarg dropped — removed in sklearn 1.7+.
    LogisticRegression auto-selects multinomial for multiclass targets.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_test = int(len(features) * test_frac)
    te = perm[:n_test]
    tr = perm[n_test:]
    scaler = StandardScaler().fit(features[tr])
    lr = LogisticRegression(
        C=C,
        max_iter=1_000,
    ).fit(scaler.transform(features[tr]), hours[tr])
    return float(lr.score(scaler.transform(features[te]), hours[te]))

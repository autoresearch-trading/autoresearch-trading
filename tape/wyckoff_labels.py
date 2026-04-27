# tape/wyckoff_labels.py
"""Per-window Wyckoff binary labels for representation-quality probes.

These labels are computed entirely within a single 200-event window using
post-feature-engineering values (the (200, 17) tensor stored in
`shard["features"]`). They are deterministic, audit-friendly, and require
no out-of-window state.

Used by:
  - scripts/run_condition_c1.py — Wyckoff absorption probe (logistic
    regression on frozen embeddings, label = is_absorption per window)
  - scripts/run_condition_c3.py — ARI cluster–Wyckoff alignment (k-means
    cluster vs each of {is_absorption, is_buying_climax, is_selling_climax,
    is_stressed})
  - scripts/run_condition_c4.py — embedding trajectory (climax_score>3.0
    seed criterion to select event windows)

Operationalization notes
------------------------
The post-Gate-2 pre-registration (commit `c28bc17`) defines is_absorption
formally:

    is_absorption = (mean(effort_vs_result[-100:]) > 1.5) AND
                    (std(log_return[-100:]) < 0.5 * rolling_std_log_return) AND
                    (mean(log_total_qty[-100:]) > 0.5)

`rolling_std_log_return` is interpreted here as the std of log_return over
the FULL 200-event window — a within-window baseline. This makes the label
self-contained and reproducible without an external rolling state.

For the other 3 Wyckoff variants in C3 (buying climax, selling climax,
stressed), we mirror the per-event definitions in
`scripts/step0_validate.py`, adapted to per-window operationalization:

    is_buying_climax  = max(climax_score[150:200]) > 2.5 AND
                        mean(log_return[150:200]) > 0 AND
                        mean(log_return[50:150])  > 0
    is_selling_climax = max(climax_score[150:200]) > 2.5 AND
                        mean(log_return[150:200]) < 0 AND
                        mean(log_return[50:150])  < 0
    is_stressed       = max(climax_score[150:200]) > 3.0 AND
                        std(log_return[100:200])  > 1.5 * std(log_return[:200])

These formulae operate on the LAST half of the window (events 100–200) for
recency, with the FULL window providing the volatility baseline.
"""

from __future__ import annotations

import numpy as np

from tape.constants import FEATURE_NAMES

# Cached feature column indices.
_LOG_RET = FEATURE_NAMES.index("log_return")
_LOG_QTY = FEATURE_NAMES.index("log_total_qty")
_EVR = FEATURE_NAMES.index("effort_vs_result")
_CLIMAX = FEATURE_NAMES.index("climax_score")

# Window slicing constants.
_WINDOW_LEN = 200
_RECENT_HALF = 100  # last 100 events of the window
_LAST_QUARTER = 150  # last 50 events
_PRIOR_HUNDRED = 50  # mid 100 events (50:150)


def is_absorption_window(W: np.ndarray) -> bool:
    """Per-window is_absorption (binary).

    W : (200, 17) float32 — post-FE feature window.

    Returns True iff:
      - mean(effort_vs_result over last 100 events) > 1.5
      - std(log_return over last 100 events) < 0.5 * std(log_return over full window)
      - mean(log_total_qty over last 100 events) > 0.5
    """
    if W.shape[0] != _WINDOW_LEN:
        raise ValueError(f"expected window length {_WINDOW_LEN}, got {W.shape[0]}")
    evr_recent = W[_RECENT_HALF:, _EVR]
    ret_recent = W[_RECENT_HALF:, _LOG_RET]
    qty_recent = W[_RECENT_HALF:, _LOG_QTY]
    ret_full = W[:, _LOG_RET]
    full_std = float(np.std(ret_full)) + 1e-12
    return bool(
        float(np.mean(evr_recent)) > 1.5
        and float(np.std(ret_recent)) < 0.5 * full_std
        and float(np.mean(qty_recent)) > 0.5
    )


def is_buying_climax_window(W: np.ndarray) -> bool:
    """Per-window is_buying_climax: high climax + sustained uptrend."""
    if W.shape[0] != _WINDOW_LEN:
        raise ValueError(f"expected window length {_WINDOW_LEN}, got {W.shape[0]}")
    climax_last = W[_LAST_QUARTER:, _CLIMAX]
    ret_last = W[_LAST_QUARTER:, _LOG_RET]
    ret_prior = W[_PRIOR_HUNDRED:_LAST_QUARTER, _LOG_RET]
    return bool(
        float(np.max(climax_last)) > 2.5
        and float(np.mean(ret_last)) > 0.0
        and float(np.mean(ret_prior)) > 0.0
    )


def is_selling_climax_window(W: np.ndarray) -> bool:
    """Per-window is_selling_climax: high climax + sustained downtrend."""
    if W.shape[0] != _WINDOW_LEN:
        raise ValueError(f"expected window length {_WINDOW_LEN}, got {W.shape[0]}")
    climax_last = W[_LAST_QUARTER:, _CLIMAX]
    ret_last = W[_LAST_QUARTER:, _LOG_RET]
    ret_prior = W[_PRIOR_HUNDRED:_LAST_QUARTER, _LOG_RET]
    return bool(
        float(np.max(climax_last)) > 2.5
        and float(np.mean(ret_last)) < 0.0
        and float(np.mean(ret_prior)) < 0.0
    )


def is_stressed_window(W: np.ndarray) -> bool:
    """Per-window is_stressed: very high climax + elevated vol."""
    if W.shape[0] != _WINDOW_LEN:
        raise ValueError(f"expected window length {_WINDOW_LEN}, got {W.shape[0]}")
    climax_last = W[_LAST_QUARTER:, _CLIMAX]
    ret_last_100 = W[_RECENT_HALF:, _LOG_RET]
    ret_full = W[:, _LOG_RET]
    full_std = float(np.std(ret_full)) + 1e-12
    return bool(
        float(np.max(climax_last)) > 3.0
        and float(np.std(ret_last_100)) > 1.5 * full_std
    )


def climax_seed_score(W: np.ndarray) -> float:
    """Per-window climax-event seed score for C4.

    Returns max(climax_score) over the LAST 50 events. The C4 seed
    criterion is `score > 3.0`; ranking by this score lets us deterministically
    pick the strongest climax-event candidates without manual annotation.
    """
    if W.shape[0] != _WINDOW_LEN:
        raise ValueError(f"expected window length {_WINDOW_LEN}, got {W.shape[0]}")
    return float(np.max(W[_LAST_QUARTER:, _CLIMAX]))


def all_labels(W: np.ndarray) -> dict[str, bool]:
    """Compute all four C3 Wyckoff labels for one window."""
    return {
        "is_absorption": is_absorption_window(W),
        "is_buying_climax": is_buying_climax_window(W),
        "is_selling_climax": is_selling_climax_window(W),
        "is_stressed": is_stressed_window(W),
    }

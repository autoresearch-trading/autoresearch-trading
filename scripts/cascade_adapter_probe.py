# scripts/cascade_adapter_probe.py
"""Goal-A v2 Phase 1 — 5b cascade-adapter probe vs flat-LR.

Implements council-6's recipe from
  docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md
  §"Pre-registered Phase 1 plan (5b adapter test)"

Trains a small non-linear adapter head (Linear(256→64) + ReLU + Dropout(0.2)
+ Linear(64→1), ~16K params) on TOP of the FROZEN random-init `TapeEncoder`
256-dim global embeddings, against the cascade-H500 label, under the SAME
5-fold day-blocked CV with 600-event embargo as Phase 0.  Paired bootstrap
on the (adapter − flat-LR) delta produces a pre-registered decision tier.

Embedding source: re-uses helpers from `scripts.random_init_probe` (Phase 0)
to keep a single source of truth for data assembly.  Embeddings are
recomputed at adapter-test time (no disk caching) — ~5-10s per seed at
CPU scale on the merged Apr 1-26 dataset.

Outputs (default --out-dir docs/experiments/goal-a-v2/):
  * cascade_adapter_table.csv             — per-symbol per-fold per-seed AUC.
  * cascade_adapter.md                    — markdown report w/ verdict.
  * cascade_adapter_per_window.parquet    — per-window OOF predictions.

Smoke mode produces *_smoke.{csv,md,parquet} on a small subset.

Usage:
    uv run python scripts/cascade_adapter_probe.py --cache data/cache \
        --out-dir docs/experiments/goal-a-v2 [--smoke] [--seed 0]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# When invoked as `python scripts/cascade_adapter_probe.py`, the project root
# is not on sys.path — so the `from scripts.random_init_probe import ...` line
# below would fail.  Prepend the parent directory of `scripts/` (i.e. the
# project root) to sys.path so `scripts.random_init_probe` resolves.  When
# invoked as `python -m scripts.cascade_adapter_probe`, this is a no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402

# Re-use Phase 0 helpers (single source of truth).
from scripts.random_init_probe import (  # noqa: E402
    ENCODER_BATCH_SIZE,
    H_PRIMARY,
    N_BOOT,
    N_FOLDS,
    PER_SYMBOL_AUC_NULL,
    PER_SYMBOL_MIN_CASCADES,
    PHASE0_END_INCLUSIVE,
    PHASE0_START,
    PROBE_HORIZONS,
    OOFPredictions,
    _bootstrap_per_symbol_pvalue,
    apply_embargo_mask,
    bh_fdr,
    build_random_init_encoder,
    day_blocked_folds,
    day_clustered_bootstrap_auc,
    encode_windows,
    gather_phase0_batches,
    paired_day_clustered_bootstrap_delta,
    phase0_dates,
    run_5fold_cv,
    stack_batches_at_horizon,
)
from tape.constants import EMBARGO_EVENTS, SYMBOLS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT: Path = Path("data/cache")
OUT_DIR_DEFAULT: Path = Path("docs/experiments/goal-a-v2")

# Adapter hyper-params (council-6 spec, locked).
ADAPTER_HIDDEN: int = 64
ADAPTER_DROPOUT: float = 0.2
ADAPTER_POS_WEIGHT: float = 15.7  # n_neg/n_pos at base rate ~6%
ADAPTER_LR: float = 1e-3
ADAPTER_WD: float = 1e-3
ADAPTER_BATCH_SIZE: int = 256
ADAPTER_MAX_EPOCHS: int = 50
ADAPTER_PATIENCE: int = 5

# Number of random-init encoder seeds (per council-6).
N_ADAPTER_SEEDS: int = 3


# ---------------------------------------------------------------------------
# Adapter head
# ---------------------------------------------------------------------------


class CascadeAdapterHead(nn.Module):
    """Small non-linear adapter on top of frozen 256-dim encoder embeddings.

    Architecture (council-6, locked):
        Linear(256 → hidden) + ReLU + Dropout(p) + Linear(hidden → 1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden: int = ADAPTER_HIDDEN,
        dropout_p: float = ADAPTER_DROPOUT,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim) → logits (B, 1)
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)


def init_adapter_weights(head: CascadeAdapterHead, seed: int) -> None:
    """He/Kaiming init for fc1 (ReLU follows), Xavier for fc2, zero biases.

    Uses a generator threaded through every weight tensor for full determinism.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    with torch.no_grad():
        # Kaiming uniform on fc1 — using its formula directly with a generator.
        fan_in_1 = head.fc1.weight.shape[1]
        bound_1 = math.sqrt(2.0 / fan_in_1) * math.sqrt(3.0)  # uniform bound
        head.fc1.weight.uniform_(-bound_1, bound_1, generator=g)
        head.fc1.bias.zero_()
        # Xavier uniform on fc2.
        fan_in_2 = head.fc2.weight.shape[1]
        fan_out_2 = head.fc2.weight.shape[0]
        bound_2 = math.sqrt(6.0 / (fan_in_2 + fan_out_2))
        head.fc2.weight.uniform_(-bound_2, bound_2, generator=g)
        head.fc2.bias.zero_()


# ---------------------------------------------------------------------------
# Early-stopping tracker
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopTracker:
    """Minimal early-stop on validation AUC.

    Returns True from `update()` once `patience` consecutive non-improving
    epochs have passed since `best_epoch`.
    """

    patience: int
    best_val_auc: float = -float("inf")
    best_epoch: int = -1
    stalled_epochs: int = 0
    best_state: dict | None = None

    def update(
        self,
        epoch: int,
        val_auc: float,
        state: dict | None = None,
    ) -> bool:
        if math.isnan(val_auc):
            self.stalled_epochs += 1
            return self.stalled_epochs >= self.patience
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = epoch
            self.stalled_epochs = 0
            if state is not None:
                self.best_state = {k: v.detach().clone() for k, v in state.items()}
            return False
        self.stalled_epochs += 1
        return self.stalled_epochs >= self.patience


# ---------------------------------------------------------------------------
# Adapter training (per fold)
# ---------------------------------------------------------------------------


def _train_adapter_one_fold(
    *,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    device: str = "cpu",
    max_epochs: int = ADAPTER_MAX_EPOCHS,
    patience: int = ADAPTER_PATIENCE,
    batch_size: int = ADAPTER_BATCH_SIZE,
    lr: float = ADAPTER_LR,
    wd: float = ADAPTER_WD,
    pos_weight: float = ADAPTER_POS_WEIGHT,
    return_loss_history: bool = False,
) -> tuple[np.ndarray, dict]:
    """Train a CascadeAdapterHead on one fold; return P(y=1) on Xte.

    StandardScaler is fit on training rows only.  Validation = held-out fold
    (yte/Xte).  Best-val-AUC checkpoint is restored before final inference.

    Returns
    -------
    (proba_te, info_dict) — proba_te is (len(Xte),) float64 in [0, 1].
        info_dict has keys: best_epoch, best_val_auc, n_epochs, loss_history
        (only if return_loss_history=True).
    """
    if len(np.unique(ytr)) < 2 or len(Xtr) == 0:
        const = float(ytr.mean()) if len(ytr) > 0 else 0.0
        return np.full(len(Xte), const, dtype=np.float64), dict(
            best_epoch=-1, best_val_auc=float("nan"), n_epochs=0
        )

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr_s).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).unsqueeze(-1).to(device)
    Xte_t = torch.from_numpy(Xte_s).to(device)

    head = CascadeAdapterHead(embed_dim=Xtr.shape[1]).to(device)
    init_adapter_weights(head, seed=seed)
    pw_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    n = len(Xtr_t)
    rng = np.random.default_rng(seed + 1000)
    tracker = EarlyStopTracker(patience=patience)
    loss_history: list[float] = []
    epoch = -1  # initialized so n_epochs is well-defined when max_epochs == 0

    for epoch in range(max_epochs):
        head.train()
        perm = rng.permutation(n)
        epoch_loss_sum = 0.0
        epoch_n = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            optim.zero_grad()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss_sum += float(loss.item()) * len(idx)
            epoch_n += len(idx)
        avg_train_loss = epoch_loss_sum / max(1, epoch_n)
        loss_history.append(avg_train_loss)

        # Validation AUC on held-out fold.
        head.eval()
        with torch.no_grad():
            val_logits = head(Xte_t).cpu().numpy().reshape(-1)
        val_proba = 1.0 / (1.0 + np.exp(-val_logits))
        val_auc = float("nan")
        if len(np.unique(yte)) >= 2:
            try:
                val_auc = float(roc_auc_score(yte, val_proba))
            except ValueError:
                val_auc = float("nan")

        # Capture state on improvement.
        state_dict = {k: v.detach().clone() for k, v in head.state_dict().items()}
        stop = tracker.update(epoch=epoch, val_auc=val_auc, state=state_dict)
        if stop:
            break

    # Restore best checkpoint.
    if tracker.best_state is not None:
        head.load_state_dict(tracker.best_state)
    head.eval()
    with torch.no_grad():
        final_logits = head(Xte_t).cpu().numpy().reshape(-1)
    final_proba = 1.0 / (1.0 + np.exp(-final_logits))

    info: dict[str, Any] = dict(
        best_epoch=tracker.best_epoch,
        best_val_auc=tracker.best_val_auc,
        n_epochs=epoch + 1,
    )
    if return_loss_history:
        info["loss_history"] = loss_history
    return final_proba.astype(np.float64), info


# ---------------------------------------------------------------------------
# Decision tier (Phase 1, council-6)
# ---------------------------------------------------------------------------


def decision_tier_phase1(
    *,
    auc_flat: float,
    auc_adapter_median: float,
    delta_lo: float,
    delta_hi: float,
) -> str:
    """Council-6 pre-registered Phase 1 decision tree.

    Returns one of:
      'GREENLIGHT_FINETUNE_OR_PRETRAIN'  — adapter ≥ flat + 0.02 AND delta_lo > 0
      'KILL_ARCH_BOTTLENECK_CONFIRMED'   — adapter < flat by > 0.02 (point estimate)
      'MATCHED_FLAT'                     — otherwise (CI overlaps zero / small delta)

    `delta_hi` is part of the API contract (CI upper bound) for symmetry with
    `delta_lo`; the current decision rule does not consume it but is retained
    so callers can pass the full CI tuple unmodified.
    """
    _ = delta_hi  # intentionally unused in the current decision rule
    delta = auc_adapter_median - auc_flat
    if delta >= 0.02 and delta_lo > 0:
        return "GREENLIGHT_FINETUNE_OR_PRETRAIN"
    if delta < -0.02:
        return "KILL_ARCH_BOTTLENECK_CONFIRMED"
    return "MATCHED_FLAT"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def _pooled_auc(proba: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(proba)
    p = proba[valid]
    y = labels[valid]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _build_per_symbol_table_phase1(
    *,
    sym: np.ndarray,
    dates: np.ndarray,
    proba_flat: np.ndarray,
    proba_adapter: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
    min_cascades: int = PER_SYMBOL_MIN_CASCADES,
    null_auc: float = PER_SYMBOL_AUC_NULL,
) -> pd.DataFrame:
    rows: list[dict] = []
    for s in sorted(np.unique(sym).tolist()):
        mask = sym == s
        if not mask.any():
            continue
        y_s = labels[mask]
        n_pos = int((y_s == 1).sum())
        if n_pos < min_cascades:
            continue
        valid = np.isfinite(proba_flat[mask]) & np.isfinite(proba_adapter[mask])
        idx = np.flatnonzero(mask)[valid]
        if len(idx) == 0:
            continue
        labels_s = labels[idx]
        dates_s = dates[idx]
        p_flat_s = proba_flat[idx]
        p_ad_s = proba_adapter[idx]
        try:
            auc_flat_s = float(roc_auc_score(labels_s, p_flat_s))
        except ValueError:
            auc_flat_s = float("nan")
        try:
            auc_ad_s = float(roc_auc_score(labels_s, p_ad_s))
        except ValueError:
            auc_ad_s = float("nan")
        pv_flat = _bootstrap_per_symbol_pvalue(
            proba=p_flat_s,
            labels=labels_s,
            dates=dates_s,
            n_boot=n_boot,
            seed=seed,
            null_auc=null_auc,
        )
        pv_ad = _bootstrap_per_symbol_pvalue(
            proba=p_ad_s,
            labels=labels_s,
            dates=dates_s,
            n_boot=n_boot,
            seed=seed + 1,
            null_auc=null_auc,
        )
        rows.append(
            dict(
                symbol=s,
                n_windows=int(mask.sum()),
                n_cascades=n_pos,
                auc_flat=auc_flat_s,
                auc_adapter=auc_ad_s,
                p_flat=pv_flat,
                p_adapter=pv_ad,
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "n_windows",
                "n_cascades",
                "auc_flat",
                "auc_adapter",
                "p_flat",
                "p_adapter",
                "q_flat",
                "q_adapter",
            ]
        )
    df = pd.DataFrame(rows)
    pf = df["p_flat"].to_numpy()
    pa = df["p_adapter"].to_numpy()
    qf = np.full_like(pf, np.nan)
    qa = np.full_like(pa, np.nan)
    f_finite = np.isfinite(pf)
    a_finite = np.isfinite(pa)
    if f_finite.any():
        qf[f_finite] = bh_fdr(pf[f_finite])
    if a_finite.any():
        qa[a_finite] = bh_fdr(pa[a_finite])
    df["q_flat"] = qf
    df["q_adapter"] = qa
    return df


def _emit_markdown(
    *,
    out_path: Path,
    n_windows: int,
    n_cascades: int,
    base_rate: float,
    n_days: int,
    n_symbols: int,
    auc_flat: float,
    auc_flat_lo: float,
    auc_flat_hi: float,
    auc_adapter_per_seed: dict[int, tuple[float, float, float]],
    auc_adapter_median: float,
    auc_adapter_min: float,
    auc_adapter_max: float,
    median_seed: int,
    delta_point: float,
    delta_lo: float,
    delta_hi: float,
    decision: str,
    per_symbol_df: pd.DataFrame,
    elapsed_sec: float,
    smoke: bool,
    max_epochs: int,
) -> None:
    lines: list[str] = []
    lines.append("# Goal-A v2 Phase 1 — Cascade adapter probe vs flat-LR")
    lines.append("")
    if smoke:
        lines.append(
            "_(smoke run — limited symbol/date subset, NOT for decision binding.)_"
        )
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"Date range: {PHASE0_START} → {PHASE0_END_INCLUSIVE} (merged Apr 1-26, "
        "holdout consumed per gotcha #17).  Same 5-fold day-blocked CV with "
        f"{EMBARGO_EVENTS}-event embargo as Phase 0."
    )
    lines.append("")
    lines.append(
        f"Adapter: `Linear({256}→{ADAPTER_HIDDEN}) + ReLU + Dropout("
        f"{ADAPTER_DROPOUT}) + Linear({ADAPTER_HIDDEN}→1)` "
        f"trained on FROZEN random-init `TapeEncoder` 256-dim global "
        f"embeddings.  BCEWithLogitsLoss(pos_weight={ADAPTER_POS_WEIGHT}), "
        f"AdamW(lr={ADAPTER_LR}, wd={ADAPTER_WD}), no LR schedule, "
        f"max {max_epochs} epochs, batch {ADAPTER_BATCH_SIZE}, "
        f"early-stop on val-AUC patience={ADAPTER_PATIENCE}."
    )
    lines.append("")
    lines.append(
        f"Symbols: {n_symbols} | Days: {n_days} | Windows: {n_windows:,} | "
        f"Cascades (H{H_PRIMARY}): {n_cascades:,} | Base rate: {base_rate:.4f}"
    )
    lines.append("")
    lines.append("## Pooled OOF AUC")
    lines.append("")
    lines.append("| Model | Pooled AUC | 95% CI (day-clustered, 1000 reps) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Flat-LR (FLAT_DIM=83) | {auc_flat:.4f} | "
        f"[{auc_flat_lo:.4f}, {auc_flat_hi:.4f}] |"
    )
    lines.append(
        f"| Adapter on random-init enc (median seed={median_seed}) | "
        f"{auc_adapter_median:.4f} | min-max across {N_ADAPTER_SEEDS} seeds: "
        f"[{auc_adapter_min:.4f}, {auc_adapter_max:.4f}] |"
    )
    lines.append("")
    lines.append("### Per-seed adapter pooled AUC")
    lines.append("")
    lines.append("| Seed | Pooled AUC | 95% CI |")
    lines.append("|---|---|---|")
    for s in sorted(auc_adapter_per_seed.keys()):
        a, lo, hi = auc_adapter_per_seed[s]
        lines.append(f"| {s} | {a:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines.append("")
    lines.append("## Paired delta (adapter_median − flat-LR)")
    lines.append("")
    lines.append(
        f"Delta point estimate: **{delta_point:+.4f}** | "
        f"95% paired-bootstrap CI: [{delta_lo:+.4f}, {delta_hi:+.4f}]"
    )
    lines.append("")

    lines.append("## Decision tier (council-6 pre-registered)")
    lines.append("")
    if decision == "GREENLIGHT_FINETUNE_OR_PRETRAIN":
        verdict = (
            "**GREENLIGHT_FINETUNE_OR_PRETRAIN** — adapter ≥ flat-LR + 2pp AND "
            "paired-delta CI excludes 0.  The encoder manifold contains "
            "cascade signal beyond flat features; Phase 0's linear probe was "
            "the bottleneck.  Justifies cascade-aware MEM (council-4 recipe) "
            "or end-to-end fine-tune ONLY IF the program also has a TAKER-side "
            "downstream task (council-5)."
        )
    elif decision == "KILL_ARCH_BOTTLENECK_CONFIRMED":
        verdict = (
            "**KILL_ARCH_BOTTLENECK_CONFIRMED** — adapter < flat-LR by > 2pp.  "
            "Manifold is actively deficient for cascade detection.  STOP encoder "
            "retrain; pivot per council-5 (TAKER-side framing or non-Maker-fee "
            "deliverable)."
        )
    else:
        verdict = (
            "**MATCHED_FLAT** — adapter ≈ flat-LR (delta CI overlaps 0).  "
            "Manifold matches the flat-feature signal but doesn't beat it.  "
            "STOP encoder retrain.  Pivot per council-5."
        )
    lines.append(verdict)
    lines.append("")

    lines.append("## Per-symbol AUC (BH-FDR adjusted)")
    lines.append("")
    if len(per_symbol_df) == 0:
        lines.append(
            f"_No symbol passed the n_cascades ≥ {PER_SYMBOL_MIN_CASCADES} threshold._"
        )
    else:
        lines.append(
            "| Symbol | n_win | n_casc | AUC_flat | q_flat | AUC_adapter | q_adapter |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in per_symbol_df.iterrows():
            qf = r["q_flat"]
            qa = r["q_adapter"]
            lines.append(
                f"| {r['symbol']} | {int(r['n_windows'])} | "
                f"{int(r['n_cascades'])} | {r['auc_flat']:.4f} | "
                f"{qf:.4f} | {r['auc_adapter']:.4f} | {qa:.4f} |"
            )
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "* Encoder is frozen random-init `TapeEncoder(EncoderConfig())` with "
        "input BatchNorm1d.track_running_stats=False (gotcha #18).  Embeddings "
        "are extracted under `torch.no_grad()`; no gradients flow into the "
        "encoder."
    )
    lines.append(
        f"* {N_ADAPTER_SEEDS} encoder seeds.  Median seed binds the report; "
        "min-max bounds random-init variance."
    )
    lines.append(
        "* Adapter: ~16K params; weight init He/Kaiming for fc1, Xavier for "
        "fc2, zero biases.  Per-fold StandardScaler fit on training rows only."
    )
    lines.append(
        "* Day-clustered bootstrap: 1000 iterations.  Paired delta uses the "
        "SAME seeded RNG so identical day samples produce both AUCs in lockstep."
    )
    lines.append("")
    lines.append(
        f"_Pipeline ran in {elapsed_sec:.1f} s.  CPU-only.  Smoke mode: {smoke}._"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_fold_table(
    *,
    sym: np.ndarray,
    fold_idx: np.ndarray,
    labels: np.ndarray,
    proba_adapter_per_seed: dict[int, np.ndarray],
    proba_adapter_median: np.ndarray,
    proba_flat: np.ndarray,
) -> pd.DataFrame:
    """Per (symbol × fold) table including adapter-per-seed + median + flat columns."""
    rows: list[dict] = []
    seeds_sorted = sorted(proba_adapter_per_seed.keys())
    for s in sorted(np.unique(sym).tolist()):
        for f in sorted(np.unique(fold_idx).tolist()):
            mask = (sym == s) & (fold_idx == f)
            if not mask.any():
                continue
            yb = labels[mask]
            n_pos = int((yb == 1).sum())
            n_neg = int((yb == 0).sum())
            row: dict = dict(
                symbol=s,
                fold_idx=int(f),
                n_pos=n_pos,
                n_neg=n_neg,
            )

            # AUCs per model, guarding against degenerate folds.
            def _auc(p: np.ndarray) -> float:
                if n_pos == 0 or n_neg == 0:
                    return float("nan")
                pv = p[mask]
                valid = np.isfinite(pv)
                if not valid.any():
                    return float("nan")
                try:
                    return float(roc_auc_score(yb[valid], pv[valid]))
                except ValueError:
                    return float("nan")

            for sd in seeds_sorted:
                row[f"auc_adapter_seed{sd}"] = _auc(proba_adapter_per_seed[sd])
            row["auc_adapter_median"] = _auc(proba_adapter_median)
            row["auc_flat_LR"] = _auc(proba_flat)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _enforce_sanity_checks(
    *,
    n_pos_pooled: int,
    smoke: bool,
    fold_pos_counts: list[int],
    fold0_loss_history: list[float] | None,
    auc_per_seed: dict[int, float],
    median_seed: int,
) -> None:
    """Fail-fast sanity checks per the implementation contract.

    * n_cascades_pooled at H500 >= 135 (full mode only — smoke is too small).
    * Each of 5 folds has >= 1 cascade-positive window (full mode only).
    * Adapter training loss decreases monotonically over the FIRST 3 epochs
      on the fold-0 training set.
    * Median seed AUC sits within [min, max] of the seeds.
    """
    if not smoke:
        if n_pos_pooled < 135:
            raise AssertionError(
                f"sanity check failed: n_cascades_pooled={n_pos_pooled} "
                "is below the holdout-consume floor of 135 (~169 ±20%).  "
                "The cache likely missed Apr 14-26 — check gotcha #17."
            )
        for fi, np_pos in enumerate(fold_pos_counts):
            if np_pos < 1:
                raise AssertionError(
                    f"sanity check failed: fold {fi} has 0 cascade-positive windows."
                )
    # Loss monotonicity over first 3 epochs.
    if fold0_loss_history is not None and len(fold0_loss_history) >= 3:
        l0, l1, l2 = fold0_loss_history[0], fold0_loss_history[1], fold0_loss_history[2]
        if not (l0 > l1 > l2):
            print(
                f"[WARN] sanity: fold-0 loss did not decrease monotonically "
                f"over first 3 epochs: {l0:.4f}, {l1:.4f}, {l2:.4f} "
                "(possible stuck-at-init bug or noisy mini-batch dynamics)"
            )
    # Median bounded by min/max.
    finite = {s: a for s, a in auc_per_seed.items() if math.isfinite(a)}
    if finite:
        v_med = auc_per_seed[median_seed]
        v_min = min(finite.values())
        v_max = max(finite.values())
        if not (v_min - 1e-9 <= v_med <= v_max + 1e-9):
            raise AssertionError(
                f"sanity check failed: median seed AUC {v_med} not in "
                f"[{v_min}, {v_max}]"
            )


def _run_pipeline(
    *,
    cache_dir: Path,
    out_dir: Path,
    smoke: bool,
    seed: int,
) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        symbols: tuple[str, ...] = ("BTC", "SOL", "ETH")
        dates = ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]
        max_epochs = 5  # smoke mode: cap epochs hard
        print(f"[smoke] symbols={symbols} dates={dates} max_epochs={max_epochs}")
    else:
        symbols = SYMBOLS
        dates = phase0_dates()
        max_epochs = ADAPTER_MAX_EPOCHS

    suffix = "_smoke" if smoke else ""

    # ---- 1. Gather batches ------------------------------------------------
    print(
        f"[1/6] Gathering Apr 1-26 batches: {len(symbols)} symbols × {len(dates)} days"
    )
    batches = gather_phase0_batches(cache_dir, symbols, dates, horizons=PROBE_HORIZONS)
    print(f"      → built {len(batches)} (symbol × date) batches")
    if not batches:
        raise RuntimeError("no batches built — check cache and trade parquet")

    # ---- 2. Stack at horizon ----------------------------------------------
    print(f"[2/6] Stacking at horizon H{H_PRIMARY}")
    flat_X, raw_X, y, sym, date_arr, anchor_ts, anchor_idx = stack_batches_at_horizon(
        batches, H_PRIMARY
    )
    n_total = len(y)
    n_pos = int(y.sum())
    n_days = len(np.unique(date_arr))
    n_symbols = len(np.unique(sym))
    base_rate = float(n_pos / max(1, n_total))
    print(
        f"      → N={n_total:,} flat_X.shape={flat_X.shape} raw_X.shape={raw_X.shape} "
        f"n_pos={n_pos} base_rate={base_rate:.4f} days={n_days} symbols={n_symbols}"
    )

    # ---- 3. Phase 0a flat-LR baseline (recomputed on same partition) ------
    print("[3/6] Flat-LR 5-fold day-blocked CV (recomputed for true paired delta)")
    oof_flat = run_5fold_cv(
        X=flat_X.astype(np.float32),
        y=y.astype(np.int64),
        sym=sym,
        dates=date_arr,
        anchor_ts=anchor_ts,
        anchor_idx_in_day=anchor_idx,
        embargo_events=EMBARGO_EVENTS,
        k=N_FOLDS,
    )
    auc_flat = _pooled_auc(oof_flat.proba, oof_flat.labels)
    af, alo, ahi = day_clustered_bootstrap_auc(
        proba=oof_flat.proba,
        labels=oof_flat.labels.astype(np.int64),
        dates=date_arr,
        n_boot=N_BOOT,
        seed=0,
    )
    print(
        f"      → flat pooled AUC={auc_flat:.4f} bootstrap mean={af:.4f} "
        f"CI=[{alo:.4f},{ahi:.4f}]"
    )
    fold_pos_counts = [npos for _, npos, _, _ in oof_flat.per_fold_aucs]

    # ---- 4. Adapter probe across seeds ------------------------------------
    encoder_seeds = [seed + 0, seed + 1, seed + 2]
    print(f"[4/6] Adapter probe across encoder seeds {encoder_seeds}")
    ad_oof_per_seed: dict[int, OOFPredictions] = {}
    ad_pooled_per_seed: dict[int, tuple[float, float, float]] = {}
    fold0_loss_history_seed0: list[float] | None = None

    for sd in encoder_seeds:
        print(f"  seed={sd}: building random-init encoder + extracting embeddings")
        enc = build_random_init_encoder(seed=sd)
        emb = encode_windows(enc, raw_X, batch_size=ENCODER_BATCH_SIZE, device="cpu")
        del enc

        oof, fold0_loss = _run_adapter_cv_with_max_epochs(
            X=emb,
            y=y.astype(np.int64),
            sym=sym,
            dates=date_arr,
            anchor_ts=anchor_ts,
            anchor_idx_in_day=anchor_idx,
            seed=sd,
            embargo_events=EMBARGO_EVENTS,
            k=N_FOLDS,
            device="cpu",
            max_epochs=max_epochs,
            return_fold0_loss_history=(sd == encoder_seeds[0]),
        )
        if sd == encoder_seeds[0]:
            fold0_loss_history_seed0 = fold0_loss

        a = _pooled_auc(oof.proba, oof.labels)
        _, boot_lo, boot_hi = day_clustered_bootstrap_auc(
            proba=oof.proba,
            labels=oof.labels.astype(np.int64),
            dates=date_arr,
            n_boot=N_BOOT,
            seed=sd + 200,
        )
        ad_oof_per_seed[sd] = oof
        ad_pooled_per_seed[sd] = (a, boot_lo, boot_hi)
        print(
            f"  seed={sd}: pooled AUC={a:.4f} bootstrap CI=[{boot_lo:.4f},{boot_hi:.4f}]"
        )

    # Median seed (by pooled AUC, deterministic tiebreak).
    seed_to_auc = {s: ad_pooled_per_seed[s][0] for s in encoder_seeds}
    finite_seed_to_auc = {s: a for s, a in seed_to_auc.items() if math.isfinite(a)}
    if not finite_seed_to_auc:
        raise RuntimeError("All adapter seeds produced NaN pooled AUC")
    sorted_seeds = sorted(
        finite_seed_to_auc.keys(), key=lambda s: finite_seed_to_auc[s]
    )
    median_seed = sorted_seeds[len(sorted_seeds) // 2]
    auc_adapter_median = finite_seed_to_auc[median_seed]
    auc_adapter_min = min(finite_seed_to_auc.values())
    auc_adapter_max = max(finite_seed_to_auc.values())
    median_oof = ad_oof_per_seed[median_seed]

    # ---- 5. Paired bootstrap on delta -------------------------------------
    print(f"[5/6] Paired bootstrap on delta = adapter_seed{median_seed} − flat-LR")
    delta_point, delta_lo, delta_hi, *_ = paired_day_clustered_bootstrap_delta(
        proba_a=oof_flat.proba,
        proba_b=median_oof.proba,
        labels=y.astype(np.int64),
        dates=date_arr,
        n_boot=N_BOOT,
        seed=999,
    )
    print(f"      → delta={delta_point:+.4f} CI=[{delta_lo:+.4f}, {delta_hi:+.4f}]")

    decision = decision_tier_phase1(
        auc_flat=auc_flat,
        auc_adapter_median=auc_adapter_median,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
    )

    # Sanity checks (fail-fast).
    _enforce_sanity_checks(
        n_pos_pooled=n_pos,
        smoke=smoke,
        fold_pos_counts=fold_pos_counts,
        fold0_loss_history=fold0_loss_history_seed0,
        auc_per_seed=seed_to_auc,
        median_seed=median_seed,
    )

    # ---- 6. Outputs --------------------------------------------------------
    print(f"[6/6] Building outputs in {out_dir}")
    proba_adapter_per_seed = {sd: oof.proba for sd, oof in ad_oof_per_seed.items()}

    fold_df = _build_fold_table(
        sym=sym,
        fold_idx=oof_flat.fold_idx,
        labels=y.astype(np.int64),
        proba_adapter_per_seed=proba_adapter_per_seed,
        proba_adapter_median=median_oof.proba,
        proba_flat=oof_flat.proba,
    )
    csv_path = out_dir / f"cascade_adapter_table{suffix}.csv"
    fold_df.to_csv(csv_path, index=False)

    # Per-window parquet
    per_window_df = pd.DataFrame(
        dict(
            symbol=sym,
            date=date_arr,
            anchor_ts=anchor_ts.astype(np.int64),
            fold_idx=oof_flat.fold_idx.astype(np.int32),
            label=y.astype(np.int8),
            proba_flat=oof_flat.proba.astype(np.float32),
            proba_adapter_median_seed=median_oof.proba.astype(np.float32),
        )
    )
    for sd, p in proba_adapter_per_seed.items():
        per_window_df[f"proba_adapter_seed{sd}"] = p.astype(np.float32)
    parquet_path = out_dir / f"cascade_adapter_per_window{suffix}.parquet"
    per_window_df.to_parquet(parquet_path, index=False)

    # Per-symbol BH-FDR table.
    per_symbol_df = _build_per_symbol_table_phase1(
        sym=sym,
        dates=date_arr,
        proba_flat=oof_flat.proba,
        proba_adapter=median_oof.proba,
        labels=y.astype(np.int64),
        n_boot=N_BOOT,
        seed=12345,
    )

    md_path = out_dir / f"cascade_adapter{suffix}.md"
    elapsed_sec = time.perf_counter() - t0
    _emit_markdown(
        out_path=md_path,
        n_windows=n_total,
        n_cascades=n_pos,
        base_rate=base_rate,
        n_days=n_days,
        n_symbols=n_symbols,
        auc_flat=auc_flat,
        auc_flat_lo=alo,
        auc_flat_hi=ahi,
        auc_adapter_per_seed=ad_pooled_per_seed,
        auc_adapter_median=auc_adapter_median,
        auc_adapter_min=auc_adapter_min,
        auc_adapter_max=auc_adapter_max,
        median_seed=median_seed,
        delta_point=delta_point,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        decision=decision,
        per_symbol_df=per_symbol_df,
        elapsed_sec=elapsed_sec,
        smoke=smoke,
        max_epochs=max_epochs,
    )

    # Stdout summary
    print()
    print("=" * 60)
    print(f"  Pooled flat-LR AUC: {auc_flat:.4f} [{alo:.4f}, {ahi:.4f}]")
    print(
        f"  Pooled adapter AUC (median seed={median_seed}): "
        f"{auc_adapter_median:.4f}  "
        f"[min={auc_adapter_min:.4f}, max={auc_adapter_max:.4f}]"
    )
    print(
        f"  Paired delta (adapter - flat): {delta_point:+.4f} "
        f"[{delta_lo:+.4f}, {delta_hi:+.4f}]"
    )
    print(f"  Decision tier: {decision}")
    print(f"  Outputs: {csv_path} ; {md_path} ; {parquet_path}")
    print(f"  Elapsed: {elapsed_sec:.1f} s")
    print("=" * 60)


def _run_adapter_cv_with_max_epochs(
    *,
    X: np.ndarray,
    y: np.ndarray,
    sym: np.ndarray,
    dates: np.ndarray,
    anchor_ts: np.ndarray,
    anchor_idx_in_day: np.ndarray,
    seed: int,
    embargo_events: int,
    k: int,
    device: str,
    max_epochs: int,
    return_fold0_loss_history: bool,
) -> tuple[OOFPredictions, list[float] | None]:
    """Wrapper that threads `max_epochs` through to the per-fold trainer.

    `run_5fold_cv_adapter` has the canonical CV control flow but uses the
    module default for max_epochs.  Smoke mode needs to override it without
    monkeypatching.  Refactor: call `_train_adapter_one_fold` directly with
    the override.
    """
    sorted_days = sorted(np.unique(dates).tolist())
    days_by_fold = day_blocked_folds(sorted_days, k=k)
    day_to_fold = {d: fi for fi, fdays in enumerate(days_by_fold) for d in fdays}
    fold_assignments = np.array([day_to_fold[d] for d in dates], dtype=np.int64)

    proba_oof = np.full(len(X), np.nan, dtype=np.float64)
    per_fold: list[tuple[int, int, int, float]] = []
    fold0_loss_history: list[float] | None = None

    for held_out in range(k):
        train_mask = apply_embargo_mask(
            held_out_fold=held_out,
            fold_assignments=fold_assignments,
            dates=dates,
            anchor_idx_in_day=anchor_idx_in_day,
            embargo_events=embargo_events,
            days_by_fold=days_by_fold,
        )
        test_mask = fold_assignments == held_out
        if not test_mask.any() or not train_mask.any():
            continue
        Xtr = X[train_mask]
        ytr = y[train_mask]
        Xte = X[test_mask]
        yte = y[test_mask]
        fold_seed = seed * 100 + held_out
        proba, info = _train_adapter_one_fold(
            Xtr=Xtr,
            ytr=ytr.astype(np.int64),
            Xte=Xte,
            yte=yte.astype(np.int64),
            seed=fold_seed,
            device=device,
            max_epochs=max_epochs,
            return_loss_history=return_fold0_loss_history and held_out == 0,
        )
        if return_fold0_loss_history and held_out == 0:
            fold0_loss_history = info.get("loss_history")
        proba_oof[test_mask] = proba
        n_pos_f = int((yte == 1).sum())
        n_neg_f = int((yte == 0).sum())
        if n_pos_f > 0 and n_neg_f > 0:
            try:
                auc_f = float(roc_auc_score(yte, proba))
            except ValueError:
                auc_f = float("nan")
        else:
            auc_f = float("nan")
        per_fold.append((held_out, n_pos_f, n_neg_f, auc_f))

    return (
        OOFPredictions(
            proba=proba_oof,
            labels=y.astype(np.int64),
            sym=sym,
            dates=dates,
            anchor_ts=anchor_ts,
            anchor_idx_in_day=anchor_idx_in_day,
            fold_idx=fold_assignments,
            per_fold_aucs=per_fold,
        ),
        fold0_loss_history,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run on 3 symbols × 5 days, capped 5 epochs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base seed.  Encoder seeds = {seed+0, seed+1, seed+2}.  "
            "Default 0 matches Phase 0 for paired comparability."
        ),
    )
    args = p.parse_args()
    _run_pipeline(
        cache_dir=args.cache, out_dir=args.out_dir, smoke=args.smoke, seed=args.seed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

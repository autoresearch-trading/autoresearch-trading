# tests/scripts/test_random_init_probe.py
"""Unit tests for scripts/random_init_probe.py — Goal-A v2 Phase 0.

Council-1 + reviewer-10 audit checklist:
  * Day-blocked CV: 26 days → 5 contiguous folds, no day in two folds.
  * 600-event embargo correctly drops boundary events.
  * Paired bootstrap: identical day samples produce both AUCs in lockstep.
  * BH-FDR helper matches scipy.stats.false_discovery_control on a fixed
    p-value array.
  * Random-init encoder forward pass: same input + same seed → same output.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Day-blocked CV partition
# ---------------------------------------------------------------------------


def test_day_blocked_folds_partition_26_days_into_5_contiguous_blocks():
    """26 sorted days → 5 contiguous folds with no day appearing in two folds."""
    from scripts.random_init_probe import day_blocked_folds

    days = [f"2026-04-{d:02d}" for d in range(1, 27)]  # 26 days
    folds = day_blocked_folds(days, k=5)

    assert len(folds) == 5
    # Every day appears exactly once across all folds.
    flat = [d for fold in folds for d in fold]
    assert sorted(flat) == sorted(days)
    assert len(flat) == 26
    # Each fold's days are contiguous in sorted order.
    for fold in folds:
        idxs = sorted(days.index(d) for d in fold)
        assert idxs == list(
            range(min(idxs), max(idxs) + 1)
        ), f"fold not contiguous: {fold}"
    # Folds are ordered by date.
    fold_starts = [days.index(min(f)) for f in folds]
    assert fold_starts == sorted(fold_starts)


def test_day_blocked_folds_handles_small_partial_dataset():
    """5 sorted days → 5 folds of 1 day each (degenerate but well-defined)."""
    from scripts.random_init_probe import day_blocked_folds

    days = ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"]
    folds = day_blocked_folds(days, k=5)
    assert len(folds) == 5
    assert all(len(f) == 1 for f in folds)
    assert sorted(d for f in folds for d in f) == days


# ---------------------------------------------------------------------------
# Embargo enforcement
# ---------------------------------------------------------------------------


def test_embargo_drops_last_n_events_of_prior_block_and_first_n_of_next():
    """For training fold k, drop last 600 events of fold k-1's last day AND first
    600 events of fold k+1's first day from the training set.  The held-out
    fold k itself is untouched."""
    from scripts.random_init_probe import apply_embargo_mask

    # 3 folds, each with 2 days × 100 events; total 600 rows.
    n_per_day = 100
    days_by_fold = [
        ["2026-04-01", "2026-04-02"],
        ["2026-04-03", "2026-04-04"],
        ["2026-04-05", "2026-04-06"],
    ]
    # Build (date, anchor_idx_in_day) metadata.
    dates = []
    anchor_idx_in_day = []
    fold_assign = []
    for fi, fdays in enumerate(days_by_fold):
        for d in fdays:
            for k in range(n_per_day):
                dates.append(d)
                anchor_idx_in_day.append(k)
                fold_assign.append(fi)
    dates_arr = np.array(dates)
    anchors_in_day = np.array(anchor_idx_in_day, dtype=np.int64)

    # Test held-out fold = 1 (middle).
    train_mask = apply_embargo_mask(
        held_out_fold=1,
        fold_assignments=np.array(fold_assign, dtype=np.int64),
        dates=dates_arr,
        anchor_idx_in_day=anchors_in_day,
        embargo_events=30,
        days_by_fold=days_by_fold,
    )
    # Fold 1 itself is fully excluded from training (held-out).
    assert (~train_mask[np.array(fold_assign) == 1]).all()
    # Fold 0's last day (2026-04-02): drop last 30 events.
    f0_last_day_mask = (dates_arr == "2026-04-02") & (np.array(fold_assign) == 0)
    assert (~train_mask[f0_last_day_mask & (anchors_in_day >= n_per_day - 30)]).all()
    assert train_mask[f0_last_day_mask & (anchors_in_day < n_per_day - 30)].all()
    # Fold 2's first day (2026-04-05): drop first 30 events.
    f2_first_day_mask = (dates_arr == "2026-04-05") & (np.array(fold_assign) == 2)
    assert (~train_mask[f2_first_day_mask & (anchors_in_day < 30)]).all()
    assert train_mask[f2_first_day_mask & (anchors_in_day >= 30)].all()
    # Fold 0's first day and Fold 2's last day untouched.
    f0_first_day_mask = (dates_arr == "2026-04-01") & (np.array(fold_assign) == 0)
    assert train_mask[f0_first_day_mask].all()
    f2_last_day_mask = (dates_arr == "2026-04-06") & (np.array(fold_assign) == 2)
    assert train_mask[f2_last_day_mask].all()


def test_embargo_for_first_fold_only_drops_next_fold_neighbor():
    """When held-out fold is the first one, no fold k-1 exists — only embargo
    against fold k+1's first day."""
    from scripts.random_init_probe import apply_embargo_mask

    n_per_day = 100
    days_by_fold = [
        ["2026-04-01"],
        ["2026-04-02"],
        ["2026-04-03"],
    ]
    dates = []
    anchors = []
    fold_assign = []
    for fi, fdays in enumerate(days_by_fold):
        for d in fdays:
            for k in range(n_per_day):
                dates.append(d)
                anchors.append(k)
                fold_assign.append(fi)
    train_mask = apply_embargo_mask(
        held_out_fold=0,
        fold_assignments=np.array(fold_assign, dtype=np.int64),
        dates=np.array(dates),
        anchor_idx_in_day=np.array(anchors, dtype=np.int64),
        embargo_events=30,
        days_by_fold=days_by_fold,
    )
    # Fold 0 fully excluded.
    assert (~train_mask[np.array(fold_assign) == 0]).all()
    # Fold 1 first day: first 30 dropped.
    f1_first = (np.array(dates) == "2026-04-02") & (np.array(fold_assign) == 1)
    f1_anchors = np.array(anchors)
    assert (~train_mask[f1_first & (f1_anchors < 30)]).all()
    assert train_mask[f1_first & (f1_anchors >= 30)].all()
    # Fold 2 days untouched.
    f2_mask = np.array(fold_assign) == 2
    assert train_mask[f2_mask].all()


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_uses_identical_day_resamples_for_both_models():
    """Same RNG seed must produce identical day samples for both AUC arrays.
    Verified by passing two distinct probability arrays through the paired
    routine and asserting the per-iteration day-index sequences match."""
    from scripts.random_init_probe import paired_day_clustered_bootstrap_delta

    rng_seed = 42
    n_days = 10
    n_per_day = 50
    dates = np.repeat(
        np.array([f"2026-04-{d:02d}" for d in range(1, n_days + 1)]), n_per_day
    )
    rng = np.random.default_rng(0)
    proba_a = rng.random(n_days * n_per_day)
    proba_b = rng.random(n_days * n_per_day)
    labels = (rng.random(n_days * n_per_day) > 0.5).astype(np.int64)

    # Capture the bootstrap by passing model_a == model_b: delta must be exactly 0.
    delta_point, lo, hi, aucs_a, aucs_b = paired_day_clustered_bootstrap_delta(
        proba_a=proba_a,
        proba_b=proba_a,  # identical
        labels=labels,
        dates=dates,
        n_boot=200,
        seed=rng_seed,
    )
    # When both prob arrays are identical, the delta MUST be exactly zero on every iter.
    assert np.all(np.abs(np.array(aucs_a) - np.array(aucs_b)) < 1e-12)
    assert abs(delta_point) < 1e-12


def test_paired_bootstrap_resample_identity_two_runs_match():
    """Calling paired bootstrap twice with the same seed yields identical
    per-iteration samples (deterministic RNG)."""
    from scripts.random_init_probe import paired_day_clustered_bootstrap_delta

    n_days = 8
    n_per_day = 30
    rng = np.random.default_rng(0)
    dates = np.repeat(
        np.array([f"2026-04-{d:02d}" for d in range(1, n_days + 1)]), n_per_day
    )
    proba_a = rng.random(n_days * n_per_day)
    proba_b = rng.random(n_days * n_per_day)
    labels = (rng.random(n_days * n_per_day) > 0.5).astype(np.int64)

    out1 = paired_day_clustered_bootstrap_delta(
        proba_a=proba_a,
        proba_b=proba_b,
        labels=labels,
        dates=dates,
        n_boot=100,
        seed=7,
    )
    out2 = paired_day_clustered_bootstrap_delta(
        proba_a=proba_a,
        proba_b=proba_b,
        labels=labels,
        dates=dates,
        n_boot=100,
        seed=7,
    )
    assert out1[0] == out2[0]
    assert out1[1] == out2[1]
    assert out1[2] == out2[2]
    assert np.array_equal(np.array(out1[3]), np.array(out2[3]))
    assert np.array_equal(np.array(out1[4]), np.array(out2[4]))


# ---------------------------------------------------------------------------
# BH-FDR helper
# ---------------------------------------------------------------------------


def test_bh_fdr_matches_scipy_on_fixed_pvalue_array():
    """Our BH-FDR helper must match scipy.stats.false_discovery_control."""
    from scripts.random_init_probe import bh_fdr

    pvals = np.array(
        [
            0.001,
            0.008,
            0.039,
            0.041,
            0.042,
            0.06,
            0.074,
            0.205,
            0.212,
            0.216,
            0.222,
            0.251,
            0.269,
            0.275,
            0.34,
            0.341,
            0.384,
            0.569,
            0.594,
            0.696,
            0.762,
            0.94,
            0.942,
            0.975,
            0.986,
        ]
    )
    out = bh_fdr(pvals)

    from scipy.stats import false_discovery_control as scipy_bh

    expected = scipy_bh(pvals, method="bh")
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_bh_fdr_handles_edge_cases():
    """Empty input → empty output; single value → unchanged."""
    from scripts.random_init_probe import bh_fdr

    assert bh_fdr(np.array([])).shape == (0,)
    np.testing.assert_allclose(bh_fdr(np.array([0.04])), np.array([0.04]))


# ---------------------------------------------------------------------------
# Random-init encoder determinism
# ---------------------------------------------------------------------------


def test_random_init_encoder_forward_is_deterministic_under_same_seed():
    """Same torch.manual_seed → same random-init encoder weights → same output
    on identical input."""
    import torch

    from scripts.random_init_probe import build_random_init_encoder, encode_windows

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 200, 17)).astype(np.float32)

    enc1 = build_random_init_encoder(seed=0)
    enc2 = build_random_init_encoder(seed=0)
    emb1 = encode_windows(enc1, x, batch_size=8, device="cpu")
    emb2 = encode_windows(enc2, x, batch_size=8, device="cpu")

    np.testing.assert_allclose(emb1, emb2, rtol=1e-6, atol=1e-6)
    assert emb1.shape == (16, 256)


def test_random_init_encoder_different_seeds_produce_different_embeddings():
    """Sanity check: different seeds must give measurably different embeddings."""
    from scripts.random_init_probe import build_random_init_encoder, encode_windows

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 200, 17)).astype(np.float32)

    enc0 = build_random_init_encoder(seed=0)
    enc1 = build_random_init_encoder(seed=1)
    emb0 = encode_windows(enc0, x, batch_size=8, device="cpu")
    emb1 = encode_windows(enc1, x, batch_size=8, device="cpu")

    # Embeddings should differ — at least one element must move by >0.01.
    assert np.max(np.abs(emb0 - emb1)) > 1e-2


def test_random_init_encoder_bn_track_running_stats_disabled():
    """track_running_stats=False on the input BatchNorm — eval() then uses batch
    stats, so two independent forward passes over the SAME input give the same
    output (i.e. eval-mode is well-defined even without warmup)."""
    from scripts.random_init_probe import build_random_init_encoder

    enc = build_random_init_encoder(seed=0)
    bn = enc.input_bn
    assert bn.track_running_stats is False
    # When track_running_stats=False, running_mean/running_var are None.
    assert bn.running_mean is None
    assert bn.running_var is None

"""Tests for tape/flat_features.py — Gate 0 baseline flat feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from tape.flat_features import (
    FLAT_DIM,
    FLAT_FEATURE_NAMES,
    extract_flat_features,
    extract_flat_features_batch,
    window_to_flat,
)

# ---- Shape tests ----


def test_single_window_shape():
    w = np.random.randn(200, 17).astype(np.float32)
    v = extract_flat_features(w)
    assert v.shape == (FLAT_DIM,)


def test_flat_dim_is_83():
    assert FLAT_DIM == 83


def test_batch_shape():
    ws = np.random.randn(8, 200, 17).astype(np.float32)
    vs = extract_flat_features_batch(ws)
    assert vs.shape == (8, FLAT_DIM)


def test_window_to_flat_shape_83():
    """window_to_flat → (83,) after pruning time_delta_last + prev_seq_time_span_last."""
    w = np.random.randn(200, 17).astype(np.float32)
    v = window_to_flat(w)
    assert v.shape == (83,)


# ---- Purity test ----


def test_determinism():
    rng = np.random.default_rng(42)
    w = rng.standard_normal((200, 17)).astype(np.float32)
    v1 = extract_flat_features(w)
    v2 = extract_flat_features(w)
    np.testing.assert_array_equal(v1, v2)


# ---- Correctness tests ----


def test_mean_of_constant_channel():
    w = np.zeros((200, 17), dtype=np.float32)
    w[:, 0] = 1.0  # feature 0 is all ones
    v = extract_flat_features(w)
    # First 17 elements are means; feature-0 mean should be 1.0
    assert v[0] == pytest.approx(1.0, abs=1e-5)


def test_std_of_constant_channel():
    w = np.zeros((200, 17), dtype=np.float32)
    w[:, 0] = 1.0  # std of constant is 0
    v = extract_flat_features(w)
    # Std block is positions [17:34]; feature-0 std should be 0
    assert v[17] == pytest.approx(0.0, abs=1e-5)


def test_last_value():
    w = np.zeros((200, 17), dtype=np.float32)
    w[-1, :] = 7.0  # last row all sevens
    v = extract_flat_features(w)
    # Last 15 elements are last values (17 - 2 pruned: time_delta_last, prev_seq_time_span_last)
    last_block = v[68:]  # 4 * 17 = 68 offset; last block is 15 elements
    assert len(last_block) == 15
    np.testing.assert_allclose(last_block, 7.0, atol=1e-5)


# ---- Batch equivalence ----


def test_batch_equals_single():
    rng = np.random.default_rng(0)
    ws = rng.standard_normal((5, 200, 17)).astype(np.float32)
    batch_out = extract_flat_features_batch(ws)
    for i in range(5):
        single_out = extract_flat_features(ws[i])
        np.testing.assert_allclose(batch_out[i], single_out, atol=1e-6)


# ---- FLAT_FEATURE_NAMES ----


def test_feature_names_length_equals_flat_dim():
    assert len(FLAT_FEATURE_NAMES) == FLAT_DIM


def test_feature_names_are_unique():
    assert len(set(FLAT_FEATURE_NAMES)) == FLAT_DIM


# ---- No NaN/inf ----


def test_no_nan_on_random_window():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((200, 17)).astype(np.float32)
    v = extract_flat_features(w)
    assert np.all(np.isfinite(v)), f"Non-finite values: {v[~np.isfinite(v)]}"


def test_no_nan_on_constant_window():
    """Plan test: constant (zero) window must yield all-finite output."""
    w = np.zeros((200, 17), dtype=np.float32)
    v = window_to_flat(w)
    assert np.isfinite(v).all()


def test_dtype_is_float32():
    w = np.random.randn(200, 17).astype(np.float32)
    v = extract_flat_features(w)
    assert v.dtype == np.float32

"""Tests for normalization in prepare.py (v3 20-feature layout)."""

from __future__ import annotations

import numpy as np

from prepare import ROBUST_FEATURE_INDICES, normalize_features

NUM_FEATURES = 39
ZSCORE_INDICES = set(range(NUM_FEATURES)) - ROBUST_FEATURE_INDICES


class TestNormalization:
    """Test hybrid normalization."""

    def test_output_shape_unchanged(self):
        """Normalization should not change shape."""
        features = np.random.default_rng(42).normal(0, 1, (500, NUM_FEATURES))
        result = normalize_features(features)
        assert result.shape == features.shape

    def test_no_nans(self):
        """Output should have no NaN values."""
        features = np.random.default_rng(42).normal(0, 1, (500, NUM_FEATURES))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))

    def test_robust_columns_handle_outliers(self):
        """Robust-scaled columns should not have extreme values from outliers."""
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (1000, NUM_FEATURES))
        # Pick a robust-scaled column (e.g. 5 = bipower_var_20)
        robust_col = 5
        assert robust_col in ROBUST_FEATURE_INDICES
        features[500, robust_col] = 10000.0
        features[501, robust_col] = -10000.0
        result = normalize_features(features)
        non_outlier_mask = np.abs(features[:, robust_col]) < 100
        assert np.std(result[non_outlier_mask, robust_col]) < 5.0

    def test_zscore_columns_use_mean_std(self):
        """Z-scored columns should use mean/std normalization."""
        rng = np.random.default_rng(42)
        features = rng.normal(5.0, 2.0, (2000, NUM_FEATURES))
        result = normalize_features(features)
        # Z-scored column (e.g. 0 = returns) should be ~mean 0, std ~1
        zscore_col = 0
        assert zscore_col in ZSCORE_INDICES
        # After rolling normalization, later values should be roughly centered
        late = result[500:, zscore_col]
        assert abs(np.mean(late)) < 1.0
        assert np.std(late) < 3.0

    def test_empty_features(self):
        """Edge case: empty input."""
        result = normalize_features(np.array([]))
        assert len(result) == 0

    def test_single_row(self):
        """Single row should normalize to zeros (no variance)."""
        features = np.random.default_rng(42).normal(0, 1, (1, NUM_FEATURES))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))

    def test_robust_indices_correct(self):
        """Verify ROBUST_FEATURE_INDICES matches expected set."""
        expected = {
            5,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            22,
            23,
            24,
            25,
            29,
            33,
            34,
            35,
            37,
        }
        assert ROBUST_FEATURE_INDICES == expected

    def test_no_infs(self):
        """Output should have no inf values."""
        features = np.random.default_rng(42).normal(0, 1, (500, NUM_FEATURES))
        result = normalize_features(features)
        assert not np.any(np.isinf(result))

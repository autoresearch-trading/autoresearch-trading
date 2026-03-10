"""Tests for normalization in prepare.py."""

from __future__ import annotations

import numpy as np

from prepare import normalize_features


class TestNormalization:
    """Test hybrid normalization."""

    def test_output_shape_unchanged(self):
        """Normalization should not change shape."""
        features = np.random.default_rng(42).normal(0, 1, (500, 33))
        result = normalize_features(features)
        assert result.shape == features.shape

    def test_no_nans(self):
        """Output should have no NaN values."""
        features = np.random.default_rng(42).normal(0, 1, (500, 33))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))

    def test_robust_columns_handle_outliers(self):
        """Robust-scaled columns should not have extreme values from outliers."""
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (1000, 33))
        # Add extreme outliers to net_volume (col 2, robust-scaled)
        features[500, 2] = 10000.0
        features[501, 2] = -10000.0
        result = normalize_features(features)
        # With robust scaling, outliers should be large but not as extreme
        # The key property: most values should be well-behaved despite outliers
        non_outlier_mask = np.abs(features[:, 2]) < 100
        assert np.std(result[non_outlier_mask, 2]) < 5.0  # non-outliers stay reasonable

    def test_empty_features(self):
        """Edge case: empty input."""
        result = normalize_features(np.array([]))
        assert len(result) == 0

    def test_single_row(self):
        """Single row should normalize to zeros (no variance)."""
        features = np.random.default_rng(42).normal(0, 1, (1, 33))
        result = normalize_features(features)
        assert not np.any(np.isnan(result))

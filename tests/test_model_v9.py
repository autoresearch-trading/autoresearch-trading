"""Tests for v9 HybridClassifier."""

import torch

from train import HybridClassifier


def test_hybrid_forward_shape():
    model = HybridClassifier(
        obs_shape=(75, 5), n_classes=3, hidden_dim=128, num_layers=2
    )
    x = torch.randn(4, 75, 5)
    out = model(x)
    assert out.shape == (4, 3)


def test_hybrid_param_count():
    model = HybridClassifier(
        obs_shape=(75, 5), n_classes=3, hidden_dim=128, num_layers=2
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 60_000 < n_params < 80_000  # ~68K (flat=385+8=393 combined, TCN=808)


def test_hybrid_different_window():
    model = HybridClassifier(
        obs_shape=(50, 5), n_classes=3, hidden_dim=128, num_layers=2
    )
    x = torch.randn(2, 50, 5)
    out = model(x)
    assert out.shape == (2, 3)

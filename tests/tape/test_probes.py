# tests/tape/test_probes.py
import numpy as np

from tape.probes import (
    direction_probe_h100,
    hour_of_day_probe,
    symbol_identity_probe,
)


def test_direction_probe_returns_per_symbol_balanced_acc():
    rng = np.random.default_rng(0)
    # 2 symbols, 5000 windows each, 256-dim embeddings, binary labels.
    # 5000 >= min_train(2000) + embargo(600) + n_folds(3) * min_test(500) = 4100.
    feats = {
        "BTC": rng.standard_normal((5000, 256)).astype(np.float32),
        "ETH": rng.standard_normal((5000, 256)).astype(np.float32),
    }
    labels = {
        "BTC": rng.integers(0, 2, size=5000).astype(np.int64),
        "ETH": rng.integers(0, 2, size=5000).astype(np.int64),
    }
    masks = {
        "BTC": np.ones(5000, dtype=bool),
        "ETH": np.ones(5000, dtype=bool),
    }
    out = direction_probe_h100(feats, labels, masks)
    assert set(out.keys()) == {"BTC", "ETH"}
    # Balanced accuracy on random labels should be ~0.50
    assert 0.45 <= out["BTC"] <= 0.55
    assert 0.45 <= out["ETH"] <= 0.55


def test_symbol_identity_probe_low_on_random_features():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((1000, 256)).astype(np.float32)
    sym_ids = rng.integers(0, 25, size=1000).astype(np.int64)
    acc = symbol_identity_probe(feats, sym_ids, n_symbols=25)
    # Random features -> ~1/25 = 4%
    assert acc < 0.10


def test_hour_of_day_probe_low_on_random_features():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((2000, 256)).astype(np.float32)
    hours = rng.integers(0, 24, size=2000).astype(np.int64)
    acc = hour_of_day_probe(feats, hours)
    # Random features -> ~1/24 ≈ 4.2%
    assert acc < 0.10

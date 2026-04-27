# tests/scripts/test_cascade_adapter_probe.py
"""Unit tests for scripts/cascade_adapter_probe.py — Goal-A v2 Phase 1 (5b).

Council-6 audit checklist:
  * Adapter forward pass is deterministic under same torch seed.
  * Training loss decreases monotonically over the first 3 batches on synthetic
    data (no stuck-at-init bug).
  * BCEWithLogitsLoss with pos_weight reweights the positive class as expected
    (numeric agreement with the analytic formulation).
  * Early-stopping on a stalled val-AUC curve halts at epoch (best + patience).
  * decision_tier_phase1 produces the three pre-registered tiers correctly.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Adapter head determinism
# ---------------------------------------------------------------------------


def test_adapter_head_forward_determinism_same_seed():
    """Same torch.manual_seed → same adapter weights → same logits on same input."""
    from scripts.cascade_adapter_probe import CascadeAdapterHead, init_adapter_weights

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((32, 256)).astype(np.float32))

    torch.manual_seed(123)
    head_a = CascadeAdapterHead(embed_dim=256, hidden=64, dropout_p=0.2)
    init_adapter_weights(head_a, seed=123)
    head_a.eval()

    torch.manual_seed(123)
    head_b = CascadeAdapterHead(embed_dim=256, hidden=64, dropout_p=0.2)
    init_adapter_weights(head_b, seed=123)
    head_b.eval()

    with torch.no_grad():
        logits_a = head_a(x)
        logits_b = head_b(x)

    assert logits_a.shape == (32, 1)
    np.testing.assert_allclose(logits_a.numpy(), logits_b.numpy(), rtol=1e-6, atol=1e-6)


def test_adapter_different_seeds_produce_different_logits():
    """Sanity: different init seeds must yield measurably different logits."""
    from scripts.cascade_adapter_probe import CascadeAdapterHead, init_adapter_weights

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((16, 256)).astype(np.float32))

    head_a = CascadeAdapterHead()
    init_adapter_weights(head_a, seed=0)
    head_a.eval()
    head_b = CascadeAdapterHead()
    init_adapter_weights(head_b, seed=1)
    head_b.eval()

    with torch.no_grad():
        diff = (head_a(x) - head_b(x)).abs().max().item()
    assert diff > 1e-3, f"different-seed logits too similar: max diff {diff}"


# ---------------------------------------------------------------------------
# Training-loss-decrease sanity check
# ---------------------------------------------------------------------------


def test_adapter_training_loss_decreases_3_epochs():
    """On synthetic separable data, BCE loss must drop over 3 training steps."""
    from scripts.cascade_adapter_probe import CascadeAdapterHead, init_adapter_weights

    torch.manual_seed(0)
    n = 256
    embed_dim = 256
    rng = np.random.default_rng(0)
    # Construct labels then build embeddings whose first dim is correlated with y.
    y = (rng.random(n) > 0.5).astype(np.float32)
    x = rng.standard_normal((n, embed_dim)).astype(np.float32)
    x[:, 0] += 4.0 * (y - 0.5)  # strong signal in first dim

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).unsqueeze(-1)

    head = CascadeAdapterHead()
    init_adapter_weights(head, seed=0)
    head.train()
    pos_weight = torch.tensor([15.7])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)

    losses: list[float] = []
    for _ in range(3):
        optim.zero_grad()
        logits = head(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert (
        losses[0] > losses[1] > losses[2]
    ), f"loss did not decrease monotonically: {losses}"


# ---------------------------------------------------------------------------
# pos_weight reweighting numerics
# ---------------------------------------------------------------------------


def test_adapter_pos_weight_reweighting():
    """Hand-compute BCE-with-pos-weight on a tiny batch and compare to torch.

    With pos_weight w, BCEWithLogitsLoss returns mean over batch of:
        l_n = -[ w * y_n * log(sigmoid(x_n)) + (1 - y_n) * log(1 - sigmoid(x_n)) ]
    """

    logits = torch.tensor([[0.5], [-0.3], [1.2], [-2.0]])
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    pos_weight = torch.tensor([15.7])

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    actual = bce(logits, targets).item()

    # Hand calculation
    sig = torch.sigmoid(logits)
    eps = 1e-12
    log_sig = torch.log(sig + eps)
    log_one_minus = torch.log(1.0 - sig + eps)
    losses = -(pos_weight * targets * log_sig + (1.0 - targets) * log_one_minus)
    expected = losses.mean().item()

    assert (
        abs(actual - expected) < 1e-4
    ), f"pos_weight numeric mismatch: actual={actual} expected={expected}"


# ---------------------------------------------------------------------------
# Early-stopping
# ---------------------------------------------------------------------------


def test_early_stopping_patience_5_stops_after_best_plus_patience():
    """Stall val-AUC after epoch 1: tracker should fire stop at epoch 6 (best=1, patience=5)."""
    from scripts.cascade_adapter_probe import EarlyStopTracker

    tracker = EarlyStopTracker(patience=5)
    # Epoch 0: AUC 0.55 (initial best)
    assert not tracker.update(epoch=0, val_auc=0.55)
    # Epoch 1: AUC 0.70 — new best
    assert not tracker.update(epoch=1, val_auc=0.70)
    # Stalled epochs 2..5: still no stop, but counter increments.
    for ep in range(2, 6):
        stop = tracker.update(epoch=ep, val_auc=0.65)
        assert not stop, f"stopped prematurely at epoch {ep}"
    # Epoch 6: 5 stalled epochs → triggers stop
    stop = tracker.update(epoch=6, val_auc=0.65)
    assert stop, "should stop at epoch 6 (best=1, patience=5)"
    assert tracker.best_epoch == 1
    assert abs(tracker.best_val_auc - 0.70) < 1e-9


def test_early_stopping_resets_on_improvement():
    """If AUC improves, the patience counter resets."""
    from scripts.cascade_adapter_probe import EarlyStopTracker

    tracker = EarlyStopTracker(patience=3)
    tracker.update(epoch=0, val_auc=0.5)
    tracker.update(epoch=1, val_auc=0.55)  # best
    # Two stalled epochs.
    assert not tracker.update(epoch=2, val_auc=0.50)
    assert not tracker.update(epoch=3, val_auc=0.50)
    # Improvement — reset counter.
    assert not tracker.update(epoch=4, val_auc=0.60)  # new best
    # After 3 more stalls, finally stop.
    for ep in range(5, 7):
        assert not tracker.update(epoch=ep, val_auc=0.55)
    assert tracker.update(epoch=7, val_auc=0.55)
    assert tracker.best_epoch == 4


# ---------------------------------------------------------------------------
# Decision tier
# ---------------------------------------------------------------------------


def test_decision_tier_phase1_thresholds():
    """Three pre-registered cases per council-6 synthesis."""
    from scripts.cascade_adapter_probe import decision_tier_phase1

    # GREENLIGHT: adapter beats flat by >= 0.02 AND delta_lo > 0.
    assert (
        decision_tier_phase1(
            auc_flat=0.83, auc_adapter_median=0.86, delta_lo=0.01, delta_hi=0.05
        )
        == "GREENLIGHT_FINETUNE_OR_PRETRAIN"
    )
    # MATCHED: delta CI overlaps zero.
    assert (
        decision_tier_phase1(
            auc_flat=0.83, auc_adapter_median=0.84, delta_lo=-0.01, delta_hi=0.03
        )
        == "MATCHED_FLAT"
    )
    # KILL: adapter loses by > 0.02.
    assert (
        decision_tier_phase1(
            auc_flat=0.83, auc_adapter_median=0.80, delta_lo=-0.05, delta_hi=-0.01
        )
        == "KILL_ARCH_BOTTLENECK_CONFIRMED"
    )
    # Edge: small positive delta but CI includes zero → MATCHED, not GREENLIGHT.
    assert (
        decision_tier_phase1(
            auc_flat=0.83, auc_adapter_median=0.835, delta_lo=-0.01, delta_hi=0.02
        )
        == "MATCHED_FLAT"
    )
    # Edge: delta exactly -0.02 → MATCHED (only > 0.02 shortfall is KILL).
    assert (
        decision_tier_phase1(
            auc_flat=0.83, auc_adapter_median=0.81, delta_lo=-0.05, delta_hi=0.01
        )
        == "MATCHED_FLAT"
    )

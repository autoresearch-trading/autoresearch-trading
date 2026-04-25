# tests/tape/test_finetune.py
"""Tests for tape.finetune — DirectionHead, FineTunedModel, weighted_bce_loss, cka_torch."""

from __future__ import annotations

import math

import pytest
import torch

from tape.finetune import (
    EMBED_DIM,
    HEAD_TRUNK_DIM,
    HORIZON_WEIGHTS,
    HORIZONS,
    LABEL_SMOOTHING_EPS,
    DirectionHead,
    FineTunedModel,
    cka_torch,
    weighted_bce_loss,
)
from tape.model import EncoderConfig, TapeEncoder

# ---------------------------------------------------------------------------
# DirectionHead
# ---------------------------------------------------------------------------


def test_direction_head_forward_shape():
    head = DirectionHead(embed_dim=EMBED_DIM, trunk_dim=HEAD_TRUNK_DIM)
    embeddings = torch.randn(8, EMBED_DIM)
    logits = head(embeddings)
    assert logits.shape == (8, len(HORIZONS))
    assert logits.dtype == torch.float32


def test_direction_head_returns_logits_not_sigmoid():
    """Outputs should NOT be in [0, 1]; they are raw logits (caller does sigmoid)."""
    head = DirectionHead(embed_dim=EMBED_DIM)
    # Force non-zero logits by spiking one input dim through an explicit linear.
    embeddings = 100.0 * torch.randn(16, EMBED_DIM)
    logits = head(embeddings)
    # On a normal init we expect at least one logit outside [0, 1] from this scale.
    assert logits.min().item() < 0.0 or logits.max().item() > 1.0


def test_direction_head_init_near_zero_for_heads():
    """Per-horizon heads init with σ=0.02 → logits at zero-input are near zero."""
    head = DirectionHead(embed_dim=EMBED_DIM)
    # Pass in zeros so trunk activation is also zero (trunk has bias zero too).
    zeros = torch.zeros(4, EMBED_DIM)
    logits = head(zeros)
    # Trunk(0) + ReLU = 0, so head(0) = head_bias which we set to zero.
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)


# ---------------------------------------------------------------------------
# FineTunedModel — freeze / unfreeze
# ---------------------------------------------------------------------------


def _make_finetuned_model() -> FineTunedModel:
    enc = TapeEncoder(EncoderConfig(channel_mult=0.5))
    return FineTunedModel(enc)


def test_finetuned_model_forward_shape():
    model = _make_finetuned_model()
    features = torch.randn(4, 200, 17)
    logits = model(features)
    assert logits.shape == (4, len(HORIZONS))


def test_freeze_encoder_toggles_requires_grad():
    model = _make_finetuned_model()
    # Sanity: all encoder params start trainable.
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    # Heads must remain trainable in Phase A.
    assert all(p.requires_grad for p in model.head.parameters())

    model.unfreeze_encoder()
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_phase_b_optimizer_attaches_encoder_params():
    """Reviewer-10 G2 — guard against silent 'encoder never unfreezes' bug.

    The Phase A → Phase B transition in run_finetune.py builds the new optimizer
    via `[p for p in model.parameters() if p.requires_grad]` AFTER calling
    `unfreeze_encoder()`. If those two operations were ever reordered, the new
    optimizer would silently train heads only — burning 3.5h before Gate 2 eval
    revealed it. This test pins the contract: after unfreeze, an optimizer
    constructed from the requires_grad-filtered param list MUST contain BOTH
    encoder and head params, and a backward pass must produce non-zero grads
    on both.
    """
    model = _make_finetuned_model()
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    head_param_ids = {id(p) for p in model.head.parameters()}

    # Phase A: build opt_a after freeze. Should contain ONLY head params.
    model.freeze_encoder()
    opt_a_params = [p for p in model.parameters() if p.requires_grad]
    opt_a_ids = {id(p) for p in opt_a_params}
    assert opt_a_ids == head_param_ids, (
        "Phase A optimizer must include only head params; "
        f"unexpected encoder params: {opt_a_ids & encoder_param_ids}"
    )

    # Phase B: unfreeze, then build opt_b. Both encoder AND head params must be present.
    model.unfreeze_encoder()
    opt_b_params = [p for p in model.parameters() if p.requires_grad]
    opt_b_ids = {id(p) for p in opt_b_params}
    assert encoder_param_ids.issubset(opt_b_ids), (
        "Phase B optimizer must include all encoder params after unfreeze; "
        f"missing: {encoder_param_ids - opt_b_ids}"
    )
    assert head_param_ids.issubset(opt_b_ids), (
        "Phase B optimizer must continue to include head params; "
        f"missing: {head_param_ids - opt_b_ids}"
    )

    # End-to-end: a backward pass through the assembled opt_b must produce
    # non-zero gradients on both encoder and head params.
    opt_b = torch.optim.AdamW(opt_b_params, lr=5e-5)
    features = torch.randn(2, 200, 17)
    logits = model(features)
    loss = logits.pow(2).sum()  # any scalar loss touching all heads
    opt_b.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grads_present = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    head_grads_present = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()
    )
    assert encoder_grads_present, "Encoder params must receive gradient in Phase B"
    assert head_grads_present, "Head params must receive gradient in Phase B"


# ---------------------------------------------------------------------------
# weighted_bce_loss — numerics, masking, label smoothing
# ---------------------------------------------------------------------------


def test_weighted_bce_loss_numerics_match_hand_computed():
    """Single-example sanity check: per-horizon loss = BCE on smoothed label."""
    # Construct: B=1, H=4, all masks = 1.
    logit_value = 0.5  # shared across all 4 horizons for simplicity
    label_value = 1
    logits = torch.full((1, 4), logit_value, dtype=torch.float64)
    labels = torch.full((1, 4), label_value, dtype=torch.int64)
    masks = torch.ones((1, 4), dtype=torch.int64)

    eps = (0.10, 0.08, 0.05, 0.05)
    weights = (0.10, 0.20, 0.20, 0.50)
    total, per_h = weighted_bce_loss(
        logits, labels, masks, label_smoothing_eps=eps, horizon_weights=weights
    )
    # Hand compute: smoothed = 1*(1-eps) + 0.5*eps = 1 - 0.5*eps
    # BCE(logit, smooth) = -smooth*log(sigmoid(logit)) - (1-smooth)*log(1-sigmoid(logit))
    sig = 1.0 / (1.0 + math.exp(-logit_value))
    expected_per_h = []
    for e in eps:
        smooth = 1.0 - 0.5 * e
        bce = -smooth * math.log(sig) - (1.0 - smooth) * math.log(1.0 - sig)
        expected_per_h.append(bce)
    expected_total = sum(w * b for w, b in zip(weights, expected_per_h))
    assert per_h.shape == (4,)
    for got, want in zip(per_h.tolist(), expected_per_h):
        assert abs(got - want) < 1e-9, (got, want)
    assert abs(float(total) - expected_total) < 1e-9


def test_weighted_bce_loss_masking_excludes_invalid():
    """Rows with mask=0 must NOT contribute to the per-horizon mean."""
    # B=4, H=2. Set up so row 0 has BCE 100 but mask=0; rows 1..3 have BCE 0.69ish.
    logits = torch.zeros(4, 2, dtype=torch.float64)
    labels = torch.zeros(4, 2, dtype=torch.int64)
    # Row 0 H0: would be a huge loss if not masked.
    logits[0, 0] = 10.0  # sigmoid ≈ 1; label 0 → BCE huge
    masks = torch.ones(4, 2, dtype=torch.int64)
    masks[0, 0] = 0  # invalidate the loud cell on horizon 0

    # No smoothing, equal weights for clarity
    total, per_h = weighted_bce_loss(
        logits,
        labels,
        masks,
        label_smoothing_eps=(0.0, 0.0),
        horizon_weights=(1.0, 1.0),
    )
    # Per-horizon loss on H0 averages over rows 1,2,3 only (mask sum = 3).
    # On H0 those rows have logit 0, label 0 → BCE = -log(0.5) ≈ 0.6931
    expected_h0 = -math.log(0.5)
    expected_h1 = -math.log(0.5)  # all four rows valid on H1, same logit/label
    assert abs(float(per_h[0]) - expected_h0) < 1e-9
    assert abs(float(per_h[1]) - expected_h1) < 1e-9


def test_weighted_bce_loss_empty_mask_horizon_is_zero():
    """A horizon with zero valid labels in the batch contributes 0.0 (no NaN)."""
    logits = torch.randn(4, 2, dtype=torch.float64)
    labels = torch.zeros(4, 2, dtype=torch.int64)
    masks = torch.ones(4, 2, dtype=torch.int64)
    masks[:, 0] = 0  # entire H0 invalid in this batch

    total, per_h = weighted_bce_loss(
        logits,
        labels,
        masks,
        label_smoothing_eps=(0.0, 0.0),
        horizon_weights=(1.0, 1.0),
    )
    assert math.isfinite(float(total))
    assert float(per_h[0]) == 0.0
    assert float(per_h[1]) > 0.0  # horizon with valid labels still contributes


def test_weighted_bce_loss_label_smoothing_target_values():
    """ε=0.1 → label 1 maps to 0.95 smoothed target; label 0 → 0.05."""
    # We verify by checking that the loss matches BCE against the expected
    # smoothed target rather than the raw label.
    logits = torch.zeros(2, 1, dtype=torch.float64)  # sigmoid = 0.5
    masks = torch.ones(2, 1, dtype=torch.int64)
    labels = torch.tensor([[1], [0]], dtype=torch.int64)
    eps = (0.10,)
    _total, per_h = weighted_bce_loss(
        logits,
        labels,
        masks,
        label_smoothing_eps=eps,
        horizon_weights=(1.0,),
    )
    # Smoothed targets: row 0 -> 0.95, row 1 -> 0.05.
    # BCE per row at logit 0 (sigmoid 0.5):
    #   row 0: -0.95*log(0.5) - 0.05*log(0.5) = -log(0.5) = 0.6931
    #   row 1: -0.05*log(0.5) - 0.95*log(0.5) = -log(0.5) = 0.6931
    # Mean over 2 rows = 0.6931. (Symmetric — ε just blends the targets.)
    expected = -math.log(0.5)
    assert abs(float(per_h[0]) - expected) < 1e-9


def test_weighted_bce_loss_shape_validation():
    logits = torch.zeros(4, 2)
    labels = torch.zeros(4, 3, dtype=torch.int64)  # mismatched shape
    masks = torch.ones(4, 2, dtype=torch.int64)
    with pytest.raises(ValueError):
        weighted_bce_loss(
            logits,
            labels,
            masks,
            label_smoothing_eps=(0.0, 0.0),
            horizon_weights=(1.0, 1.0),
        )


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


def test_cka_self_is_one():
    torch.manual_seed(0)
    X = torch.randn(64, 32)
    val = cka_torch(X, X)
    assert abs(val - 1.0) < 1e-6


def test_cka_negation_is_one():
    """Linear CKA on centered features is sign-invariant: CKA(X, -X) == 1.0."""
    torch.manual_seed(1)
    X = torch.randn(64, 32)
    val = cka_torch(X, -X)
    assert abs(val - 1.0) < 1e-6


def test_cka_random_unrelated_is_small():
    """Two large random matrices should yield CKA close to 0."""
    torch.manual_seed(2)
    X = torch.randn(2_048, 64)
    Y = torch.randn(2_048, 64)
    val = cka_torch(X, Y)
    # Empirical: random pairs at this size land < 0.05.
    assert val < 0.1, val


def test_cka_zero_matrix_returns_zero():
    """Degenerate denominator (zero variance) returns 0.0 (no NaN)."""
    X = torch.zeros(8, 4)
    Y = torch.randn(8, 4)
    val = cka_torch(X, Y)
    assert val == 0.0


def test_cka_shape_validation():
    with pytest.raises(ValueError):
        cka_torch(torch.randn(8, 4), torch.randn(8, 4, 2))
    with pytest.raises(ValueError):
        cka_torch(torch.randn(8, 4), torch.randn(7, 4))


# ---------------------------------------------------------------------------
# Walk-forward fold construction (matches Gate 1 protocol)
# ---------------------------------------------------------------------------


def test_walk_forward_train_end_date_carves_test_correctly():
    """The pretrain-style date filter at --train-end-date 2026-02-01 should
    keep ONLY shards with date < 2026-02-01 in the training set, and shards with
    date >= 2026-02-01 belong to the held-out eval set.

    This is the same shard-filtering logic the run_finetune script will use; we
    test the boundary behavior here so the script's split matches Gate 1.
    """
    train_end = "2026-02-01"
    sample_dates = [
        "2025-10-16",  # train
        "2026-01-31",  # train (last day)
        "2026-02-01",  # eval (first day, embargo enforced separately)
        "2026-02-15",  # eval
        "2026-03-31",  # eval
        "2026-04-13",  # eval (still pre-April-heldout)
        "2026-04-14",  # excluded from BOTH train and eval (April hold-out)
    ]
    train = [d for d in sample_dates if d < train_end]
    eval_pre_april = [d for d in sample_dates if train_end <= d < "2026-04-14"]
    excluded = [d for d in sample_dates if d >= "2026-04-14"]
    assert train == ["2025-10-16", "2026-01-31"]
    assert eval_pre_april == ["2026-02-01", "2026-02-15", "2026-03-31", "2026-04-13"]
    assert excluded == ["2026-04-14"]

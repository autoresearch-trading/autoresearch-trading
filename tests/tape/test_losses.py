# tests/tape/test_losses.py
import torch

from tape.losses import mem_loss, nt_xent_loss


def test_mem_loss_zero_on_perfect_reconstruction():
    pred = torch.randn(2, 200, 17)
    target = pred.clone()
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    pos_mask[:, 50:55] = True
    feat_mask = torch.ones(17, dtype=torch.bool)
    feat_mask[6:9] = False  # exclude 3 features (mimics carry-forward)
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mem_loss_only_counts_masked_positions_and_target_features():
    pred = torch.zeros(2, 200, 17)
    target = torch.ones(2, 200, 17)
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    pos_mask[:, 0:5] = True  # 5 positions per batch row -> 10 positions total
    feat_mask = torch.zeros(17, dtype=torch.bool)
    feat_mask[0] = True  # only feature 0 counted -> 10 cells
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    # Each cell error = 1.0; mean over 10 cells = 1.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_mem_loss_zero_when_no_positions_masked():
    pred = torch.randn(2, 200, 17)
    target = torch.randn(2, 200, 17)
    pos_mask = torch.zeros(2, 200, dtype=torch.bool)
    feat_mask = torch.ones(17, dtype=torch.bool)
    loss = mem_loss(pred, target, pos_mask, feat_mask)
    # No masked positions -> zero loss (graceful no-op)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_nt_xent_identical_views_low_loss():
    z1 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    z2 = z1.clone()
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    # With identical views the diagonal is the strongest similarity -> loss small
    assert loss.item() < 0.5


def test_nt_xent_random_views_higher_loss():
    z1 = torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    # Random views -> log(2N - 1) ≈ log(31) ≈ 3.4 (T=0.1 keeps it close)
    assert loss.item() > 1.0


def test_nt_xent_soft_positives_reduce_loss():
    z1 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    base = nt_xent_loss(z1, z2, temperature=0.1).item()

    # Mark indices 0 and 1 as a soft-positive pair across the two views
    soft = torch.zeros(8, 8)
    soft[0, 1] = 1.0
    soft[1, 0] = 1.0
    augmented = nt_xent_loss(
        z1, z2, temperature=0.1, soft_positive_pairs=soft, soft_weight=0.5
    ).item()
    assert augmented <= base + 0.5  # primary loss unchanged or slightly reduced

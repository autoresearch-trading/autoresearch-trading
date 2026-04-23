# tests/tape/test_masking.py
import numpy as np
import torch

from tape.constants import (
    FEATURE_NAMES,
    MEM_BLOCK_LEN,
    MEM_EXCLUDED_FEATURES,
    MEM_MASK_FRACTION,
    MEM_RANDOM_MASK_FEATURES,
)
from tape.masking import block_mask, build_mem_target_mask, random_mask


def test_block_mask_fraction_is_approximately_correct():
    rng = np.random.default_rng(0)
    masks = [
        block_mask(
            window_len=200, block_len=MEM_BLOCK_LEN, fraction=MEM_MASK_FRACTION, rng=rng
        )
        for _ in range(50)
    ]
    rates = [m.mean() for m in masks]
    avg = float(np.mean(rates))
    # Allow a ±5pp tolerance — block masking with discrete blocks is granular.
    assert MEM_MASK_FRACTION - 0.05 <= avg <= MEM_MASK_FRACTION + 0.05


def test_block_mask_blocks_are_contiguous_runs_of_block_len():
    rng = np.random.default_rng(1)
    m = block_mask(window_len=200, block_len=5, fraction=0.15, rng=rng)
    # Every contiguous True run should have length divisible by block_len
    # (we may have adjacent blocks merging — len % 5 == 0).
    runs: list[int] = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    for r in runs:
        assert r % 5 == 0, runs


def test_random_mask_independence_per_position():
    rng = np.random.default_rng(0)
    m = random_mask(window_len=200, fraction=0.15, rng=rng)
    # No structural test beyond rate — random masks are i.i.d. per position.
    assert 0.10 <= m.mean() <= 0.20


def test_build_mem_target_mask_excludes_carry_forward_features():
    target = build_mem_target_mask()
    assert target.dtype == torch.bool
    assert target.shape == (17,)
    excluded = {FEATURE_NAMES.index(name) for name in MEM_EXCLUDED_FEATURES}
    for i, name in enumerate(FEATURE_NAMES):
        assert target[i].item() == (name not in MEM_EXCLUDED_FEATURES), (i, name)
    # Check excluded features are False
    for ei in excluded:
        assert not target[ei]
    # Check there are exactly 14 trues (17 - 3 excluded)
    assert int(target.sum()) == 14

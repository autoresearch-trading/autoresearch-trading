---
title: MEM Block Size 20, Not 5
date: 2026-04-10
status: accepted
decided_by: council-6, council-5
sources:
  - docs/council-reviews/2026-04-10-round5-council-6-pretraining-mechanics.md
  - docs/council-reviews/2026-04-10-round5-council-5-impl-risks.md
last_updated: 2026-04-10
---

# Decision: MEM Block Size 20, Not 5

## What Was Decided

Increase masked block size from 5 consecutive events to 20 consecutive events.
Increase masking rate from 15% to 20% (4 blocks of 20 per 200-event window).

## Why

The dilated CNN has receptive field RF=253, covering the entire 200-event input.
With 5-event blocks, layer-1 convolutions (k=5, d=1) can reconstruct masked
positions from positions p-1 and p+5 — pure local interpolation without learning
any tape structure. This is analogous to masking 1 pixel in a 256×256 image and
asking a CNN to fill it.

With 20-event blocks, the gap requires layers 3+ (dilation ≥ 4) to bridge,
forcing the model to use long-range context from across the window.

## Alternatives Considered

1. **Higher masking ratio (40%) with 5-event blocks:** Creates overlapping masked
   regions that prevent interpolation. Rejected: more complex, same effect as
   larger blocks but less interpretable.
2. **Tiered masking (easy + hard):** 15% easy masks for convergence + 15% hard
   masks for difficulty. Rejected: adds complexity for marginal benefit over
   simply using larger blocks.

## Impact

Changes masking from `num_blocks = int(200 * 0.15 / 5) = 6` to
`num_blocks = int(200 * 0.20 / 20) = 2` (with rounding, typically 4 blocks).
Each masked region is longer but there are fewer of them.

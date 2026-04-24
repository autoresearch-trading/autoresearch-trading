---
title: Step 3 Run-0 Collapse Diagnosis — Three Probe Bugs Invalidated the Signal
date: 2026-04-23
status: completed
result: inconclusive
sources:
  - docs/council-reviews/council-5-step3-run0-falsifiability.md
  - docs/council-reviews/council-6-step3-run0-diagnosis.md
  - docs/council-reviews/council-6-step3-model-size.md
last_updated: 2026-04-24
---

# Experiment: Step 3 Run-0 Collapse Diagnosis

## Hypothesis

Run-0 of Step 3 pretraining (commit `34b8f18`, MPS 1.74h wall-clock, epochs
1–8 early-stopped) produced logs reading like objective collapse: symbol-ID
probe at 0.971, direction probe at 0.499, MEM 0.803→0.513→0.736. Was this
a real objective failure, or was the measurement itself broken?

## Setup

- **Checkpoint under review:** `runs/step3-r1/` (commit `34b8f18`).
- **Observed signals:** MEM 0.803→0.513→0.736 (non-monotone); contrastive
  5.01→3.88 (monotone); direction probe 0.499; symbol-ID 0.971; hour-of-day
  probe 0.587.
- **Council-5 diagnostic:** fine-grained code review of
  `tape/pretrain.py`, `tape/augment.py`, `scripts/run_pretrain.py`,
  `tape/dataset.py`, `tape/probes.py`.

## Result

**Three probe bugs and one recipe bug, each independently invalidating the
signal.**

### Bug A — SimCLR jitter was not applied (recipe, load-bearing)
`tape/pretrain.py::pretrain_step` generated both contrastive views via
`apply_augment_pipeline(b.clone(), ...)` on the already-sliced 200-event
window. `tape/augment.py::make_views_from_context` existed as dead code.
Consequence: **v1 and v2 were identical 200-event sequences with two
different draws of feature-level noise.** Symbol ID, date, UTC hour — all
perfectly invariant across the positive pair. The contrastive objective
trained was "make embeddings noise-robust," not "make temporally-shifted
views of the same regime close." Symbol-ID probe 0.971 is the exact
pathology this bug produces.

### Bug B — Hour-of-day probe measured the wrong thing
`scripts/run_pretrain.py::_run_probe_trio` line 257 used
`int((item.get("start", 0) // 3600) % 24)`. `item["start"]` is the
event-index within the shard (0, 50, 100, ...), not a Unix timestamp.
Dividing by 3600 gives mostly 0. **0.587 was measuring within-day
event-index bucket predictability, not UTC hour.** The <10% spec threshold
was never evaluated. Secondary issue: `TapeDataset.__getitem__` never
emitted `ts_first_ms`, so the soft-positive matrix for cross-symbol
contrastive used the 0 fallback too.

### Bug C — Direction probe covered 3 symbols, not 24
`_run_probe_trio` iterated `dataset[i]` linearly up to 50K unshuffled. Shard
order is sorted alphabetically × date. **First 50K windows covered only 2Z,
AAVE, ASTER** (three illiquid alts). The 0.499 "balanced acc mean" was a
3-symbol average, not 25-symbol evidence.

### Bug D — MEM trajectory partially masked by contrastive reward-hacking
MEM 0.803 → 0.513 → 0.736 combined with contrastive 5.01 → 3.88 monotone
means the encoder found a shortcut for contrastive and was willing to
degrade MEM to hold it. The early-stop rule ("<1% MEM improvement over last
20% of epochs") fired on a regression, not a plateau — it couldn't
distinguish "converged" from "the other head won."

## Additional Bugs Surfaced by Council-6 (same round)

Council-6 model-size review identified orthogonal bugs:

- **MEM flow order:** `enc(v1)` was called on the FULL, UNMASKED input; the
  mask was built afterward and only selected loss positions. Encoder saw
  ground truth at masked positions → MEM became trivial identity task.
  Fixed in commit `29f23c0`. See
  [MEM pretraining concept](../concepts/mem-pretraining.md).
- **NT-Xent temperature was 0.10 (ImageNet default).** Accepted decision
  is τ=0.5→0.3. See
  [NT-Xent temperature decision](../decisions/ntxent-temperature.md).
- **Block masking defaulted to 5 events at 15%.** Accepted decision is
  block_len=20 at 20% fraction. See
  [MEM block size decision](../decisions/mem-block-size-20.md).
- **Loss weight annealing not implemented** — static 0.70/0.30 vs spec
  0.90→0.60 / 0.10→0.40.
- **Gradient clipping absent** — accepted decision is max_norm=1.0.

## What We Learned

1. **Run-0 was not a real objective failure, it was three broken probes.**
   Council-5 correctly gated run-1 on bug fixes + a cheap sanity matrix.
2. **All five code-level bugs were latent on CUDA too.** CPU-only smoke
   tests did not catch them; an H100 run would have crashed or measured
   the same artifacts.
3. **The bug-fix commit chain** `117187d` (probe bugs) + `bda524e`
   (diagnostic tooling + `--train-end-date` flag) + `29f23c0` (MEM flow
   order) enabled run-2's Gate 1 pass. See
   [Gate 1 pass experiment](gate1-pass-feb-mar-h500.md).
4. **The methodology lesson** — "encoder looks collapsed" is a hypothesis,
   not a conclusion. Audit the probe before auditing the encoder.

## Verdict

**Inconclusive on run-0 (uninformative), but the council reviews produced a
definitive fix path that unblocked run-2.** Run-0 is not evidence for or
against the objective; it is evidence that the measurement pipeline was
broken. Step 3 run-2 (same hyperparameters, same data, fixed probes) passed
Gate 1.

## Related

- [Gate 1 pass experiment](gate1-pass-feb-mar-h500.md)
- [MEM pretraining concept](../concepts/mem-pretraining.md)
- [NT-Xent temperature decision](../decisions/ntxent-temperature.md)
- [MEM block size decision](../decisions/mem-block-size-20.md)

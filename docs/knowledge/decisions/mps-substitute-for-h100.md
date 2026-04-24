---
title: Apple Silicon MPS Accepted as Pretraining Substrate
date: 2026-04-23
status: accepted
decided_by: lead-0 (empirical validation via step3 run-2)
sources:
  - docs/experiments/step3-run-2-gate1-pass.md
  - CLAUDE.md (Training section)
  - .claude/agents/lead-0.md
last_updated: 2026-04-24
---

# Decision: Apple Silicon MPS Accepted as Pretraining Substrate

## What Was Decided

Pretraining on Apple Silicon MPS at batch 256 is an **accepted substitute for
H100 cloud GPU** for the current 400K-param encoder.

Measured targets:
- **H100**: ~2.5 hours / 30 epochs at batch 256 (~$6 at spot pricing).
- **M4 Pro MPS**: ~5h 17m / 30 epochs at batch 256 ($0, local, commit `96722b4`).

The 3× wall-clock ratio does NOT justify the cloud spend for a <5M-param model
at 24h compute cap. Pick the cheaper target unless compute requirements
change (e.g., hierarchical Level-2 model, larger pool, longer context).

## Why

bf16 mixed precision and `torch.compile(mode="reduce-overhead")` are
CUDA-only. On MPS both auto-disable: MPS runs fp32, no kernel fusion. The
wall-clock penalty on M-series is ~1.5× the bf16+compile version, which
brings a 2.5h H100 run to a 5–7h M4 run — still comfortably within the 24h
compute cap at batch 256.

Step 3 run-2 (2026-04-23, the Gate 1 pass run) validated this end-to-end:
5h 17m, 30 epochs, $0, all four Gate 1 binding conditions passed on Feb and
Mar 2026.

## Requirements for MPS Correctness

Latent bug fixed in commit `af2bee1`: `torch.from_numpy(...)` in `tape/augment.py`
and `tape/pretrain.py` defaulted to CPU tensors and crashed when arithmetic'd
with device tensors. This bug was latent on CUDA too — CPU-only smoke tests
hid it; a real H100 run would have crashed at the first augmentation step. All
numpy-sourced tensors must `.to(window.device)` before arithmetic.

Additionally, `torch.linalg.svdvals` is unimplemented on MPS in PyTorch 2.10;
`effective_rank()` falls back to CPU for this once-per-step diagnostic
(overhead negligible).

Device selection in `scripts/run_pretrain.py` prefers CUDA → MPS → CPU, and
`cfg.use_torch_compile` is force-set to False off-CUDA.

## When H100 Is Still Preferred

- Training >1M-param encoder (kernel fusion gains compound).
- Training run >12h on MPS (MPS thermal throttling starts to matter).
- Multi-seed CKA stability runs where wall-clock parallelism dominates.
- Any run where bf16 precision becomes load-bearing (larger context, bigger
  batch).

## Alternatives Considered

1. **H100 default.** Rejected for this model size — the 3× speedup doesn't
   justify $6/run when most pretraining iterations will be configuration
   sweeps.
2. **MPS default for all runs.** Rejected — future hierarchical or larger
   models will need CUDA bf16 + compile.
3. **MPS for dev, H100 for "final" runs.** Partially adopted — the Gate 1 pass
   run was intentionally done on MPS to prove it; future universality-targeting
   runs may re-evaluate.

## Impact

- `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
  "Pre-Registration" section explicitly lists MPS as an accepted substitute.
- `CLAUDE.md` "Training" section documents the measured targets.
- `.claude/agents/lead-0.md` directive updated to pick cheaper hardware by
  default.
- Step 3 run-2 (`runs/step3-r2/`) is the reference reproduction.

## Related

- [Gate 1 pass experiment](../experiments/gate1-pass-feb-mar-h500.md)
- [Pivot to representation learning](pivot-to-representation-learning.md)

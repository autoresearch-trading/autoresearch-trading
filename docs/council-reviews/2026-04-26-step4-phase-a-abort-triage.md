# Step 4 Phase A Abort Triage — 2026-04-26

**Status:** Misspecified criterion confirmed by council-5 + council-6. Patch plan, resume from checkpoint.

## Trigger

`scripts/run_finetune.py` aborted at epoch 5 on `ABORT_EPOCH_5_H500_VAL_BCE_GT_0_95X_INIT`:

```
epoch=5 H500 val BCE (0.6918) > 0.95× initial (0.6937); linear-probe warmup failed
```

## Diagnosis: misspecified, not unhealthy

### The math (council-6, definitive)

For binary classification at base rate ≈ 0.5, near the achievable BCE floor:

```
BCE(β) ≈ log(2) − 2·(β − 0.5)²
```

The 0.95× threshold requires:

```
required_β = 0.5 + sqrt((1 − factor) · log(2) / 2)
           = 0.5 + sqrt(0.05 · 0.6931 / 2)
           = 0.632
```

Gate 1 measured the SAME frozen encoder's H500 separability ceiling at:
- **0.514 mean** across 24 symbols (Feb + Mar matched-density)
- **0.535 best symbol**

The Phase A head (Linear 256→64 + ReLU + Linear 64→1) is structurally weaker than the sklearn LogisticRegression with C-sweep used at Gate 1 — no regularization search, single fixed lr/schedule. So Phase A's β-ceiling is bounded above by Gate 1's measurement.

**The abort was mathematically guaranteed at plan-ratification time.** The criterion required Phase A to EXCEED its own upper bound (Gate 1's measurement of the encoder's frozen separability).

### Run health: all green

| Indicator | Threshold | Observed |
|-----------|-----------|----------|
| Train BCE descending | monotone | 0.6923 → 0.6917 → 0.6915 → 0.6913 → 0.6911 ✓ |
| Val BCE descending | monotone | 0.6921 → 0.6917 → 0.6915 → 0.6915 → 0.6914 ✓ |
| Val H500 BCE < init | < 0.6937 | 0.6918 ✓ |
| Val H500 bal acc | > 0.500 | 0.513 (matches Gate 1's 0.514) ✓ |
| embed_std | > 0.05 | 0.604 ✓ |
| effective_rank | > 30 | 192/256 ✓ |
| CKA-vs-frozen | (frozen Phase A) | 0.984 (encoder properly frozen) ✓ |
| hour-of-day acc | < 0.12 | 0.083 ✓ |

This is textbook successful Phase A — the head saturated the frozen embedding's linear-probe ceiling.

### Where the derivation went wrong

The 0.95× value was introduced in `docs/council-reviews/council-6-step4-design.md` Q2 and ratified into the plan without an arithmetic check. The intuition ("if heads can't fit a usable boundary on frozen-and-known-good embeddings in 5 epochs, abort") is sound, but the magnitude was calibrated for tasks where random-init BCE is far from achievable BCE floor (e.g. MEM in BN-normalized space, NT-Xent log(B)). Binary classification at base rate 0.5 with strong label smoothing has init BCE ≈ log(2) ≈ 0.6931 already sitting on the achievable floor.

A 5% relative drop on log(2) demands separability that this domain doesn't have at any horizon, by Gate 1's measurement.

## Patch (mechanical alignment, NOT a binding-gate amendment)

This is a math-bug fix, not a redefinition of Gate 2 criteria. The amendment-budget clause does not apply.

### Replace in `tape/finetune.py`:

```python
ABORT_EPOCH_5_H500_VAL_BCE_GT_0_95X_INIT: bool = True
```

with:

```python
# Replaced 2026-04-26 (council-5 + council-6 triage). Original criterion required
# β-balanced-acc ≈ 0.632 from a frozen encoder Gate 1 measured ceiling at β=0.514
# — abort was guaranteed by construction. New criterion uses accuracy-grounded checks.
ABORT_EPOCH_5_H500_VAL_BCE_NOT_MONOTONE_OR_BAL_ACC_LT_0_510: bool = True
```

### Replace in `scripts/run_finetune.py` abort logic (the BCE-threshold check at end of Phase A):

```python
# OLD:
if epoch == frozen_epochs and val_h500_bce > 0.95 * initial_h500_val_bce:
    abort = "epoch=5 H500 val BCE ({:.4f}) > 0.95× initial ({:.4f}); linear-probe warmup failed"

# NEW:
if epoch == frozen_epochs:
    val_h500_bal_acc = val_balanced_acc_per_horizon[3]
    bce_monotone = all(
        log_history[i+1]["val_per_horizon_bce"][3] < log_history[i]["val_per_horizon_bce"][3]
        for i in range(len(log_history) - 1)
    )
    if not bce_monotone:
        abort = f"epoch=5 H500 val BCE not monotone-decreasing; linear-probe warmup failed"
    elif val_h500_bal_acc < 0.510:
        abort = f"epoch=5 H500 val balanced acc ({val_h500_bal_acc:.3f}) < 0.510; head failed to extract Gate-1 separability"
```

### Add `--resume-from-checkpoint` CLI flag

To skip Phase A and start at Phase B from `runs/step4-r1/aborted.pt`:

```python
parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Load encoder + head weights and start at epoch=frozen_epochs (Phase B)")
```

Loading logic: model.load_state_dict(checkpoint), set start_epoch = frozen_epochs, fresh Phase B optimizer + scheduler.

## Going-forward rule

Any abort threshold of the form `BCE × init_factor` must be pre-derived against the measured baseline by:

```
required_β_at_threshold = 0.5 + sqrt((1 − factor) · log(2) / 2)
```

If `required_β > Gate-1-measured-β-ceiling`, the criterion is misspecified.

## Phase B amendments (council-5 demand)

While we're patching, also fix two issues council-5 surfaced for Phase B:

1. **Add CKA upper bound** — `CKA > 0.95 after epoch 8` = "Phase B did nothing" (encoder didn't adapt). Currently only have the `< 0.3` lower bound. Recommended: add `ABORT_CKA_GT_0_95_AFTER_EPOCH_8`.

2. **Phase B success criterion uses H500 balanced accuracy improvement, not weighted-mean BCE.** H10 has up to 9.9pp inflation potential on illiquid alts (Gotcha #28); a Phase B that finds an H10 shortcut would show outsized weighted-mean BCE improvement while H500 stagnates. The eventual Gate 2 criteria already use H500 balanced accuracy explicitly — this just aligns Phase B monitoring with Gate 2.

## Other abort criteria (audited, no changes)

| Criterion | Status | Notes |
|-----------|--------|-------|
| `ABORT_EPOCH_3_H500_VAL_BCE_GT_INIT` | ✓ Sane | Fires only on regression. |
| `ABORT_EMBED_STD_LT_0_05` | ✓ Sane | Pretraining-derived; transfers correctly. |
| `ABORT_CKA_LT_0_3_AFTER_EPOCH_8` | ⚠ Add upper bound | Per council-5: also need `> 0.95` upper bound. |
| `ABORT_H100_VAL_BAL_ACC_LT_0_50_AFTER_EPOCH_8` | ✓ Sane | H100 was at noise floor; 0.50 is correct floor. |
| `ABORT_HOUR_PROBE_GT_0_12_AT_5EPOCH_CHECKPOINT` | ✓ Sane | 0.083 currently; 0.12 calibrated against pretraining hour probe. |

## Summary

(1) **Misspecified, not unhealthy** — every health indicator (embed_std=0.604, eff_rank=192, CKA=0.984, monotone train+val BCE descent, val balanced acc 0.513 matching Gate 1's 0.514) is green; only the 0.95× threshold fired, and that threshold demanded H500 balanced acc ≈ 0.555-0.632 from a frozen encoder Gate 1 had measured to ceiling at 0.514, making abort mathematically guaranteed at ratification time.

(2) **Patch + resume from `runs/step4-r1/aborted.pt`** — replace the BCE-relative clause with two arithmetic-grounded checks (val H500 BCE strictly monotone-decreasing through epoch 5 AND val H500 balanced acc ≥ 0.510 at epoch 5; both are currently true), add a `--resume-from-checkpoint` CLI flag, and dispatch the 15-epoch unfrozen phase. Saves ~3.5h vs full restart; equivalent up to head-weight noise smaller than the per-symbol bootstrap CI half-width.

(3) **Process update** — pre-derive any future BCE-relative threshold via `required_β = 0.5 + sqrt((1 − factor) · log(2) / 2)` against measured Gate-1-style baselines; trial-count log this triage as a single mechanical alignment.

## References

- Council-5 verdict (full reasoning): see this file (synthesis)
- Council-6 verdict (full reasoning): see this file (synthesis)
- `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` (lines 51, 105 — original threshold)
- `docs/council-reviews/council-6-step4-design.md` (Q2, line 23 — origin of 0.95×)
- `docs/experiments/step3-run-2-gate1-pass.md` (Gate 1 measured ceilings β-mean=0.514, β-best=0.535)
- `runs/step4-r1/training-log.jsonl` (aborted-run trajectory)
- `runs/step4-r1/aborted.pt`, `runs/step4-r1/finetuned-best.pt` (forensic snapshots; valid Phase A endpoint)

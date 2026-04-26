# Step 4 Phase B CKA-Abort Triage — 2026-04-26

**Status:** Misspecified threshold confirmed by council-5 + council-6. Comprehensive amendment to all Phase B abort criteria, then resume from Phase A end.

## Trigger

`scripts/run_finetune.py` (resumed from `runs/step4-r1/aborted.pt`) aborted at epoch 8 on the **NEW** `ABORT_CKA_GT_0_95_AFTER_EPOCH_8` criterion that lead-0 added in this morning's patch:

```
CKA-vs-frozen frozen: 0.9569 > 0.95 after epoch 8 (Phase B did nothing)
```

This is the **second consecutive math-bug abort** in 24 hours. Per the falsifiability discipline established in this morning's triage doc, council was called rather than patching quietly.

## Diagnosis: misspecified, run is healthy

### The Phase B trajectory (3 epochs of unfreeze at lr=5e-5)

| Epoch | val H500 BCE | H500 bal acc | embed_std | eff_rank | CKA-vs-frozen |
|-------|--------------|--------------|-----------|----------|---------------|
| 5 (Phase A end) | 0.6918 | 0.513 | 0.604 | 192 | 0.984 |
| 6 (PB E1) | 0.6918 | 0.516 | 0.609 | 191 | 0.984 |
| 7 (PB E2) | 0.6918 | 0.514 | 0.623 | 197 | 0.980 |
| 8 (PB E3) | 0.6915 | 0.519 | 0.633 | 202 | 0.957 (ABORT) |

Every other signal is healthy:
- Monotone val H500 BCE descent (0.6918 → 0.6915)
- H500 bal acc climbing +0.6pp in 3 epochs (already exceeds Gate 1's frozen ceiling 0.514)
- embed_std climbing 0.604 → 0.633 (encoder rebalancing, opposite of collapse)
- eff_rank climbing 192 → 202 (more expressive embedding)
- CKA dropping 0.984 → 0.957 (encoder IS adapting)
- hour-of-day probe 0.069 (well below 0.12 floor)

### The math (council-6, definitive)

**OneCycleLR with `pct_start=0.05` over 15 Phase B epochs places peak lr at epoch 5.75** (i.e., 0.75 epochs into Phase B — almost immediate). Lead-0's threshold-pick assumed peak at epoch 12, which is wrong.

**Integrated-lr fraction by epoch 8: ~42%, not ~20%.** The encoder has already absorbed nearly half of its total Phase B gradient budget by the time the abort fires.

**Expected CKA trajectory (council-6 derivation):**

| Epoch | Integrated-lr fraction | Expected CKA (linear) | Expected CKA (sqrt) |
|-------|------------------------|------------------------|----------------------|
| 5 (PA end) | 0.00 | 1.000 | 1.000 |
| 6 (PB E1) | 0.07 | 0.993 | 0.972 |
| 7 (PB E2) | 0.21 | 0.979 | 0.948 |
| 8 (PB E3) | 0.42 | 0.958 | 0.917 |
| 12 (PB E7) | 0.74 | 0.926 | 0.853 |
| 19 (PB E14) | 1.00 | 0.900 | 0.820 |

**Observed 0.957 at epoch 8 lies exactly on the linear-regime trajectory.** The threshold 0.95 demanded behavior the lr schedule structurally cannot produce by epoch 8.

### Root cause (same as morning bug)

Both today's bugs share root cause: **abort thresholds picked as round numbers in a unit (BCE-fraction, CKA-fraction) whose dynamic range was implicitly assumed to be ImageNet-scale, but is compressed in our setting** (BCE near log(2) floor; CKA near 1.0 because lr is 10× lower than pretraining, encoder rotates much slower).

This morning's BCE bug demanded β-balanced-acc ≈ 0.632 from a frozen encoder Gate 1 had measured at 0.514. Tonight's CKA bug demanded ~3× faster encoder rotation than the lr schedule supports.

## Patch (mechanical alignment, NOT a binding-gate amendment)

Per council-5: this is the **second** mechanical alignment in 24 hours. **One more abort in this run triggers full stop, complete trajectory simulation, re-launch from scratch.** Gate 1 thresholds and Gate 2 binding criteria remain LOCKED.

### Going-forward rule (generalized; supersedes morning's narrower BCE-only rule)

> Every abort threshold MUST be accompanied by a written one-paragraph trajectory model: (a) what known-good baseline behavior should look like at the check epoch, (b) what failure-mode behavior should look like, (c) the gap between them, and (d) where the threshold sits in that gap. If the threshold is closer than 1 trajectory-step to known-good behavior, it is misspecified. If no trajectory model can be written — because the dynamics are unknown — the threshold belongs at end-of-training, not mid-training.

### Amendment 1 — CKA upper bound

**Drop:** `ABORT_CKA_GT_0_95_AFTER_EPOCH_8`

**Add:** `ABORT_CKA_END_OF_PHASE_B` — at epoch ≥ epochs−2 (i.e., last two epochs of training): CKA > 0.95 = "Phase B did nothing." This is the intent of council-5's morning demand, correctly placed where the data supports the threshold.

**Add:** `ABORT_CKA_RATE_TOO_SLOW` — at epoch ≥ 8 (post-warmup): if (last_3_CKA_deltas).max() < 0.005 (i.e., CKA dropping by less than 0.5pp/epoch for 3 consecutive epochs), encoder is stuck. Catches "encoder didn't move" without falsely aborting on "encoder hasn't moved YET."

### Amendment 2 — H100 floor (predicted bug #3 by council-5)

**Drop:** `ABORT_H100_VAL_BAL_ACC_LT_0_50_AFTER_EPOCH_8` (zero-margin against Gate 1's measured H100 noise floor; bootstrap CI half-width on H100 is ~0.5-1.5pp, so a single dip below 0.50 is within noise).

**Add:** `ABORT_H100_BAL_ACC_TRAILING_DEGRADATION` — trailing-5-epoch mean of H100 val balanced acc must not drop below (Phase A end H100 bal acc) − 1.0pp. The right epistemic shape: "H100 is degrading," not "H100 dipped below an arbitrary threshold once."

### Amendment 3 — Phase B success criterion (both councils)

**Add:** `ABORT_PHASE_B_NO_H500_BAL_ACC_GAIN` — at epoch 15 (E10 of Phase B), H500 val bal_acc must be ≥ Phase-A-end H500 bal_acc + 1.0pp. Soft warn at epoch 12: if not yet ≥ +0.5pp, log warning. Phase B is justified ONLY if it improves the metric it's claimed to improve; a 1.0pp gain is ~3σ given val-fold bal_acc noise of ~0.3pp.

### Amendment 4 — Effective rank floor (council-6 insurance)

**Add:** `ABORT_EFF_RANK_LT_50_AFTER_EPOCH_8` — eff_rank below 50 (1/4 of starting value) after epoch 8 = catastrophic rank collapse. Currently 202. Cheap insurance.

### Amendment 5 — Hour probe at all PB checkpoints (council-6)

**Already implemented:** the existing 5-epoch checkpoint logic covers all of E5, E10, E15. No code change needed; the implementation already triggers at every 5-epoch boundary including those mid-Phase-B.

## Resume strategy

**Restart from Phase A end** (`runs/step4-r1/aborted.pt`), NOT from Phase B epoch 8 (`runs/step4-r1-phase-b/aborted.pt`). Reason: `aborted.pt` does not checkpoint scheduler state (`scheduler_state_dict` and `step_count` not in saved keys), so resuming Phase B from epoch 8 would re-run OneCycleLR from `last_epoch=0` rather than `last_epoch=3` — corrupted lr schedule. Council-6's verdict: "a corrupted scheduler resume produces a worse outcome than a clean 3-epoch redo."

Cost: re-run 3 Phase B epochs (~30-45 min on M4 Pro). Benefit: clean audit chain.

Output dir: `runs/step4-r1-phase-b-v2/` (preserves prior Phase B trail forensically).

## Audit of remaining live abort criteria

| Criterion | Trajectory model | Verdict |
|---|---|---|
| `EPOCH_3_H500_VAL_BCE_GT_INIT` | Known-good: BCE descending; failure: not descending. Far on failure side. | ✓ OK |
| `EPOCH_5_H500_VAL_BCE_NOT_MONOTONE` | Known-good: monotone descent; failure: oscillation. Explicit gap. | ✓ OK |
| `EPOCH_5_H500_VAL_BAL_ACC_LT_0_510` | Known-good: ≥ 0.514 (Gate 1); failure: 0.500. Threshold 0.510 = halfway. | ✓ OK |
| `EMBED_STD_LT_0_05` | Pretraining-derived; healthy >0.5; failure → 1e-3. Far on failure side. | ✓ OK |
| `CKA_LT_0_3_AFTER_EPOCH_8` | Known-good: 0.957 → 0.83; failure: catastrophic forgetting → 0. Threshold 0.3 far on failure side. | ✓ OK |
| `HOUR_PROBE_GT_0_12_AT_5EPOCH_CHECKPOINT` | Known-good: 0.06-0.09; failure: 0.20+. Threshold 0.12 in gap. | ✓ OK |

The H100 floor was the only one with zero margin from known-good. After Amendment 2 it's resolved.

## Falsifiability line (council-5)

**Math-bug-fix (allowed):** the threshold demanded behavior no healthy run could produce, derivable from baselines that pre-date the run. This morning's BCE bug: Gate 1's β=0.514 ceiling existed before run-1. Tonight's CKA bug: the OneCycleLR trajectory model is derivable from the lr schedule, which existed before run-2.

**Result-fitting (forbidden):** relax a correctly-calibrated threshold to pass observed data.

**The bright lines:**
1. Gate 1 thresholds (51.4% / 1.0pp / 1.0pp / hour<10%) are LOCKED
2. Gate 2 binding criteria (0.5pp/15+/24, regression check, 0.3pp/13+/24 vs frozen-LR) are LOCKED
3. ONE more math-bug abort in this run = full stop, complete trajectory simulation, re-launch
4. Final Gate 2 evaluation runs ONCE — no reruns "to confirm"

## References

- Morning's triage (same root cause): `docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md`
- Council-5 verdict (full): integrated above (synthesis)
- Council-6 verdict (full): integrated above (synthesis)
- `runs/step4-r1/aborted.pt` (Phase A end checkpoint — resume target)
- `runs/step4-r1-phase-b/aborted.pt` (Phase B epoch 8 — preserved forensically, NOT resume target)
- `runs/step4-r1-phase-b/training-log.jsonl` (3-epoch Phase B trajectory)

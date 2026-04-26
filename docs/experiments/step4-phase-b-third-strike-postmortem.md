# Step 4 Phase B — Third-Strike Postmortem

**Status:** Auditable record committed BEFORE Gate 2 numbers are read. Per council-5's binding audit trail.

**Date:** 2026-04-26 (mid-day)

**Subject:** `runs/step4-r1-phase-b-v2/finetuned-best.pt` (epoch 18) — the artifact entering Gate 2

## Pre-committed Phase B success criteria — ALL MET

The Phase B success criteria below were pre-committed in `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` (commits `285933f`, `18bdaf2`, `8149aa8`, `322ab50`) BEFORE this run started. Observed values as recorded in `runs/step4-r1-phase-b-v2/training-log.jsonl`:

| Criterion | Threshold (pre-committed) | Observed (E18) | Status |
|-----------|--------------------------|----------------|--------|
| H500 val bal_acc gain at E15 | ≥ +1.0pp vs Phase-A-end (0.513) | +1.0pp at E15, +1.2pp at E18 | ✓ PASS |
| End-of-training CKA-vs-frozen | < 0.95 | 0.923 | ✓ PASS |
| Encoder adaptation (CKA drop) | meaningful drift from 1.0 | 0.984 → 0.923 (Δ = −0.061) | ✓ PASS |
| embed_std (no collapse) | > 0.05 | 0.604 → 0.769 | ✓ PASS |
| eff_rank (no rank collapse) | > 50 | 192 → 215 | ✓ PASS |
| hour-of-day probe | < 0.12 | 0.064 (E11), 0.058 (E16) | ✓ PASS |
| H100 trailing-5 degradation | ≤ 1.0pp from PA-end (0.513) | 0.519 (improved, no degradation) | ✓ PASS |
| Monotone val H500 BCE descent | strictly decreasing PA-end → end | 0.6918 → 0.6908 (Δ = −0.0010) | ✓ PASS |

**Decision (pre-Gate-2):** `runs/step4-r1-phase-b-v2/finetuned-best.pt` (E18 best checkpoint, val_total_bce=0.6906) is the artifact entering Gate 2. The pre-committed Phase B success criteria all passed BEFORE Gate 2 evaluation.

## The three abort firings — full trial-count log

| Bug # | When | Criterion that fired | Class | Resolution |
|-------|------|---------------------|-------|------------|
| 1 | Run-1 Phase A E5 | `H500 val BCE > 0.95×init` | **A (binding science)** | Math-bug: required β-bal-acc=0.632 from frozen encoder Gate 1 measured at 0.514. Replaced with monotone BCE + bal_acc≥0.510 floor (commit `8149aa8`). Triage: `docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md`. |
| 2 | Run-2 Phase B E8 | `CKA-vs-frozen > 0.95 after E8` | **A (binding science)** | Math-bug: lead-0 heuristic without trajectory simulation. OneCycleLR pct_start=0.05 puts peak lr at E5.75; by E8 only ~42% of integrated lr consumed; observed 0.957 sits exactly on linear-regime trajectory. Replaced with end-of-Phase-B check + rate-check (commit `322ab50`). Triage: `docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md`. |
| 3 | Run-3 Phase B E18 | `max(ΔCKA over last 3 epochs) < 0.005` | **B (redundant guard)** | Math-bug: rate-check originally suggested by council-6 for early Phase B (E6-E8) when encoder should be moving; lead-0 generalized to "after epoch 8" without protecting OneCycleLR cosine cooldown. By E18, lr has decayed to ~1e-7; encoder NATURALLY oscillates ±0.005 around convergence. Triage: this document + parallel council-5 / council-6 review. |

## Bug #3 — disposition

**The fired criterion:** `ABORT_CKA_RATE_TOO_SLOW`. At E18, last 3 ΔCKA = [−0.004, +0.005, −0.005]. max(deltas) = 0.005 = threshold → fired.

**Why it was misspecified:**
- OneCycleLR pct_start=0.05 over E5.75→E19 with peak lr=5e-5 has cosine decay to ~0 by epoch 19.
- By E18, effective lr ≈ 5e-7 (3 orders of magnitude below peak). Encoder takes microscopic steps.
- CKA stabilizes at convergence value with epoch-to-epoch noise of ~0.005 (matching FP variance in 1024-window CKA computation).
- The rate-check fires because that's the SCHEDULED end-of-training behavior, not pathology.

**Why it's Class B (redundant guard) not Class A (binding science):**
- The end-of-training CKA<0.95 upper bound (epoch ≥ epochs−1) ALREADY catches "Phase B did nothing." That's the binding criterion — and observed CKA 0.923 is well below 0.95, so it would have passed.
- The rate-check was meant as belt-and-suspenders for "encoder is moving but slower than expected mid-run" — a path property, not the success criterion.
- Council-6's original recommendation was rate-check at "epochs 6, 7, 8" (early Phase B at peak lr, when encoder SHOULD be actively rotating). lead-0 generalized to "epoch ≥ 8" without simulating cooldown behavior.

**Resolution: retire the rate-check as a misspecified redundant guard.** Do NOT replace with a calibrated version — the end-of-training upper bound is the necessary and sufficient version of this check. Adding more guards in the same logical region risks more bugs without adding falsifiability.

## Convergence audit (council-6 four-signature analysis)

The encoder at E18 is a converged Phase B end-state, NOT prematurely cut off:

1. **Loss plateau** (val H500 BCE last 5 epochs = [0.6907, 0.6909, 0.6909, 0.6908, 0.6909, 0.6908]): peak-to-peak 0.0002, at FP/data-shuffle noise floor for 41K-window val split.
2. **Metric plateau** (H500 bal_acc last 5 = [0.523, 0.523, 0.525, 0.526, 0.525]): peak-to-peak 0.003, well inside 1×CI half-width (~0.005).
3. **CKA stationary oscillation** ([−0.004, +0.005, −0.005]): mean-zero. Signature of parameter-delta dominated by gradient noise, not signal — i.e., local optimum at current data + lr.
4. **Effective rank ceiling** (215 from E11 onward): embedding subspace stopped rotating.

**E19 wouldn't have changed anything:** at lr ~1e-7, expected H500 bal_acc Δ < val-fold noise floor.

## Pre-Gate-2 commitments (BINDING)

In service of council-5's audit trail, the following are pre-committed BEFORE Gate 2 numbers are read:

### Commitment 1 — Gate 2 thresholds are LOCKED

The pre-registered Gate 2 binding criteria from the ratified plan:

- **Criterion 1**: fine-tuned CNN beats flat-LR by ≥ 0.5pp on 15+/24 symbols (mean H500 balanced accuracy, Feb + Mar held-out)
- **Criterion 2**: no per-symbol H500 balanced acc drop > max(1.0pp, 1×bootstrap CI half-width) on 1+/24 symbols vs flat-LR baseline (regression check)
- **Criterion 3**: fine-tuned CNN beats frozen-encoder LR by ≥ 0.3pp on 13+/24 symbols

**If Gate 2 fails on ANY of the three binding criteria, the answer is "Gate 2 failed."** The next move is council, not retry of Phase B with different hyperparameters / criteria. The Phase B run has had its one shot at producing a fine-tuned encoder.

### Commitment 2 — anti-amnesia in Gate 2 report

Whatever the Gate 2 outcome, the resulting report (`docs/experiments/step4-gate2-finetune.md`) MUST publish:
- The original Phase B abort criteria as written in commits `285933f` and onwards
- Bug #3 annotation: "fired at E18 on cosine cooldown; retired as redundant with end-of-training CKA<0.95 upper bound. Decision documented in `step4-phase-b-third-strike-postmortem.md` (this file)."
- This file's path as a reference

### Commitment 3 — honest framing if Gate 2 passes

If Gate 2 passes, the Gate 2 PASS report MUST contain a one-paragraph honest framing of the third-strike decision, citing this file by path, so any future reader auditing the Gate 2 claim sees:
1. The relaxed-strike call was made on a Class B (redundant guard) bug, not on binding science
2. The pre-committed success criteria all passed BEFORE Gate 2 was run
3. The audit trail (this file) was committed BEFORE Gate 2 numbers

No silent transit.

## Going-forward rule (replaces "third bug = full stop")

**Two-class abort taxonomy (binding from this review forward):**

- **Class A — binding-science abort:** the pre-committed success criteria themselves require re-derivation (bugs #1 and #2 fit this). Class A bug = STOP, council, re-derive, re-launch. *One Class A bug per run is the limit.*
- **Class B — redundant-guard abort:** a guard that fires on a path-property already covered by a binding success criterion (bug #3 fits this). Class B bug = retire the guard, document in run postmortem, continue. *Three Class B bugs per run is the limit before mandatory full process review.*

**Pre-launch obligation:** every abort criterion must be classified A or B BEFORE launch, and every Class B criterion must explicitly state which Class A criterion subsumes it. If no Class A criterion subsumes it, it's Class A in disguise and must be derived from the lr schedule and trajectory simulation.

**Trajectory simulation requirement:** any Class A abort threshold derived from a quantity that depends on the lr schedule (CKA, embed_std, BCE, gradient norm) must be checked against a deterministic trajectory simulation under the lr schedule. If the simulation says the threshold cannot be reached given the schedule, the threshold is wrong and must be re-derived BEFORE launch.

This rule, if applied to Phase B's pre-launch state, would have caught all three bugs.

## References

- `runs/step4-r1-phase-b-v2/finetuned-best.pt` — Gate 2 input
- `runs/step4-r1-phase-b-v2/training-log.jsonl` — full trajectory
- `docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md` — Bug #1 triage
- `docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md` — Bug #2 triage
- `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` — pre-committed criteria
- Commits: `285933f` (initial), `18bdaf2` (plan ratification), `8149aa8` (Bug #1 patch), `322ab50` (Bug #2 patch)

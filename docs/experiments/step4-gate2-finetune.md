# Step 4 Gate 2 — Fine-Tuned CNN Evaluation

**Date:** 2026-04-26 (PM)
**Verdict:** **GATE 2 FAILED** on all three binding criteria, both held-out months.
**Pre-Gate-2 audit trail:** `docs/experiments/step4-phase-b-third-strike-postmortem.md` (committed `f2f50dc` BEFORE this evaluation read numbers).

## Honest framing (per pre-committed audit trail)

This Step 4 fine-tune run experienced three abort-criterion math bugs (one in Phase A, two in Phase B). The bugs are documented in:
- `docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md` (Bug #1: BCE-relative threshold required β=0.632 from frozen encoder Gate 1 measured at 0.514)
- `docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md` (Bug #2: CKA upper-bound at epoch 8 demanded ~3× faster encoder rotation than the lr schedule supports)
- `docs/experiments/step4-phase-b-third-strike-postmortem.md` (Bug #3: rate-of-change check fired during scheduled cosine cooldown — Class B redundant guard)

Council-5 + council-6 jointly adjudicated bug #3 as Class B (redundant guard structurally subsumed by the end-of-Phase-B CKA<0.95 upper bound, which the run passed at CKA 0.923). The pre-committed Phase B success criteria all passed BEFORE Gate 2 was run; Gate 2 was the final binding falsification gate.

**Gate 2 failed.** Per pre-committed audit trail, no retry. The next move is council to determine whether the encoder pretraining itself is the bottleneck, or whether the fine-tuning approach is the wrong tool for this problem.

## Gate 2 setup

- **Fine-tuned checkpoint:** `runs/step4-r1-phase-b-v2/finetuned-best.pt` (E18 best, val_total_bce=0.6906)
- **Frozen-encoder baseline checkpoint:** `runs/step3-r2/encoder-best.pt` (Gate 1 PASS reference)
- **Held-out months:** 2026-02, 2026-03 (matched-density per spec amendment 2026-04-25)
- **Symbols:** 24 (all PRETRAINING_SYMBOLS, AVAX excluded as Gate 3 hold-out)
- **Horizon:** H500 (primary per spec amendment)
- **Comparators:** flat-LR (LogisticRegression on 83-dim flat features), frozen-encoder LR (Gate 1 protocol with C-search), fine-tuned CNN
- **Bootstrap:** 1000 resamples per cell, 95% percentile CI

## Aggregate result

| Comparator | Mean bal_acc (n=48 cells) | Median |
|------------|----------------------------|--------|
| flat-LR baseline | 0.5115 | 0.5039 |
| frozen-encoder LR | 0.5061 | 0.5037 |
| **fine-tuned CNN** | **0.4947** | **0.4853** |
| **CNN − flat-LR** | **−0.017** | — |
| **CNN − frozen-encoder LR** | **−0.011** | — |

**The fine-tuned CNN underperforms flat-LR by 1.7pp and the pre-fine-tuning frozen baseline by 1.1pp on the held-out months.** Phase B fine-tuning made the model worse, not better, on Feb+Mar — the +1.2pp val-fold gain (random 90/10 split) was overfitting to in-distribution label imbalance.

## Per-criterion verdict

| Criterion | Threshold | Feb 2026 | Mar 2026 | Overall |
|-----------|-----------|----------|----------|---------|
| C1: CNN > flat-LR by ≥0.5pp on 15+/24 | 15+/24 | 7/24 | 10/24 | **FAIL** |
| C2: No symbol regression > max(1.0pp, 1×CI) on 1+/24 | ≤ 0/24 violations | 16/24 violations | 12/24 violations | **FAIL** |
| C3: CNN > frozen-encoder LR by ≥0.3pp on 13+/24 | 13+/24 | 8/24 | 12/24 | **FAIL** |

## Per-symbol failure pattern (diagnostic)

**Worst 10 CNN regressions vs flat-LR (large signal lost):**

| Month | Symbol | flat-LR | CNN | Δ |
|-------|--------|---------|-----|---|
| 2026-02 | SUI | 0.626 | 0.477 | **−0.148** |
| 2026-02 | LTC | 0.595 | 0.458 | **−0.136** |
| 2026-03 | 2Z | 0.633 | 0.499 | **−0.134** |
| 2026-03 | LINK | 0.605 | 0.510 | −0.095 |
| 2026-03 | LTC | 0.539 | 0.451 | −0.088 |
| 2026-02 | XPL | 0.500 | 0.423 | −0.077 |
| 2026-02 | SOL | 0.543 | 0.467 | −0.076 |
| 2026-03 | PENGU | 0.553 | 0.479 | −0.074 |
| 2026-02 | ETH | 0.543 | 0.470 | −0.074 |
| 2026-02 | HYPE | 0.534 | 0.464 | −0.070 |

**Best 8 CNN gains vs flat-LR (illiquid + below-random flat baselines):**

| Month | Symbol | flat-LR | CNN | Δ |
|-------|--------|---------|-----|---|
| 2026-02 | KPEPE | 0.443 | 0.530 | +0.088 |
| 2026-03 | KPEPE | 0.480 | 0.566 | +0.086 |
| 2026-03 | AAVE | 0.471 | 0.552 | +0.081 |
| 2026-02 | KBONK | 0.405 | 0.473 | +0.068 |
| 2026-03 | HYPE | 0.449 | 0.517 | +0.068 |
| 2026-02 | BNB | 0.477 | 0.542 | +0.066 |
| 2026-03 | FARTCOIN | 0.484 | 0.544 | +0.060 |
| 2026-03 | BNB | 0.524 | 0.576 | +0.052 |

**Pattern: CNN is a "regression to 0.50" predictor.** Flat-LR baselines that are FAR from 0.500 (in either direction) get pulled toward 0.500 by the fine-tuned CNN. Where flat-LR is well above 0.500 (real signal), the CNN destroys signal. Where flat-LR is well below 0.500 (anti-signal from class imbalance), the CNN looks like it helps but it's just defaulting to 0.500.

This is consistent with the encoder having learned representations that don't transfer to held-out months — the +1.2pp val-fold gain (random 90/10 split, in-distribution) was the encoder fitting to label imbalance specific to the training period, not extracting transferable directional signal.

## What this means for the project

**Step 4 fine-tuning is FALSIFIED on the binding Gate 2 criteria.** The CNN encoder + supervised fine-tuning approach, as configured, does not produce a model that beats flat features + logistic regression on held-out months at H500.

**This does NOT falsify the encoder pretraining itself** — Gate 1 (frozen-encoder LR on H500) PASSED on the same Feb + Mar window. The frozen-encoder LR mean of 0.506 vs flat-LR 0.512 (a 0.6pp gap, marginal) is the actual ceiling of this encoder's transferable signal. Fine-tuning made it worse.

**Open questions for council:**
1. Was the fine-tuning architecture wrong? Linear-trunk-then-per-horizon-head is conventional but maybe the wrong inductive bias for tape data.
2. Was the loss-weight schedule wrong? Pre-committed to (0.10, 0.20, 0.20, 0.50) for H10/H50/H100/H500, but maybe H10's 9.9pp inflation potential dominated the gradient signal even at low weight.
3. Is the encoder representation too symbol-specific to transfer? Phase B showed +1.2pp on the in-distribution val fold; the same encoder loses 1.7pp on Feb+Mar. The encoder may be memorizing per-symbol artifacts.
4. Should we move to a different downstream task (clustering, retrieval, regime classification) where the encoder's representation quality can be assessed without the temporal-transfer constraint?

## Anti-amnesia: original Phase B abort criteria

Per council-5's pre-committed audit trail, every Gate 2 report MUST publish the original Phase B abort criteria as written, with retired guard #3 annotated:

**Phase B abort criteria as ratified in commit `285933f`:**
1. ✓ Epoch 3: H500 val BCE > training-init BCE → Class A
2. ✓ Epoch 5: H500 val BCE not monotone-decreasing → Class A (replaces bug #1's `0.95×init`)
3. ✓ Epoch 5: H500 val balanced acc < 0.510 → Class A
4. ✓ Embed std < 0.05 → Class A
5. ✓ CKA-vs-frozen < 0.3 (epoch ≥ 8) → Class A
6. ⊗ **CKA-vs-frozen > 0.95 after epoch 8** → was Class B, **REPLACED 2026-04-26 PM** with end-of-training upper bound (Class A) + rate-check (Class B, **RETIRED at bug #3**)
7. ✓ End-of-training (epoch ≥ epochs−1): CKA-vs-frozen > 0.95 → Class A
8. ⊗ **max(ΔCKA over last 3 epochs) < 0.005 after epoch 8** → Class B (**RETIRED 2026-04-26 PM** as bug #3 — fired on cosine cooldown which is scheduled lr-driven stationarity, not pathology; structurally subsumed by criterion 7)
9. ✓ Effective rank < 50 (epoch ≥ 8) → Class A
10. ✓ H100 trailing-5-epoch degradation > 1.0pp from PA-end (replaces absolute < 0.50 floor) → Class A
11. ✓ Epoch 15: H500 bal_acc gain < 1.0pp vs PA-end → Class A
12. ✓ Hour-of-day probe > 0.12 at any 5-epoch checkpoint → Class A

**Bug #3 audit:** the rate-of-change check (criterion #8) fired at E18 with last 3 ΔCKA = [−0.004, +0.005, −0.005], max = 0.005 = threshold. OneCycleLR cosine decay puts effective lr at ~5e-7 by E18 (3 orders of magnitude below peak); encoder takes microscopic steps and CKA stabilizes at convergence value with epoch-to-epoch noise of ~0.005 (matching FP variance in 1024-window CKA computation). The criterion fired on the lr schedule's prescribed end-of-training stationarity, not on encoder pathology. Council adjudicated as Class B redundant guard structurally subsumed by criterion #7 (end-of-training CKA<0.95, which the run passed at 0.923).

## Files

- `runs/step4-r1-gate2/gate2-eval.json` — full results (per-symbol bal_acc, deltas, bootstrap CIs, verdict)
- `runs/step4-r1-gate2/gate2.log` — execution log
- `runs/step4-r1-phase-b-v2/finetuned-best.pt` — fine-tuned checkpoint (E18)
- `runs/step4-r1-phase-b-v2/training-log.jsonl` — full Phase B trajectory (E5→E18)
- `docs/experiments/step4-phase-b-third-strike-postmortem.md` — pre-Gate-2 audit trail
- `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` — pre-committed plan
- Commits: `285933f`, `18bdaf2` (initial + ratification), `8149aa8`, `322ab50`, `f2f50dc` (three patches + postmortem)

## Trial-count log (per spec §"Trial-count log")

The Step 4 / Gate 2 evaluation pipeline was run ONCE on `runs/step4-r1-phase-b-v2/finetuned-best.pt`. No re-runs. The fine-tune run #1 aborted at E5 (`runs/step4-r1/aborted.pt`); fine-tune run #2 aborted at E8 (`runs/step4-r1-phase-b/aborted.pt`); fine-tune run #3 aborted at E18 (`runs/step4-r1-phase-b-v2/aborted.pt`) — three Phase B training runs, ONE Gate 2 evaluation. The first two training-run aborts produced no usable Gate 2 input; only the third produced `finetuned-best.pt`, which was evaluated once and failed.

## Conclusion

**Gate 2 FAILED. Step 4 fine-tuning falsified.** Encoder pretraining (Gate 1) remains valid. Next move: council on alternative downstream tasks or architectural revisions before any retry.

# Step 4 Plan — Fine-Tuning + Gate 2 (RATIFIED v2)

**Date:** 2026-04-24
**Author:** lead-0 (auto mode)
**Reviewers:** council-5 (`docs/council-reviews/council-5-step4-falsifiability.md`), council-6 (`docs/council-reviews/council-6-step4-design.md`)
**Status:** RATIFIED v2 — all required edits applied; ready for builder-8

## Objective

Pass Gate 2 with a fine-tuned encoder + direction heads on the same matched-density Feb+Mar 2026 H500 held-out protocol used for Gate 1. Per amended Gate 2 (council-aligned), three criteria must hold; see "Walk-forward evaluation" below.

## Pre-flight: Gate 2 spec drift to align (mechanical, not amendment-budget)

The 2026-04-24 spec amendment v2 fixed Gate 1, Gate 3, Gate 4 horizon and held-out language but did NOT touch Gate 2 (line 347–351) or the Fine-Tuning section (line 271–278). Three drift points need consistency-aligned fixes — these are NOT new binding-gate amendments under the amendment-budget clause; they are mechanical alignment with the already-ratified amendment.

| Drift point | Pre-amendment language | Aligned language |
|---|---|---|
| Line 349 "primary horizon" | unspecified, implicitly H100 | **H500** (per amendment horizon-selection rule) |
| Line 349 "15+ symbols" | implicitly 15+/25 | **15+/24** (AVAX excluded) |
| Line 277 loss weights 0.10/0.20/0.50/0.20 | H100 primary, H500 de-weighted | **0.10/0.20/0.20/0.50** (H500 primary, H100 ancillary) |

**The third drift was the substantive one and was reviewed by council-6 (Q1):** clean swap to `0.10/0.20/0.20/0.50` selected as the binding schedule. Council-5 concurrence (Q1): defensible IFF pre-committed before training. **Pre-commitment locked here, in this plan, before builder-8 implements:** loss weights are `0.10/0.20/0.20/0.50` for H10/H50/H100/H500. No fall-back schedule. Failure of this schedule must be debugged at the architectural level (catastrophic forgetting, head capacity), NOT by re-shooting Gate 2 with different weights.

## Architecture

```
Encoder (frozen → unfrozen)  →  256-dim global embedding
                                       ↓
                                Linear(256 → 64)  + ReLU         ← shared trunk (council-6 Q5)
                                       ↓
                                4 × Linear(64 → 1) + sigmoid     ← per-horizon heads
                                       ↓
                                BCE loss per horizon
                                       ↓
                                weighted sum (0.10 H10, 0.20 H50, 0.20 H100, 0.50 H500)
```

Direction heads use a shared 256→64 trunk feeding 4 separate `Linear(64→1)` (per spec line 273; council-6 Q5 confirmed shared trunk is correct over independent heads — multi-task pedagogy via shared trunk + 4.4% trainable head fraction is more conservative than 17.4% with independent heads). Total head params: `(256·64 + 64) + 4·(64+1) = 16,704` extra params on a 376K-param encoder.

## Training schedule

| Phase | Epochs | Encoder | LR | Loss |
|---|---|---|---|---|
| Linear-probe warmup | 5 | **frozen** | 1e-3 (heads only) | weighted BCE |
| Joint fine-tune | 15 | **unfrozen** | **5e-5** (10× lower than pretraining max_lr=1e-3) | weighted BCE |

Total: 20 epochs. Estimated ~3.5h on M4 Pro MPS at batch 256.

Optimizer: AdamW. Scheduler: OneCycleLR with `max_lr=5e-5`, `pct_start=0.05` for the unfrozen phase. Gradient clipping `max_norm=1.0`.

**Council-6 Q2 abort criterion (NEW):** at end-of-epoch-5 (last frozen epoch), the per-horizon val BCE for H500 must drop below `0.95 × initial_random_BCE`. If heads cannot fit a usable boundary on frozen-and-known-good embeddings in 5 epochs, abort — the fix is longer warmup or head-architecture rethink, NOT encoder unfreeze.

Label smoothing: ε = 0.10 / 0.08 / 0.05 / 0.05 for H10/H50/H100/H500 (per spec line 276). **Rationale corrected per council-6 Q6 #4:** H500 has the BEST signal-to-noise (per the amendment's horizon-selection rule), so we trust those labels MOST → less smoothing. (The earlier "highest noise floor → less smoothing" rationale was backward; higher noise = MORE smoothing, not less.)

## Walk-forward evaluation

Per amended Gate 1: train on Oct 16 – Jan 31, evaluate on Feb 2026 AND Mar 2026 independently at H500. **Council-5 sub-edit applied:** 600-event embargo between train tail (Jan 31) and test head (Feb 1) per spec line 278 (gotcha #12). Train/test splits use calendar-month boundaries.

### Gate 2 binding criteria (THREE conditions, all must hold)

**Criterion 1 (council-aligned amendment):** Fine-tuned CNN balanced accuracy ≥ flat-LR balanced accuracy + **0.5pp** on **15+/24** symbols at H500, separately on Feb AND Mar (no adjudication, no averaging, no "close enough" — same AND-rule as Gate 1).

**Criterion 2 (CI-aware regression check, council-5 required edit #2):** Per-symbol bootstrap 95% CI on fine-tuned CNN's balanced accuracy must NOT regress below the Gate 1 frozen-encoder LR's per-symbol bootstrap CI lower bound by more than `max(1.0pp, 1× CI half-width)` on any single symbol AND the point-estimate must not regress ≥1.0pp. Both Gate 1 baseline and Gate 2 candidate use the **same 1000-resample bootstrap protocol** (per `concepts/bootstrap-methodology.md`). Catches catastrophic forgetting at the freeze→unfreeze transition.

**Criterion 3 (NEW — council-5 required edit #4):** Fine-tuned CNN must beat **frozen-encoder LR** (the Gate 1 winner, NOT just flat-LR) by ≥ **0.3pp** balanced accuracy on **13+/24** symbols at H500, on BOTH Feb and Mar independently. **This is the test of fine-tuning's marginal contribution.** Without it, Gate 2 is a guaranteed pass — frozen-encoder LR already beats flat-LR by +1.9–2.3pp at Gate 1, and any non-broken fine-tune inherits that margin against flat-LR for free. Criterion 1 falsifies "the encoder is useful" (already established); Criterion 3 falsifies "fine-tuning adds marginal value."

The 0.3pp / 13+/24 numbers: 0.3pp is one CI half-width (smallest detectable effect at our sample size); 13/24 = 54.2% (slightly above majority — a real but achievable bar).

**Both Feb AND Mar must satisfy ALL THREE criteria. Failure on any criterion or any month = Gate 2 FAILS. No adjudication.**

### Comparator measurement protocol (council-6 Q6 #5)

Both the "frozen-encoder LR" comparator (Criterion 2 + Criterion 3) and the "flat-LR" comparator (Criterion 1) MUST use sklearn `LogisticRegression` with `C ∈ {0.001, 0.01, 0.1}` cross-validated, EXACTLY as Gate 1 used. Do NOT use the PyTorch fine-tune heads as the encoder-LR comparator — that compares apples (sklearn LR + C-search) to oranges (PyTorch trained-from-scratch heads at fixed lr/schedule). The fine-tuned CNN's encoder is forward-passed to extract 256-dim embeddings, then a fresh sklearn LR is trained on those embeddings against the same labels.

### Trial-count log (council-5 required edit #3, anti-amnesia)

Gate 2 writeup MUST publish a trial-count log of every prior evaluation of `runs/step3-r2/encoder-best.pt` against Feb+Mar:
1. Gate 1 H500 frozen-encoder LR (binding) — `step3-run-2-gate1-pass.md`
2. Gate 3 informational AVAX bootstrap (4 cells) — `step5-gate3-triage.md`
3. Cluster cohesion 6-anchor probe (Feb only) — `step5-cluster-cohesion.md`
4. Surrogate sweep 5 symbols × 2 months × 2 horizons (20 cells) — `step5-surrogate-sweep.md`
5. Gate 2 fine-tuned CNN (this step)

Note in the writeup that Gate 2 pass/fail must be interpreted in the light of these prior evaluations. NO retroactive Bonferroni inflation — anti-amnesia hygiene only.

## Monitoring during training

### Every epoch

- Train loss per horizon (BCE, label-smoothed)
- Val loss per horizon (held-out 10% of training distribution, NOT Feb/Mar — that's reserved for final eval)
- Embedding std on a fixed val batch (collapse alarm < 0.05; abort if held)
- Effective rank of 256×B embedding matrix (alarm < 30)
- **CKA between frozen-checkpoint embeddings and live-encoder embeddings on a fixed 1024-window val batch** (council-6 Q3 leading indicator). Alarm if CKA < 0.5 by epoch 12; ABORT if CKA < 0.3 at any epoch ≥ 8. The single most useful in-training signal for catastrophic forgetting / MEM-geometry drift.

### Every 5 epochs

- Linear probe on frozen-snapshot embeddings against H500 Feb (lagging indicator of forgetting; the per-symbol regression check is computed end-of-training only)
- Hour-of-day probe (must stay <10%; ABORT at >0.12 at any 5-epoch checkpoint)

### Numeric abort criteria (council-6 Q6 #2 required edit)

**ABORT and stop training if any of:**
- End of epoch 3: H500 val BCE > training-init BCE → heads aren't learning
- End of epoch 5: H500 val BCE has not dropped below `0.95 × initial_random_BCE` → linear-probe warmup failed
- Any epoch: embedding std drops below 0.05 → collapse
- Any epoch ≥ 8: CKA-vs-frozen drops below 0.3 → catastrophic geometry drift
- Any epoch after 8: H100 val balanced accuracy drops below 0.50 → shared trunk anti-fitting H100 (council-6 Q4 replacement for original "H100 val loss > 1.5× train loss")
- Any 5-epoch checkpoint: hour-of-day probe > 0.12 → session leakage re-emergence

## Failure modes guarded

1. **Catastrophic forgetting / MEM-geometry drift.** Mitigations: 5-epoch frozen warmup + 10× lower lr + grad clip 1.0 + CKA-vs-frozen monitor (leading indicator) + per-symbol regression check (Criterion 2, lagging).
2. **H500 over-fit at the cost of H100/H50.** Council-6 Q4 reframed: tolerable as long as H100 val balanced acc stays above 0.50. Monitor + abort at <0.50 after epoch 8.
3. **Hour-of-day re-emergence.** Hour probe every 5 epochs; abort at >0.12.
4. **Class-imbalance label exploit.** Label smoothing partially mitigates. Report per-symbol per-horizon class priors in eval output (already in `avax_gate3_probe.py` output schema — reuse).

## Deliverables

1. **Code:**
   - `tape/finetune.py` — `DirectionHead` (shared 256→64 trunk + 4 separate Linear(64→1)), `FineTunedModel` wrapper (encoder + heads), weighted-BCE loss with label smoothing, freeze/unfreeze utility, **CKA-vs-frozen-checkpoint monitor utility** (per council-6 Q3).
   - `scripts/run_finetune.py` — CLI with `--checkpoint`, `--epochs 20`, `--frozen-epochs 5`, `--batch-size 256`, `--lr-frozen 1e-3`, `--lr-unfrozen 5e-5`, `--out-dir`, `--seed`, `--train-end-date 2026-02-01`. Writes per-epoch monitoring to `training-log.jsonl`. Writes best-val-BCE checkpoint to `finetuned-best.pt`.
   - `scripts/run_gate2_eval.py` — runs THREE comparators on Feb+Mar held-out: (a) flat-LR, (b) frozen-encoder LR (sklearn with C-search, identical to Gate 1 protocol), (c) fine-tuned CNN. All three with per-symbol bootstrap 95% CI (1000 resamples). Writes `gate2-eval.json`.
2. **Tests:**
   - `tests/test_finetune.py` — heads forward shape, BCE numerics, freeze toggles `requires_grad`, label-smoothing ε ranges, walk-forward fold construction, CKA computation correctness, abort-criteria triggers.
3. **Writeup:** `docs/experiments/step4-gate2-finetune.md` — config, training trajectory (per-epoch loss + CKA + embedding std + effective rank), per-symbol Gate 2 verdict (all 3 criteria, both months), trial-count log per the anti-amnesia clause.
4. **Spec patch (mechanical alignment, NOT new binding-gate amendment):** update spec lines 273–278 + 347–351 to reflect H500-primary loss weights, 15+/24 count, three Gate 2 criteria. Cite this plan as the rationale.

## Dispatch order (post-council)

1. ✅ Council review parallel — DONE 2026-04-24 (council-5 + council-6 reviews committed; this plan ratified v2 with all required edits applied).
2. **Now:** dispatch builder-8 to implement `tape/finetune.py`, `scripts/run_finetune.py`, `scripts/run_gate2_eval.py`, `tests/test_finetune.py`.
3. reviewer-10 reviews against this plan.
4. Run on M4 Pro MPS (~3.5h). Monitor abort criteria from epoch 0.
5. Eval on Feb + Mar held-out using `run_gate2_eval.py`, write up to `step4-gate2-finetune.md`, commit, update state.md, mechanical-align spec sections.

## Open questions resolved

| Open question | Resolution | Source |
|---|---|---|
| 1. Loss-weight schedule | **Pre-committed: `0.10/0.20/0.20/0.50` (clean swap, Option A)** | council-6 Q1, council-5 Q1 |
| 2. Freeze duration | **5 epochs + abort criterion (H500 val BCE < 0.95× random-init by end of epoch 5)** | council-6 Q2 |
| 3. Per-symbol regression threshold | **`max(1.0pp, 1× bootstrap CI half-width)` on CI lower bound, AND ≥1.0pp on point estimate** | council-5 Q2 / required edit #2 |
| 4. Eval data overlap | **Trial-count log in writeup (no Bonferroni). Discriminating power added via Criterion 3 (fine-tuned CNN vs frozen-encoder LR).** | council-5 Q3 / required edits #3 + #4 |

All required edits from both councils applied. Plan is ratified for implementation.

# Step 4 Plan — Fine-Tuning + Gate 2

**Date:** 2026-04-24
**Author:** lead-0 (auto mode)
**Reviewers (pre-launch):** council-5 (skeptic), council-6 (DL researcher / primary architect)
**Implementer:** builder-8
**Status:** DRAFT — awaiting council review

## Objective

Pass Gate 2 with a fine-tuned encoder + direction heads on the same matched-density Feb+Mar 2026 H500 held-out protocol used for Gate 1. Threshold: balanced accuracy ≥ +0.5pp over flat-feature LR on **15+/24 symbols** (AVAX excluded), per Gate 2's amendment-aligned definition.

## Pre-flight: Gate 2 spec drift to align

The 2026-04-24 spec amendment v2 fixed Gate 1, Gate 3, Gate 4 horizon and held-out language but did NOT touch Gate 2 (line 347–351) or the Fine-Tuning section (line 271–278). Three drift points need consistency-aligned fixes — these are NOT new binding-gate amendments under the amendment-budget clause; they are mechanical alignment with the already-ratified amendment.

| Drift point | Pre-amendment language | Aligned language |
|---|---|---|
| Line 349 "primary horizon" | unspecified, implicitly H100 | **H500** (per amendment horizon-selection rule) |
| Line 349 "15+ symbols" | implicitly 15+/25 | **15+/24** (AVAX excluded) |
| Line 277 loss weights 0.10/0.20/0.50/0.20 | H100 primary, H500 de-weighted | **0.10/0.20/0.20/0.50** (H500 primary, H100 ancillary) |

The third drift is the substantive one — loss weighting affects what the encoder optimizes for during fine-tuning. Council-6 should validate the rebalance.

## Architecture

Per spec line 273:
```
Encoder (frozen → unfrozen)  →  256-dim global embedding
                                       ↓
                                Linear(256 → 64)  + ReLU
                                       ↓
                                4 × Linear(64 → 1) + sigmoid
                                       ↓
                                BCE loss per horizon
                                       ↓
                                weighted sum (0.10 H10, 0.20 H50, 0.20 H100, 0.50 H500)
```

Direction heads are 4 separate `Linear(64 → 1)` (not a shared 4-output Linear) per spec — each head can learn its own bias/scale. Total head params: `(256·64 + 64) + 4·(64+1) = 16,704` extra params.

## Training schedule (per spec line 271–278)

| Phase | Epochs | Encoder | LR | Loss |
|---|---|---|---|---|
| Linear-probe warmup | 5 | **frozen** | 1e-3 (heads only) | weighted BCE |
| Joint fine-tune | 15 | **unfrozen** | **5e-5** (10× lower than pretraining max_lr=1e-3) | weighted BCE |

Total: 20 epochs. Estimated ~3.5h on M4 Pro MPS at batch 256 (similar to pretraining at 5h 17m / 30 epochs, scaled down).

Optimizer: AdamW. Scheduler: OneCycleLR with `max_lr=5e-5`, `pct_start=0.05` for the unfrozen phase (small warmup since lr is already small). Gradient clipping `max_norm=1.0` (same as pretraining anti-collapse).

Label smoothing: ε = 0.10 / 0.08 / 0.05 / 0.05 for H10/H50/H100/H500 (per spec line 276 — H500 gets the lowest smoothing because it's primary AND because long-horizon labels have the highest noise floor, so we trust them most directly).

## Walk-forward evaluation

Per amended Gate 1: train on Oct 16 – Jan 31, evaluate on Feb 2026 AND Mar 2026 independently at H500.

Gate 2 binding criteria (NEW, council-aligned):
1. **Fine-tuned CNN balanced accuracy ≥ flat-LR balanced accuracy + 0.5pp on 15+/24 symbols** at H500, separately on Feb AND Mar (no adjudication, per Gate 1 AND-rule).
2. **Fine-tuned CNN does NOT regress vs frozen-encoder LR Gate 1 result** by more than 1.0pp on any single symbol. This catches catastrophic forgetting at the freeze→unfreeze transition. (Council-5 likely will require this — fine-tuning that LOSES Gate 1 signal is a fail even if it beats flat LR.)

Both Feb and Mar must satisfy both criteria. Failure on either month = Gate 2 FAILS.

## Monitoring during training

Every epoch:
- Train loss per horizon (BCE, label-smoothed)
- Val loss per horizon (held-out 10% of training distribution, NOT Feb/Mar held-out — that's reserved for final eval)
- Embedding std on a fixed val batch (collapse alarm < 0.05 same as pretraining)
- Effective rank of 256×B embedding matrix (alarm < 30)

Every 5 epochs:
- Linear probe on frozen-snapshot embeddings against H500 Feb (early warning of forgetting)
- Hour-of-day probe (must stay <10% — if fine-tuning re-introduces session leakage, abort)

## Failure modes to guard

1. **Catastrophic forgetting at unfreeze.** Encoder representations could drift to over-fit direction labels, losing the Gate 1 signal. Mitigation: 5-epoch linear-probe warmup + 10× lower lr + per-symbol regression check.
2. **H500 over-fit at the cost of H100/H50.** Loss weighting 0.50/0.20 puts dominant gradient on H500. If H100 head completely fails to learn, that's an indication the joint structure isn't pedagogically useful. Monitor per-horizon val loss; if H100 val loss > 1.5× train loss the regularization story is broken.
3. **Hour-of-day re-emergence.** Fine-tuning on direction labels could re-introduce session-of-day shortcuts. Hour probe every 5 epochs is an early-warning.
4. **Class-imbalance label exploit.** Per-symbol label balance varies (Gate 1 measured 0.404–0.554 class priors). Label smoothing partially mitigates but the BCE loss's class-imbalance bias is real. Report per-symbol per-horizon class priors in eval output.

## Deliverables

1. **Code**:
   - `tape/finetune.py` — DirectionHead module (4 × Linear(64→1)), FineTunedModel wrapper (encoder + heads), weighted-BCE loss, freeze/unfreeze utility
   - `scripts/run_finetune.py` — CLI with `--checkpoint`, `--epochs`, `--batch-size`, `--lr-frozen`, `--lr-unfrozen`, `--out-dir`, `--seed`, `--train-end-date 2026-02-01`
2. **Tests**:
   - `tests/test_finetune.py` — heads forward shape, BCE numerics, freeze toggles encoder.requires_grad, label-smoothing ε ranges, walk-forward fold construction
3. **Evaluation**:
   - Reuse `scripts/temporal_stability.py` style for held-out eval. New script `scripts/run_gate2_eval.py` runs fine-tuned encoder + flat-LR baseline on Feb+Mar held-out, reports per-symbol per-horizon balanced accuracy + bootstrap CI.
4. **Writeup**: `docs/experiments/step4-gate2-finetune.md` — config, training trajectory, per-symbol Gate 2 verdict.
5. **Spec patch (mechanical alignment, not amendment)**: update lines 273–278 + 349 to reflect H500-primary loss weights and 15+/24 count. Cite this plan as the rationale.

## Dispatch order

1. **Council review (parallel, this step):**
   - **council-6**: validate loss-weight rebalance (0.10/0.20/0.20/0.50) — is H500-primary the right call for fine-tuning, or is multi-horizon supervision better with weight on H100 for pedagogical-regularization reasons? Validate freeze/unfreeze schedule (5 + 15 epochs).
   - **council-5**: stress-test the Gate 2 criteria. Specifically: (a) is the per-symbol regression check (Criterion 2) a real falsifier or a soft floor that any plausible fine-tune passes? (b) is the loss-weight rebalance pass-chasing or amendment-following?
2. **Apply critiques** (sequential).
3. **builder-8** implements `tape/finetune.py`, `scripts/run_finetune.py`, `scripts/run_gate2_eval.py`, tests.
4. **reviewer-10** reviews against this plan.
5. **Run on M4 Pro MPS** (~3.5h estimate). Monitor first 2 epochs for collapse / forgetting signs. Abort + iterate if probe degrades.
6. **Eval on Feb + Mar held-out**, write up, commit, update state.md.

## Open questions for council

1. **Loss weights (council-6).** Should the H500-primary rebalance be a clean 0.10/0.20/0.20/0.50 swap, or a softer 0.10/0.20/0.30/0.40, or even an annealed schedule (start with H100 weight 0.50 to leverage the easier short-horizon signal, anneal to H500 weight 0.50 over 20 epochs)? Each has a different inductive bias.
2. **Freeze duration (council-6).** Spec says 5 epochs frozen. Given total budget is 20 epochs, this is 25% of training as linear-probe-only. Is that the right ratio, or does the empirical Gate 1 result (encoder-LR already strongly outperforming flat-LR) suggest a shorter warmup is fine?
3. **Per-symbol regression check threshold (council-5).** I propose `<1.0pp` regression on any single symbol vs Gate 1's per-symbol result. Too strict (Gate 2 fine-tuning has a free lunch and we want to catch any backsliding) or too loose (1pp is within Gate 1's measured per-symbol noise on illiquid syms)?
4. **Eval data overlap (council-5).** Gate 2 eval reuses the same Feb+Mar held-out as Gate 1. The encoder has now been "seen" through the lens of two probe tasks (linear probe + fine-tune). Does this constitute multiple-testing on the same data? If yes, what's the corrected pass threshold?

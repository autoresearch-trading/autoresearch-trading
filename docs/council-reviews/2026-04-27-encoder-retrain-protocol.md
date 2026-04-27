# Council Review — Cascade-Onset Encoder Retrain Protocol (Goal-A v2)

**Date:** 2026-04-27 (PM)
**Trigger:** Goal-A v2 kickoff. Flat-LR baseline OOS AUC = 0.778 [0.732, 0.833] at H500
on Apr 14-26 (n=96 cascades). Proposal in state.md: retrain v1's 376K-param dilated CNN
end-to-end with BCE on cascade label, target OOS AUC ≥ 0.85 to flip strategy economics
under maker fees.

**Reviewers:** council-1 (methodology), council-5 (skeptic), council-6 (DL architect).
All ran in parallel.

## Convergent recommendations

1. **Day-clustered bootstrap is mandatory.** Day is the cluster unit. Per-window
   bootstrap understates uncertainty when cascades cluster intraday.
2. **Paired bootstrap** (resample days; recompute encoder AUC and flat-LR AUC on
   the same fold) is the right comparison structure. ~2× power vs separate
   bootstraps when models are correlated, which they will be.
3. **Pre-register the success bar before training.** No sliding goalposts.
4. **Run a low-cost ablation FIRST** before committing to expensive end-to-end
   training. The expensive option is unfalsifiable at this n.

## The load-bearing first experiment (all three agree)

**Frozen random-init encoder linear probe vs flat-LR, under unified 5-fold
day-blocked CV on the merged Apr 1-26 dataset.**

- Forward-pass every cached window through a freshly-initialized
  `TapeEncoder(EncoderConfig())` (no training). Extract 256-dim global embeddings.
- Fit `LogisticRegression(class_weight='balanced', C=1.0)` on the embeddings
  with cascade-H500 label. Same CV folds applied to flat-LR (83-dim) baseline.
- Pooled-across-symbols AUC + day-clustered bootstrap CI for both. Paired
  bootstrap on the delta `(AUC_encoder - AUC_flat_LR)`.
- Cost: < 30 CPU-minutes. No GPU needed.

This single number arbitrates the program's next step.

## Decision tree from the random-init probe (council-6, sharpened)

| Random-init probe result vs flat-LR | Interpretation | Next move |
|---|---|---|
| Probe ≥ flat-LR + 2pp, paired-bootstrap CI of delta excludes 0 | Encoder representation already extractable — NTK regime / lottery-ticket effect | Light end-to-end fine-tune (~$0-2 CPU/MPS, no pretraining needed) |
| Probe ≈ flat-LR (delta CI overlaps 0) | Architecture matches flat-feature signal, but doesn't beat it | Decide based on Tier A/B/C result of fine-tune attempt; pretraining unjustified |
| Probe < flat-LR by > 2pp | Architecture is the bottleneck for linear extraction | Either (a) MEM+SimCLR pretrain → re-probe; (b) end-to-end with strong regularization. Pretraining bet only makes sense if the gap is large. |

## Methodology rules (council-1)

1. **Re-evaluate flat-LR under the SAME 5-fold day-blocked CV** on merged Apr 1-26.
   The published OOS-0.778 number from commit `b0de994` is biased — that was the
   realized draw, not a fresh sample. **Retire 0.778 as the reference baseline**;
   replace with the 5-fold day-blocked CV pooled-OOF AUC computed on the same
   partitions the encoder probe sees.
2. **5-fold day-blocked CV** with 600-event embargo at fold boundaries. Cascades
   propagate at second-to-minute scale; cross-fold leakage is the dominant risk.
   Pool out-of-fold predictions across all 26 days, then bootstrap by day.
3. **LOOCV is wrong.** 26 folds × ~6% base rate × ~30 windows/day-symbol → tiny
   per-fold positive counts, per-fold AUC variance dominates.
4. **Per-symbol AUC reporting:** primary endpoint = pooled cross-symbol AUC.
   Per-symbol cells are exploratory. If reported, use BH-FDR at q=0.10 on
   per-symbol p-values, NOT Bonferroni (which over-corrects given symbol
   correlation).
5. **Trial count:** state.md prohibits hyperparameter search. Run encoder probe
   ONCE with state.md-frozen hyperparameters. If any architecture variant is
   tried and discarded, log it for Deflated AUC accounting (Bailey & Lopez de
   Prado 2014).

## Falsifiability re-frame (council-5)

The "encoder beats 0.778 to win" framing is unfalsifiable at n=96. With holdout
consumed and CI half-width ~5pp, AUC 0.79–0.85 is a coin flip dressed as a
result. Reframe:

> **The deliverable is "isolate whether the encoder carries cascade signal beyond
> hand features" — NOT "lift OOS AUC above 0.85."** Strategy economics flip is a
> separate question requiring fresh data accrual.

**Three-tier pre-registered outcome (encoder retrain phase, conditional on
positive random-init probe):**

| Tier | Criteria | Action |
|---|---|---|
| **A** (publishable) | Encoder OOS AUC lower bound (day-clustered, 1000 reps) > 0.833 AND > flat-LR by ≥ 3pp on ≥ 3 of {SUI, AVAX, PENGU, XRP} AND ≥ 2 NEW symbols cross AUC > 0.65 | Proceed to top-1% precision evaluation; consider strategy retest |
| **B** (interesting, not bankable) | Point estimate 0.81–0.85, paired-bootstrap delta CI overlaps 0, OR new-symbol gain count < 2 | File as "encoder matches flat-LR, no causal claim." Stop the program. |
| **C** (kill) | Point ≤ 0.78 OR CI lower bound ≤ 0.73 OR per-symbol gain concentrates entirely on the same SUI/AVAX/PENGU/XRP that flat-LR already captures | Encoder adds nothing. Ship flat-LR or stop. |

## Architecture / regularization (council-6, conditional on going end-to-end)

**Recipe (only if probe greenlights end-to-end):**
- `pos_weight = n_neg / n_pos` in `BCEWithLogitsLoss` (~15.7× at base rate 6%);
  prefer over focal loss (γ adds untunable hyperparam at n=169).
- AdamW `weight_decay = 5e-4` (raise from v1's 1e-4 — labeled regime is ~10K
  windows, not 3.5M).
- Dropout 0.1 after blocks 3-6 (mid-stack, NOT just first two as in v1).
- `max_lr = 3e-4` (drop from v1's 1e-3).
- OneCycleLR shape preserved.
- Day-clustered 5-fold val on Apr 1-13 ONLY for early stopping (do NOT touch
  Apr 14-26 — the consumed-but-not-fully-leaked period). Patience 3 epochs on
  val AUC, NOT val BCE.
- Embedding strategy: keep `concat[GAP, last_position]` — cascade onset is a
  right-edge label, structurally aligned with last_position channel.
- Embedding-collapse guard: std<0.05, effective-rank<30 aborts.
- Cap: 3 seeds, 1 H100-day max. Report median seed, not best.

**The MEM+SimCLR pretrain leg is NOT justified up-front** — random-init probe
result decides.

## Open methodology question (deferred until probe runs)

If random-init probe ≈ flat-LR, the second-most-valuable experiment is a
MEM-only pretrain (no SimCLR) → re-probe. SimCLR's instance-discrimination
objective doesn't obviously align with cascade detection, while MEM's
reconstruction objective at least biases the encoder toward feature-fidelity.
The ablation budget should reflect this if we get there.

## Pre-launch checklist for builder-8

- [ ] Implement random-init encoder forward-pass over all merged Apr 1-26 cache shards
- [ ] Implement 5-fold day-blocked CV partition (k=5, ~5 days/block, ordered by date)
- [ ] Implement 600-event embargo at fold boundaries
- [ ] Implement paired day-clustered bootstrap (1000 iters; resample 26 days with replacement)
- [ ] Re-evaluate flat-LR (83-dim) under the SAME folds — both numbers must come from one run
- [ ] Pre-commit the protocol (this file + experiment plan) BEFORE running
- [ ] Single-seed run; log all unrealized variants for trial-count accounting
- [ ] Output: pooled AUC for both, per-symbol AUC table (BH-FDR adjusted), paired delta CI

**Hard constraint (anti-amnesia):** Apr 14-26 is in the merged dataset; the holdout
has been deliberately consumed (gotcha #17). The CV protocol is the ONLY defensible
evaluation now — no separate "OOS test" remains.

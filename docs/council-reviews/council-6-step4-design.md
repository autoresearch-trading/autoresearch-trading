# Council-6 Review — Step 4 Fine-Tuning Plan

**Date:** 2026-04-24
**Reviewer:** council-6 (DL researcher / SSL pretraining architect)
**Plan:** `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` (commit `ae7a970`)
**Encoder:** `runs/step3-r2/encoder-best.pt` (376K params, epoch-6 MEM-min, Gate 1 PASS at +1.9–2.3pp over PCA on Feb+Mar H500)
**Verdict:** AMEND (2 substantive changes) then ship.

## Q1: Loss weights — pick Option A (clean swap to `0.10/0.20/0.20/0.50`)

The annealed schedule (Option C) is over-engineered. Curriculum-by-horizon assumes easier short-horizon supervision provides "scaffolding" gradients that a harder long-horizon objective benefits from later — that is an LM/RL story, not a multi-horizon binary-classification one. With four sigmoid heads sharing a 256→64 trunk, all four gradients flow through the SAME shared parameters every step from epoch 0; the encoder is already getting H100/H500 co-supervision regardless of weight ratio. Annealing adds two new hyperparameters (anneal start/end), one new abort surface, and zero falsifiable benefit.

The softer rebalance (Option B, `0.10/0.20/0.30/0.40`) is also weakly rejected: the amendment explicitly designated H500 as primary via the horizon-selection rule, and council-5's finding showed H100 is at noise floor for this encoder on this data. Putting 0.30 on H100 invests gradient capacity in a head that has no measurable signal — that's not regularization, that's wasted gradient.

Multi-horizon supervision regularizes via shared-trunk co-adaptation + gradient noise from per-horizon disagreement; both mechanisms preserved at any non-degenerate weight vector. `0.10/0.20/0.20/0.50` satisfies both.

## Q2: Freeze duration — keep 5 epochs, add a warm-up abort

The empirical Gate 1 argument cuts both ways: the heads CAN probe usefully — but the linear probe used sklearn LogisticRegression with C-tuning, NOT a randomly-initialized PyTorch `Linear(256→64) + ReLU → Linear(64→1)` trained from scratch. The fine-tune heads are different objects (5× more params, ReLU non-linearity, no closed-form regularization, BCE loss, label smoothing). Five epochs gives those PyTorch heads time to fit before the encoder's 376K params start updating around them.

3 epochs is too short (~4,800 steps at batch 256, heads still in linear-tangent regime). 10 epochs wastes half the fine-tune budget on linear-probe-equivalent training.

**Add this:** abort the unfreeze transition if at end-of-epoch-5 the per-horizon val BCE for H500 hasn't dropped below `0.95 × initial_random_BCE`. If heads can't fit a usable boundary on frozen-and-known-good embeddings in 5 epochs, unfreezing the encoder is wrong — that's where a longer warmup or head-architecture rethink is needed, not encoder drift.

## Q3: Catastrophic forgetting — moderate risk; the dominant failure is "drift away from MEM minimum" not classical CF

Classical catastrophic forgetting (Kirkpatrick 2017) is about losing capability on prior task A while learning task B. Here task A (MEM + SimCLR) is gone — there is no MEM/contrastive loss in the fine-tune objective. What we risk is the encoder losing the **inductive geometry** that MEM forced (locally-smooth-in-event-time feature reconstruction) in favor of a direction-discriminative geometry that overfits per-symbol H500 signal.

Three reasons this is moderate (not severe) on this encoder:
1. 376K params + 5e-5 LR + 15 epochs ≈ 11.25K steps × 5e-5 effective updates, gradient-clipped at 1.0 → bounded encoder shift. Empirically on similarly-sized SSL CNNs (wav2vec 2.0 small, BYOL-A), CKA against pretraining checkpoint after fine-tune drops 0.2–0.4 — geometry shifts but doesn't collapse.
2. Gate 1's strongest signals are on illiquid alts (FARTCOIN +0.15, SUI +0.17, PUMP). Those are exactly where H500 label distribution is most class-imbalanced (priors 0.404–0.554), so encoder inductive features will be most stressed by BCE-on-imbalanced-labels. Forgetting will surface there first.
3. Cluster cohesion already showed encoder learned per-symbol features (delta +0.037, symbol-ID 0.934). Fine-tune is unlikely to make this WORSE — encoder already privileges per-symbol structure, which is what direction-prediction-per-symbol wants.

**Live monitoring signal (single most useful):** **CKA between fixed val-batch's frozen-checkpoint embeddings and live-encoder embeddings, every epoch on the same 1024 windows.** Threshold: alarm if CKA < 0.5 mid-fine-tune (epoch 12 of 20), abort if CKA < 0.3. Cheap (one forward pass per epoch, ~5s on MPS), directly measures geometric drift, distinguishes "encoder is learning to discriminate" (CKA high, just rotates) from "encoder is forgetting MEM structure" (CKA decays toward random).

Per-symbol regression-vs-Gate-1 (plan's Criterion 2) is a useful end-of-training check but it's a **lagging** indicator — by the time you measure it, the encoder is already trained. CKA-vs-frozen is the leading indicator.

## Q4: H500 over-fit cost on H100 — tolerable

With weights `0.10/0.20/0.20/0.50`, H100 head's `Linear(64→1)` gets 5× less gradient than H500's. That head will fit less well — its individual val BCE will be higher than under `0.10/0.20/0.50/0.20`. **Mechanically guaranteed and NOT a sign of bad shared representation.**

What WOULD be a sign of bad shared representation: H100 val BCE INCREASING during fine-tune (head unlearning), or H100 val balanced acc dropping below 0.50 (worse than chance). Either means the shared 256→64 trunk is being driven by H500 to encode features anti-correlated with H100.

Plan failure-mode #2's "H100 val loss > 1.5× train loss" threshold is the wrong shape — it catches H100 OVERFIT, not H100 anti-fit. **Replace with: alarm if H100 val balanced accuracy drops below 0.50 at any epoch after epoch 8.**

The H100-degrades-but-stays-above-chance scenario is EXPECTED and FINE — amendment designated H500 as the only binding horizon. We accept this trade.

## Q5: Direction-head architecture — keep shared trunk

Two reasons:
1. **Multi-task pedagogy IS the point of the shared trunk.** Four fully-independent heads are mathematically equivalent to four single-horizon fine-tunes that happen to share an encoder. Shared 256→64 forces encoder to encode horizon-averaged feature representation rather than four per-horizon specialists. That's the regularization Q4 asked about.
2. **Param budget cuts same direction.** Spec: `256·64 + 64 + 4·(64+1) = 16,704` head params on 376K encoder = 4.4% trainable head fraction. Independent-heads alternative: 65,540 params = 17.4%. At 17.4% fraction, heads themselves can over-fit H500 signal on illiquid alts (FARTCOIN, SUI, PUMP have ~3K windows on Feb test) — shared-trunk version is genuinely more conservative.

Independent-heads only right call if horizons require IRRECONCILABLE feature representations. On this data they don't.

## Q6: Missing from plan (5 load-bearing items)

1. **No frozen-checkpoint CKA monitor (per Q3).** Single most useful in-training signal for catastrophic forgetting is missing. Add to monitoring section.
2. **No numeric abort criteria — only "monitor and iterate."** Need explicit thresholds: (a) abort if H500 val BCE > training-init BCE at end of epoch 3 (heads aren't learning); (b) abort if embedding std drops below 0.05 (collapse); (c) abort if hour-of-day probe re-emerges above 0.12 at any 5-epoch checkpoint (session leakage).
3. **No ablation infrastructure.** Gate 2 binary tells you whether fine-tune helped, not WHY. At minimum log frozen-encoder + LR-trained-now (Gate 1 baseline re-measured at fine-tune eval moment) alongside fine-tuned model — separates "fine-tune signal" from "Gate 1 already gave us this." Cost ~5min eval time.
4. **Label smoothing schedule rationale is shaky.** Spec's `0.10/0.08/0.05/0.05` (decreasing with horizon) plan describes as "H500 has lowest smoothing because it's primary AND because long-horizon labels have highest noise floor." Second half is BACKWARD — higher noise floor = LESS confident labels = MORE smoothing, not less. Right read: H500 has BEST signal-to-noise, so we trust those labels MOST → less smoothing. Re-write rationale.
5. **Criterion 2 measurement protocol must use sklearn LR with C-search to match Gate 1.** Otherwise comparing apples (sklearn LR + C-search) to oranges (PyTorch trained-from-scratch heads at fixed lr/schedule). Must nail this down.

## Summary for orchestrator

(1) **Loss weights:** Option A clean swap to `0.10/0.20/0.20/0.50`; annealing adds hyperparameters with no falsifiable benefit and 0.30 on H100 invests gradient on noise-floor head. (2) **Freeze duration:** keep 5 epochs but add end-of-warmup abort (H500 val BCE must drop below 0.95× random-init or abort). (3) **Single design change:** add per-epoch CKA-vs-frozen-checkpoint monitor on fixed 1024-window val batch as leading indicator of catastrophic forgetting (alarm < 0.5 mid-fine-tune, abort < 0.3); plan's per-symbol regression check is lagging only. (4) **AMEND** — incorporate CKA monitor, numeric abort criteria, ablation-vs-frozen-LR baseline; ship after.

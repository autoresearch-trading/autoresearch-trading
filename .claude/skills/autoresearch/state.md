# Research State

## Status (2026-04-27 PM-late — Phase 0 complete, ARCH_BOTTLENECK verdict)

**Phase 0 result (commit `3110abc`).** Random-init `TapeEncoder` linear probe vs
unified-CV flat-LR on merged Apr 1-26 cascade-H500. Flat-LR pooled AUC =
**0.8373 [0.8087, 0.8652]** (retires the prior 0.778 OOS reference; council-1
was right — that number was biased low). Random-init encoder pooled AUC =
**0.6463 [0.5802, 0.7246]** (median seed=1; per-seed range [0.6330, 0.6952]).
Paired delta = **−0.1812 [−0.2594, −0.1063]** — random CNN embeddings LOSE
to 83-dim hand features by ~18pp, CI firmly below zero. n_cascades=169 confirms
holdout-consume integrity. 27s CPU wall-clock.

**Decision tree fires: ARCH_BOTTLENECK.** Hand-engineered features dominate
random CNN embeddings; the architecture cannot linearly extract cascade signal
without training. Per the ratified plan, this routes to one of:
- (3a) MEM-only pretrain → re-probe (~$10 H100-half-day, ~2.5h)
- (3b) End-to-end with strong reg (council-6 recipe: pos_weight, wd=5e-4,
  dropout 0.1 throughout stack, max_lr=3e-4, 3 seeds)

These are mutually exclusive. Council decision pending — see the open question
below.

## Status (2026-04-27 PM — Goal A v2 active)

**Active program: cascade-precursor encoder.** First definitive empirical positive of the program. Flat-LR baseline on 83-dim flat features predicts liquidation-cascade onset (Pacifica `cause` flag) at OOS AUC = 0.778 on Apr 14-26 (n=96 cascades, day-clustered CI [0.732, 0.833]; in-sample AUC 0.815 on Apr 1-13). Distinguishable from shuffled-OOS baseline at H500. Per-symbol concentration: SUI/AVAX/PENGU/XRP carry the signal (mid-cap alts); BTC/HYPE/ETH at chance. Strategy not yet directly tradeable at flat-LR precision (top-1% precision = 25.4% OOS, marginal-long net-negative); encoder retrain is the load-bearing test for whether a learned tape representation lifts AUC into tradeable territory.

**v1 (closed 2026-04-27 AM):** direction-prediction representation learning. +1pp linearly-extractable signal at H500, fee-blocked at every framing tested. Tag `v1-program-closed`. Frozen state in `state-v1-closed.md`.

## Goal-A v2 feasibility chain (8 artifacts, all in `docs/experiments/goal-a-feasibility/`)

| Test | Result | Artifact |
|---|---|---|
| Per-symbol oracle headroom | Direction-prediction fee-blocked under taker | `headroom_table.csv`, accuracy-stress `survivors.md` |
| Maker-mode cost band | Flips alive at maker fee, but slip=0 assumption fragile | `maker_sensitivity.md` |
| Maker adverse-selection sim | Killed by Maker's Dilemma (breakeven=0.311) | `maker_adverse_selection.md` |
| Open-imbalance extreme regime (novelty test #1) | Chance-level in extreme tail; standard OFI captures same signal | `open_imbalance.md` |
| v1 encoder confidence-conditional | Top-quintile at chance; v1 directional kill robust | `encoder_confidence.md` |
| Cascade synthetic-label feasibility | Misframed (volatility ≠ cascades) | `cascade_precursor.md` |
| Cascade real-label LR (in-sample) | AUC=0.815, robust to day-clustering | `cascade_precursor_real.md`, `_robustness.md` |
| Cascade direction LR | Failed (AUC=0.441 < majority); marginal-long net-negative | `cascade_direction.md`, `cascade_marginal_long.md` |
| **Cascade real-label OOS** (Apr 14-26) | **AUC=0.778, generalizes** | `cascade_precursor_oos.md` |

## Next concrete move (REVISED 2026-04-27 PM after council round)

**Phase 0: Random-init encoder linear probe vs unified-CV flat-LR.** CPU-minutes
test that arbitrates whether the encoder-retrain program is worth GPU compute.
Council-1 + council-5 + council-6 converged on this in
`docs/council-reviews/2026-04-27-encoder-retrain-protocol.md`. End-to-end
fine-tune is unfalsifiable at n=96 OOS cascades — must run the cheap probe first.

**Plan:** `docs/experiments/goal-a-v2/2026-04-27-random-init-probe-plan.md`.
Two phases:
- Phase 0a: re-evaluate flat-LR (83-dim) under unified 5-fold day-blocked CV
  on merged Apr 1-26 (retires the 0.778 reference baseline; old OOS number was
  biased by holdout consumption).
- Phase 0b: forward-pass each (200, 17) window through frozen random-init
  `TapeEncoder` → 256-dim global embedding. `LogisticRegression` head. Same
  CV partition. Paired day-clustered bootstrap on the delta.

**Decision tree from Phase 0b** (post-result, see protocol doc):
1. Probe ≥ flat-LR + 2pp, delta CI excludes 0 → light end-to-end fine-tune,
   skip pretraining.
2. Probe ≈ flat-LR (delta CI overlaps 0) → end-to-end fine-tune once at
   council-6 regularization recipe; if it fails Tier A, stop the program.
3. Probe < flat-LR by > 2pp → architecture bottleneck; decide pretrain vs
   end-to-end with reg.

**Three-tier success bar (council-5)** when we get to the encoder-retrain phase:
- Tier A: OOS AUC lower bound (day-clustered) > 0.833 AND > flat-LR by ≥ 3pp
  on ≥ 3 of {SUI, AVAX, PENGU, XRP} AND ≥ 2 NEW symbols cross AUC > 0.65.
- Tier B: point 0.81–0.85, delta CI overlaps 0 — file as "no causal claim", stop.
- Tier C: point ≤ 0.78 OR lower bound ≤ 0.73 OR per-symbol gain entirely on the
  same SUI/AVAX/PENGU/XRP — kill.

**Owner:** dispatch to builder-8 (implement Phase 0a + 0b pipeline), then
reviewer-10, then validator-11 to run + grade.

**Compute:** Phase 0 = < 30 CPU-minutes, no GPU. Encoder retrain compute
budget unlocked only if Phase 0 result is actionable per decision tree.

**Pre-dispatch obligations (binding):**
- Re-evaluate flat-LR (83-dim) under SAME 5-fold day-blocked CV partition
  encoder probe will use. Both numbers come from one run.
- 600-event embargo at fold boundaries.
- 3 random encoder seeds {0, 1, 2}, report MEDIAN.
- BatchNorm gotcha: `model.eval()` + decide between `track_running_stats=False`
  vs single warmup pass; document.
- Per CLAUDE.md gotcha #17: April 14-26 holdout consumed — Phase 0 evaluates
  ONLY via the unified CV protocol. No "OOS test" remains.

**v2 architecture decisions (deferred, not yet ratified):**
- v1's CNN code is in `tape/model.py` (`TapeEncoder`, `EncoderConfig`),
  `tape/finetune.py` (`FineTunedModel`, `DirectionHead`), `tape/pretrain.py`
  (MEM+SimCLR loop). Cascade-prediction head needs a `CascadeHead` analog
  (single-output BCE) — implement only when Phase 0 greenlights it.

## Stack — extended

- `tape/` package — data pipeline, OB alignment, dedup, 17-feature builder, sampler, walk-forward splits. 289+ tests.
- **Cache: 4453 `.npz` shards** at `data/cache/` (rebuilt 2026-04-27 with `--consume-holdout`), 25 symbols × Oct 16 → Apr 26.
- Knowledge base at `docs/knowledge/INDEX.md` — frozen v1 findings, cite for context.
- Goal-A v2 feasibility chain at `docs/experiments/goal-a-feasibility/` — 8 artifacts above plus per-window parquets (gitignored).
- Pacifica fee schedule research: `docs/research/pacifica-fee-schedule-2026-04-27.md` (4bp taker, +1.5bp maker, post-only TIF=ALO/TOB available).

## Council round 2 (2026-04-27 PM-late) — Three-voice synthesis

Council-4, council-5, council-6 weighed in on pretrain vs end-to-end. They
ANSWER DIFFERENT QUESTIONS, layered rather than conflicting:

- **Council-5: STOP.** Both pretrain-first and end-to-end-first are
  unfalsifiable at n=169 with consumed holdout, AND the **Maker's Dilemma is
  the actual blocker** (E[realized|filled] = -7.89bp; encoder cannot solve
  adverse selection regardless of AUC).
- **Council-6: Run 5b first** — non-linear adapter on frozen random-init
  encoder. $0, < 1 CPU-hour. Cleanly arbitrates whether the 18pp gap is a
  linearity artifact (cheap fix) or manifold deficiency (justifies
  cascade-aware MEM).
- **Council-4 (phenomenology):** the cascade signature is a "liquidity-depletion
  ramp punctuated by a Composite Operator exit" — SLOW summary-statistic
  regime change with FAST trigger. Hand features (mean/last/max) capture this
  trivially; CNN claw-back expected at 3-8pp at most. **MEM is genuinely
  hobbled** (excludes kyle_lambda/cum_ofi_5/delta_imbalance_L1 — the
  cascade-relevant axes); **SimCLR ±25 jitter is actively harmful** for
  right-edge cascade detection. Recommends cascade-aware pretexts: re-include
  OFI features in MEM, drop SimCLR, custom "distance-to-climax regression".

Synthesis: `docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md`.

## Next concrete move (revised 2026-04-27 PM-late)

**Phase 1: 5b adapter test** (council-6's recipe). $0, < 1 CPU-hour.

- Reuse the random-init `TapeEncoder` embeddings from Phase 0 (3 seeds).
- Train a small adapter head: `Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)`,
  ~16K params, BCEWithLogitsLoss(pos_weight=15.7), AdamW(lr=1e-3, wd=1e-3),
  50 epochs, batch 256, early stop on pooled val AUC patience 5.
- Same 5-fold day-blocked CV + 600-event embargo as Phase 0.
- Paired day-clustered bootstrap on (adapter − flat-LR) delta.

**Pre-registered outcomes (council-5 falsifiability):**
- **GREENLIGHT** Tier A: adapter ≥ flat-LR + 0.02 AND paired-delta CI excludes 0
  → cascade-aware MEM justified, but ONLY if Goal-A v3 has a TAKER-side or
  non-Maker-fee downstream framing the user wants to pursue.
- **MATCHED** Tier B: adapter ≈ flat-LR (delta CI overlaps 0) → manifold
  neutral. STOP encoder retrain.
- **KILL** Tier C: adapter < flat-LR by > 0.02 → STOP and pivot.

**Strategy-economics audit (in parallel):** validate council-5's "encoder
cannot solve Maker's Dilemma" claim. Revisit
`docs/experiments/goal-a-feasibility/maker_adverse_selection.md`. If 5b's
pass case lands, the user makes the research-direction call (Goal-A v3
TAKER-side framing vs wind down).

**Owner:** builder-8 implements; reviewer-10 audits; validator-11 grades.

## Open question (2026-04-27 PM-late) — Pretrain vs End-to-End given 18pp gap (DEFERRED — see council round 2 synthesis above)

ARCH_BOTTLENECK fired with delta = −0.1812. The plan's decision tree leaves
"pretrain vs end-to-end" as a council decision parameterized on gap size. 18pp
is LARGE (the plan's example was "say <0.70" → "the pretraining bet only makes
sense if the linear-extractability gap is large"). Empirically the gap meets
that threshold.

The case for pretrain-first (MEM+SimCLR, then re-probe):
- Pretraining is a representation question, not a fine-tuning question. If
  random embeddings can't extract cascade signal, MEM+SimCLR's pretext task
  may build a basis where they can — the next probe re-arbitrates.
- Risk: pretraining objectives weren't designed for cascade detection.
  cum_ofi_5 / delta_imbalance_L1 are EXCLUDED from MEM targets (they're
  trivially copyable from neighbors), but those are the most cascade-relevant
  features (council-6 noted this).
- Cost: ~$10 H100-half-day, ~2.5h.

The case for end-to-end-with-reg first:
- 169 positives is a small-data regime, but BCE end-to-end at council-6's
  recipe (pos_weight, wd=5e-4, dropout, max_lr=3e-4, 3 seeds) is also CPU-feasible.
- Direct optimization on the cascade label may extract signal that MEM/SimCLR
  pretexts wouldn't.
- Risk: 376K params on 169 positives is overfit-prone. If end-to-end fails
  Tier-A, we still don't know whether pretraining would have helped — sequential
  ablation cost.
- Cost: < $5 if MPS-feasible.

Council needs to weigh in (council-5, council-6, possibly council-4).

## Audit trail

- v1 program end-state: `docs/experiments/step4-program-end-state.md` (commit `30bbc3b`). Closure tag: `v1-program-closed`.
- v1 frozen state: `.claude/skills/autoresearch/state-v1-closed.md`.
- v1 spec (do not edit): `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`.
- Goal-A v2 feasibility commits: `4ae3102`, `b4132cd`, `9509d1a`, `2c7dee7`, `1fa7063`, `e5ae29c`, `b80fa2e`, `f3a7b49`, `60f06d9`, `e2715ec`, `7231019`, `6113de9`, `b0de994`.
- Holdout consumption commit: pending (cache rebuild + `--consume-holdout` flag in this session).
- Goal-A v2 Phase 0 commits: `64e3587` (impl), `694d14c` (cleanups), `3110abc` (run + result).

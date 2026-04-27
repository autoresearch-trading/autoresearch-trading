# Research State

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

## Audit trail

- v1 program end-state: `docs/experiments/step4-program-end-state.md` (commit `30bbc3b`). Closure tag: `v1-program-closed`.
- v1 frozen state: `.claude/skills/autoresearch/state-v1-closed.md`.
- v1 spec (do not edit): `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`.
- Goal-A v2 feasibility commits: `4ae3102`, `b4132cd`, `9509d1a`, `2c7dee7`, `1fa7063`, `e5ae29c`, `b80fa2e`, `f3a7b49`, `60f06d9`, `e2715ec`, `7231019`, `6113de9`, `b0de994`.
- Holdout consumption commit: pending (cache rebuild + `--consume-holdout` flag in this session).

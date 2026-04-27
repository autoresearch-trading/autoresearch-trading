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

## Next concrete move

**Encoder retrain on cascade-onset target.** v1's CNN architecture (376K param dilated CNN, RF=253), pretraining objective and head swapped: end-to-end binary cross-entropy on the merged Apr 1-26 cascade label, walk-forward day-blocked CV. Goal: lift OOS AUC from 0.778 (flat-LR baseline) to 0.85+, which would push top-1% precision into tradeable range under maker fee economics (1.5bp/side per researcher-14's Pacifica fee-schedule research).

**Owner:** dispatch to builder-8 + runpod-7 (H100 likely needed). ~1-2 days work, ~$10-20 compute.

**Pre-dispatch obligations:**
- The cascade label has only ~169 events at H500 across the full Apr 1-26 dataset. Small-data regime — need careful regularization, no hyperparameter search, train/val splits day-blocked not random.
- Anti-amnesia: April 14-26 holdout has been DELIBERATELY consumed (2026-04-27, commit `b0de994` and the cache rebuild). No untouched cascade-labeled holdout remains. Future evaluation requires either (a) waiting for new data accrual past Apr 27, or (b) splitting the merged dataset.
- v1's CNN code is in `tape/encoder.py` + `tape/pretrain.py`. The cascade-prediction head needs to be added. Re-use BatchNorm + dilated conv stack; replace MEM decoder + SimCLR projection head with a binary classification head over global-pool embedding.

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

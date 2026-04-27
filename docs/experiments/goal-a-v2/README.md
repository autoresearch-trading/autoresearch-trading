# Goal-A v2 — Cascade-Onset Encoder Retrain Program

This directory holds the artifacts for Goal-A v2 (cascade-precursor encoder
retrain).  v1 (direction-prediction representation learning) closed at tag
`v1-program-closed`; see `docs/experiments/step4-program-end-state.md`.

## Plan documents

* **Phase 0 ratified plan:** `2026-04-27-random-init-probe-plan.md`
* **Council protocol:** `../../council-reviews/2026-04-27-encoder-retrain-protocol.md`

## Phase 0 — Random-init encoder linear probe

Single decision-arbitrating experiment: does a frozen, randomly-initialized
`TapeEncoder` carry cascade-onset signal beyond the 83-dim hand features?

* **Script:** `scripts/random_init_probe.py`
* **Tests:** `tests/scripts/test_random_init_probe.py`
* **Protocol:** 5-fold day-blocked CV on merged Apr 1-26 dataset (holdout
  consumed on 2026-04-27 commit `b0de994`), 600-event embargo at fold
  boundaries, paired day-clustered bootstrap on the encoder − flat-LR delta
  (1000 reps, resample 26 days with replacement).
* **Decision tree:** see plan §Decision Logic — three-tier outcome
  `GREENLIGHT_FINETUNE` / `MATCHED_FLAT` / `ARCH_BOTTLENECK`.

### Invocation

```bash
# Pre-flight smoke (3 symbols × 5 days, ~5 s)
uv run python scripts/random_init_probe.py --smoke

# Full run (25 symbols × 26 days, < 30 min CPU)
uv run python scripts/random_init_probe.py \
    --cache data/cache \
    --out-dir docs/experiments/goal-a-v2
```

### Artifacts produced

* `random_init_probe_table.csv` — per-(symbol × fold × model) AUC.
* `random_init_probe.md` — markdown report with pooled AUC, paired delta CI,
  per-symbol BH-FDR table, and decision-tier verdict.
* `random_init_probe_per_window.parquet` — per-window OOF predictions
  (gitignored).

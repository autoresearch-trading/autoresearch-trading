# Experiment: Random-Init Encoder Linear Probe vs Flat-LR (Goal-A v2, Phase 0)

## Hypothesis

A 256-dim global embedding from a *frozen, randomly-initialized* `TapeEncoder`
either (a) carries cascade-onset signal beyond what 83-dim hand features extract,
or (b) does not. The result arbitrates whether the cascade-onset encoder retrain
program is worth GPU compute. CPU-minutes test, no training involved.

**Why this experiment exists:** Council-6 + council-5 agree the proposed
end-to-end fine-tune is unfalsifiable at n=96 OOS cascades; running the linear
probe first is the load-bearing decision and costs CPU-minutes.

## Scoring

- **Primary endpoint:** pooled cross-symbol AUC at H500 under 5-fold day-blocked CV.
- **Secondary endpoints:** per-symbol AUC (BH-FDR adjusted), pooled top-1% precision.
- **Comparison:** paired day-clustered bootstrap delta `(AUC_encoder - AUC_flat_LR)`,
  1000 iters, resample 26 days with replacement.
- **Decision rule:** see the 3-row table in
  `docs/council-reviews/2026-04-27-encoder-retrain-protocol.md` §Decision tree.

## Phases

### Phase 0a: Re-evaluate flat-LR baseline under unified CV partitions (control)

Council-1 obligation: the published OOS-AUC=0.778 from `b0de994` is biased
because it was a single-shot evaluation on the held-out segment. Replace with
a proper 5-fold day-blocked CV pooled-OOF AUC computed on the merged Apr 1-26
dataset.

| Run | Config delta vs `cascade_precursor_probe.py` | Purpose |
|---|---|---|
| 1 | 5-fold day-blocked CV on merged Apr 1-26 (`LogisticRegression(C=1.0, class_weight='balanced')` on FLAT_DIM=83 features) | Establish unified flat-LR baseline that encoder probe will be compared against. Retire the 0.778 reference. |

### Phase 0b: Random-init encoder linear probe

Inherits the CV partition from Phase 0a. Feature swap only — same labels, same
folds, same bootstrap protocol.

| Run | Config delta | Purpose |
|---|---|---|
| 1 | Frozen `TapeEncoder(EncoderConfig())` random init, forward-pass each (200, 17) window → 256-dim global emb. `LogisticRegression(C=1.0, class_weight='balanced')` head. Same 5-fold day-blocked CV. | Test whether random-init representation has cascade signal that flat features miss. |

## Decision Logic

After Phase 0b completes, compare to Phase 0a baseline:

1. **Encoder ≥ flat-LR + 2pp AND paired-bootstrap delta CI excludes 0**:
   → Proceed to *light* end-to-end fine-tune. Skip MEM+SimCLR pretraining.
   Random-init probe already extracted the signal; gradient flow may give
   modest additional lift.
2. **Encoder ≈ flat-LR (delta CI overlaps 0)**:
   → The architecture matches the flat-feature signal but doesn't beat it.
   Consider end-to-end fine-tune ONCE (single seed, frozen hyperparams) to
   probe capacity gain. If that fails the Tier-A bar, stop the program.
3. **Encoder < flat-LR by > 2pp**:
   → Architecture is the bottleneck. Decide:
     - (3a) MEM-only pretrain → re-probe. Cost: ~$10 H100-half-day.
     - (3b) End-to-end with strong reg (council-6 recipe).
   These are mutually exclusive — pick one based on the size of the gap.

## Budget

- Compute: < 30 CPU-minutes (encoder forward pass on ~10K windows; sklearn LR fit).
- Wall-clock: < 1h including pipeline implementation + run + analysis.
- No GPU.
- 1 seed (state.md prohibits hyperparameter search). Encoder weights are random
  init only — but to be defensible, run with 3 random encoder seeds and report
  median + min-max range to bound the random-init variance.

## Output Contract

Script writes:
1. `docs/experiments/goal-a-v2/random_init_probe_table.csv` — per-symbol per-fold
   AUC for both flat-LR and encoder-LR, plus pooled.
2. `docs/experiments/goal-a-v2/random_init_probe.md` — markdown report:
   pooled AUC ± CI for flat-LR baseline (re-evaluated, retire 0.778),
   pooled AUC ± CI for encoder probe, paired bootstrap delta CI, per-symbol
   AUC table with BH-FDR adjusted q-values, decision tier.
3. `docs/experiments/goal-a-v2/random_init_probe_per_window.parquet` — per-window
   predictions (gitignored, for downstream analysis).

Stdout summary: pooled AUC values, paired delta point + CI, decision-tier verdict.

## Gotchas

- **Random-init variance.** A single random seed of `TapeEncoder` could give
  unlucky/lucky embeddings. Use 3 seeds {0, 1, 2}; report MEDIAN as the binding
  number, with min-max as a sanity range. Council-5 cap of 3 seeds applies.
- **BatchNorm at inference.** Per CLAUDE.md gotcha #18: `model.eval()` for the
  full forward pass; running stats matter even with random init because BN1d
  is the first layer. Set `track_running_stats=False` OR run a single warmup
  pass on a representative batch to populate running stats. Document the choice.
- **Embedding extraction.** Per `tape/model.py`: `forward(x)` returns
  `(per_pos, global_emb)`. We use `global_emb` (256-dim). Don't accidentally
  use `per_pos`.
- **Cascade label parity.** Use `_real_cascade_label_with_event_ts` exactly as
  in `cascade_precursor_probe.py`. Do not re-derive — bug surface.
- **APRIL_HELDOUT_START guard.** The merged Apr 1-26 dataset means the
  function `_load_liq_ts_for_symbol_date` will refuse to load Apr 14+ unless
  the `consume_holdout=True` path is taken. Match the same convention used in
  `cascade_precursor_oos.md` (commit `b0de994`).
- **Day-clustered bootstrap with k=5 folds.** When pooling out-of-fold
  predictions and bootstrapping by day, ensure each day appears exactly once
  in the OOF prediction array. The bootstrap then resamples 26 days with
  replacement on the OOF predictions.
- **Embargo enforcement.** 600-event embargo at fold boundaries: when training
  fold k, drop the LAST 600 events of fold k-1's last day AND the FIRST 600
  events of fold k+1's first day from the training set. The held-out fold k
  itself has no training-side neighbor to embargo against.

## Ratification

This plan ratified by council-1, council-5, council-6 in
`docs/council-reviews/2026-04-27-encoder-retrain-protocol.md`. Owner:
builder-8 (implementation), reviewer-10 (code review against this plan + the
council protocol doc), validator-11 (run + grade against decision tree).

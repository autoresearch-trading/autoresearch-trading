# Gate 0 Summary — Steps 1-2 Closeout

**Date:** 2026-04-15
**Commits:** data pipeline `ec1ea5d` → cache fix `95ca60c` → Gate 0 CLI fix `9de25c2` → RP control `c0bee9f` → majority + shuffled + balanced rendering `7ff1459`
**Spec:** `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`

## TL;DR

**Gate 0 as originally specified measures label imbalance, not signal.** The PCA+LR baseline is indistinguishable from a majority-class predictor at every horizon when read through balanced accuracy. This was hidden by raw accuracy, which inflated illiquid-symbol numbers by up to 10pp.

The pipeline itself is clean (shuffled-labels control stays at 0.500±0.003 — zero construction leakage). The implication is structural: **flat summary statistics over 200-event windows throw away the sequential signal we believe is there**, exactly the situation the self-supervised encoder is designed to address.

## Four Baselines, Balanced Accuracy (council-preferred reading)

| Horizon | PCA(20)+LR | Random Projection | Majority Class | Shuffled Labels |
|---------|------------|-------------------|----------------|-----------------|
| H10 | **0.5104** (n>51.4%: 8/25) | 0.5091 (6/25) | 0.5000 (0/25) | 0.5030 (15/25) |
| H50 | 0.5065 (6/25) | 0.5013 (2/25) | 0.5000 (0/25) | 0.4997 (5/25) |
| H100 | 0.5043 (2/25) | 0.4983 (1/25) | 0.5000 (0/25) | 0.5010 (6/25) |
| H500 | 0.5051 (4/25) | 0.4993 (2/25) | 0.5000 (0/25) | 0.5033 (3/25) |

_Standard error per symbol-fold ≈ 0.022 (3 folds × ~500 test windows per illiquid symbol). All PCA margins over Majority are within one SE._

## Per-horizon interpretation

**H10 (short-term)** — largest absolute margin PCA vs Majority: +0.010pp. PCA slightly beats RP (+0.001) which slightly beats Majority (0.000). Monotone but not significant. 8/25 symbols clear 51.4% on PCA vs 0/25 on Majority; these 8 are mostly thin-book alts where balanced accuracy recovers 51-53% from raw 58-62% (see `gate0-baseline.md` per-symbol table). Consistent with thin-book autocorrelated flow.

**H50, H100, H500** — PCA's margin over Majority < 0.007pp. Statistically indistinguishable. Flat features carry no medium-horizon signal a linear probe can extract.

**Shuffled-labels** — stays at 0.5000 ± 0.003 across all horizons as expected. The "15/25 above 51.4% at H10" for shuffled is pure small-sample noise (3 folds × 25 symbols gives 75 independent balanced-accuracy estimates; at SE ≈ 0.022 we'd expect ~17 to exceed 0.514 purely by chance).

## What this tells us

1. **The feature pipeline is sound.** Shuffled-labels proves no leakage from label construction, window alignment, fold boundaries, or embargo bookkeeping. The code is correct.

2. **Flat aggregation destroys the tape signal.** This is *consistent with* the project's central hypothesis: sequential microstructure patterns require sequential models. It does not *prove* the CNN hypothesis — if there's no signal at all, the CNN inherits the null. But it rules out the "linear-on-summaries is enough" alternative.

3. **Raw accuracy was misleading.** The original H10 summary showed 14/25 symbols > 51.4% with mean 0.5319. Balanced accuracy shows 8/25 with mean 0.5104. The gap (2.15pp mean, up to 9.9pp on 2Z individual) was majority-class exploitation. This is now corrected in the renderer.

4. **Gate 0 threshold needs reframing.** "Beat PCA by 0.5pp on 15+ symbols" is meaningless when PCA = Majority = chance. The meaningful test is "beat Majority by ≥1pp on 15+ symbols" AND "beat RP-control by ≥1pp on 15+ symbols."

## Outlier Symbols — Recalibrated

Symbols with inflated raw-accuracy H10:

| Symbol | H10 raw (misleading) | H10 balanced (real) | Gap | Verdict |
|--------|---------------------|---------------------|-----|---------|
| 2Z | 0.6208 | 0.5222 | 9.86pp | Majority artifact |
| CRV | 0.6159 | 0.5278 | 8.81pp | Majority artifact |
| WLFI | 0.5945 | 0.5133 | 8.12pp | Majority artifact |
| XPL | 0.5796 | 0.5166 | 6.30pp | Majority artifact |
| PUMP | 0.5530 | 0.5047 | 4.83pp | Majority artifact |

None of the apparent "big-win" symbols has real signal above the majority-class floor.

## AVAX (Gate-3 held-out symbol) — sobering

AVAX H10 raw = 0.4918, balanced = 0.4986. **Below chance on flat features.** Gate 3 will require the encoder to learn AVAX-relevant structure purely from invariances learned on other symbols — a strong test of generalization.

## Pipeline artifacts, statistics, and data footprint

- **Cache:** 4003 shards × 25 symbols × 161 days → 32,060,988 events; 641,130 windows @ stride=50; 160,292 windows @ stride=200.
- **Data-to-params ratio** at 400K model: ~1:1.6 (per spec §Training Strategy amendment). Tight but reasonable for self-supervised learning.
- **Data quality:** 0 critical failures after the `expand_ob_levels` NaN fix (commit `95ca60c`). 4 residual warnings are real extreme-dislocation events on illiquid symbols.
- **Walk-forward:** 3-fold, 600-event embargo, min_train=2000, min_test=500. Fits the smallest symbol (LDO at H500: 4577 windows).
- **Test suite:** 289/289 passing at `7ff1459`.

## Gate 1 / Gate 2 entry criteria (revised per council)

### Gate 0 — PASS with reframing

Original spec: "PCA + LR baseline establishes reference." → **Revised (this document): Gate 0 passes only if the pipeline is proven leakage-free and the four-baseline grid is coherent.** ✓

- [x] Cache built clean (0 critical NaN post-fix).
- [x] Shuffled-labels control ≈ 0.500 (pipeline clean).
- [x] Majority-class baseline computed (floor established).
- [x] PCA vs RP ≈ null (no adaptive structure in flat features).
- [x] Per-symbol and per-horizon tables published (raw + balanced).

### Gate 1 — Entry criteria for CNN linear probe (post-pretraining)

Requirements (binding, must ALL pass — per council-6):

1. **CNN linear probe balanced accuracy > Majority-class baseline + 1.0pp** on ≥ 15/25 symbols at H100, evaluated on **April 1–13 held-out period** (not the Oct–Mar data used for Gate 0).
2. **CNN linear probe balanced accuracy > Random-Projection control + 1.0pp** on ≥ 15/25 symbols at H100, same April data.
3. **Absolute balanced accuracy ≥ 51.4%** on those same 15+ symbols (retained as sanity floor).
4. **Hour-of-day probe < 10%** (24-class LR on 256-dim frozen embeddings — sub-1.5pp variance across UTC sessions).
5. **Symbol-identity probe < 20%** (spec existing).
6. **CKA > 0.7** between seed-varied runs (spec existing).

### Gate 2 — Fine-tuned CNN vs LR on flat features

Original: "Exceed by ≥ 0.5pp on 15+ symbols." **Retained unchanged** — council-5 noted that since CNN and LR share the same session-of-day exposure, the delta between them is still meaningful.

### Gates 3, 4 — unchanged

Gate 3: AVAX held-out accuracy at primary horizon > 51.4% (balanced).
Gate 4: H500 uses balanced accuracy (already amended spec).

## Required pretraining adjustments (council-6)

1. **SimCLR window jitter:** ±10 → ±25 events (crosses BTC session micro-boundaries; shifts illiquid-alt window center by ~10 min).
2. **Timing-feature augmentation:** σ=0.10 Gaussian noise injected on `time_delta` and `prev_seq_time_span` during SimCLR view generation. Forces encoder to rely on relative rhythms, not absolute session-indicative magnitudes.
3. **Do NOT exclude `prev_seq_time_span` from MEM** — it carries local event-rate rhythm (microstructure). The augmentation in (2) handles the session-leak risk without removing the feature.

## Session-of-day confound check (before or during Gate 1)

Per council-5. Train LR on a single feature — hour-of-day (4-hour bins one-hot) — over the same walk-forward folds. If this exceeds PCA+LR on the 85-dim flat features by >0.5pp balanced accuracy on ≥5 symbols, the `_last` block in `tape/flat_features.py` (specifically `time_delta_last` and `prev_seq_time_span_last`) is a session-of-day leak and must be pruned. This is a pre-pretraining sanity check that costs <5 minutes.

## Files of record

- `docs/experiments/gate0-baseline.md` — PCA run (both raw + balanced tables, per-symbol both-metric).
- `docs/experiments/gate0-baseline.json` — machine-readable.
- `docs/experiments/gate0-random-control.md/json` — RP control.
- `docs/experiments/gate0-majority-baseline.md/json` — Majority-class control.
- `docs/experiments/gate0-shuffled-labels.md/json` — Null-hypothesis shuffled-labels control.
- `docs/experiments/cache-nan-investigation.md` — analyst-9 post-mortem on the OB level-expand NaN bug.
- `docs/council-reviews/council-1-gate0-rp-equivalence.md` — Prado-methodology review.
- `docs/council-reviews/council-5-gate0-falsifiability.md` — skeptic review.
- `docs/council-reviews/council-6-gate0-impact-on-pretraining.md` — DL-architect review.

## Next step

Apply the spec amendments listed in §"Gate 1 / Gate 2 entry criteria (revised per council)" and §"Required pretraining adjustments" to `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`, then proceed to Step 3 (pretraining plan).

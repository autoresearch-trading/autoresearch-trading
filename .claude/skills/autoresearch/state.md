# Research State — Representation Learning Branch

## Environment
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- Branch: `main`
- Stack: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB
- Data: 40GB raw trades, 161 days (2025-10-16 → 2026-03-25 pre-April), 25 symbols (AVAX held out from pretraining)
- Primary metric: representation quality (probing tasks, cluster analysis, balanced accuracy at ALL horizons)
- Compute cap: 1 H100-day before evaluation gates

## Current State (2026-04-23)

**Steps 0, 1, 2 COMPLETE. Step 3 CODE COMPLETE** — all 12 implementation tasks landed on `main`, 324 tests green, CPU smoke ran 13.5 min end-to-end with no NaN and checkpoint written. **Ready for Task 13 (H100 launch).**

**Next session: launch the H100 run on RunPod.** Before dispatching `runpod-7`, the user must create two R2 buckets (see "Pre-launch chores" below) and confirm RunPod billing.

### Completed steps
- **Step 0** — data validation, label validation, falsifiability prereqs, base-rate stationarity measurements. Spec amendments applied.
- **Step 1** — full tape pipeline (`tape/` package: constants, ob_align, dedup, events, features_trade, features_ob, labels, cache, windowing, dataset, sampler, flat_features, splits). 289/289 tests passing.
- **Step 2 (Gate 0)** — 4 baselines published: PCA(20)+LR, Random Projection, Majority-class, Shuffled-labels. All indistinguishable from chance on balanced accuracy. Pipeline clean (shuffled 0.500±0.003). Re-run on 83-dim post-prune (2026-04-23) — qualitative result unchanged.
- **Step 3 (code)** — 12 tasks, 14 commits on `main`. Encoder + MEM + contrastive + probes + pretrain loop + CLI + probe runner + checkpoint export + RunPod scaffold. Council-6 corrections baked in (mask-first-then-encode MEM flow, τ schedule, loss annealing, grad clip, bf16, `torch.compile`).

### Cache state
- 4003 `.npz` shards at `data/cache/`, 32.06M events, 641K windows @ stride=50.
- Smallest symbol: LDO at H500 = 4577 windows (bounds the walk-forward fold params).
- 0 critical validator failures (after `expand_ob_levels` NaN fix at commit `95ca60c`).
- 4 residual warnings = real extreme-dislocation events on illiquid symbols on specific days (not bugs).

### Council round 6 (2026-04-15) — spec deltas applied
- Gate 0 reframed: 4-baseline publishing grid, not threshold-gate.
- Gate 1: 4-condition binding stop-gate — ≥51.4% balanced acc on 15+/25 AND beat Majority+1pp AND beat RP+1pp AND hour-of-day probe <10% with <1.5pp session variance.
- Balanced accuracy now universal metric (not just H500).
- SimCLR augmentations strengthened: window jitter ±10→±25, new σ=0.10 timing-feature noise on `time_delta` and `prev_seq_time_span`.
- Hour-of-day probe added to pretraining monitoring (every 5 epochs).
- Pre-pretraining session-of-day confound check added.

### Council-6 review + correction plan (2026-04-23, commit `c16a0f4`)
Written Step 3 plan was reviewed by council-6 before execution. Outcome: `docs/council-reviews/council-6-step3-model-size.md`.
- **Model size:** `channel_mult=1.0` (~400K params, within 500K cap). 625K exceeds cap without meaningful gain; 200K risks collapse via effective-rank saturation.
- **Three blocking plan defects surfaced and fixed in one coordinated amendment:**
  1. **MEM identity-task bug** — plan's `pretrain_step` encoded UNMASKED input then applied the mask only to the loss; decoder could trivially copy ground truth. Fixed to mask-first-then-encode (BN full input → zero-fill masked positions in BN-normalized space → encode MASKED input → decode → MSE). `docs/knowledge/concepts/mem-pretraining.md`.
  2. **NT-Xent τ=0.10** — the ImageNet default. Replaced with `schedule_tau(epoch)` annealing 0.5→0.3 by epoch 10. `docs/knowledge/decisions/ntxent-temperature.md`.
  3. **Block size 5 / fraction 0.15** — RF=253 made 5-event gaps solvable by local interpolation. Bumped to 20/0.20 in `tape/constants.py`. `docs/knowledge/decisions/mem-block-size-20.md`.
- **Four ancillary flags applied in the same amendment:**
  - Loss annealing: MEM 0.90→0.60, contrastive 0.10→0.40 over 20 epochs.
  - Gradient clipping `max_norm=1.0` (projection-head anti-collapse).
  - bf16 autocast + `torch.compile(encoder, mode="reduce-overhead")` — ~1.8× H100 throughput.
  - Embedding collapse threshold 1e-4 → 0.05; effective-rank monitor (flag < 20 at epoch 5, < 30 at epoch 10).

### Session-of-day leak — RESOLVED (2026-04-23)
- Confound check (Task 7, commit `a6845de`) triggered on 5/25 symbols at threshold:
  - LTC (+1.63pp), HYPE (+1.17pp), WLFI (+1.12pp), BNB (+0.74pp), PENGU (+0.62pp)
- **FLAT_DIM is now 83** (was 85): `time_delta_last` and `prev_seq_time_span_last` pruned from `tape/flat_features.py` (commit `800d1a2`).
- Gate 0 re-run on 83-dim:
  - PCA+LR H100 bal 0.5043 → 0.5037 (−0.06pp, commit `ea4f6f4`)
  - RP H100 bal 0.4983 → 0.5020 (+0.37pp, commit `04a9283`)
  - All within 0.5pp — qualitative result unchanged, baselines still ≈ chance.
- `docs/experiments/gate0-summary.md` regenerated (commit `e440bef`).
- Raw sequence: `time_delta` and `prev_seq_time_span` still in full (200, 17) tensor for encoder; mitigated via SimCLR timing-noise augmentation (σ=0.10) and ±25 jitter.
- CLAUDE.md gotcha #32 updated from conditional to historical (commit `3ae3109`).

## Architecture (post-council-6 correction)
- Self-supervised encoder: ~400K params, dilated CNN (kernel=5, dilations 1/2/4/8/16/32), 256-dim embeddings.
- **MEM:** 20-event blocks, 20% of events masked. Weight annealed 0.90→0.60 over 20 epochs. Mask-first-then-encode flow mandatory.
- **Contrastive:** SimCLR on global embeddings, weight annealed 0.10→0.40. NT-Xent τ=0.5 → τ=0.3 by epoch 10.
- **Training infra:** AdamW, OneCycleLR(max_lr=1e-3, pct=0.2), grad clip 1.0, bf16 autocast, `torch.compile(encoder)`.
- **Fine-tuning:** freeze 5 epochs → unfreeze at lr=5e-5.

## Evaluation Gates (current, post-round-6 + 2026-04-23)

| Gate | Test | Threshold | Status |
|------|------|-----------|--------|
| 0 | 4-baseline grid (PCA, RP, Majority, Shuffled) on 83-dim flat | Publishes noise floor; shuffled≈0.500 required | **DONE 2026-04-15, re-run 2026-04-23** |
| 1 | Linear probe on frozen embeddings, H100 balanced acc, April 1-13 | All 4: ≥51.4% on 15+; +1pp vs Majority on 15+; +1pp vs RP on 15+; hour-of-day probe <10% w/ <1.5pp session variance | Blocked on Task 13 |
| 2 | Fine-tuned CNN vs LR on 83-dim flat features | ≥ 0.5pp on 15+ symbols | Blocked on Task 13 |
| 3 | AVAX (held out) at H100 | > 51.4% balanced | Blocked on Task 13 |
| 4 | Temporal stability (months 1-4 vs 5-6) | < 3pp drop on 10+ symbols, balanced acc | Blocked on Task 13 |

## Step 3 implementation commits (2026-04-23, chronological)

| Commit | Subject |
|--------|---------|
| `0b4216d` | feat(tape): TapeEncoder + MEMDecoder + ProjectionHead |
| `e2f5d60` | feat(tape): block/random MEM masking + 14-feature target mask |
| `c16a0f4` | **spec: apply council-6 correction plan for Step 3** (major amendment) |
| `ee05f17` | feat(tape): SimCLR view generator with ±25 jitter |
| `b569697` | feat(tape): MEM masked MSE + NT-Xent with cross-symbol soft positives |
| `d8e2413` | feat(tape): cross-symbol soft-positive matrix with AVAX exclusion |
| `84e0381` | feat(tape): direction/symbol/hour-of-day frozen-embedding probes |
| `a6845de` | experiment: pre-pretraining session-of-day confound check |
| `800d1a2` | fix(flat_features): prune time_delta_last + prev_seq_time_span_last (85→83) |
| `29f23c0` | feat(tape): MEM+SimCLR pretrain step (mask-first-then-encode) |
| `ea4f6f4` | experiment: re-run Gate 0 PCA+LR baseline on 83-dim |
| `04a9283` | experiment: re-run Gate 0 RP control on 83-dim |
| `e440bef` | analysis: regenerate Gate 0 summary on 83-dim |
| `3ae3109` | chore: record _last-features prune + Gate 0 re-run |
| `bce88d9` | feat(scripts): pretraining CLI (also bundled `fix(sampler)` for `EqualSymbolSampler.set_epoch` rebuild) |
| `7f44b14` | feat(scripts): standalone frozen-embedding probe runner |
| `41b1b56` | feat(scripts): checkpoint export with provenance stamping |
| `660285b` | chore(runpod): Dockerfile + launch.sh scaffold for Step 3 H100 run |

## Pre-launch chores (status)

1. **R2 buckets — CREATED 2026-04-23T14:49Z** via cloudflare-api MCP:
   - `pacifica-cache` (empty, ready for `rclone sync data/cache`)
   - `pacifica-models` (empty, ready for run output)
   - `pacifica-trading-data` (pre-existing, raw data — do not touch)
2. **Confirm RunPod billing** — an H100 80GB on-demand is ~$2/hr; 24h cap ≈ $48. Ensure credit or hard limit is set.
3. **Optional:** set `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` environment variables in the RunPod secret store, or confirm rclone config will be copied into the container via `launch.sh`.

## Next Session — Task 13 (H100 launch) + Tasks 14–15

Entry prompt for the new session:

> Resume tape representation learning project on `main` branch. Status per `.claude/skills/autoresearch/state.md` — Step 3 CODE COMPLETE (15 commits on main, 324 tests green, CPU smoke validated, R2 buckets `pacifica-cache` + `pacifica-models` already created and empty). Ready for **Task 13 (H100 pretraining launch on RunPod)**. Confirm RunPod billing is configured, then:
>
> 1. Push local cache to R2: `rclone sync data/cache r2:pacifica-cache/v1 --transfers 32 --checkers 64 --size-only`.
> 2. Dispatch `runpod-7` to build `runpod/Dockerfile`, push via the `flash` skill, launch an H100 80GB pod with env `R2_CACHE_PREFIX=r2:pacifica-cache/v1`, `OUT_PREFIX=r2:pacifica-models/step3-run-0`, `EPOCHS=30`, `BATCH_SIZE=256`, `CHANNEL_MULT=1.0`, `MAX_H100_HOURS=23.0`, `SEED=0`.
> 3. **Monitor the first 2 epochs** — council-6 diagnostic: MEM loss at epoch 2 should be < 0.6 in BN-normalized space. If > 0.8, abort (underfit). Also verify embedding_std ≥ 0.05 (no collapse) and effective_rank ≥ 20 at epoch 5.
> 4. On completion: pull `encoder-gate1.pt` + `april-probe-report.json` to `runs/step3-r1/`. Execute **Task 14** (Gate 1 pre-flight on returned checkpoint, per plan lines 2810–2853) and **Task 15** (CLAUDE.md + state.md handoff for Step 4, per lines 2855–end).
>
> Plan reference: `docs/superpowers/plans/2026-04-15-step3-pretraining.md` Tasks 13–15.
> Council-6 diagnostics: `docs/council-reviews/council-6-step3-model-size.md` §Q3 (early-signal thresholds).
> CLAUDE.md gotchas 21–33 all current as of 2026-04-23.
> R2 access: the Cloudflare MCP is connected; use `cloudflare-api.execute` for any bucket ops that rclone can't cover.

## Key References for Task 13
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md` (amended 2026-04-23 with council-6 corrections)
- Step 3 plan: `docs/superpowers/plans/2026-04-15-step3-pretraining.md`
- Council-6 review: `docs/council-reviews/council-6-step3-model-size.md`
- Knowledge base: `docs/knowledge/INDEX.md`
- RunPod scaffold: `runpod/` (Dockerfile, launch.sh, requirements-runpod.txt, README.md, .dockerignore)
- Smoke-test evidence: Task 9 CPU smoke ran 13.5 min end-to-end, no NaN, checkpoint written (log surfaced during `bce88d9`)

## Deviations recorded during Step 3 code tasks (for future audit)

1. **Task 3 augment (`ee05f17`):** `make_views_from_context` guard relaxed from `raise` to clamp when caller provides invalid (center, ctx, jitter). Function is currently dead code — `pretrain_step` calls `apply_augment_pipeline` directly. Revisit when dataset-layer context materialization is wired.
2. **Task 4 losses (`b569697`):**
   - NaN guard added in soft cross-entropy (`0 * -inf = NaN`).
   - Soft-loss averaging changed from `[has_soft].mean()` to `.mean()` over all 2N rows. Scales contribution with batch composition instead of weighing soft pairs equally. Defensible but worth re-examining if contrastive behavior is unexpected.
3. **Task 6 probes (`84e0381`):** Dropped `multi_class="multinomial"` kwarg — removed in sklearn 1.8.0. LogisticRegression auto-selects multinomial for multiclass labels.
4. **Task 8 pretrain (`29f23c0`):** `LRScheduler` alias for deprecated `_LRScheduler`; `cast(int, ...)` for `opt.state["_step"]`; `effective_rank` guard for rank-1 degenerate case; `OneCycleLR` `max(10, total_steps)` floor to avoid ZeroDivisionError at tiny `total_steps`.
5. **Task 9 CLI (`bce88d9`):** Bundled `fix(sampler)` in the same commit — `EqualSymbolSampler.set_epoch()` wasn't rebuilding `_by_symbol` after `dataset.set_epoch()` rebuilt `_refs`, causing IndexError in DataLoader workers. **Real production bug caught by the live CPU smoke.**
6. **Task 10 probe runner (`7f44b14`):** Test skip condition changed from `cache exists` to `_has_april_shards()` — local cache has only pre-April dates (April is the Gate 1 probe-evaluation window, downloaded separately). Will exercise for real in-container during Task 13.
7. **Task 12 RunPod scaffold (`660285b`):** `launch.sh` uses the council-6 annealed-weight flags (not the plan's original `--mem-weight` / `--contrastive-weight`). `export_checkpoint.py` patched to fall back to `GIT_SHA` env var when `.git/` is absent inside the container.

## Key Context
- Prior supervised MLP on main branch hit a local optimum — motivated the pivot to representation learning
- 100-trade batching destroyed tape signals — this branch works with raw order events
- `is_open` has autocorrelation half-life of 20 trades (strongest persistent signal)
- Gate 0 round 6: flat aggregation destroys sequential signal (consistent with CNN hypothesis but doesn't prove it). Pipeline is leakage-free.
- Step 3 end-to-end CPU smoke (2026-04-23, Task 9): 13.5 min wall-clock, no NaN, checkpoint written — validates the full MEM+SimCLR pipeline at batch=8, 2 epochs.

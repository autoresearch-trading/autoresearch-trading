# Research State — Representation Learning Branch

## Environment
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- Branch: `main`
- Stack: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB
- Data: 40GB raw trades, 161 days (2025-10-16 → 2026-03-25 pre-April), 25 symbols (AVAX held out from pretraining)
- Primary metric: representation quality (probing tasks, cluster analysis, balanced accuracy at ALL horizons)
- Compute cap: 1 H100-day before evaluation gates

## Current State (2026-04-15)

**Steps 0, 1, 2 COMPLETE.** Cache materialized, Gate 0 published (reframed as noise-floor grid, not threshold-gate), council round 6 finalized, spec amended, CLAUDE.md synced, knowledge base compiled.

**Step 3 (pretraining plan + execution) is the next session.**

### Completed steps
- **Step 0** — data validation, label validation, falsifiability prereqs, base-rate stationarity measurements. Spec amendments applied.
- **Step 1** — full tape pipeline (`tape/` package: constants, ob_align, dedup, events, features_trade, features_ob, labels, cache, windowing, dataset, sampler, flat_features, splits). 289/289 tests passing.
- **Step 2 (Gate 0)** — 4 baselines published: PCA(20)+LR, Random Projection, Majority-class, Shuffled-labels. All indistinguishable from chance on balanced accuracy. Pipeline clean (shuffled 0.500±0.003).

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

### Session-of-day leak — RESOLVED (2026-04-23, commit `800d1a2`)
- **FLAT_DIM is now 83** (was 85): `time_delta_last` and `prev_seq_time_span_last` pruned from `tape/flat_features.py` after confound check triggered on 5/25 symbols
- Gate 0 re-run on 83-dim: PCA+LR H100 bal 0.5043→0.5037 (-0.06pp); RP H100 bal 0.4983→0.5020 (+0.37pp). All within 0.5pp — qualitative result unchanged (commits `ea4f6f4`, `04a9283`)
- Raw sequence: `time_delta` and `prev_seq_time_span` still in full (200, 17) tensor for encoder; mitigated via SimCLR timing-noise augmentation (σ=0.10)

## Architecture (unchanged from spec)
- Self-supervised encoder: ~400K params, dilated CNN (kernel=5, dilations 1/2/4/8/16/32), 256-dim embeddings
- Pretraining: MEM (block masking, 20-event blocks, 15% of events, weight 0.70) + SimCLR contrastive (weight 0.30)
- Fine-tuning: freeze 5 epochs → unfreeze at lr=5e-5

## Evaluation Gates (current, post-round-6)

| Gate | Test | Threshold | Status |
|------|------|-----------|--------|
| 0 | 4-baseline grid (PCA, RP, Majority, Shuffled) | Publishes noise floor; shuffled≈0.500 required | **DONE 2026-04-15** |
| 1 | Linear probe on frozen embeddings, H100 balanced acc, April 1-13 | All 4: ≥51.4% on 15+; +1pp vs Majority on 15+; +1pp vs RP on 15+; hour-of-day probe <10% w/ <1.5pp session variance | Not started |
| 2 | Fine-tuned CNN vs LR on flat features | ≥ 0.5pp on 15+ symbols | Not started |
| 3 | AVAX (held out) at H100 | > 51.4% balanced | Not started |
| 4 | Temporal stability (months 1-4 vs 5-6) | < 3pp drop on 10+ symbols, balanced acc | Not started |

## Next Session — Step 3 Pretraining

Entry prompt for the new session:

> Resume tape representation learning project on `main` branch. Status per `.claude/skills/autoresearch/state.md` — Steps 0-2 complete, cache at `data/cache/`, Gate 0 published (all 4 baselines ≈ chance on balanced accuracy; pipeline clean). **Task: write the Step 3 pretraining implementation plan** per `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md` (§Architecture, §Training, §Evaluation Gates — post-round-6 amendments). Use `superpowers:writing-plans` skill. Plan must incorporate: SimCLR augmentation recipe (±25 jitter + σ=0.10 timing noise), hour-of-day probe every 5 epochs, RunPod H100 execution (use `runpod-7`), 1 H100-day compute cap before gates. Council-6 model-size review is open (1:1.6 data-to-params ratio at 400K is tight — consider 200K or 600K variants?).

## Key References for Step 3
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- Council round 6 reviews: `docs/council-reviews/council-{1,5,6}-gate0-*.md`
- Gate 0 summary: `docs/experiments/gate0-summary.md`
- Knowledge base: `docs/knowledge/INDEX.md`
- CLAUDE.md: all gotchas current through 2026-04-15 (33 gotchas total)

## Open Questions for Step 3 Planning
1. Model size — council-6 review on 400K params given 1:1.6 data-to-params ratio. Consider 200K or 600K variants?
2. Should MEM reconstruction loss weight 0.70 / contrastive 0.30 be re-examined given Gate 0 findings?
3. Pretraining epochs — spec says 20-40, stop on < 1% MEM improvement over last 20%. Keep?
4. Batch size 256 — fits H100 at 400K params?

## Key Context
- Prior supervised MLP on main branch hit a local optimum — motivated the pivot to representation learning
- 100-trade batching destroyed tape signals — this branch works with raw order events
- `is_open` has autocorrelation half-life of 20 trades (strongest persistent signal)
- Gate 0 round 6: flat aggregation destroys sequential signal (consistent with CNN hypothesis but doesn't prove it). Pipeline is leakage-free.

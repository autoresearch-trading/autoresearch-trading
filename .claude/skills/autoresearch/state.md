# Research State — Representation Learning Branch

## Environment
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- Branch: `main`
- Stack: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB
- Data: 40GB raw trades, 161 days (2025-10-16 → 2026-03-25 pre-April), 25 symbols (AVAX held out from pretraining)
- Primary metric: representation quality (probing tasks, cluster analysis, balanced accuracy at ALL horizons)
- Compute cap: 1 H100-day before evaluation gates

## Current State (2026-04-27 — PROGRAM AT END-STATE; user goal = useful tape repr → A; cross-symbol re-pretrain is the load-bearing experiment)

**Goal (user-stated 2026-04-27):** "figure out a model that learns useful tape representations and leads to A" where A = profitable paper-trading.

**Where we are:** Program reaches publishable end-state. Current encoder cannot ladder to A (council unanimous: +1pp at H500 against 6bp Pacifica fees + N=5 DSR is fee-blocked at every horizon). **Two clean options:**

- **Option α (close program):** Publish Gate 1/2/4 + calibration discovery + per-horizon table + RankMe + Path D writeup as the research finding. Headline: *"+1pp linearly-extractable direction signal at H500, temporally stable, sub-fee under DEX perp realism."* Goal-A abandoned for this stack. `docs/experiments/step4-program-end-state.md` (commit `23517fc`) is the publishable artifact.

- **Option β (cross-symbol re-pretrain):** Fresh spec from scratch (NOT a continuation of post-Gate-2 pre-reg per c-5's STOP A clause). Council-4's path C levers: widen `LIQUID_CONTRASTIVE_SYMBOLS` 6→12-15 (still excluding AVAX held-out + memecoins KBONK/KPEPE/PENGU), anneal soft-positive weight 0.5→1.0, add cluster-cohesion ≥+0.10 as a TRAINING-TIME early-stop diagnostic. C-2's add-on: replace direction-MEM with execution-cost-aware objective (predict signed_and_thresholded_mid_move conditional on size > tradeable_min). Pre-register cohesion delta ≥+0.10 as binding training-time gate AND ≥+5pp accuracy at H100 over RP control as binding evaluation gate (c-2-derived from fee-floor arithmetic: gross edge ~0.7bps × 100 trades = 70bps absorbing 60bps fees + 10bps slippage). Cost: ~6h MPS + ~2 weeks evaluation + amendment-budget cost.

**Joint probability of Option β laddering to A** (council estimate): ~10-25% under generous assumptions. Three conditions must all hit: (i) cohesion delta improves to ≥+0.10 (~50-60%), (ii) new edge size ≥+5pp at H100 over RP (~30-50%, NOT automatic — Cont-de Larrard says universalizing could reduce per-symbol signal), (iii) fine-tuning works on the new geometry (~70% conditional).

**Cross-symbol problem is the SINGLE MOST ACTIONABLE LEVER the program identified.** It's the upstream cause of: (1) Gate 2 fine-tuning failure (no shared trunk for fine-tuning to specialize), (2) +1pp ceiling (LR can only extract per-symbol locally-readable signal), (3) the unresolved interpretation between "encoder reads tape" vs "encoder reads per-symbol direction priors" (multi-probe battery couldn't decide because labels never fired). If user pursues Option β, this is the load-bearing experiment, not the trading evaluation.

**Awaiting user decision on next session: Option α (close + publish) vs Option β (fresh cross-symbol re-pretrain spec).**

---

## Previous state — 2026-04-26 PM (preserved for audit)

**Steps 0, 1, 2, 3 complete. Gate 1 PASSES on Feb + Mar H500. Gate 3 triage: EXONERATED. Cluster cohesion: UNEARNED UNIVERSALITY. Spec amendment v2 RATIFIED. Step 4 fine-tuning FAILED Gate 2 on all three binding criteria, both held-out months.**

### Step 4 Gate 2 verdict (2026-04-26 PM) — FAIL

`runs/step4-r1-phase-b-v2/finetuned-best.pt` (E18, val_total_bce=0.6906) failed Gate 2:

| Criterion | Threshold | Feb 2026 | Mar 2026 |
|-----------|-----------|----------|----------|
| C1: vs flat-LR ≥0.5pp on 15+/24 | 15+/24 | **7/24 FAIL** | **10/24 FAIL** |
| C2: no per-symbol regression > 1pp | 0 violations | **16/24 FAIL** | **12/24 FAIL** |
| C3: vs frozen-encoder LR ≥0.3pp on 13+/24 | 13+/24 | **8/24 FAIL** | **12/24 FAIL** |

Aggregate H500 bal_acc (mean across 48 cells):
- flat-LR: 0.5115
- frozen-encoder LR (Gate 1 protocol): 0.5061
- fine-tuned CNN: **0.4947** (underperforms flat-LR by 1.7pp)

**Diagnostic pattern:** CNN regresses to 0.50. Liquid symbols where flat-LR was high (SUI 0.626, LTC 0.595, 2Z 0.633, ETH 0.543, LINK 0.605) lost 7-15pp under fine-tuning. Illiquid alts where flat-LR was below random (KPEPE, KBONK, AAVE, FARTCOIN) gained 6-9pp — but those are mean-reversion to chance, not signal extraction. The +1.2pp val-fold gain (random 90/10 split, in-distribution) was overfitting to training-period label imbalance. Walk-forward held-out evaluation correctly falsified.

**Gate 1 (frozen encoder) remains valid.** Pretraining is not falsified — only the fine-tuning approach as configured.

**Three abort-criterion math bugs** during Phase A + Phase B of run #1:
- **Bug #1 (Class A):** AM Phase A `BCE > 0.95×init` required β=0.632 from frozen encoder Gate 1 measured at 0.514. Patched in commit `8149aa8`. Triage: `docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md`.
- **Bug #2 (Class A):** PM Phase B `CKA > 0.95 after epoch 8` demanded ~3× faster encoder rotation than the lr schedule supports. Patched in commit `322ab50`. Triage: `docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md`.
- **Bug #3 (Class B redundant guard):** Phase B `max(ΔCKA over last 3) < 0.005` fired during scheduled OneCycleLR cosine cooldown. Council-5 + council-6 jointly adjudicated as Class B (structurally subsumed by end-of-Phase-B CKA<0.95 upper bound, which the run passed). Pre-Gate-2 postmortem committed BEFORE eval (commit `f2f50dc`). New abort taxonomy added to `lead-0.md`.

**Per pre-committed audit trail, no retry of Phase B with different hyperparameters.** Next move: council to determine whether the encoder pretraining itself is the bottleneck OR whether fine-tuning is the wrong downstream task for this representation.

### Open questions for council (post-Gate-2 FAIL)

1. **Architecture wrong?** Linear-trunk-then-per-horizon-head may be the wrong inductive bias for tape data.
2. **Loss-weight schedule wrong?** H10's 9.9pp class-imbalance inflation potential may have dominated the gradient signal even at weight 0.10.
3. **Encoder too symbol-specific?** Phase B showed +1.2pp on in-distribution val fold; same encoder loses 1.7pp on Feb+Mar held-out. Encoder may be memorizing per-symbol artifacts rather than transferable signal.
4. **Different downstream task?** Clustering, retrieval, regime classification — tasks where the encoder's representation quality can be assessed without temporal-transfer constraint.

### Step 4 deliverables (committed)

- **Code:** `tape/finetune.py` (DirectionHead + FineTunedModel + weighted_bce_loss + cka_torch), `scripts/run_finetune.py` (two-phase trainer with patched abort criteria + `--resume-from-checkpoint`), `scripts/run_gate2_eval.py` (three-comparator eval with bootstrap CIs).
- **Tests:** 17/17 passing in `tests/tape/test_finetune.py`.
- **Plan:** `docs/superpowers/plans/2026-04-24-step4-fine-tuning.md` (ratified, then patched twice with mechanical alignments).
- **Pre-Gate-2 postmortem:** `docs/experiments/step4-phase-b-third-strike-postmortem.md`.
- **Gate 2 results writeup:** `docs/experiments/step4-gate2-finetune.md` (with anti-amnesia + honest framing).
- **Process update:** `lead-0.md` now has the Class A / Class B abort taxonomy.

### Entry prompt for next session

> Resume tape representation learning on `main`. Step 4 (Gate 2) FAILED on 2026-04-26 PM — fine-tuned CNN underperformed flat-LR by 1.7pp on Feb+Mar held-out at H500 (mean bal_acc 0.4947 vs 0.5115). Diagnostic: CNN regresses to 0.50 — liquid symbols with high flat-LR baselines lost 7-15pp under fine-tuning. The +1.2pp val-fold gain was overfitting to in-distribution label imbalance, falsified by walk-forward held-out eval. Per pre-committed audit trail, no retry. Three abort-criterion math bugs on the way to Gate 2 (two Class A patched, one Class B retired); new Class A/Class B taxonomy in `lead-0.md`. Gate 1 (frozen encoder) remains valid. **Next move: council to determine whether the encoder pretraining itself is the bottleneck, or whether fine-tuning is the wrong downstream approach for this representation.** Reference `docs/experiments/step4-gate2-finetune.md` for the full Gate 2 writeup. Open questions: (a) architecture wrong, (b) loss-weight schedule wrong, (c) encoder too symbol-specific to transfer, (d) move to a different downstream task (clustering/retrieval/regime classification).

- **Checkpoint:** `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params)
- **Gate 1 pass writeup:** `docs/experiments/step3-run-2-gate1-pass.md`
- **Gate 3 AVAX probe writeup:** `docs/experiments/step5-gate3-avax-probe.md` (initial)
- **Gate 3 triage:** `docs/experiments/step5-gate3-triage.md` (bootstrap CIs + in-sample control → EXONERATED)
- **Council reviews:** `docs/council-reviews/council-5-gate3-avax-falsifiability.md`, `docs/council-reviews/council-3-avax-microstructure.md`
- **Landmark commits:**
  - `117187d` — Phase-1 fixes (council-5 Bug B/C + no early-stop + best-val checkpoint + ts_first_ms)
  - `bda524e` — Phase-1b diagnostic tooling + `--train-end-date` flag
  - `96722b4` — Gate 1 pass documentation
  - `5acde01` — Gate 3 AVAX probe script
  - `3833e35` — Gate 3 AVAX apparent fail (pre-triage)
  - `ea07bda` — probe upgrade: bootstrap CIs + N=50 shuffled-null + `--target-symbols`
  - `2c7ebc2` — Gate 3 triage: EXONERATED
  - `5bd3e6d` — cluster cohesion diagnostic script
  - `5e13e7e` — cluster cohesion experiment: 6 liquid anchors, Feb 2026
  - `d2dac9b` — cluster cohesion: drop unused mask params

### Gate 1 results (run-2, 2026-04-23 overnight, local M4 Pro MPS, $0)

| Condition | Feb (held-out, 21K windows) | Mar (held-out, 16K windows) |
|---|---|---|
| 1. ≥51.4% on 15+/24 | **15/24 ✓** | **17/24 ✓** |
| 2. > Majority+1pp | +3.03pp ✓ | +3.12pp ✓ |
| 3. > RP+1pp | +1.91pp ✓ | +2.29pp ✓ |
| 4. Hour-of-day <10% | 0.06-0.09 ✓ | 0.06-0.09 ✓ |

Encoder beats PCA+LR on 17/24 symbols (Feb) / 14/24 (Mar). Consistent winners: AAVE, BNB, CRV, FARTCOIN, HYPE, KBONK, PUMP, SUI, WLFI, XRP.

### Key methodological findings (from 2026-04-23 diagnostic work)

- **H100 is at noise floor** for all predictors on this data. H500 is the primary horizon where SSL adds value over flat baselines. Gate 1 evaluated at H500, not H100.
- **April 1-13 is underpowered for Gate 1** at stride=200 (60-150 windows/symbol, below the 200 `min_valid` threshold). Plus a 4× density shift on liquid symbols. Feb + Mar at matched density are the right held-out window.
- **Run-0's apparent shortcut collapse was 2/3 measurement artifact** — three probe bugs (hour probe used event-index, direction probe only saw 3 alphabet-first symbols, `ts_first_ms` missing from dataset). See `docs/council-reviews/council-5-step3-run0-falsifiability.md` for the full diagnosis.
- **Local MPS is production-viable** for pretraining at this model size. 5h 17m / $0 vs ~2.5h / ~$6 on H100.

### Gate 3 triage summary (2026-04-24)

Ran `scripts/avax_gate3_probe.py` → Gate 3 appeared to FAIL on AVAX at stride=50.
Council-5 + council-3 reviews both demanded bootstrap CIs + in-sample control before drawing conclusions. Triage results:

- **AVAX stride=50 (Feb+Mar × H100+H500):** encoder vs PCA CIs overlap on 4/4 cells (narrowest overlap 0.004pp on Mar H500, sign PCA > encoder). Encoder never clears 51.4% at CI lower bound. Shuffled null (N=50) is clean at μ≈0.500, σ≈0.02–0.03.
- **In-sample control LINK+LTC (n_test ~660–880, ~2× AVAX):** encoder fails to beat majority on 3/4 cells, CI-overlaps PCA on 4/4 cells, never clears 51.4%. Same failure pattern as AVAX.
- **In-sample control AAVE:** same stride=200-style "lucky cell" pattern (Feb H100 encoder >> PCA; Mar H100 PCA >> encoder; both inside each other's CIs).

**Gate 3 triage verdict:** EXONERATED. AVAX is not anomalous — the 1-month single-symbol probe at ~400–900 test windows is underpowered for the encoder's ~1–2pp Gate-1 signal regardless of which symbol is held out. The Gate-1 pass was visible at 63K windows across 25 symbols (~80× this n_test).

### Cluster cohesion diagnostic (2026-04-24) — UNEARNED UNIVERSALITY

Ran `scripts/cluster_cohesion.py` on encoder-best.pt across the 6 liquid SimCLR anchors (BTC/ETH/SOL/BNB/LINK/LTC), Feb 2026, stride=200 eval, 8,660 windows in 3,676 (symbol, date, hour) buckets. Writeup: `docs/experiments/step5-cluster-cohesion.md`.

Four cosine populations (L2-normalized 256-dim embeddings):

| Population | mean | std | n_pairs |
|---|---|---|---|
| within_symbol (same sym, same hour) | 0.895 | 0.065 | 10,984 |
| same_symbol_diff_hour | 0.836 | 0.093 | 50,000 |
| cross_symbol_same_hour (**what SimCLR trained for**) | 0.734 | 0.148 | 50,000 |
| cross_symbol_diff_hour (random cross-symbol baseline) | 0.697 | 0.147 | 50,000 |

- **SimCLR delta = +0.037** (cross_symbol_same_hour − cross_symbol_diff_hour). Below council-5's `+0.1` "some_invariance" delta threshold.
- **Symbol-identity delta = +0.139** (same_symbol_diff_hour − cross_symbol_diff_hour). **4× stronger** than the SimCLR alignment signal.
- **Symbol-ID probe on 6 liquid anchors = 0.934 balanced accuracy** (spec threshold <0.20). Encoder is nearly symbol-separable.
- **The literal `>0.6` "strong_invariance" threshold fires (0.734) but is a false positive** — every population lives on a narrow cone (mean >0.69), so the absolute threshold is inflated by a global offset.

**Interpretation:** unearned universality. SimCLR on 6-of-24 symbols with soft-positive weight 0.5 did NOT force a symbol-invariant tape geometry. The encoder learned symbol-conditional features. The Gate 3 AVAX failure was training-dynamics-overdetermined — the "universal tape representations" spec framing was stronger than the training config actually earned. **Gate 1 pass stands unchallenged** (it's evidence of per-symbol feature quality, not universality).

### Spec amendment v2 (2026-04-24) — RATIFIED

Two-commit amendment series on `docs/superpowers/specs/2026-04-10-*.md`:

- **`b1f4065`** (amendment v1) — initial draft: Gate 1 → H500 + Feb+Mar matched-density; Gate 3 retired to informational; symbol-ID <20% reframed aspirational; stale FLAT_DIM 85→83 + session-of-day check execution record; MPS substitute for H100 noted.
- **`9c91f85`** (amendment v2) — applied all convergent council-1 + council-5 review fixes. Writeups: `docs/council-reviews/council-1-amendment-2026-04-24.md`, `council-5-amendment-2026-04-24.md`.

v2 closed these loopholes:

| Loophole | Fix |
|---|---|
| "matched-density" undefined (5 uses) | Binding definition: 0.7–1.3× training windows-per-symbol-per-day at stride=200 |
| Feb-AND-Mar re-sampling drift | Months are specifically Feb+Mar 2026; no substitution without re-pre-registration |
| "This training config" generic escape hatch | Only symbol-ID probe qualifies; all other diagnostics binding unless re-pre-registered |
| H500 horizon post-hoc | Horizon-selection rule binding on future runs: shortest horizon where PCA+LR ≥ 0.505 |
| Gate 3 drift-back | Re-activation criteria binding: n_test ≥2000, CI non-overlap, cluster delta ≥+0.10, pre-declared |
| "Not retroactive rationalization" weak defense | Replaced with honest underpower framing + cite pre-dispatched thresholds |
| Gate 4 "months 1-4 vs 5-6" drift | Rewritten as Oct-Nov vs Dec-Jan, evaluated on Feb+Mar fold |
| Horizon drift across gates | Explicit structure: H500 binding (G1, G4), H100 informational, H10/H50 baseline-only |
| Amendment drift budget | Third binding-gate amendment requires out-of-band review |
| One-passes-one-fails adjudication | "No adjudication, no averaging, no close enough" |
| April 1-13 anti-amnesia | Original pre-reg window must be reported alongside amended one forever |
| 15+/24 mislabeled as loosening | Noted as mechanically tighter (60.0% → 62.5%) |
| AVAX exclusion non-transitive | Extended to any future run citing this program's pre-registration |
| Per-symbol surrogate sweep | Pre-committed as Step-6 interpretation diagnostic (n=1 → n=5 on transfer claim) |

### Next session priorities (post-amendment)

1. **Step 4 (Gate 2) fine-tuning** — unblocked. Freeze encoder 5 epochs → unfreeze at lr=5e-5, add 4 direction heads, walk-forward eval on Feb+Mar held-out at H500. Gate 2 threshold: ≥0.5pp over flat LR on 15+/24 symbols (amended count). The Gate 1 +1.9–2.3pp margin already exceeds this floor, so Gate 2 is likely to pass — but fine-tuning has its own failure modes (catastrophic forgetting at freeze→unfreeze transition).
2. **Knowledge-base compilation — DONE 2026-04-24** (commit `f5f3e29`). 3 new concepts, 8 new decisions, 5 new experiment summaries, 3 concept updates + 1 decision marked superseded. INDEX rebuilt with 15 concepts / 22 decisions / 9 experiments. Next session can cite `docs/knowledge/INDEX.md` as the canonical reference.
3. **Step-6 per-symbol surrogate sweep — DISCHARGED 2026-04-24** (commit `5958179`, writeup `docs/experiments/step5-surrogate-sweep.md`). Ran 5 in-sample symbols (ASTER, LDO, DOGE, PENGU, UNI) individually under the same stride=50 protocol. Encoder CI strictly above PCA CI on **1/20 cells** (ASTER Feb H500) — precisely the chance rate under the null. AVAX's 0/4 CI-separation result is inside the surrogate distribution. Transfer claim now has n=5 behind it; Gate 3 reframe is correct.
4. **(Optional) R2 upload of encoder-best.pt** — rclone CreateBucket 403; needs bucket-level perms on pacifica-models.
5. **(Future) training-dynamics research question parked:** can cross-symbol invariance be raised by widening LIQUID_CONTRASTIVE_SYMBOLS 6→12-15, annealing soft-positive weight 0.5→1.0, training longer? Not a blocker for current research program; would require a new pre-registered experiment under the amendment-budget clause.

### Entry prompt for next session

> Resume tape representation learning on `main`. Session of 2026-04-24 closed out three milestones: (a) Gate 3 AVAX triage → EXONERATED (`2c7ebc2`); (b) Cluster cohesion → UNEARNED UNIVERSALITY (`5e13e7e`); (c) Spec amendment v1+v2 ratified after council-1 + council-5 review (`b1f4065` + `9c91f85`); (d) Per-symbol surrogate sweep discharged pre-commitment with 1/20 chance-rate CI separations (`5958179`); (e) Knowledge base compiled (`f5f3e29`, 15 concepts / 22 decisions / 9 experiments in `docs/knowledge/INDEX.md`). Gate 1 PASSES on Feb + Mar H500. Checkpoint: `runs/step3-r2/encoder-best.pt` (376K params). The amendment-budget clause is now live. **Only remaining Phase 1 milestone: Step 4 (Gate 2) fine-tuning** — freeze encoder 5 epochs → unfreeze at lr=5e-5 → 4 direction heads → walk-forward eval on Feb+Mar H500; Gate 2 threshold ≥0.5pp over flat LR on 15+/24 (the Gate 1 +1.9-2.3pp margin likely carries through, but fine-tuning has its own failure modes — catastrophic forgetting at freeze→unfreeze, overfitting to direction labels — to monitor). Reference `docs/knowledge/INDEX.md` for canonical decisions and concepts before dispatching builder-8.

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

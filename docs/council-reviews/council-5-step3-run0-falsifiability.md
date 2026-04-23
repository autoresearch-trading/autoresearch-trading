# Council-5 — Step 3 Run-0 Falsifiability Review

**Commit under test:** `34b8f18`
**Run:** `runs/step3-r1/` (epochs 1-8, early-stopped, 1.74h MPS wall-clock)
**Date:** 2026-04-23

## TL;DR

**Run-0 is uninformative, not failing.** The training recipe has three
measurement bugs and one execution bug that each, independently, invalidate
the signal the orchestrator is reading off the log. The objective itself is
under genuine suspicion, but *we cannot yet distinguish* "objective is
mis-specified" from "the recipe never ran the objective the spec describes."
**Do not launch run-1 until the bugs are fixed and a cheap ablation rules
out the recipe explanation.**

---

## 1. Four bugs that invalidate run-0's signal

### Bug A — SimCLR jitter is not actually applied (recipe-level, load-bearing)

`tape/pretrain.py::pretrain_step` lines 293-298 generates both views via
`apply_augment_pipeline(b.clone(), ...)` on the already-sliced 200-event
window. `tape/augment.py::make_views_from_context` is never called.
`apply_augment_pipeline` has no spatial shift — only per-feature noise,
time-dilation on one channel, per-cell dropout.

Consequence: **v1 and v2 are the identical 200-event sequence with two
different draws of feature-level noise.** Symbol id, date, UTC hour, event
identities — all perfectly invariant across the positive pair. The
contrastive objective as trained is "make the embedding noise-robust," not
"make two temporally-shifted views of the same regime close." **Symbol-id
probe 0.971 is the exact pathology this bug produces.**

### Bug B — Hour-of-day probe measures the wrong thing

`scripts/run_pretrain.py::_run_probe_trio` line 257:
```python
hours.append(int((item.get("start", 0) // 3600) % 24))
```

`item["start"]` is the **event index within the shard** (typically 0, 50,
100, ..., a few thousand per day at stride=50). Dividing by 3600 gives
mostly 0. **`probe_hour_of_day_acc = 0.587` is measuring within-day event-
index bucket predictability, not UTC hour.** The `<10%` spec threshold is
not being evaluated.

Downstream: `_collate` at line 72 uses the correct `ts_first_ms` for
contrastive soft-positives, but `TapeDataset.__getitem__` never emits
`ts_first_ms` → the default 0 fallback propagates, breaking soft-positive
matrix too.

### Bug C — Direction probe covers ~3 symbols, not 24

`_run_probe_trio` iterates `dataset[i]` linearly up to 50K unshuffled. Shard
order is sorted alphabetically × date. **First 50K windows cover only 2Z,
AAVE, ASTER.** The log confirms: `probe_dir_h100_per_symbol` at epoch 5 has
exactly three keys. The 0.499 "balanced acc mean" is a 3-illiquid-alts
average, not 25-symbol evidence.

### Bug D — MEM trajectory is partially masked by contrastive reward-hacking

MEM 0.803 → 0.513 → 0.736 combined with con 5.01 → 3.88 monotone means the
encoder found a shortcut for contrastive and is willing to degrade MEM to
hold it. The early-stop rule (`<1% MEM improvement over last 20% of
epochs`) fired on a regression, not a plateau — the criterion can't
distinguish "converged" from "the other head won."

---

## 2. Direct challenges

### Q1 — Falsifiability

Given Bugs A/B/C invalidate the observables, the current run cannot answer
this at all.

**Primary falsifiable metric for run-1:** direction probe balanced accuracy
on all 24 pretraining symbols at epoch 10, exceeding 0.510 on at least 8/24
symbols.

Secondary instrumental signals (must be visible):
- Symbol-id probe drops below 0.50 by epoch 10 (from 0.971).
- Corrected UTC-hour probe stays below 0.15 by epoch 10.
- Per-symbol MEM breakdown: `log_return` and `effort_vs_result` MSE drop
  below per-symbol variance floor by epoch 5.

### Q2 — Confound vs shortcut

Steelman benign: tape reader can guess "this is BTC" from pace and depth —
symbol-id >> 0.04 chance is expected.

Steelman damning: 0.971 is not "residual leakage," it is "the 256-dim
embedding is a near-perfect symbol classifier." Combined with Bug A (the
contrastive objective *was* rewarding this) and direction at chance, the
measurements are coherent: **the encoder is a symbol classifier on
training symbols.**

**Commitment: damning for the recipe as trained, indeterminate for the
objective.** After Bug A, symbol-id should decay. If it doesn't decay below
0.50 by epoch 10 in a clean re-run, the objective is damned.

### Q3 — Is the objective mis-specified?

The theoretical concern (±25 jitter + σ=0.10 timing noise leaves
symbol/date/hour invariant → rewards shortcut) is **correct even under the
correct recipe.** Bug A makes the shortcut bite harder, but doesn't
eliminate the critique.

**Disagree with pure-MEM remedy.** Pure MEM on this data will likely fail
Gate 1 for a different reason: MEM over 14 features with 4 blocks of 20
masked events in 200 is a very local reconstruction task. MEM 0.803 →
0.513 likely reflects local slope/rhythm interpolation — Gate 0 already
showed local interpolation is not future-information.

**Real remedy: stronger positive-pair policy.** The spec already names
"same-date same-hour cross-symbol soft positives" for 6 liquid symbols.
Soft-weight 0.5 against primary same-window positive means shortcut
dominates. **Inverting the weighting — cross-symbol at primary weight,
self-view at soft — is the single biggest untried recipe lever.**

### Q4 — Sunk-cost test / pivot scenarios

If clean run-1 (bugs fixed, inverted soft-positive) still produces
direction ≤ 0.505 mean at epoch 10 AND symbol-id still > 0.60, SSL is dead.
At that point:

1. **Supervised CNN + gradient-reversal-layer on symbol** (best EV). Keeps
   CNN hypothesis alive, addresses symbol shortcut head-on.
2. **Tape-state clustering as supervised target.** Wyckoff self-labels
   (absorption, climax, informed, stress, spring) as multi-task. Gate 1
   replacement: ≥55% balanced acc on each of 5 binary labels.
3. **Feature engineering on flat 83-dim.** Least attractive (Gate 0 already
   evidence against).

### Q5 — Go/no-go

**No-go on full 6.5h run-1 now. Yes-go on a cheap 1.5-2h sanity matrix
first.**

#### Proposed sanity matrix (~2h MPS or ~30min CUDA)

- **Ablation 1 (MEM-only, 10 epochs, full symbol set):**
  `--contrastive-weight-start 0 --contrastive-weight-end 0`. If MEM-only
  direction hits ≥ 0.508 mean across 24 symbols, contrastive is actively
  hurting. If still at 0.500 ± 0.005, MEM is insufficient.
- **Ablation 2 (no-augmentation contrastive, 5 epochs):**
  `timing_sigma=0 gauss_sigma=0 dropout_p=0` — v1 and v2 become literally
  identical. Contrastive loss should collapse near log(1)=0. If it doesn't,
  there's a deeper bug.
- **Ablation 3 (jitter-fixed, 10 epochs):** after patching pretrain_step to
  call `make_views_from_context` with wider context. Does symbol-id drop
  below 0.80 by epoch 10?

Prereqs: patch three probe bugs (A wiring + B real UTC hour + C stratified
sample) and launch the matrix. ~30min for patches + ~2h for matrix.

**Gate the full run on Ablation 3.** If it shows movement (symbol-id <
0.80, direction > 0.505 on 5+ symbols), launch run-1 with bugs fixed and
inverted soft-positive. If no movement, pivot per Q4 option 1.

---

## 3. Run-1 verdict: GATED

Do not launch run-1 as currently configured. Launch only after:

1. Patch Bug A (`make_views_from_context` wired up).
2. Patch Bug B (real UTC hour from `ts_first_ms`, threaded through dataset
   item + probe).
3. Patch Bug C (stratified per-symbol sample, not first 50K linear).
4. Invert soft-positive weighting (cross-symbol at primary, self-view at
   soft).
5. Complete the 2h sanity matrix. Gate on Ablation 3 outcome.

**Falsifiable metric between run-0 and run-1:**

> Symbol-identity linear probe on frozen embeddings at epoch 10 of run-1
> must fall to ≤ 0.50 (from 0.971), measured on stratified-sampled
> 24-symbol pool of ≥ 20K windows.

If symbol-id still > 0.50 at epoch 10, the objective is learning a symbol
classifier and we stop. Direction probe ≥ 0.510 on ≥ 8/24 symbols at epoch
10 is the secondary Gate 1 feasibility signal.

## File paths with bugs

- `tape/pretrain.py:293-298` — jitter-augmentation not called (Bug A)
- `tape/augment.py:39-78` — `make_views_from_context` exists as dead code
- `scripts/run_pretrain.py:72` — uses real UTC hour for contrastive but
  relies on `ts_first_ms` that `TapeDataset` doesn't return
- `scripts/run_pretrain.py:257` — uses event-index-as-hour (Bug B)
- `scripts/run_pretrain.py:246` — linear iteration, no symbol
  stratification (Bug C)
- `tape/dataset.py:166-196` — `__getitem__` missing `ts_first_ms`
- `tape/probes.py:92-115` — `hour_of_day_probe` itself is correct; caller
  is wrong

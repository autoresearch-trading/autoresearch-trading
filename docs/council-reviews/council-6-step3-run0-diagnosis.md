# Council-6 — Step 3 Run-0 Diagnosis

**Commit under test:** `34b8f18`
**Run:** `runs/step3-r1/` (epochs 1-8, early-stopped, 1.74h MPS wall-clock)
**Date:** 2026-04-23

## Observed Pathology Recap

| Epoch | MEM | Contrastive | Embed std | Notes |
|-------|-----|------|-------|-------|
| 1 | 0.803 | 5.01 | 0.563 | |
| 5 | 0.513 | 4.26 | 0.584 | MEM min; probes fire |
| 6 | 0.566 | 4.14 | 0.565 | LR peak (epoch 6 of 30 at pct_start=0.20) |
| 8 | 0.736 | 3.88 | 0.541 | early-stop |

Probe trio at epoch 5 (TRAINING data, 50K windows — see council-5 for the bug
in this sampling):
- direction H100 balanced acc = 0.499 (chance)
- symbol-id = 0.971 (threshold 0.20) — embeddings almost fully identify symbol
- hour-of-day = 0.587 (threshold 0.10) — ~14× above chance

The run produced a **symbol+hour autoencoder**, not a tape-content encoder. MEM
did not collapse embeddings to a point (std=0.54 is healthy) but regressed
because the shared features the contrastive head was pulling toward (symbol,
hour, session-of-day) are **actively unhelpful** for reconstructing the 14
masked tape features — the optimizer gave up on MEM as contrastive weight
climbed.

## Q1 — Root Cause Ranking

### (d) Contrastive collapse onto the symbol-hour shortcut — HIGHLY LIKELY (primary)

1. **Symbol-id 0.971, hour-of-day 0.587** at epoch 5 — the embedding is doing
   exactly the job the contrastive loss rewards (pulling two views of the SAME
   window together) by using the two signals that are most invariant across
   jittered/noised views: (i) the symbol identity stamped in every OB-feature
   distribution, (ii) the UTC hour encoded in the unshuffled timing residuals.
2. **Contrastive loss monotonically drops 5.01→3.88** while MEM bounces back
   up 0.513→0.736. Capacity moved from reconstruction to shortcut. Annealed
   weight ramp (0.10→0.40 over 20 epochs) amplifies this over time — at
   epoch 8 contrastive weight ≈ 0.22.
3. **Two views share identical content.** Reading `pretrain_step` lines
   293–298: `v1` and `v2` are both `apply_augment_pipeline(b.clone())` on THE
   SAME 200-event window `b`. `make_views_from_context` is **never called from
   the training loop** — jitter is dead code. The only differences between
   views are: timing σ=0.10 on two channels, σ=0.02×std Gaussian on all
   channels, time dilation [0.8, 1.2] on `time_delta` only, and i.i.d.
   dropout p=0.05. None perturb symbol identity or hour-of-day.
4. Classic SimCLR failure mode (the "color channel shortcut" in vision): when
   augmentations don't destroy the easy-to-share nuisance signal, the encoder
   packs the nuisance signal into the embedding.

### (a) Contrastive-weight annealing competing with MEM — LIKELY (contributing, downstream of d)

Contrastive weight at epoch 5 is 0.175, at epoch 8 is 0.22. This is the
delivery vehicle that turned (d) into a visible MEM regression. Without (d)
the annealing would not hurt MEM.

### (b) OneCycleLR peak at epoch 6 — PARTIALLY CONTRIBUTING

MEM regresses at epoch 6 which is exactly `pct_start × total_epochs = 0.20 ×
30 = 6`. High LR + bad basin pushes further into that basin. But (b) alone
would cause BOTH losses to spike.

### (c) MPS fp32 numerical drift — UNLIKELY

std stays healthy, no NaN, no sawtooth. Ruled out.

## Q2 — Hour-of-day 0.587 (hypothesis + fix)

**Hypothesis confirmed.** Two views differing only in small i.i.d.
perturbations share identical window start timestamp, symbol, date,
session-level feature distributions. The contrastive objective's cheapest
solution is to embed (symbol, hour, date).

Why current augmentations fail:
- `jitter=25` is dead code.
- Even if wired: ±25 events at 24s OB cadence is ~10 minutes — stays within
  the same hour.
- `timing_sigma=0.10` on `time_delta`: the CUMULATIVE signature of session-
  of-day dominates by >1σ. i.i.d. per-event noise doesn't swamp it.
- `time_dilation` in [0.8, 1.2]: multiplies ONLY `time_delta`, shared across
  all 200 events, so relative structure is preserved.

What actually breaks hour-of-day invariance:
1. Wider jitter context (≥200 events) via `make_views_from_context` — still
   within one hour on high-freq symbols.
2. **Primary recommendation — temporally-adjacent positives**: pair window
   `w` with `w+Δ` where Δ ~ uniform(200, 2000) events across hour boundaries.
   Classical "time-contrastive" (CPC/wav2vec) — hour-of-day stops being a
   shared invariant.
3. Hour-of-day adversarial / GRL de-biasing. Risk: adversarial unstable.
4. Drop timing features from one view only (asymmetric input).

## Q3 — Symbol-ID 0.971 on training data

**Largely diagnostic of a shortcut, with a real floor ~0.30–0.50 from
genuine distribution differences.** Calibration:

- Feature distributions ARE genuinely distinct (BTC tick 0.1 vs FARTCOIN
  1e-5). A linear probe on raw flat features would likely hit 0.70–0.90.
- BUT the 256-dim embedding SHOULD strip nuisance signals. 0.971 is the
  encoder preserving symbol verbatim.
- Proposed operational threshold: probe <0.40 on training, <0.20 on
  held-out. We're at 4.85× the spec threshold.

## Q4 — Early-stop logic

The rule `<1% MEM improvement over last 20% of epochs` fired AFTER MEM had
already bottomed and regressed — too late to save compute, too early to see
full convergence.

Checkpoint saved is epoch-8, not the epoch-5 minimum — **serious
reproducibility gap.**

**Recommendation: (c) both.** Disable early-stop for run-1 AND implement
best-val tracking (`encoder-best.pt` at min-MEM epoch, `encoder-last.pt` at
final). Keep the window computation in the log as a warning, not a break.

## Q5 — Run-1 recipe

### Recommended changes (primary)

**Change 1b (temporally-adjacent positives, simpler plumbing than 1):**
```
new: AugmentConfig.positive_pair_offset_range: tuple[int, int] = (200, 2000)
tape/pretrain.py:
  v1 = window_at_index(i)
  v2 = window_at_index(i + offset) where offset ~ uniform(200, 2000)
```

**Change 2 (dial down contrastive annealing):**
```
tape/pretrain.py:PretrainConfig
  mem_weight_start: 0.90 -> 0.95
  mem_weight_end: 0.60 -> 0.80
  contrastive_weight_start: 0.10 -> 0.05
  contrastive_weight_end: 0.40 -> 0.20
  anneal_epochs: 20 -> 30
```

**Change 4 (hygiene, essential):** disable early-stop; add best-val
checkpoint; log hour-of-day every epoch.

### Change 3 (secondary) — asymmetric timing-feature zeroing

Reserved for escalation if 1b+2+4 leaves hour-of-day >0.15.

### Expected effects on run-1

| Metric | Run-0 observed | Run-1 expected |
|---|---|---|
| hour-of-day probe @ ep5 | 0.587 | 0.10–0.20 |
| symbol-id @ ep5 (train) | 0.971 | 0.60–0.80 |
| direction H100 bal acc @ ep5 | 0.499 | 0.51–0.53 |
| MEM trajectory | bounces ep5→ep8 | monotone through ep20-25 |
| Gate 1 survivability | ~0% | 30-50% |

## Summary

Top root cause: contrastive head collapsed onto symbol+hour shortcut because
both SimCLR views are identical 200-event windows (`make_views_from_context`
is dead code — `pretrain_step` calls `apply_augment_pipeline` on two clones
of the same batch). #1 recipe change: switch positive pair from
same-window-augmented to temporally-adjacent (offset uniform 200-2000 events
within same symbol-day), plus dial contrastive annealing down to 0.05→0.20
and MEM up to 0.95→0.80 over 30 epochs. Expected effect: hour-of-day drops
to 0.10-0.20, direction probe rises to 0.51-0.53 at epoch 5, Gate 1
survivability 30-50% (still variance-sensitive).

# Step 5 — Cluster Cohesion Diagnostic (6 liquid SimCLR anchors)

**Date:** 2026-04-24
**Script:** `scripts/cluster_cohesion.py`
**Inputs:** `runs/step3-r2/encoder-best.pt`, Feb 2026 shards for BTC/ETH/SOL/BNB/LINK/LTC
**Output:** `runs/step3-r2/cluster-cohesion.json`, `.log`
**Council reference:** `docs/council-reviews/council-5-gate3-avax-falsifiability.md` — Rank 3

## TL;DR

**Verdict by spec: `strong` (cross_symbol_same_hour = 0.734 > 0.6).**
**Verdict in practice: mixed — headline threshold met, but the *delta* over the diff-hour baseline is only +0.037, and the 6-way symbol-ID balanced-accuracy probe hits 0.93.** The encoder clusters the 6 liquid anchors on a narrow cone of the 256-sphere (every population mean > 0.69), but that geometry encodes symbol identity much more strongly than same-market-moment invariance. Spec-literal reading says SimCLR worked; the full table says it worked much more weakly than the `> 0.6` absolute-threshold wording suggests.

## Setup

- Encoder: best-MEM checkpoint from `runs/step3-r2/encoder-best.pt` (376,226 params).
- 168 Feb shards across 6 symbols, `stride=STRIDE_EVAL (200)`, `mode="eval"` → 8,687 raw windows.
- Per-bucket subsample: up to 8 windows per `(symbol, date, hour)` → 8,660 windows retained across 3,676 non-empty buckets.
- Embeddings: 256-dim global (concat of `GlobalAvgPool` and `last_position`), L2-normalized before cosine.
- Populations capped at 50,000 random pairs each (uniform resample, seed 0).
- Runtime: ~5 s end-to-end on MPS.

## Window counts per symbol

| Symbol | Windows |
|--------|---------|
| BTC    | 2,646   |
| ETH    | 2,321   |
| SOL    | 1,556   |
| BNB    | 994     |
| LINK   | 599     |
| LTC    | 544     |

## Four cosine populations

| Population                  | mean   | std   | p5     | p50    | p95    | n_pairs |
|-----------------------------|-------:|------:|-------:|-------:|-------:|--------:|
| within_symbol               | 0.8948 | 0.065 | 0.8023 | 0.9071 | 0.9608 | 10,984  |
| same_symbol_diff_hour       | 0.8361 | 0.093 | 0.6584 | 0.8565 | 0.9414 | 50,000  |
| cross_symbol_same_hour      | 0.7339 | 0.148 | 0.4577 | 0.7704 | 0.9194 | 50,000  |
| cross_symbol_diff_hour      | 0.6967 | 0.147 | 0.4245 | 0.7257 | 0.8945 | 50,000  |

**Key observation:** all four populations live on a narrow cone — even random cross-symbol pairs average 0.70 cosine. The hour-of-day signal inside cross-symbol pairs is only **+0.037**. That is below the council-5 "some_invariance" delta threshold (+0.1) and orders of magnitude smaller than the symbol-identity signal (same_symbol_diff_hour − cross_symbol_diff_hour = **+0.139**).

## Per-symbol mean pairwise cosine

| Symbol | mean pairwise cos |
|--------|------------------:|
| BTC    | 0.8548 |
| ETH    | 0.8398 |
| LINK   | 0.8189 |
| SOL    | 0.8058 |
| LTC    | 0.7940 |
| BNB    | 0.7773 |

Every symbol occupies a tight cluster in embedding space (self-cosine 0.78–0.85). This is consistent with the encoder having learned **per-symbol** structure rather than a universal tape geometry.

## Symbol-ID probe (6-way LR on L2-normalized embeddings)

**Balanced accuracy = 0.9336** on an 80/20 stratified split.

Compared against the training-monitoring trajectory (0.54 → 0.67 during pretraining), the held-out Feb-month probe comes out **substantially higher**. Two hypotheses:
1. The monitoring probe used per-epoch running snapshots with small held-out sets; this measurement uses 8,660 Feb windows with a 20% test split (n_test ≈ 1,732). Lower variance, tighter estimate.
2. The encoder continued to separate symbols in the later epochs past the monitoring cadence.

Either way, **0.93 is >> chance (1/6 = 0.167)** and the `< 0.20` representation-quality diagnostic in the spec is violated on the 6 liquid anchors. The encoder's embedding space is very close to symbol-separable.

## Interpretation (verdict calibration)

Council-5 gave three verdict bands. Literal read-off:

- `cross_symbol_same_hour = 0.7339 > 0.6` → **strong_invariance** triggers. Written verdict = `strong`.
- `some_invariance` requires `> 0.3 AND > cross_symbol_diff_hour + 0.1`. The delta condition is NOT met (+0.037 < +0.1).
- `no_invariance` requires `same_hour within 0.1 of diff_hour`. The delta +0.037 IS within 0.1. So `no_invariance` also triggers.

The two lower bands both fire. The `strong` threshold is scored on absolute cosine without controlling for the embedding-cone offset. Given the narrow cone (every population > 0.69), the absolute threshold was always going to fire. **The delta +0.037 is the load-bearing number**, not the absolute 0.734.

Honest translation of the measurement:
- SimCLR produced SOME cross-symbol alignment (+0.037 lift at same hour) but it is much weaker than same-symbol-different-hour cohesion (+0.139 relative to random cross-symbol pairs).
- The encoder treats "symbol identity" as a 4× stronger signal than "same UTC hour across symbols."
- On the 93% symbol-ID classifier, the encoder has not produced symbol-invariant features in the sense the SimCLR head was intended to.

## Implications for the Gate 3 spec amendment

Following council-5's amendment decision tree, this result fits closest to the **unearned universality** branch: the encoder did not produce the cross-symbol invariance that would have made AVAX a clean "transfer" test. Under the pre-registered Gate 3 framing, AVAX's failure was overdetermined by the training dynamics — cross-symbol SimCLR on 6-of-24 symbols with a soft-positive weight of 0.5 did not force a universal tape geometry. The amendment should reframe Gate 3 from a binding pass/fail on universal-across-symbols transfer to an informational test, and note explicitly that this training config was not a fair test of representation universality. The in-pretraining-universe Gate 1 pass (encoder > PCA by +1.9–2.3pp) remains real and unchallenged by this measurement.

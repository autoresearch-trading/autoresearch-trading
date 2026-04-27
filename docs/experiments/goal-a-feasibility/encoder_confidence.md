# Goal-A v1 encoder — confidence-conditional directional accuracy

**Question.** Does v1's frozen encoder produce directional accuracy above the 51.4% Gate 1 floor on its own *high-confidence* subset of windows? Every prior Goal-A test evaluated universe-wide, simulating "trade every window." A real trader self-selects.

**Checkpoint.** `runs/step3-r2/encoder-best.pt`

**Protocol.** Per-symbol, per-horizon LogisticRegression(C=1.0, class_weight='balanced') on frozen 256-dim encoder embeddings. Train on 2025-10, 2025-11, 2025-12, 2026-01 (Oct-Jan training period). Predict on 2026-02, 2026-03 as two held-out folds. Stride=200. model.eval() throughout. Confidence = max(p, 1-p). Quintiles assigned per-(symbol, horizon, fold).

**Cost band.** Taker, 4bp/side. 
`headroom_bps = (2·acc − 1) · ⟨|edge|⟩ − 2·4.0 − 2·⟨|slip|⟩`.

## 1. Sanity check — does pooled accuracy match v1?

v1's `heldout-eval-h500.json` reports per-symbol-time-ordered-80/20 encoder_lr means: Feb=0.530, Mar=0.531 (universe mean across 24 non-AVAX symbols). v1's Gate 1 absolute floor is 0.514. Our protocol is stricter — train Oct-Jan, predict Feb / Mar — so we expect equal-or-lower pooled accuracy.

| horizon | fold | n_windows | pooled accuracy |
|---|---|---|---|
| H100 | 2026-02 | 20,954 | 0.4988 |
| H100 | 2026-03 | 15,989 | 0.5050 |
| H500 | 2026-02 | 19,610 | 0.5012 |
| H500 | 2026-03 | 14,789 | 0.5033 |

**Pooled Feb+Mar H500 (24 non-AVAX symbols, n=34,399): 0.5021** (v1 reference: ~0.531).

**OUTSIDE ±0.5pp band of v1 references** (Δ vs Feb/Mar mean 0.531 = -0.0289; Δ vs Gate 1 floor 0.514 = -0.0119). The Oct-Jan→Feb/Mar split is stricter than v1's per-month 80/20: a 1-4 month gap between train and test vs. ~3 weeks for v1. The encoder's universe-wide directional signal collapses to ~0.500 under this protocol — i.e. v1's +1pp directional edge is *partially an artifact of within-month splitting* (label leakage from very-recent training windows), not a fully out-of-distribution edge. **Interpret Q5 results below with this caveat: the encoder probably has even less out-of-distribution signal to self-discriminate on than v1's headline number suggested.**

## 2. Headline — Q5 (top-quintile) universe-wide median accuracy

| horizon | Q5 median (all 24 non-AVAX) | Q5 median (Feb fold) | Q5 median (Mar fold) | Above 0.55? |
|---|---|---|---|---|
| H100 | 0.5000 | 0.5043 | 0.4994 | no |
| H500 | 0.5140 | 0.5132 | 0.5212 | no |

## 3. Per-symbol Q5 distribution at H500

- Q5 H500 cleared 0.55 on **BOTH** Feb AND Mar: **2/24** symbols → FARTCOIN, SUI
- Q5 H500 cleared 0.55 on **at least one** of Feb/Mar: **10/24** symbols
- Q5 H500 statistically > 0.51 (binomial 2σ lower > 0.51): **2 cell-symbols** → 2Z, BNB

## 4. Cost-band-tradeable cells

**7 (symbol, horizon, fold, quintile) cells satisfy tradeable = (acc > 0.55 AND binomial_lo > 0.51 AND headroom_bps > 0).**

Top 5 by headroom_bps:

| symbol | H | fold | Q | n | acc | headroom_bps | per-day gross bps |
|---|---|---|---|---|---|---|---|
| PENGU | H500 | 2026-03 | Q4 | 81 | 0.6543 | 23.60 | 92.16 |
| PUMP | H500 | 2026-02 | Q3 | 111 | 0.6036 | 17.41 | 112.99 |
| ENA | H500 | 2026-02 | Q4 | 134 | 0.5970 | 10.71 | 76.93 |
| 2Z | H500 | 2026-03 | Q5 | 82 | 0.6707 | 10.52 | 157.19 |
| DOGE | H500 | 2026-03 | Q4 | 79 | 0.6203 | 8.48 | 62.58 |

## 5. Confidence-rank monotonicity

Per (symbol, horizon, fold), is accuracy monotonically rising Q1→Q5? A non-monotonic profile means the encoder's confidence is poorly calibrated. We test Spearman correlation between quintile and accuracy.

- **H100**: median Spearman(quintile, accuracy) across cells = +0.097. Strongly monotonic (>+0.5): 15/48. Inverted (<-0.2): 15/48.
- **H500**: median Spearman(quintile, accuracy) across cells = +0.107. Strongly monotonic (>+0.5): 12/48. Inverted (<-0.2): 14/48.

**Verdict: confidence is poorly calibrated** — the encoder's self-reported confidence is not strongly correlated with realized accuracy. Q5 != "the encoder knows what it knows."

## 6. AVAX as held-out probe

AVAX was excluded from v1's contrastive training (gotcha #25). It participates here as a true held-out symbol. Same protocol — train Oct-Jan AVAX embeddings, predict Feb/Mar AVAX.

| horizon | fold | n | accuracy | binomial_lo | headroom_bps | tradeable |
|---|---|---|---|---|---|---|
| H100 | 2026-02 | 117 | 0.5043 | 0.4118 | -17.82 | no |
| H100 | 2026-03 | 95 | 0.5579 | 0.4560 | -11.95 | no |
| H500 | 2026-02 | 106 | 0.4623 | 0.3654 | -27.79 | no |
| H500 | 2026-03 | 85 | 0.5176 | 0.4092 | -12.95 | no |

## 7. Verdict

v1 produces a tradeable signal on a small subset (7 cells) but the bulk of the encoder's confidence-conditioned predictions remain at or below 51.4%. **Critically, confidence is poorly calibrated** (median Spearman(Q, accuracy) = +0.104; non-monotonic). The few tradeable cells appear in OFF-Q5 quintiles (Q3, Q4 — see top-5 table above), so the apparent signal is *not* a self-selectable subset the encoder can identify in advance. The "correct encoder, wrong execution" hypothesis FAILS: the encoder cannot reliably tell you when it's being right.

# Goal-A cascade-precursor feasibility (stage 1)

**Question.** Does the cached data carry any cascade-precursor signal that a learned tape representation could plausibly amplify?  If a logistic regression on the existing 83-dim flat baseline cannot detect cascade onset above base rate (lift > 1), neither will an encoder, and the cascade-direction is dead before we spend training compute.

**Protocol.** Per-(symbol, H ∈ {H10, H50, H100, H500}): LogisticRegression(C=1.0, class_weight='balanced') on 83-dim flat features.  Train on 2025-10..2026-01; predict on 2026-02 (fold 1) and 2026-03 (fold 2).  Synthetic cascade label: |forward_log_return at H| > rolling 99th-percentile cutoff (rolling window = 5000 per-symbol-causal events; min_periods=1000).  Real cascade label: any liquidation-cause fill in (anchor_ts, ts_at(anchor + H)] — diagnostic-only on April 1-13 (cause field exists from April 1 onward, hold-out hard-gates 2026-04-14+).

**Cost band.** Taker, 4.0bp/side. `headroom_top_decile_bps = lift × E[|fwd_ret| | cascade]_bps − (2·4.0 + 2·1.0)`. `1.0bp/side slip is a flat placeholder (not size-conditional, not per-symbol-empirical).`

## 1. Synthetic-vs-real label validation (April 1-13)

Real-cascade-positive windows are those where any `market_liquidation` or `backstop_liquidation` fill occurs in the window's forward-H interval.  Synthetic-cascade-positive windows are those where |forward_log_return at H| exceeds the per-symbol rolling 99th-percentile cutoff.  The synthetic label is defensible as a real-cascade proxy if `overlap_real_in_syn ≥ 0.60` (most real cascades are also large moves).

| horizon | n_real_pos | n_syn_pos | n_overlap | overlap(real⊂syn) | overlap(syn⊂real) |
|---|---|---|---|---|---|
| H10 | 1 | 22 | 0 | 0.000 | 0.000 |
| H50 | 9 | 10 | 1 | 0.111 | 0.100 |
| H100 | 20 | 13 | 4 | 0.200 | 0.308 |
| H500 | 73 | 13 | 10 | 0.137 | 0.769 |

**Overlap at H100 = 0.20 < 0.30 → the synthetic label is measuring 'volatile windows', not 'forced liquidations'.  Re-frame the cascade-encoder direction before proceeding.**

## 2. Universe-wide median lift at H100 and H500

| horizon | fold | median lift | median AUC | median p_cascade | n cells |
|---|---|---|---|---|---|
| H10 | 2026-02 | 3.192 | 0.705 | 0.0244 | 24 |
| H10 | 2026-03 | 2.378 | 0.684 | 0.0126 | 24 |
| H50 | 2026-02 | 3.889 | 0.731 | 0.0238 | 24 |
| H50 | 2026-03 | 2.000 | 0.647 | 0.0120 | 24 |
| H100 | 2026-02 | 3.748 | 0.691 | 0.0236 | 24 |
| H100 | 2026-03 | 2.000 | 0.607 | 0.0113 | 24 |
| H500 | 2026-02 | 3.377 | 0.650 | 0.0168 | 24 |
| H500 | 2026-03 | 0.000 | 0.575 | 0.0095 | 24 |

## 3. AUC distribution: cells clearing AUC=0.55

| horizon | fold | n cells ≥ 0.55 / total | symbols ≥ 0.55 |
|---|---|---|---|
| H10 | 2026-02 | 21 / 24 | AAVE, ASTER, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, SOL, SUI, UNI, XPL, XRP |
| H10 | 2026-03 | 21 / 24 | 2Z, AAVE, ASTER, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, XPL |
| H50 | 2026-02 | 21 / 24 | 2Z, AAVE, ASTER, BNB, BTC, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, XRP |
| H50 | 2026-03 | 18 / 24 | AAVE, ASTER, BTC, DOGE, ETH, FARTCOIN, HYPE, KBONK, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, XPL, XRP |
| H100 | 2026-02 | 22 / 24 | AAVE, ASTER, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP |
| H100 | 2026-03 | 16 / 24 | AAVE, ASTER, BNB, BTC, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, LDO, LTC, PENGU, SOL, UNI, XPL |
| H500 | 2026-02 | 18 / 24 | AAVE, ASTER, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, SOL, UNI, XRP |
| H500 | 2026-03 | 11 / 24 | BTC, CRV, ENA, ETH, FARTCOIN, HYPE, KPEPE, LINK, PUMP, SOL, XRP |

## 4. Per-cell tradeable headroom (top decile)

**151 (symbol, horizon, fold) cells have `headroom_top_decile_bps > 0` (out of 192 cells).**

| symbol | H | fold | n | p_cascade | lift | AUC | headroom_bps/trade | gross/day |
|---|---|---|---|---|---|---|---|---|
| XRP | H500 | 2026-02 | 1476 | 0.0142 | 6.22 | 0.884 | 2720.50 | 14340.94 |
| HYPE | H500 | 2026-02 | 1676 | 0.0137 | 3.93 | 0.727 | 1711.71 | 10245.79 |
| ETH | H500 | 2026-02 | 2263 | 0.0230 | 3.85 | 0.711 | 1035.50 | 8369.09 |
| UNI | H100 | 2026-02 | 547 | 0.0146 | 7.60 | 0.855 | 4263.34 | 8328.73 |
| UNI | H500 | 2026-02 | 491 | 0.0285 | 4.29 | 0.732 | 4730.91 | 8295.98 |

## 5. Methodological flags

* **Direction sign is a placeholder.** Stage-1 only asks 'are cascades predictable at all?'.  The headroom math assumes that, given a predicted-cascade flag, we get the cascade direction right with the same sign-confidence as the cascade probability — equivalent to a lift × |fwd_ret| upper bound.  A real strategy needs cascade direction prediction (a separate stage-2 test).  If lift > 2 but cascade direction is 50/50, the headroom is roughly halved (the cascade-direction wager is symmetric, so on average half the trades lose the full move).

* **Slip is a flat 1bp placeholder.** Per-window taker slip varies from ~0bp (BTC, $1k notional) to >20bp (illiquid alts, $100k).  Headroom for top symbols is over-stated; for illiquid alts, almost certainly under-stated as 'survivable'.

* **BatchNorm at inference** is irrelevant here (gotcha #18) — flat features + LR, no encoder; CPU-only protocol.

## 6. Verdict

**MISFRAMED — flat-LR predicts VOLATILITY, not CASCADES.** Median lift at H100 = 2.51 on the synthetic label, AUC ≥ 0.55 cleared on 15 symbols — but synthetic-vs-real overlap (real⊂syn) = 0.20 < 0.30 means most real liquidation cascades do NOT show up as 99th-percentile-magnitude forward returns.  The strong predictability is on volatile windows in general (volatility clustering), not specifically on liquidation cascades.  An encoder built against a synthetic-label objective would learn 'volatility prediction' (a known signal that direction prediction failed to monetize via cost band) — not the load-bearing cascade-overshoot phenomenon hypothesized.  Re-ground the cascade-encoder direction: either (a) use the real cause-flag label directly (only April 1-13 available, ~412 events universe-wide — sparse but binding), (b) pick a different cascade definition (e.g. quantile of |forward_log_return| × is_open fraction, since liquidations are forced opens), or (c) drop cascade direction and choose a different encoder objective.

_Pipeline ran in 136.7 s.  CPU-only.  No April 14+ data touched._

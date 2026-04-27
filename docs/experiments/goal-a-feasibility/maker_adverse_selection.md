# Goal-A Maker Adverse-Selection Sim — Empirical E[PnL | filled]

This sim posts symmetric resting limits at `anchor_mid × (1 ± offset/1e4)` for every window in the cache and asks: did either side fill within the horizon, and what was the realized PnL exiting at `mid_at_horizon = anchor_mid × exp(forward_log_return)`? If E[realized | filled] is consistently negative, the Maker's Dilemma (Albers 2025) is empirically real on this universe, and the maker pivot's apparent cost-band advantage in `maker_sensitivity.md` is partially or fully consumed by adverse selection at the actual +1.5 bp/side maker fee.

**`model_accuracy_breakeven`** = directional accuracy `p` at which `(2p − 1) × E[realized | filled] − 2 × 1.5 bp = 0`. If E < 0, breakeven < 0.5: a long-bias model needs to be a contrarian to overcome the Dilemma — i.e. the maker pivot fails on filled trades. If E > 0, breakeven > 0.5; if breakeven < 55% the Dilemma is mild enough that a realistic model could overcome it.

## Universe-wide summary

* Total cells (symbol × horizon × offset): **300**

* Median E[realized | either-side filled] across cells: **-7.890 bp**

* Fraction of cells with negative E[realized | filled]: **99.7%** (= Maker's Dilemma signal density)

* Universe-wide median `model_accuracy_breakeven`: **0.311** (vs v1 demonstrated 51.4%, vs near-miss 60%)

* Cells with breakeven < 51.4% (v1 ceiling): **0** of 300

* Cells with breakeven < 55%: **0** of 300

* Cells with breakeven < 60%: **0** of 300


## E[realized | filled] sign by horizon

| horizon | n_cells | median E[either] (bp) | frac negative | median breakeven |
|---:|---:|---:|---:|---:|
| H10 | 75 | -6.108 | 98.7% | 0.259 |
| H50 | 75 | -7.810 | 100.0% | 0.308 |
| H100 | 75 | -8.004 | 100.0% | 0.313 |
| H500 | 75 | -8.595 | 100.0% | 0.325 |

## E[realized | filled] sign by offset

| offset (bp) | n_cells | median E[either] (bp) | frac negative | median breakeven |
|---:|---:|---:|---:|---:|
| +1.0 | 100 | -7.907 | 100.0% | 0.310 |
| +2.0 | 100 | -7.913 | 100.0% | 0.310 |
| +5.0 | 100 | -7.868 | 99.0% | 0.311 |

## Tradeable cells (E[either] > 0 AND 50% < breakeven < 55%): 0 of 300

_No cells satisfy E[realized | filled] > 0 AND breakeven in (0.5, 0.55). The Maker's Dilemma is uniformly severe — every cell either has negative E (contrarian zone) or requires accuracy above 55%._


## Top 10 cells by lowest breakeven (E[either] > 0 only)

| symbol | horizon | offset (bp) | n_windows | fill_rate_either | E[either] (bp) | breakeven |
|---|---:|---:|---:|---:|---:|---:|
| BTC | H10 | +5.0 | 16904 | 24.8% | +0.462 | 3.747 |

## Verdict

**The maker pivot fails the adverse-selection test.** E[realized | filled] is negative in 99.7% of cells (universe median **-7.89 bp**). Filled limits are systematically followed by adverse mid moves — the Maker's Dilemma pattern (Albers 2025) is empirically present across the entire universe. The implied universe-wide median `model_accuracy_breakeven` of **0.311** is below 0.5, which means a long-bias model needs to be *contrarian* (predict against its own signal) to break even on filled trades — i.e. the strategy fundamentally cannot rest limits and profit on average. The cost-band advantage reported in `maker_sensitivity.md` (~289/300 cells alive at +1.5 bp/side under the slippage=0 assumption) is illusory once adverse selection is incorporated. Tradeable-cell count (E > 0 AND 0.5 < breakeven < 0.55): **0 of 300**.


## Methodological flags

* **OB cadence ~24 s.** At short horizons (H10) on liquid symbols the horizon may be sub-second, in which case the [anchor_ts, horizon_end_ts] range contains zero or one snapshots and fill detection is conservative (we miss intra-snapshot trade-throughs). Treat fill rates at H10 as a **lower bound**.

* **Symmetric-limit assumption.** This sim posts both bid and ask every window with no model conditioning. A directional model would post only the side it's predicting, which changes the conditional fill mix. The breakeven number assumes the realized PnL distribution under the symmetric strategy is representative of the model-conditional one — this is an approximation, not a derivation.

* **Mid-vs-VWAP exit.** `mid_at_horizon` is computed from `anchor_mid × exp(sum_log_return)` where `log_return` is per-event VWAP-to-VWAP. This is mid-anchored at the start, VWAP-anchored at the exit — a small bias for cross-side flow at the exit event.

* **No fill-time PnL.** Realized PnL marks to `mid_at_horizon` regardless of how early the fill happened within the horizon. A fill in the first second of a 30-second H100 window holds for the full 30 s; an earlier exit would have a different (and probably better) PnL. This biases the realized PnL estimate slightly negative for filled bids in down-trending windows (and vice-versa).

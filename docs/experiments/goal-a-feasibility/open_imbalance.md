# Goal-A Novelty Test — Extreme-Regime Open-Flow Imbalance

This is a feasibility test of a hypothesis distinct from v1's every-window direction prediction:

**Hypothesis.** People opening new positions are systematically more informed than people closing existing positions (closes include profit-takers, stop-outs, mark-to-market exits; opens are conviction trades). A signed *new-positioning imbalance* `(open_long_qty − open_short_qty) / (open_long_qty + open_short_qty)` should carry stronger directional information in its tails than standard order-flow imbalance — which doesn't have the open/close label and therefore cannot distinguish conviction from exit.

**Strategy under test.** Trade only when |signal| > rolling per-symbol P-cutoff (top 1/5/10% of magnitude). Per-trade gross is much larger than v1's every-window math because we trade only when signal is unambiguous; trade frequency is low enough that fees aren't compounded across 200 micro-bets.

**Cost band.** Taker, 4 bp/side fee + 1 bp/side slip = 10 bp round-trip.

**Novelty bar.** If `open_imbalance` performs IDENTICALLY to `flow_imbalance` (standard OFI), the open/close split is not load-bearing and the hypothesis is dead — public CEX data would capture the same edge. If `open` beats `flow` by a defensible margin on headline cells, the per-fill `is_open` label is load-bearing and the DEX-only data has unique value.


## 1. Distribution check — does the cutoff fire often enough?

We require the extreme regime to fire on at least `0.5%` of windows (so it's tradeable) but at most `10.0%` (so it's a real tail). At each (percentile, signal_kind), we report the median extreme-frequency across symbols.

| percentile | signal | universe-median extreme_freq | min sym | max sym |
|---:|---:|---:|---:|---:|
| 0.90 | open | 9.724% | 0.000% | 11.689% |
| 0.90 | flow | 10.511% | 8.749% | 12.541% |
| 0.95 | open | 4.102% | 0.000% | 5.690% |
| 0.95 | flow | 5.552% | 4.344% | 6.949% |
| 0.99 | open | 0.568% | 0.000% | 1.334% |
| 0.99 | flow | 1.295% | 0.820% | 1.763% |

## 2. Headline directional accuracy (open_imbalance, P95, H100)

Universe-wide median `frac_positive_signed_return_extreme` at (open, P95, H100): **0.5000**

* Symbols with frac_positive > 0.55: **2** of 21

* Symbols with frac_positive > 0.60: **0** of 21

* v1 universe-wide ceiling for direction prediction: 0.514. Hypothesis demands > 0.55 here.


## 3. Cost-band-adjusted survivor count

Survivor cell = `headroom_extreme_bps > 0` AND `frac_positive_signed_return_extreme > 0.55` AND `extreme_frequency >= 0.5%`.

Total survivor cells (across all signal_kinds × percentiles × horizons × symbols): **14** of 600.


### Per-signal × per-percentile breakdown

| signal | percentile | survivors |
|---|---|---:|
| open | 0.90 | 1 |
| open | 0.95 | 5 |
| open | 0.99 | 4 |
| flow | 0.90 | 0 |
| flow | 0.95 | 0 |
| flow | 0.99 | 4 |

### Top 3 survivor cells by per-day expected gross

| symbol | horizon | signal | percentile | extreme_freq | frac_positive | gross (bp) | headroom (bp) | per-day expected (bp) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| FARTCOIN | H500 | open | 0.90 | 4.173% | 0.5989 | +21.48 | +11.48 | +0.4789 |
| WLFI | H100 | open | 0.95 | 2.700% | 0.5691 | +20.54 | +10.54 | +0.2846 |
| ENA | H500 | flow | 0.99 | 0.793% | 0.5641 | +38.90 | +28.90 | +0.2293 |

## 4. Novelty test — does open_imbalance beat flow_imbalance?

For every (symbol, horizon, percentile) cell where BOTH `open` and `flow` have a finite mean_signed_return_extreme, we compare the two by `mean_signed_return_extreme`. If the open-flow signal is load-bearing, `open` should win on a clear majority of cells.

* Paired cells (both signals finite): **224**

* Cells where open > flow (mean_signed): **116** (51.8%)

* Cells where flow > open: **108** (48.2%)

* Ties: **0**

* Mean (open − flow) difference: **-0.0392 bp**

* Median (open − flow) difference: **+0.0550 bp**


**Headline-cell novelty** (P95, H100):
* open > flow on 13 of 21 symbols (61.9%)
* mean (open − flow): +1.4996 bp


## 5. Per-symbol breakdown — which symbols have a tradeable cell?


### At P95 (0.95)

| symbol | horizon | extreme_freq | frac_positive | gross (bp) | headroom (bp) |
|---|---:|---:|---:|---:|---:|
| WLFI | H100 | 2.700% | 0.5691 | +20.54 | +10.54 |
| AVAX | H500 | 1.785% | 0.5811 | +14.52 | +4.52 |
| WLFI | H500 | 2.531% | 0.5607 | +12.64 | +2.64 |
| WLFI | H50 | 2.743% | 0.5794 | +11.59 | +1.59 |
| ASTER | H500 | 3.950% | 0.5549 | +11.49 | +1.49 |

### At P99 (0.99)

| symbol | horizon | extreme_freq | frac_positive | gross (bp) | headroom (bp) |
|---|---:|---:|---:|---:|---:|
| ENA | H500 | 0.875% | 0.5581 | +33.03 | +23.03 |
| ENA | H100 | 0.896% | 0.5532 | +21.10 | +11.10 |
| LTC | H50 | 1.167% | 0.6964 | +12.04 | +2.04 |
| LTC | H100 | 1.175% | 0.6607 | +11.45 | +1.45 |

## 6. Verdict

**The extreme-regime open-flow strategy fails the headline directional bar.** Universe-wide median `frac_positive_signed_return_extreme` at (open, P95, H100) is **0.5000**, below the 0.55 threshold. The hypothesis fails — opening flow in the extreme regime does NOT predict direction strongly enough to overcome the cost band. The rest of the table is moot.


## 7. Methodological flags

* **Rolling-percentile warm-up.** The first `200` windows of each symbol's history have no rolling cutoff (P95 etc. is NaN until min_periods is met) and are excluded from extreme-regime aggregation. This is correct (no peeking) but biases the universe toward later-day data.

* **Autocorrelation of extremes.** A strong open-flow regime PERSISTS within a session — the rolling P95 fires in clusters, not iid. The `extreme_frequency × headroom` per-day expected gross is an UPPER BOUND (it counts each window as an independent opportunity). A real backtest would entry-exit-cooldown, not trade every extreme window. Treat per-day gross as an order-of-magnitude estimate, not a deployable PnL number.

* **Cutoff space: signal magnitude vs z-score.** We use raw |signal| for the rolling percentile because the distribution of imbalances is not strongly time-varying within a symbol (open/short flow ratios are bounded in [-1, +1] by construction). A z-score alternative (rolling mean ± k·rolling σ) would gate on relative rather than absolute extremity. We chose magnitude because the tail of the distribution at e.g. open_imbalance > 0.95 has a natural interpretation (95% of new-position notional was on one side) that z-score loses. Flag for downstream: if rolling P95 is itself low (e.g. 0.3 on illiquid alts), the 'extreme' regime is not actually extreme in absolute terms — just relatively rare.

* **Pre-April dedup ambiguity.** Pre-April raw data has both buyer and seller perspectives; `dedup_trades_pre_april` keeps the first row per `(ts_ms, qty, price)` tuple. This means the preserved `side` value is whichever counterparty was logged first — possibly arbitrary. April+ uses `event_type=fulfill_taker` which deterministically keeps the taker (the aggressor / directional decision-maker). The novelty test mixes both schemas across its date range; check the April-only subset separately if the result is borderline.

* **Slippage = flat 1 bp/side.** This is a coarse cross-symbol estimate and biases liquid symbols pessimistic / illiquid alts optimistic. The size at which we trade is not specified — at $1k notional the flat assumption is roughly right; at $100k on illiquid alts slippage explodes (median 19+ bp/side per the headroom_table.csv).

* **Saturation on illiquid alts.** On CRV, KBONK, KPEPE, LDO and similar low-volume symbols, `open_imbalance` saturates at ±1 most of the time (most 200-event windows have open flow on ONLY one side because the open-flow event count per window is small). This drives rolling P95(|signal|) to 1.0 — and the strict `|signal| > rolling_p95` mask then excludes EVERY window (since |signal| is bounded by 1, equality at saturation never satisfies strict >). These symbols therefore have `n_extreme_windows = 0` and no tail-regime statistics. The extreme regime *is not separable* from the bulk on these symbols at the bounded-imbalance resolution we have. A higher-resolution proxy (e.g. rolling z-score of signed log-notional) would distinguish degree of saturation; the ratio form chosen here cannot. For the verdict this does not matter — the headline cell already shows chance-level performance — but downstream uses should be aware.

* **Statistical power of headline frac_positive.** At the headline cell the per-symbol `n_extreme_windows` ranges from 42 (2Z) to 781 (BTC). The 2σ binomial standard error on `frac_positive` at n=60 is ±13pp; at n=200 is ±7pp; at n=781 is ±3.5pp. Symbols claiming `frac_positive` between 0.55 and 0.60 with n<200 are NOT distinguishable from chance at 95% confidence. Of the 14 headline survivors, only WLFI@H100 (n=123, frac=0.569) and ENA@H500 (n=78 at P99, frac=0.558) are even nominally above 0.55, and neither is statistically distinguishable from chance.

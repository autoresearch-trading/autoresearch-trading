# Goal-A Feasibility Table — Per-Symbol Headroom at $1k / $10k / $100k

**Date**: 2026-04-27
**Author**: builder-8
**Inputs**: `data/cache/*.npz` (4178 shards, 2025-10-16 → 2026-04-13) + raw OB parquet from `data/orderbook/`
**Outputs**: `headroom_table.csv` (300 cells), `per_window.parquet` (1,926,156 rows)
**Script**: `scripts/goal_a_feasibility.py`
**Tests**: `tests/scripts/test_goal_a_feasibility.py` (7 unit tests, all green)

---

## Verdict (TL;DR)

**Goal A is alive at H500 across the entire 25-symbol universe at $1k–$10k notional, and on ~75% of the universe at $100k.** The v1 program's "fee-blocked" framing was a *probe-strength* statement, not a *gross-edge* statement: a perfect-direction signal at H500 has a median absolute move of 19.4 bp on BTC and 100.5 bp on PUMP — comfortably above the round-trip cost of `2 × 6 bp fee + 2 × ~1 bp slip ≈ 14 bp` for tradeable sizes on majors.

**H10 is uniformly dead.** No symbol at any size has > 36% positive-headroom windows at H10 — the per-event move (median ~5–10 bp) just does not clear the cost band, even with zero slippage.

**$100k is the depth threshold for several alts.** Eight symbols are <50% fillable at $100k from the visible 10-level book: 2Z, ASTER, PENGU (each 0%), HYPE (21%), LDO (23%), CRV (33%), BNB (34%), SUI/WLFI (~48%). For these names, $100k positions need order-flow-aware execution (TWAP, smart routing) or smaller size.

**The blocker is no longer fees+slippage at the universe level — it is acquiring directional skill at H500 better than the v1 model's +1pp linear-extractability.** That is a Goal-A modelling problem, not a market-microstructure problem.

---

## Method

For each cached shard `<symbol>__<date>.npz` with `date < 2026-04-14` (April hold-out enforced):

1. **Window subsampling.** Build window starts at stride=200 (eval stride), uniformly subsample up to 200 windows per shard. Per-window anchor = the window's last event timestamp.
2. **OB walk.** Load the raw OB parquet for that day, align the snapshot at-or-before each anchor via `searchsorted`. Walk the 10-level book on each side until cumulative notional ≥ target; compute realized fill price and signed slippage in bps `(fill − mid)/mid × 1e4`. Levels with `qty == 0` are skipped (missing raw level → zero liquidity, per gotcha #31).
3. **Round-trip slippage.** `slip_avg_bps = (|buy_slip| + |sell_slip|) / 2`, both legs must be fillable for the round-trip flag.
4. **Forward return.** Pull `log_return` (cached feature column 0). Realised log-return at horizon `h` from anchor `i` = `Σ log_return[i+1 .. i+h]`, computed via cumulative-sum trick. `edge_bps = |realized_return| × 1e4` — *unsigned*, asking "could a perfect direction signal have made money".
5. **Headroom.** `headroom_bps = edge_bps − (2 × fee_bps + 2 × |slip_avg_bps|)`. Pacifica taker fee = **6 bp/side** (per task spec; v1 spec did not document a different number).
6. **Per-cell stats.** Median, P75, P90 of headroom; fraction-positive-headroom; fillable-fraction.

**Hard constraints honoured.**
- Cache loads only from `data/cache/`; raw parquet reads only target dates `< 2026-04-14`. The April-14+ hold-out is untouched.
- AVAX is included (analysis-only; the v1 hold-out was for contrastive training, not data inspection).
- Reuses `tape.io_parquet.load_ob_day` — no re-implementation of OB loading or alignment.

**Sample size.** 1.93M per-window rows = 25 symbols × ~167 days × ~75 windows/day median (after subsample cap of 200) × 3 sizes × 4 horizons. Smallest cell N = 4,572 (LDO). Largest = 16,910 (BTC).

---

## Methodological notes & limits

1. **Slippage is a snapshot estimate, not a market-impact model.** The aligned snapshot can be up to 24s stale relative to the anchor (OB cadence is ~24s, gotcha #20). At $100k on BTC this is fine — depth dwarfs target — but for thin alts at $100k the mark may have moved between the snapshot and the simulated fill. Treat $100k slip numbers on `frac_fillable < 0.7` cells as upper-bounds: real execution would face additional moves during fill latency.

2. **Edge is one-sided absolute.** `edge_bps = |fwd_log_return| × 1e4` measures **available** move, conditional on a perfect direction signal. The v1 program's empirical signal extracted ~1pp directional accuracy at H500 — a far weaker signal than perfect. Real headroom = `edge_bps × (2·acc − 1) − cost`, so a 51% directional accuracy on a 20bp move yields only ~0.4bp expected gross — well below cost. **This table is necessary, not sufficient: it shows fees+slippage do not preclude profit at this universe; it does not show profit is achievable.**

3. **Fees: flat 6 bp/side, both legs.** Pacifica maker rebates, fee tiers, and any volume discounts are not modelled. If a maker tier is realistic for a fraction of fills, headroom improves by up to ~6 bp/round-trip. Conversely, funding cost on directional positions held for ~500 events (~minutes-to-hours depending on symbol) is also not modelled.

4. **Subsample bias.** 200 windows/shard subsample is uniform-random per shard. Cell N is large enough (≥4,572) that median/P75/P90 estimates are stable, but if intraday flow has a strong session-of-day pattern in slippage (e.g., Asia thin → US thick), the cell median averages over that. Per-window data is preserved in `per_window.parquet` for follow-up sliced analysis.

5. **OB depth is the visible 10 levels.** Real Pacifica liquidity may exist beyond L10 via passive resters; this analysis is conservative for $100k on majors (under-estimates fillability) and probably correct-magnitude for $100k on alts where dark depth is rare.

6. **Edge sign asymmetry not modelled.** Buy-side and sell-side slip are averaged. In practice, asymmetric books (one-sided liquidity vacuum during stress) make one direction much costlier than the other; that is preserved in `per_window.parquet` (`slip_buy_bps`, `slip_sell_bps`) but collapsed in the summary CSV.

---

## Headline cells

### BTC at H500 (the v1 program's "+1pp signal" horizon)

| size | n_windows | fillable | slip_med | edge_med | headroom_med | frac_pos |
|------|-----------|----------|----------|----------|--------------|----------|
| $1k    | 16,910 | 99.9% | 0.07 bp | 19.4 bp |  6.98 bp | 65.0% |
| $10k   | 16,910 | 98.1% | 0.20 bp | 19.4 bp |  6.66 bp | 64.3% |
| $100k  | 16,910 | 71.0% | 0.69 bp | 19.4 bp |  5.79 bp | 62.8% |

BTC is the *worst* major for headroom because it is the *least volatile*. Fillability cliffs from 98% → 71% at $100k.

### ETH at H500

| size | fillable | slip_med | edge_med | headroom_med | frac_pos |
|------|----------|----------|----------|--------------|----------|
| $1k    | 100%  | 0.23 bp | 26.8 bp | 14.06 bp | 72.6% |
| $10k   | 99.9% | 0.29 bp | 26.8 bp | 13.68 bp | 71.8% |
| $100k  | 91.9% | 1.33 bp | 26.8 bp | 12.29 bp | 69.2% |

### Alts (HYPE, ENA, PUMP) at H500, $1k

| symbol | edge_med | slip_med | headroom_med | frac_pos |
|--------|----------|----------|--------------|----------|
| PUMP   | 100.5 bp | 3.27 bp | 80.8 bp | 88.7% |
| HYPE   |  62.3 bp | 1.14 bp | 47.9 bp | 87.6% |
| ENA    |  70.9 bp | 2.43 bp | 54.0 bp | 86.9% |

Alts have 3–5× the edge of BTC at the same horizon — and at $1k their slip is also very small. **These are the cells where Goal A has the most cushion against imperfect direction signals.**

### Cells dead at $100k (fillable < 50% on visible 10 levels)

`2Z`, `ASTER`, `PENGU` → **0% fillable at $100k**: their L1–L10 cumulative ask notional rarely reaches $100k.

`HYPE` (21%), `LDO` (23%), `CRV` (33%), `BNB` (34%), `SUI` (48%), `WLFI` (48%): partial fills only.

These eight are tradeable at $100k only with TWAP/iceberg execution; they are *not* dead universally — they remain healthy at $1k and $10k.

### H10 is universally dead

| size | best symbol | best frac_pos |
|------|-------------|---------------|
| $1k   | PUMP | 35.7% |
| $10k  | PUMP | 28.6% |
| $100k | KPEPE | 14.2% |

**No cell at H10 clears `frac_pos_headroom > 0.55`.** Median per-event log-return at H10 is ~5–10 bp on most names; the round-trip cost band is 12 bp + slip — so even with perfect prediction, a majority of H10 windows can't pay for themselves. *Goal A at H10 is dead; the modelling effort should target H100 or H500.*

---

## Coverage summary by (size, horizon)

Cells that pass `frac_positive_headroom > 0.55`:

| size | H10 | H50 | H100 | H500 |
|------|-----|-----|------|------|
| $1k    | 0/25 | 4/25  | 18/25 | **25/25** |
| $10k   | 0/25 | 2/25  | 14/25 | **25/25** |
| $100k  | 0/25 | 0/25  |  2/25 | 19/25 |

`headroom_table.csv` has the full 300-cell grid.

---

## Surprises vs the v1-program's "tradeable universe" assumption

1. **The 25-symbol universe is genuinely tradeable at $1k–$10k.** The v1 program ran direction probes assuming all 25 symbols were market-tradeable equivalents. At $100k, **eight names are not** — the visible book runs out before fill. The "tradeable universe" the v1 program assumed should have been a **per-size manifold**, not a flat list.

2. **BTC has the *lowest* gross headroom**, not the highest. Edge of 19 bp at H500 is the smallest in the universe; 5–6 bp headroom leaves no room for imperfect direction signals. **The most-pretrained-on symbol is the worst Goal-A candidate at H500.** Alts (HYPE/ENA/PUMP/2Z/FARTCOIN/XPL) have 3–5× the headroom band — and that's exactly where the v1 model's per-symbol Gate-1 numbers were strongest at H500.

3. **Slippage at $1k–$10k on majors is essentially noise** — BTC at $10k median 0.20 bp, ETH 0.29 bp, SOL 0.82 bp. On the 10-level book, the visible passive depth is enough to absorb $10k clean across the board. **Pacifica's perp book is deeper than the v1 program reasoned about.** The "6bp/side fee" was the dominant cost on majors; slippage is a rounding error.

4. **Alts have outsized edge but outsized cost too.** XPL at $10k slips 11.7 bp median, vs ENA at 4.0 bp. PUMP at $100k slips 16.1 bp. Cost-aware ranking changes per size: at $1k pick HYPE (1.1 bp slip, 62 bp edge); at $100k pick SOL (2.8 bp slip, 39 bp edge, 98% fillable).

5. **The v1 closure note's "fees + slippage at tradeable size" was directionally right but quantitatively wrong.** The signal isn't fee-blocked at the *universe* level. It is **signal-strength-blocked**: a 1pp accuracy edge on a 20–60 bp move = 0.2–0.6 bp expected gross, which is what the closure was implicitly observing. The fix isn't smaller fees; it is a stronger directional model **at the cells with high fillability AND high edge**.

---

## Blockers & resolutions

1. **The cache shards only contain event-aligned 8-feature OB summaries, not raw 10-level depth.** The cache is sufficient for ML training but not for book-walk simulation. **Resolved**: I load raw OB parquet via `tape.io_parquet.load_ob_day` (existing infrastructure, no re-implementation). I verified the dates we read (≤ 2026-04-13) are all pre-hold-out, so no April-14+ contamination. The script hard-skips any shard with `date >= APRIL_HELDOUT_START` defence-in-depth.

2. **Cache shard count (4178) > spec figure (4003).** The spec was written before the April 1–13 backfill. All 4178 shards have `date < 2026-04-14`, verified.

3. **Per-event vwap is not in the cache** (only `log_return`). **Resolved**: forward log-return at horizon h from anchor i is `Σ log_return[i+1..i+h]`, computed via a cumulative-sum trick. Equivalent to `log(vwap[i+h]/vwap[i])` and avoids re-loading raw trades.

4. **Edge math: signed vs absolute.** Task spec said "abs realized return × 10000 (one-sided, since we're asking 'could a perfect direction signal have made money')". I use unsigned. **Note**: `per_window.parquet` retains signed forward log-returns under no column (only `edge_bps = |return|`), but `slip_buy_bps` and `slip_sell_bps` are signed and preserved. If signed analysis is needed downstream, the per-window file has enough info to recompute: edge_bps and the buy-side slip alone tell you how a buy-then-sell round-trip plays.

5. **Subsample randomness.** Used a fixed `np.random.default_rng(0xCAFE)`; same seed → same per-window subset. Reproducible.

6. **No NaN propagation observed.** All 1.93M rows have either a finite headroom or a documented NaN (un-fillable + finite edge → headroom = NaN). Sanity-checked via `assert np.all(np.isfinite(features))` indirectly through cache builder; cache schema version 1.

---

## Files written

- `docs/experiments/goal-a-feasibility/README.md` — this file
- `docs/experiments/goal-a-feasibility/headroom_table.csv` — 300 cells, summary statistics (now augmented with per-accuracy stats — see below)
- `docs/experiments/goal-a-feasibility/per_window.parquet` — 1.93M rows, full per-window detail (preserved for follow-up slicing)
- `docs/experiments/goal-a-feasibility/survivors.md` — accuracy-regime survivor tables (added 2026-04-27)
- `scripts/goal_a_feasibility.py` — book-walk simulator + aggregation + accuracy stress
- `tests/scripts/test_goal_a_feasibility.py` — 19 unit tests, all green
- `docs/implementation/goal_a_feasibility_run.log` — run log (tee'd from full job)

---

## Survivors at realistic directional accuracy (added 2026-04-27)

The oracle verdict above (`gross_edge_bps = |return| × 1e4`) overstates feasibility because it assumes a perfect direction signal. v1's encoder hit ~+1pp at H500 (~51.4% balanced accuracy); realistic models on this data top out around 55–60%. **Expected per-round-trip PnL at directional accuracy `p`** is `(2p − 1) × |edge| − cost`, not `|edge| − cost`. We re-aggregate the same 1.93M per-window parquet under three regimes — 55% / 57.5% / 60% — and define a survivor as a `(symbol, size, horizon)` cell with `frac_pos_acc_X > 0.55` AND `headroom_X_median_bps > 0`. Full per-cell stats are appended to `headroom_table.csv` (12 new columns); survivor sub-tables and near-miss top-5s are in `survivors.md`.

### Headline verdict

**At 55% accuracy: nothing survives.** The strongest cell is HYPE $1k H500 with median headroom **−8.2 bp**, frac_pos **15.6%**. The 0.10× discount on the 62 bp HYPE edge yields 6.2 bp expected gross — short of the 14.5 bp round-trip cost band by ~8 bp. Same picture at PUMP/SOL/ENA/BNB. **A 55%-accurate model on this universe loses money on every name at every size at every horizon.**

**At 57.5% accuracy: still nothing survives.** Best cell PUMP $1k H500: median **−4.4 bp**, frac_pos **40.2%**. The 0.15× discount gives 15.1 bp expected gross vs ~21 bp round-trip cost. Even the fattest cell in the entire 300-grid is still under water at 57.5%.

**At 60% accuracy: zero strict survivors, exactly one near-miss.** PUMP $1k H500 produces a positive median headroom of **+0.51 bp** with frac_pos **51.1%** — but fails the `frac_pos > 0.55` floor by 4pp. Every other cell is negative-median. **Goal A requires a directional model materially better than 60% to be feasible at any size at any horizon, and even then only on a single illiquid name (PUMP) at a $1k position.**

### Modelling caveats (read before scoping anything from this)

1. **First-order accuracy calc.** `(2p − 1) × |edge|` treats `p` as iid and uses realized `|edge|` as the magnitude. It does not model the **joint distribution** of (signal-correctness, edge-size) — a model that's right 60% on average might be right *less* than 60% on the largest moves (regime-conditional skill). A more honest sim would condition on a model output ↔ realized-return joint, which we do not have. **Direction**: this calc is probably *optimistic* — real models tend to be worse on tail moves where most of the available edge lives.

2. **Slippage is decoupled from signal selectivity.** The same slip-bps applies to a perfect-oracle round trip and a 60%-accurate one. In reality, a 60%-accurate model that *also* trades when the book is thin is paying *more* slip than the book-snapshot estimate, because it picks up trades during stress where depth has already degraded. The slip column here is ~book-walk-at-anchor; real execution would face **adverse selection in book regime**.

3. **Stationarity flag (PUMP $1k H500 @ 0.60).** Quick check: 167 days of data, 103/167 days (61.7%) have positive median headroom on this cell — *not* outlier-driven, the cell median holds at 0.32 bp after dropping the top-5 days. But the per-day frac-positive distribution has σ=17pp and a min of 0% (a non-trivial slice of days is fully dead). **This means the +0.51 bp universe median is real but high-variance; days-of-week or volatility-regime conditioning would be the next slice.**

4. **Fees.** Same 6 bp/side flat fee assumption as the oracle table. Maker rebates would push the bar lower; we do not model them.

### What this means for next steps

- **Goal A as scoped (any-symbol, any-size, any-horizon, 55%-accurate model) is dead.** No cell in the 300-grid survives the basic `frac_pos > 0.55 AND median > 0` gate.
- **One cell could survive a stronger directional model**: PUMP $1k H500 at ≥60% accuracy. Before any program-shape commitment, the next investigation should be **per-day stationarity of that single cell's headroom** (volatility regime, day-of-week, time-of-day) — to determine whether the +0.51 bp median is a tradeable signal or an artifact of a few high-vol days. v1's CNN hit ~51% at H500 on the universe; whether per-symbol fine-tune can push PUMP to 60% is a separate empirical question that this analysis does *not* answer.
- **The 25-symbol universe assumption is the hidden killer.** v1 spread compute across 25 symbols looking for universal direction skill. The accuracy-stressed feasibility says: *no universe-level program is viable*; the only path is **single-name, $1k-scale, H500-only, with a model strong enough to clear 60% balanced accuracy on PUMP**. That is not a representation-learning program; it is a single-symbol direction-prediction problem at small notional.

### Reproducibility

Re-run the accuracy stress (does not re-walk the book, ~15s):

```bash
uv run python scripts/goal_a_feasibility.py --accuracy-stress-only
```

This regenerates `per_window.parquet` (with 3 new columns), `headroom_table.csv` (with 12 new columns), and `survivors.md` from the existing parquet. To rebuild from cache shards (slow), drop the `--accuracy-stress-only` flag.

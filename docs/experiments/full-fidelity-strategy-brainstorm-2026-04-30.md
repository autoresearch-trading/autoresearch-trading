# Full-fidelity Pacifica strategy brainstorm

**Date:** 2026-04-30  
**Goal:** find a path toward a non-HFT Pacifica/perp strategy that can plausibly reach paper-trading Sortino > 2.  
**Context:** prior direction/encoder/RL/maker/cascade-direction programs are closed. The new thing is not “more data, same labels.” The new thing is a full-fidelity archive with mark/oracle/funding/OI, BBO IDs, book order counts/nonces, trade history IDs/nonces, candles, mark candles, and raw message timing.

---

## Hard constraints from prior research

Binding evidence:

- Pacifica tape has real microstructure structure, especially cascade-onset prediction.
- Generic next-direction trading is dead under current economics.
- Maker posting as alpha is dead under tested fill-conditional adverse selection.
- Cascade-onset is not directly tradeable because cascade direction failed.
- RL/encoder work is not the bottleneck. Executable edge and cost economics are the bottleneck.

Therefore, brainstorm candidates must transform the problem into one of:

1. risk avoidance / no-trade decisions;
2. event-after-observation decisions;
3. mechanically anchored relative-value convergence;
4. execution-quality selection;
5. data products that compound into cleaner labels and lower overfit risk.

Reject candidates that are just renamed versions of:

- predict next return direction;
- generic deep sequence model;
- RL-first policy discovery;
- maker quoting because “maker fees are lower”;
- liquidation/cascade direction retry;
- broad threshold mining on consumed data.

---

## What the new full-fidelity archive unlocks

New public market-data streams now being captured:

- `prices`: funding, next_funding, mark, mid, oracle, open_interest, volume_24h, yesterday_price.
- `trades`: raw history id `h`, symbol `s`, amount `a`, price `p`, side `d`, cause `tc`, timestamp `t`, exchange nonce `li`.
- `book`: raw levels with amount `a`, order count `n`, price `p`, symbol `s`, timestamp `t`, exchange nonce `li`.
- `bbo`: order id `i`, last order id / nonce `li`, timestamp `t`, bid/ask prices and sizes.
- `candle` and `mark_price_candle` for trade-price and mark-price OHLCV.
- REST snapshots for `/info` and `/info/prices`.

This creates four new research axes:

1. Fair-value/basis axis: mark vs oracle vs mid vs BBO.
2. Crowding/carry axis: funding, next funding, OI, open/close flow.
3. Queue/market-quality axis: BBO IDs, book order counts, nonces, spread/depth/churn.
4. Forced-flow/event axis: liquidation trades, history IDs, exchange nonces, post-event refill/recovery.

---

## Ranked trading hypotheses

### 1. Toxic-regime / no-quote risk overlay

**Verdict:** do first.

**Thesis:** the proven cascade/onset signal may be most valuable as a risk-state and no-trade filter, not as a direction signal. If we can identify toxic windows before adverse selection, we can improve Sortino by not trading, canceling passive quotes, reducing size, or refusing entries.

**Why this is different from closed work:** it does not require predicting profitable direction. It asks whether a state predicts worse future execution conditions: adverse excursion, spread widening, depth collapse, fill toxicity, or slippage.

**Data used:** trades, BBO IDs, book order counts/nonces, prices/mark/oracle/OI/funding, existing cascade-onset features.

**Labels:**

- next 1/5/15m adverse excursion;
- spread widening;
- depth collapse;
- fill-conditional adverse move at touch;
- simulated taker slippage;
- liquidation/cascade probability.

**Action if true:**

- no new positions during top toxic-risk decile;
- cancel/widen passive quotes;
- reduce size;
- require bigger basis edge before taker entry.

**One-day falsification:**

- Build a per-symbol event-time panel from the live full-fidelity day.
- Score toxicity from BBO turnover, nonce jumps, order-count collapse, spread/depth instability, mark/oracle dislocation, OI/funding stress, and cascade precursor score.
- Test whether top toxic decile has at least 2x worse next-5/15m adverse excursion or slippage than bottom half.

**Hard gate:**

- top-decile toxic score must increase adverse excursion/slippage by >= 2x versus bottom half day-blocked OOS;
- maker quote-cancel overlay must reduce simulated adverse selection by >= 30% while retaining >= 50% benign quoting minutes;
- simple baseline with overlay must improve downside deviation or max drawdown by >= 25% without removing >50% of gross opportunity.

---

### 2. Mark/oracle/mid dislocation reversion

**Verdict:** do first, but keep first pass brutally small.

**Thesis:** when Pacifica mid/mark deviates from oracle by enough, there may be mechanically anchored convergence. This is not generic direction; it is a basis/fair-value spread with a clear anchor.

**Why this is different:** older data lacked mark/oracle/funding/OI. The trade is conditional on executable BBO and fair-value divergence, not on raw next-return prediction.

**Data used:** prices, BBO, book, mark_price_candle, candles, toxicity score.

**Action if true:**

- if mid/mark trades too low versus oracle and ask is executable with enough edge, buy sparse high-threshold events;
- if mid/mark trades too high versus oracle and bid is executable with enough edge, short sparse high-threshold events;
- exit at basis compression, stop-loss, or timeout.

**One-day falsification:**

- Define `basis_bps = 1e4 * (mid - oracle) / oracle` and `mark_basis_bps = 1e4 * (mark - oracle) / oracle`.
- Bucket abs basis by 5-10, 10-20, >20 bp.
- Require spread/depth/liquidity filters.
- Simulate convergence over 1/5/15/60m against BBO execution costs.

**Hard gate:**

- >= 100 independent events/month after filters;
- median convergence after 8 bp taker round trip plus measured slippage must be positive by >= 5 bp day-blocked OOS;
- 95th percentile adverse excursion before convergence cannot exceed expected convergence by >2x;
- signal must remain when high-cascade-risk windows are removed or separately modeled.

---

### 3. Stale quote / oracle cross

**Verdict:** high-upside, fast falsification, slightly more fragile than basis reversion.

**Thesis:** BBO quotes may occasionally lag oracle/mark enough that the best bid/ask is stale for seconds. Raw BBO order ID `i` and nonce `li` enable quote-age and quote-refresh detection.

**Data used:** BBO, prices, mark/oracle, trades, mark candles.

**Action if true:**

- taker buy if best ask is stale below mark/oracle by more than costs plus buffer;
- taker short if best bid is stale above mark/oracle by more than costs plus buffer;
- exit at fair-value convergence or short timeout.

**One-day falsification:**

- Align BBO and prices.
- Detect unchanged BBO ID for >= 1/3/5s while mark/oracle moves away.
- Require edge >= 8-12 bp depending on exit mode.
- Simulate entry at BBO and exit at convergence/timeout.

**Hard gate:**

- >= 20 candidate events/day across all symbols;
- median post-cost PnL > 0 in at least one pre-registered staleness/dislocation bucket;
- not explained by local collector latency or delayed receive timestamps.

---

### 4. Post-liquidation absorption / forced-flow stabilization

**Verdict:** do as event study once enough full-fidelity data accrues.

**Thesis:** cascade direction was not predictable ex ante, but after forced liquidation flow has printed, we can observe whether the book absorbs, refills, or continues to fail. The entry should wait for stabilization features, not front-run onset.

**Data used:** raw trades with `tc`, `d`, `h`, `li`; book order counts/nonces; BBO stability; mark/oracle; OI; funding.

**Action if true:**

- after liquidation burst, trade delayed reversal only if price progress per liquidation notional stalls, spread normalizes, depth refills, BBO IDs stabilize, and mark/oracle basis compresses;
- alternatively classify “do not fade” continuation regimes.

**One-day/month falsification:**

- Detect liquidation bursts with `tc in {market_liquidation, backstop_liquidation}` and nonce/time clustering.
- Compute post-burst refill/stabilization features.
- Test delayed-entry reversal/continuation PnL after full costs.

**Hard gate:**

- >= 50 independent liquidation bursts/month after clustering;
- delayed-entry rules positive after 8 bp taker round trip plus slippage;
- edge cannot require entering before stabilization features are observable;
- not dominated by one day/symbol.

---

### 5. Queue/fill-quality model for passive execution

**Verdict:** maybe; only as an execution filter for another positive signal.

**Thesis:** prior maker alpha died from adverse selection, but BBO IDs, order counts, nonces, and book churn may identify rare regimes where passive execution is not toxic.

**Data used:** BBO `i/li`, book `n/li`, trades `li/h`, mark/oracle drift.

**Action if true:**

- only use maker/passive entry when queue toxicity is low;
- cancel when BBO ID turnover, nonce jump rate, book-count collapse, or mark/oracle drift spikes.

**Hard gate:**

- lowest-toxicity decile must improve fill-conditional drift by >= 1 bp and reduce adverse selection by >= 30%;
- safe-regime fill rate must be non-trivial;
- public data must infer fills conservatively enough to trust the result.

---

### 6. Funding/OI carry with toxicity overlay

**Verdict:** maybe; lower priority because sample size is slower.

**Thesis:** funding and next funding may create a carry edge only when combined with OI/crowding/flow and toxicity filters.

**Data used:** prices funding/next_funding/OI, trades open/close flow, mark/oracle basis, BBO/book quality.

**Action if true:**

- trade sparse funding windows where expected carry exceeds execution cost and toxicity/crowding risk is low;
- avoid crowded-side expansion and cascade-risk states.

**Hard gate:**

- funding magnitude must cover realistic execution and tail risk;
- carry filter must improve post-cost PnL over naive high-funding carry;
- settlement timing must be reconstructed exactly.

---

### 7. Nonce-gap / order-count toxicity as shared filter

**Verdict:** build as feature infrastructure, not standalone alpha.

**Thesis:** exchange nonces and order-count changes can identify quote churn/cancel pressure and fragile liquidity before visible price movement.

**Use:** input into toxic-regime overlay, passive execution filter, and basis trade gate.

**Hard gate:**

- high nonce-gap states must predict worse spread/depth/slippage/adverse drift OOS;
- adds value beyond simple spread/depth amount features;
- does not filter out nearly all opportunity.

---

## Ranked product/data builds

These are not all strategies, but they make strategy discovery cleaner and less overfit.

### A. Normalized full-fidelity event joiner

**Why first:** every serious candidate needs aligned trades, BBO, book, prices, candles, and mark candles. Without this, every probe becomes a one-off join and leaks/bugs multiply.

**Deliverables:**

- bronze raw JSONL stays immutable;
- silver normalized events per channel;
- event-time joiner: `symbol, ts -> nearest BBO/book/prices/mark/candle` with lag diagnostics;
- data quality checks for nonce monotonicity, duplicate messages, missing symbols, and timestamp drift.

**Minimum success:** deterministic rebuild and >= 95% joins within acceptable lag on fresh full-fidelity days.

### B. Market-quality and toxic-regime dashboard

**Why second:** tells us where any edge can survive fees.

**Deliverables:** spread/depth/churn/nonces/order-count health, adverse-selection proxy, symbol tradability score, bad hours/symbols.

**Minimum success:** market-quality metrics explain realized slippage/adverse selection OOS.

### C. Liquidation/forced-flow monitor

**Why third:** highest continuity with the strongest prior signal.

**Deliverables:** liquidation event table, episode detector, precursor/post-event feature table, episode browser.

**Minimum success:** enough event coverage and stable labels to support walk-forward probes.

### D. Label registry

**Why:** avoid label churn and multiple-testing fog.

**Deliverables:** versioned label definitions, builder scripts, prevalence checks, overlap matrix, leakage/null controls.

**Minimum success:** each label has deterministic rebuild, prevalence bounds, no-lookahead check, and shuffled/null controls.

---

## Proposed build sequence

### Phase 0: Let full-fidelity data accumulate

Run collector continuously for at least several fresh full days. Do not tune strategy thresholds on partial first-day artifacts.

### Phase 1: Build the event joiner and data-quality harness

Files to create later:

- `scripts/build_pacifica_full_fidelity_silver.py`
- `tests/scripts/test_build_pacifica_full_fidelity_silver.py`
- `docs/experiments/full-fidelity-data-quality/README.md`

Outputs:

- normalized prices/trades/bbo/book/candle/mark-candle parquet or duckdb tables;
- join lag report;
- nonce/order-id sanity report;
- symbol/day/channel completeness report.

### Phase 2: First tradeability probe: toxic-regime/no-quote overlay

Files to create later:

- `scripts/toxic_regime_probe.py`
- `tests/scripts/test_toxic_regime_probe.py`
- `docs/experiments/toxic-regime-overlay/README.md`

Question:

Can full-fidelity microstructure predict when trading/quoting is likely to be toxic, enough to improve Sortino by avoiding left-tail regimes?

### Phase 3: First alpha probe: mark/oracle/mid basis reversion

Files to create later:

- `scripts/mark_oracle_basis_probe.py`
- `tests/scripts/test_mark_oracle_basis_probe.py`
- `docs/experiments/mark-oracle-basis-reversion/README.md`

Question:

Do rare fair-value dislocations converge enough to clear realistic taker/maker costs after liquidity and toxicity filters?

### Phase 4: Forced-flow absorption event study

Files to create later:

- `scripts/liquidation_absorption_probe.py`
- `tests/scripts/test_liquidation_absorption_probe.py`
- `docs/experiments/liquidation-absorption/README.md`

Question:

After observed liquidation bursts, can stabilization/refill features identify delayed entries with positive post-cost expectancy?

---

## What not to build now

Do not build yet:

- a big RL environment;
- a new encoder/transformer;
- a full paper-trading stack before a net-edge gate;
- a generic direction classifier using the new fields;
- a maker strategy that assumes passive fills are good;
- broad hyperparameter sweeps across the fresh archive.

Those come only after one of the sparse, economics-first probes shows positive post-cost expectancy or clear risk-reduction value.

---

## My recommendation

Start with the product that gives the most leverage and least overfit risk:

1. Build the normalized full-fidelity event joiner/data-quality layer.
2. Use it immediately for the toxic-regime/no-quote overlay probe.
3. In parallel, define mark/oracle basis events but do not trade/tune them until there are enough fresh full-fidelity days.

The most promising path to Sortino > 2 is not “predict better.” It is:

> identify when the venue is tradeable, avoid toxic regimes, and only take sparse mechanically anchored dislocations where the expected convergence is much larger than fees/slippage.

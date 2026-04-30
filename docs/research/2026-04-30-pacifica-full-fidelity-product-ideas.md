# Pacifica full-fidelity archive: non/semi-trading analytics product ideas

Context: the main objective remains profitable non-HFT trading edge with Sortino > 2, but the prior pure direction strategy line is closed. The new raw archive preserves prices/mark/oracle/funding/open interest, raw trades with liquidation/cause history, raw book with order counts/nonces, BBO, candles, and mark candles. The most useful next products should compound into a data moat, improve research labels, and expose regimes where simple strategies can later be tested with lower multiple-testing risk.

## Ranking rubric

Score each idea by expected usefulness for eventually finding profitable strategies:

- 5 = directly creates predictive labels/regime filters or execution/risk constraints likely to change strategy economics.
- 4 = creates durable data moat or research substrate that should materially accelerate strategy discovery.
- 3 = useful diagnostics/dashboarding but less directly connected to edge.
- 2 = mostly operational or external-facing value.
- 1 = nice-to-have.

## Ranked shortlist

| Rank | Idea | Type | Usefulness | Build difficulty | Why it matters for eventual strategies |
|---:|---|---|---:|---:|---|
| 1 | Liquidation cascade and forced-flow monitor | Semi-trading analytics | 5 | M | Directly extends the current Goal A v2 line. Converts `cause/li` trade fields, book depletion, mark/oracle divergence, OI, and funding into event labels and live/archival cascade risk features. |
| 2 | Risk-state oracle / market regime state machine | Semi-trading analytics | 5 | M-H | Creates a reusable regime filter: normal, crowded, fragile, squeeze-prone, deleveraging, toxic-flow, recovery. A strategy with mediocre unconditional edge can become viable if traded only in favorable states. |
| 3 | Labeling platform for tape states and outcomes | Research platform | 5 | M | Turns raw archive into clean supervised targets: cascades, absorption, informed flow, spoof/withdrawal-like behavior, mark dislocations, liquidity droughts, post-event drift/reversion. Reduces ad hoc label churn. |
| 4 | Market quality analyzer | Semi-trading analytics | 4.5 | M | Quantifies spread, depth, resiliency, order-count churn, nonces, slippage, adverse selection, latency gaps. Essential for knowing whether predicted edge survives fees/execution. |
| 5 | Flow/OI/funding crowding dashboard | Semi-trading analytics | 4.5 | M | Combines open/close side, OI deltas, funding/next funding, mark-oracle basis, and liquidation flow to identify crowded positioning and unwind pressure. |
| 6 | Full-fidelity normalized feature lake + event joiner | Data moat | 4.5 | H | Not a strategy, but the highest-leverage substrate: deterministic joins across trades/book/BBO/prices/candles/raw JSON, preserving ids/nonces/order counts. Prevents lossy one-off pipelines. |
| 7 | Mark/oracle/basis dislocation monitor | Semi-trading analytics | 4 | L-M | Captures regimes where perps decouple from oracle/mark, funding moves, and liquidation thresholds get closer. Useful for both risk filters and mean-reversion/momentum hypotheses. |
| 8 | Liquidity resiliency and book-recovery profiler | Analytics | 4 | M | Measures how fast depth/spread/BBO recover after aggressive flow/liquidations. Directly informs maker/taker selection and post-shock strategies. |
| 9 | Data quality, coverage, and exchange-microstructure observatory | Data moat | 3.5 | M | Detects collector gaps, duplicate messages, nonce discontinuities, schema drift, websocket latency, and impossible states. Protects all downstream research from silent corruption. |
| 10 | Symbol health and tradability scorecard | Dashboard | 3.5 | L-M | Ranks symbols by liquidity, fee hurdle, stable microstructure, event frequency, and edge capacity. Avoids wasting research on symbols where edge cannot monetize. |
| 11 | Replayable market tape viewer | Research UI | 3 | H | Human/agent inspection tool for cascades and anomalous flow. Helps discover new labels, but payoff is indirect unless tightly coupled to label creation. |
| 12 | External benchmark/API product | Data product | 2.5 | H | Could become a moat or monetizable product, but only after internal labels/features prove useful. Premature as a primary path to trading edge. |

## Product details

### 1. Liquidation cascade and forced-flow monitor

Deliverables:
- Archival table of liquidation events keyed by symbol, timestamp, side/direction, qty, price, `cause`, raw `li/history` fields when present, mark/oracle, OI, funding, BBO, and local book state before/after.
- Cascade episode detector: groups liquidation events into episodes by time gap, signed price move, and forced-flow notional.
- Precursor feature table at multiple horizons: 1m/5m/15m/30m and event-count horizons. Include spread/depth changes, order-count churn, OI/funding/basis, open-vs-close flow, OFI, mark-oracle distance, and recent failed recovery.
- Dashboard: live/archival heatmap of cascade risk by symbol; episode browser; precursor traces.

Why valuable:
- Direct continuation of current cascade-precursor work, using the new full-fidelity archive rather than lossy parquet only.
- Liquidations are exogenous-ish labels compared with noisy return direction labels.
- Produces tradable hypotheses: avoid maker quotes before cascades, join momentum during early cascade, fade after forced-flow exhaustion, or allocate risk only when cascade probability is low.

Measurable success:
- Label coverage: >= 95% of liquidation-flagged raw trades joined to nearest BBO/book/prices within acceptable lag.
- Episode stability: detector parameters produce similar episode counts under small threshold perturbations.
- Predictive value: OOS cascade AUC improves over current baseline of ~0.778, target >= 0.85 before strategy work.
- Economic proxy: top decile risk states have materially worse next-horizon adverse selection / drawdown than bottom decile after fees.

### 2. Risk-state oracle / market regime state machine

Deliverables:
- Per-symbol, per-minute/event state vector with calibrated scores: liquidity stress, crowding, informed flow, liquidation risk, basis stress, volatility shock, recovery/resiliency.
- Discrete state machine: normal, thin/fragile, crowded-long, crowded-short, toxic-flow, cascade-onset, forced-deleveraging, post-cascade-recovery, disconnected-oracle.
- State transition matrix and dwell-time stats by symbol and hour/day.
- Simple API/CLI: `risk_state(symbol, ts)` returns scores, labels, explanations, and nearest historical analogs.

Why valuable:
- A state oracle is an edge amplifier. It can filter strategy exposure, set maker/taker mode, choose horizons, and define clean samples for future research.
- It creates a compounding data asset: every new day improves transition/base-rate estimates.

Measurable success:
- States are stable OOS: transition/dwell distributions do not collapse under walk-forward splits.
- States separate future distributions: next 5m/30m volatility, slippage, liquidation probability, and adverse selection differ monotonically across risk quantiles.
- Strategy usefulness: applying state filters improves at least one existing weak signal’s net Sortino or drawdown profile in walk-forward tests, without retraining the signal on the test window.

### 3. Labeling platform for tape states and outcomes

Deliverables:
- A versioned label registry with definitions, SQL/Python builders, parameters, data dependencies, and validity windows.
- First labels: liquidation cascade onset, absorption, buying/selling climax, liquidity withdrawal, book stuffing/churn, informed flow, mark/oracle dislocation, funding crowding, post-shock reversal/drift, maker-toxic period.
- Label QA report: prevalence by symbol/day, autocorrelation, overlap matrix, examples, counterexamples, and leakage checks.
- Optional human/agent review workflow: sample windows, render traces, accept/reject labels, store adjudications.

Why valuable:
- The previous program suffered from weak/noisy direction labels and calibration issues. A label platform makes labels auditable and reusable.
- Enables many semi-supervised/representation-learning tasks without re-inventing targets each time.

Measurable success:
- Every label has deterministic rebuild, prevalence bounds, no-lookahead tests, and shuffled/null controls.
- At least 5 labels have enough positive examples for walk-forward probes across liquid symbols.
- Label overlap matrix reveals non-redundant states; no single trivial variable explains all labels.
- Human/agent adjudication precision for sampled positives exceeds a chosen threshold, e.g. 70-80% for heuristic labels.

### 4. Market quality analyzer

Deliverables:
- Per-symbol market-quality time series: quoted/effective spread, depth by level, BBO stability, order-count churn, nonce gaps, cancel/replace proxy, book imbalance, realized volatility, slippage curves, resiliency after shocks.
- Maker adverse-selection dashboard: expected next return after passive fill proxies under different states.
- Tradability report: fee hurdle, minimum edge needed, capacity proxy, bad hours/symbols to exclude.

Why valuable:
- Profitable non-HFT edge is likely fee/execution constrained. This product tells us where theoretical alpha can survive.
- New book order counts/nonces and BBO fields provide microstructure observability not present in the older 10-level snapshots.

Measurable success:
- Market-quality metrics explain realized slippage/adverse selection OOS.
- Produces stable symbol ranking; excluded symbols/hours show demonstrably worse net execution economics.
- Detects known stress events and liquidation episodes as market-quality deterioration before or during the event.

### 5. Flow/OI/funding crowding dashboard

Deliverables:
- Flow decomposition by symbol: open_long, close_long, open_short, close_short, liquidation flow, aggressive buy/sell proxy, OI delta, volume delta.
- Crowding indicators: OI expansion with one-sided opens, extreme funding/next funding, mark premium/discount, liquidation imbalance, failure of price to move despite effort.
- Cross-symbol heatmap and historical analog search.

Why valuable:
- Crowding and forced unwind are plausible sources of non-HFT edge. This is also interpretable and useful for risk control.
- It uses Pacifica-specific fields (`side`, OI, funding, mark/oracle) that are not generic OHLCV.

Measurable success:
- Crowding scores predict later liquidation imbalance, volatility expansion, or funding/basis normalization OOS.
- Top/bottom crowding quantiles have different forward return skew or drawdown distributions after costs.
- Dashboard highlights episodes that align with observed cascade labels.

### 6. Full-fidelity normalized feature lake + event joiner

Deliverables:
- Bronze raw JSONL catalog and silver normalized parquet tables for prices, trades, book, BBO, candle, mark candle, REST info/prices.
- Deterministic as-of joiner with lag metadata: event -> latest BBO/book/prices/mark/oracle/funding/OI/candle context.
- Schema versioning, raw payload hash, dedup keys, and reproducibility manifest.
- Feature store API for downstream labels/models.

Why valuable:
- Most future analytics require the same joins. Building the lake once prevents inconsistent research pipelines.
- Preserving raw fields plus normalized views is a durable data moat.

Measurable success:
- Rebuild is deterministic from raw archive.
- Join lag distributions are monitored and bounded.
- Existing derived parquet features can be reproduced or explained from the new full-fidelity lake.
- New fields unavailable in old parquet are exposed in documented columns.

### 7. Mark/oracle/basis dislocation monitor

Deliverables:
- Time series of mark-oracle spread, mark/last/BBO/candle discrepancies, funding and next-funding trajectory, OI/volume context.
- Dislocation events: threshold crossings, duration, magnitude, resolution path, nearby liquidations.
- Alert/report for symbols with persistent disconnects.

Why valuable:
- Liquidation mechanics and perp pricing depend on mark/oracle, not just last trade. Dislocations can create risk and opportunity.
- Can become a filter for cascade risk or a source of mean-reversion hypotheses.

Measurable success:
- Dislocation events predict liquidation probability, spread widening, or abnormal forward volatility.
- Event definitions are robust across thresholds and symbols.
- Joined event browser shows clear causal ordering rather than artifact/latency issues.

### 8. Liquidity resiliency and book-recovery profiler

Deliverables:
- Shock detector for aggressive trades/liquidations/book depletion.
- Recovery metrics: time to spread normalization, time to depth refill, BBO churn, order-count refill, price impact decay, overshoot/reversal.
- Symbol/state profiles and post-shock analog search.

Why valuable:
- Resiliency determines whether post-event continuation or fade strategies are plausible.
- Helps choose maker vs taker and holding horizon.

Measurable success:
- Recovery profiles cluster into stable regimes with different forward return/slippage distributions.
- Post-shock features predict continuation vs reversal better than simple return/vol baselines.
- Resiliency states improve execution filters for existing weak signals.

### 9. Data quality, coverage, and microstructure observatory

Deliverables:
- Coverage dashboard by channel/symbol/date/hour with message counts, gaps, reconnects, raw JSON schema changes, duplicate rates, recv/event lag.
- Nonce/order-count continuity checks where fields exist.
- Canary tests comparing REST snapshots, websocket prices, candles, and derived parquet consistency.

Why valuable:
- Silent data corruption is fatal for rare-event strategy research. This is insurance for the data moat.
- Also reveals exchange behavior changes that may themselves become regimes.

Measurable success:
- Daily automated report catches missing channels/symbols within one collection cycle.
- Known collector outages or websocket reconnects are visible in coverage metrics.
- Downstream label/model jobs refuse to run or flag low-confidence periods when coverage is insufficient.

### 10. Symbol health and tradability scorecard

Deliverables:
- Daily symbol score: liquidity, spreads, depth, event frequency, funding stability, liquidation event count, OI, data quality, fee hurdle, capacity proxy.
- Research universe selector: liquid core, experimental, exclude.
- Drift report when a symbol changes class.

Why valuable:
- Prevents wasting compute and multiple-testing budget on untradable symbols.
- Helps build cross-symbol train/test splits that respect liquidity regimes.

Measurable success:
- Score predicts whether a symbol’s weak signals survive fees/slippage.
- Score is stable enough for universe construction but responsive to real liquidity changes.
- Excluding low-score periods improves aggregate backtest quality metrics.

### 11. Replayable market tape viewer

Deliverables:
- CLI/browser timeline for a symbol/time: trades, liquidations, BBO/book depth, mark/oracle, OI/funding, candle context, state labels.
- Episode browser for cascades, dislocations, high-churn books, and anomalous flow.
- Export selected windows to label adjudication queue.

Why valuable:
- Human/agent inspection can reveal mechanisms missed by aggregate stats.
- Best used as a companion to the label platform, not a standalone product.

Measurable success:
- Reduces time to inspect an event from raw parquet/JSON spelunking to seconds.
- Generates new candidate labels/hypotheses that pass later OOS tests.
- Reviewers agree on label validity for sampled episodes at acceptable rates.

### 12. External benchmark/API product

Deliverables:
- Sanitized daily market-quality and risk-state summary API.
- Public/private benchmark reports comparing Pacifica symbols and market regimes.
- Documentation and sample notebooks.

Why valuable:
- Could monetize or attract collaborators, but only after internal metrics prove reliable.
- External-facing pressure can improve data quality discipline.

Measurable success:
- Internal users rely on the API for research before externalization.
- Metrics remain stable under schema changes and collection outages.
- External demand/collaboration emerges without distracting from strategy research.

## Recommended build sequence

1. Build the normalized full-fidelity feature lake and deterministic event joiner enough to support trades + BBO/book + prices/mark/oracle/OI/funding. Do not overbuild UI first.
2. Build the liquidation cascade/forced-flow monitor on top of the joiner, because it is closest to the current promising Goal A v2 result.
3. Create the label registry while implementing cascade labels, so future labels inherit QA/leakage controls from day one.
4. Add market quality metrics and resiliency profiles; use them to decide whether cascade predictions can be monetized as maker avoidance, taker entry, or risk gating.
5. Promote recurring metrics into the risk-state oracle and dashboards only after they show OOS separation of future risk/return/slippage distributions.

## Highest-value immediate MVP

A 2-week MVP should be:

- `silver_events`: as-of joined table for April+ trades with BBO/book/prices context and raw liquidation fields.
- `cascade_episodes`: grouped liquidation episodes with direction, size, duration, mark/oracle/OI/funding context.
- `precursor_features`: 1m/5m/15m pre-episode features plus matched negative samples.
- `market_quality_context`: spread/depth/resiliency/adverse-selection metrics around episodes.
- `report`: OOS tables showing whether precursor risk deciles separate future cascade probability, slippage, and drawdown.

Success gate for the MVP:

- If cascade-risk deciles do not separate future cascade probability and adverse execution economics OOS, pause strategy work and focus on label/feature quality.
- If they do, the next trading-adjacent step is not a full strategy search; it is testing three constrained uses: maker quote suppression, post-cascade fade/continuation classifier, and risk-state filter for existing weak signals.

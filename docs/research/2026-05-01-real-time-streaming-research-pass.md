# Real-time streaming research pass for Pacifica full-fidelity system

Date: 2026-05-01
Question: can doing real-time research on streaming data improve the overall non-HFT Pacifica paper-trading system?

## Bottom line

Yes, but the value is probably not standalone alpha.

The strongest externally grounded case is to use streaming data as a causal market-state and risk layer:

1. no-trade / no-quote toxicity overlays;
2. market-quality gates from spread, depth, volatility, and order-flow stress;
3. liquidation / forced-flow / cascade monitors;
4. mark/oracle/basis dislocation monitors;
5. live data-quality, drift, and online/offline parity checks;
6. faster event-study feedback while the archive accrues.

This matches the current repo direction: use high-frequency full-fidelity data to make slower 1m+ non-HFT decisions. Do not pivot back to generic high-frequency direction ML or latency-style trading.

## Practical interpretation for this repo

The right question is not:

> Can streaming data predict the next tick/minute?

The useful question is:

> Can causal streaming features identify states where our expected post-cost PnL, slippage, adverse selection, or drawdown is materially worse, so the system should skip, reduce size, or wait?

That means real-time research is worth doing if scoped around overlays, event studies, and monitoring.

## External evidence themes

### 1. Order-flow and book state matter, but mostly at short horizons

Cont, Kukanov, and Stoikov show that order-flow imbalance has explanatory power for short-term price changes and price impact, especially conditional on liquidity/depth.

Reference:
- Cont, Kukanov, Stoikov, "The Price Impact of Order Book Events", Journal of Financial Econometrics, 2014. https://academic.oup.com/jfec/article/12/1/47/816163

Implication:
Use order-flow imbalance, trade imbalance, and book imbalance as toxicity/risk features, not as naive directional alpha.

Candidate local tests:
- Does high OFI/toxic-flow state predict worse next 5m/15m adverse excursion?
- Does skipping high-toxicity minutes improve downside deviation or max drawdown?
- Does the effect survive day-blocked validation and symbol concentration checks?

### 2. Flow toxicity is real as a concept, but VPIN-style metrics are fragile

Easley, Lopez de Prado, and O'Hara introduced flow toxicity / VPIN. Andersen and Bondarenko later criticized VPIN robustness and flash-crash claims.

References:
- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High-frequency World", Review of Financial Studies, 2012. https://doi.org/10.1093/rfs/hhs053
- Andersen and Bondarenko, "VPIN and the Flash Crash", Journal of Financial Markets / related versions. https://ssrn.com/abstract=1881731

Implication:
Do not implement VPIN as a magic signal. Implement a small family of toxicity proxies and validate only by strategy-conditional outcomes:
- realized slippage;
- fill-conditional adverse move;
- spread/depth deterioration;
- downside excursion;
- retained opportunity after filtering.

### 3. High-frequency realized volatility and jump/stress estimates are useful for slower risk control

Realized-volatility literature supports using high-frequency data to estimate current volatility and jump states more accurately than daily bars.

References:
- Andersen, Bollerslev, Diebold, Labys, "Modeling and Forecasting Realized Volatility", Econometrica, 2003. https://doi.org/10.1111/1468-0262.00418
- Barndorff-Nielsen and Shephard, "Power and Bipower Variation with Stochastic Volatility and Jumps", Journal of Financial Econometrics, 2004. https://doi.org/10.1093/jjfinec/nbh001
- Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 2009. https://doi.org/10.1093/jjfinec/nbp001

Implication:
Real-time vol/jump/stress states should feed:
- no-trade filters;
- size scaling;
- cooldowns;
- slippage assumptions;
- post-event risk-state labels.

### 4. Crypto perps are especially suited to event/risk-state monitoring

Perpetual futures have funding, mark/index/oracle mechanics, leverage, and forced liquidation feedback loops. Streaming data can expose these states before or during dangerous regimes.

References:
- BitMEX perpetual contracts guide: https://www.bitmex.com/app/perpetualContractsGuide
- BitMEX fair price marking: https://www.bitmex.com/app/fairPriceMarking
- Binance mark price / price index docs: https://www.binance.com/en/support/faq/what-is-mark-price-and-price-index-360033525071
- Binance funding rate docs: https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031
- BIS Bulletin on crypto shocks/leverage: https://www.bis.org/publ/bisbull64.htm
- Makarov and Schoar, "Trading and Arbitrage in Cryptocurrency Markets", Journal of Financial Economics, 2020. https://doi.org/10.1016/j.jfineco.2019.07.001

Implication:
For Pacifica, prioritize:
- liquidation / forced-flow monitor;
- mark/oracle/basis dislocation monitor;
- funding/OI crowding monitor;
- spread/depth/market-quality monitor.

### 5. Streaming architecture can improve research quality without requiring HFT infrastructure

A practical architecture does not need to jump straight to Kafka/Flink. A local Kappa-like microbatch system can be enough:

- immutable raw JSONL.GZ / bronze parquet event log;
- deterministic transformations;
- event-time windows;
- watermarks / provisional vs final rows;
- incremental DuckDB/Parquet rebuilds;
- same feature code for live microbatch and historical backtest;
- online/offline feature parity checks;
- data-quality and drift reports.

References:
- Jay Kreps, "Questioning the Lambda Architecture": https://www.oreilly.com/radar/questioning-the-lambda-architecture/
- Confluent Kappa Architecture overview: https://www.confluent.io/learn/kappa-architecture/
- Apache Flink event time/watermarks: https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/
- Apache Beam windowing/watermarks: https://beam.apache.org/documentation/programming-guide/#windowing
- DuckDB ASOF joins: https://duckdb.org/docs/stable/guides/sql_features/asof_join.html
- DuckDB Parquet docs: https://duckdb.org/docs/stable/data/parquet/overview.html

Implication:
Build a microbatch/live research layer only where it improves causal feature correctness, monitoring, and feedback speed. Keep the batch rebuild as authoritative until parity is proven.

## Recommended local research pass

### Research question

Can causal 1m+ streaming features identify states where a strategy should skip, reduce size, or delay because expected post-cost economics are worse?

### Priority 1: market-stress / market-quality monitor

Features:
- spread bps;
- top-of-book and top-N depth;
- depth imbalance;
- realized vol 1m/5m/15m;
- range and return z-scores;
- trade count / volume z-scores;
- estimated price impact for target notional.

Tests:
- top-decile stress vs bottom-half stress future adverse excursion;
- slippage proxy by stress bucket;
- effect of skipping stress buckets on downside deviation and retained opportunity.

### Priority 2: toxicity / adverse-selection overlay

Features:
- signed aggressive volume imbalance;
- order-flow imbalance;
- trade intensity spike;
- depth depletion after aggressive flow;
- spread widening after flow bursts;
- volume-synchronized imbalance as optional VPIN-like diagnostic.

Tests:
- whether high toxicity predicts worse next 5m/15m outcomes;
- whether it improves no-trade overlay metrics;
- whether effects survive day-blocked validation.

### Priority 3: liquidation / forced-flow monitor

Features/events:
- liquidation/cause trade fields where available;
- burst clustering;
- price acceleration + volume spike;
- spread/depth collapse;
- OI/funding/basis context;
- post-burst depth refill and spread normalization.

Tests:
- cascade probability by precursor bucket;
- post-event continuation vs reversal after observable stabilization;
- cooldown duration that reduces drawdown without deleting too much opportunity.

### Priority 4: mark/oracle/basis dislocation monitor

Features:
- mid/oracle basis bps;
- mark/oracle basis bps;
- last/mark spread;
- basis z-score and duration;
- nearby liquidation/funding/OI context.

Tests:
- event frequency after liquidity/toxicity filters;
- convergence after 1m/5m/15m/60m;
- adverse excursion before convergence;
- causal availability of mark/oracle fields.

### Priority 5: live data-quality and parity layer

Checks:
- newest raw file age;
- channel/symbol freshness;
- missing minutes;
- duplicate ids/content;
- schema drift;
- null/impossible values;
- late/out-of-order events;
- online feature vs offline recomputation mismatch.

This is not alpha, but it prevents false alpha.

## Architecture recommendation

Do not build a full trading event loop yet.

Build a near-real-time microbatch research monitor:

1. Keep current raw collector as append-only source of truth.
2. Add bronze normalized event tables if needed.
3. Add incremental 1m feature/regime builder with event-time windows.
4. Mark rows as provisional/final with `available_ts`, `computed_at`, `watermark_ts`, `feature_version`, and quality flags.
5. Use the same code path for historical backfill and live microbatch.
6. Add parity checks comparing last N hours live features to offline recomputation.
7. Only allow a live/paper decision layer after one sparse rule or overlay passes local post-cost validation gates.

## Go / no-go recommendation

Go for an external/real-time research pass, but scope it tightly.

Go:
- no-trade overlays;
- risk-state monitors;
- event studies;
- online/offline parity;
- data-quality/drift detection;
- causal 1m+ feature computation.

No-go:
- generic high-frequency direction ML;
- next-tick/order-queue alpha;
- VPIN as a standalone signal;
- paper-trading bot before post-cost local gates;
- heavy Kafka/Flink stack before local DuckDB/Parquet microbatch proves useful.

## Concrete next implementation target

Create `scripts/watch_pacifica_realtime_research.py` or equivalent microbatch job that every 1-5 minutes:

1. inventories new raw files;
2. updates latest bronze/silver partitions for affected windows;
3. computes market-quality, toxicity, liquidation, and basis features;
4. writes a latest regime table;
5. runs quality checks;
6. emits a markdown/CSV status report;
7. does not place trades.

This gives us the benefits of streaming research while preserving the current evidence discipline.

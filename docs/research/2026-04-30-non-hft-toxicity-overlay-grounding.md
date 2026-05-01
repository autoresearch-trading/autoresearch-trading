# Non-HFT toxicity overlay external grounding

Date: 2026-04-30

## Purpose

This note is a targeted external-research pass before implementing the next Pacifica probe:

- `scripts/non_hft_toxic_overlay_probe.py`
- `tests/scripts/test_non_hft_toxic_overlay_probe.py`
- `docs/experiments/toxic-regime-overlay/README.md`

The goal is not to find a new HFT alpha. The goal is to sharpen a non-HFT risk filter: use high-frequency/full-fidelity market data to decide when a slower strategy should not trade, reduce size, or avoid passive exposure.

## Bottom line

The external literature supports building the toxic-regime/no-trade overlay next, but only if it is treated as a risk-control/event-study probe rather than a return-prediction model.

The best-supported hypothesis is:

> One-minute regimes combining order-flow imbalance, queue/book imbalance, spread/depth stress, liquidation flow, funding/OI crowding, and mark/oracle dislocation should identify windows with worse future adverse excursion or execution quality. Excluding the worst toxicity decile should improve downside-risk metrics if the state variable has economic value.

This should be evaluated against downside-risk and adverse-selection labels, not AUC.

## What the literature supports

### 1. Order-flow imbalance is a necessary baseline

Cont, Kukanov, and Stoikov show that order-book event imbalance is central to short-horizon price impact. Their result supports using order-flow imbalance as a baseline state variable before any model-heavy approach.

Sources:

- Cont, Kukanov, Stoikov, "The Price Impact of Order Book Events"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822  
  https://academic.oup.com/jfec/article/12/1/47/816163

Implication for this repo:

The overlay should not rely only on trade imbalance. It should include BBO/book-based order-flow imbalance from quote and depth changes. For Pacifica this means at least:

- best-level OFI from BBO updates;
- top-of-book bid/ask size changes;
- spread changes;
- depth-normalized OFI;
- rolling OFI over 1m/5m windows.

Probe implication:

The first implementation can use the existing 1-minute regime-state features, but the next feature upgrade should add explicit OFI/microprice/queue imbalance from partitioned `bbo` and `book` silver.

### 2. Queue imbalance and microprice are adverse-selection diagnostics

Queue imbalance and microprice literature supports the idea that top-of-book imbalance reveals short-horizon pressure. This does not mean we can scalp the next tick. It means a slower strategy can avoid entering when local book pressure is hostile.

Sources:

- Lipton, Pesavento, Sotiropoulos, "Trade Arrival Dynamics and Quote Imbalance in a Limit Order Book"  
  https://arxiv.org/abs/1312.0514
- Gould and Bonart, "Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book"  
  https://arxiv.org/abs/1512.03492
- Stoikov, "The Micro-Price: A High-Frequency Estimator of Future Prices"  
  https://ssrn.com/abstract=2970694
- Cartea, Donnelly, Jaimungal, "Enhancing Trading Strategies with Order Book Signals"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2668277

Implication for this repo:

Use queue imbalance and microprice as no-trade context, not as a standalone HFT entry signal.

Useful Pacifica features:

- `queue_imbalance = (bid_size - ask_size) / (bid_size + ask_size)`;
- `microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)`;
- `microprice_minus_mid_bps`;
- rolling imbalance mean/slope/persistence;
- contradiction flags such as intended long while microprice is below mid.

Probe implication:

The toxic overlay should eventually report ablations:

1. current combined toxicity score;
2. spread/depth only;
3. trade-flow only;
4. BBO/queue/microprice only;
5. liquidation/funding/OI only.

### 3. Flow toxicity is conceptually useful, but VPIN-style shortcuts are controversial

Easley, Lopez de Prado, and O'Hara introduced VPIN/flow-toxicity framing: toxic flow adversely selects liquidity providers. But Andersen and Bondarenko dispute VPIN's robustness and real-time warning value. That combination is important: toxicity is a valid concept, but one magic scalar should not be trusted without out-of-sample validation.

Sources:

- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High-frequency World"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
- Easley, Lopez de Prado, O'Hara, "The Microstructure of the Flash Crash"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695041
- Andersen and Bondarenko, "VPIN and the Flash Crash"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731
- Andersen and Bondarenko, "Assessing Measures of Order Flow Toxicity and Early Warning Signals for Market Turbulence"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2449004
- Easley, Lopez de Prado, O'Hara, "The Volume Clock"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858

Implication for this repo:

Do not implement "VPIN says no trade" as the whole strategy. Use toxicity as an ensemble state composed of observable stresses:

- signed trade imbalance;
- OFI;
- queue imbalance;
- spread widening;
- depth collapse;
- liquidation notional/share;
- mark/oracle/mid dislocation;
- funding/OI crowding;
- short-horizon realized volatility.

Probe implication:

The first probe should evaluate a fixed existing toxicity score and a small fixed set of thresholds. It should not mine many toxicity definitions.

### 4. Realized spread and markouts are the right labels for adverse selection

The market-microstructure literature evaluates liquidity provision and adverse selection through realized spread, price impact, and markouts. For this project, that means the right validation target is not classification accuracy. It is whether high-toxicity states precede worse signed markouts, adverse excursion, or execution conditions.

Sources:

- Glosten and Milgrom, "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders"  
  https://www.sciencedirect.com/science/article/abs/pii/0304405X85900444
- Kyle, "Continuous Auctions and Insider Trading"  
  https://www.jstor.org/stable/1913210
- Huang and Stoll, "Dealer versus Auction Markets"  
  https://www.jstor.org/stable/2329325
- Madhavan, Richardson, Roomans, "Why Do Security Prices Change?"  
  https://www.jstor.org/stable/2329302
- Menkveld, "High Frequency Trading and the New Market Makers"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1722924
- Databento microstructure glossary, "Markout"  
  https://databento.com/microstructure/markout

Implication for this repo:

The next probe should compute:

- forward return;
- forward adverse excursion;
- downside deviation;
- Sortino proxy;
- tail-loss probability;
- if/when strategy decisions exist, signed markout from hypothetical execution price.

Probe implication:

Until there is a concrete entry strategy, evaluate direction-agnostic market-state risk first:

- long adverse excursion over horizon H;
- short adverse excursion over horizon H;
- realized path volatility;
- probability of large move during or after top-toxicity windows.

Then later attach the overlay to actual candidate strategies.

### 5. Crypto/perp mechanics make toxicity more than just order-book imbalance

Perpetual futures have venue-specific stress variables: mark price, oracle/index price, funding, open interest, and forced liquidation flow. These can create toxic regimes even when ordinary trade imbalance is ambiguous.

Sources:

- BitMEX, "Fair Price Marking"  
  https://www.bitmex.com/app/fairPriceMarking
- Binance Futures, "Mark Price and Price Index"  
  https://www.binance.com/en/support/faq/mark-price-and-price-index-futures
- dYdX documentation, "Liquidations"  
  https://docs.dydx.xyz/concepts/trading/liquidations
- Paradigm, "A Guide to Designing Effective Perpetual Contracts"  
  https://www.paradigm.xyz/2021/08/a-guide-to-designing-effective-perpetual-contracts
- Makarov and Schoar, "Trading and Arbitrage in Cryptocurrency Markets"  
  https://www.nber.org/papers/w24565  
  https://doi.org/10.1016/j.jfineco.2019.07.001
- Donier and Bouchaud, "Why Do Markets Crash? Bitcoin Data Offers Unprecedented Insights"  
  https://arxiv.org/abs/1509.06461
- Alexander, Choi, Park, Sohn, "BitMEX Bitcoin Derivatives: Price Discovery, Informational Efficiency, and Hedging Effectiveness"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3353583

Implication for this repo:

Pacifica's full-fidelity archive has exactly the fields needed to test a crypto/perp-native toxicity state:

- `prices`: mark, oracle, mid, funding, next funding, open interest;
- `trades`: liquidation cause, direction, amount, price, nonce/history IDs;
- `bbo`: best bid/ask, size, order ID/nonce;
- `book`: depth levels, order counts, exchange nonce;
- `mark_price_candle`: mark OHLCV.

Useful Pacifica-specific toxicity features:

- `mark_minus_mid_bps`;
- `oracle_minus_mid_bps`;
- `mark_minus_oracle_bps`;
- `abs(mark_return_1m - mid_return_1m)`;
- `funding_z`;
- `oi_change_1m`, `oi_change_5m`;
- `funding_z * oi_change`;
- liquidation notional/share by side;
- liquidation notional divided by visible depth;
- spread/depth stress;
- book refresh/depth collapse.

Probe implication:

The first overlay can start with the current regime-state score. But the probe report should explicitly say whether the current score is missing OFI/microprice/liquidation/OI features and should treat a weak result as possibly incomplete rather than final.

## Evaluation design recommended by external sources

### 1. Pre-register a small fixed filter family

Avoid threshold mining. Test a fixed family:

- baseline: all eligible 1-minute states;
- filtered 90: exclude top 10% toxicity;
- filtered 80: exclude top 20% toxicity as sensitivity;
- filtered 70: exclude top 30% toxicity as stress/sensitivity only.

Primary decision should be top-decile exclusion, not whichever cutoff looks best.

Sources:

- Bailey et al., "The Probability of Backtest Overfitting"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- White, "A Reality Check for Data Snooping"  
  https://doi.org/10.1111/1468-0262.00177
- Hansen, "A Test for Superior Predictive Ability"  
  https://doi.org/10.1198/073500104000000045

### 2. Use chronological/purged validation, not random folds

The labels overlap across adjacent minutes. Random folds leak future-path information. Validation should be chronological with purging and embargo.

Sources:

- Lopez de Prado, "Advances in Financial Machine Learning"  
  https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- Hudson & Thames, "Cross Validation in Finance: Purging, Embargoing, Combination"  
  https://hudsonthames.org/cross-validation-in-finance-purging-embargoing-combinatorial/

Implementation implication:

For this initial diagnostic, do not pretend one partial day is a valid walk-forward backtest. The script should classify early results as `INSUFFICIENT_SAMPLE` unless day-count and event-count gates pass.

### 3. Use event-study metrics plus strategy-filter metrics

For every symbol-minute, compute future path statistics over:

- 5m;
- 15m;
- 30m;
- 60m.

For each horizon, report:

- final forward return;
- long adverse excursion;
- short adverse excursion;
- realized path volatility;
- downside deviation;
- Sortino proxy;
- tail-loss probability;
- opportunity retention.

Sources:

- MacKinlay, "Event Studies in Economics and Finance"  
  https://doi.org/10.1257/jel.35.1.13
- Newey and West, "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"  
  https://doi.org/10.2307/1913610
- Sortino and Price, "Performance Measurement in a Downside Risk Framework"  
  https://www.pm-research.com/content/iijpormgmt/17/4/59

Implementation implication:

The script should compare three populations:

1. all eligible minutes;
2. accepted minutes after excluding top toxicity;
3. removed high-toxicity minutes.

The removed population should have visibly worse adverse-excursion/tail-risk metrics. If it does not, the overlay is not economically meaningful.

## Hard gates for this repo

### Sample-size gates

Because the current full-fidelity archive only has early live data, the first run will likely be diagnostic only.

Use these gates:

- minimum 30 distinct calendar days for any serious validation;
- preferred 60+ days;
- at least 100 removed high-toxicity observations per validation fold;
- preferred 300+ per fold;
- no single day contributes more than 10% of removed high-toxicity observations;
- for strategy-attached trade tests, at least 100 baseline trades and at least 30 trades removed by the filter.

If these fail, verdict is `INSUFFICIENT_SAMPLE`, not `PASS`.

### Opportunity-retention gates

For top-decile exclusion:

- minute-level retention should be about 85-92%;
- trade-level retention must be at least 70%;
- gross exposure retention must be at least 70%;
- if the filter removes most opportunity, it fails regardless of downside improvement.

### Downside-improvement gates

For at least two horizons, including either 15m or 30m:

- 5th percentile adverse excursion improves by at least 5%;
- downside deviation improves by at least 3%;
- probability of large adverse excursion improves by at least 5% relative;
- mean forward return does not degrade by more than 5% relative or a pre-set bp tolerance;
- improvement is not driven by one symbol/day.

### Monotonicity gates

The toxicity decile curve should make sense:

- top decile should be among the worst three deciles for adverse-excursion risk in at least three of four horizons;
- Spearman correlation between toxicity decile and adverse-risk metric should have the expected sign in most horizons;
- if only one arbitrary cutoff works and deciles are random, classify as `INCONCLUSIVE_MIXED` or `FAIL`.

### Robustness gates

When there is enough history:

- chronological folds only;
- thresholds computed on train folds only;
- purged forward-label overlap;
- embargo at least max(label horizon, feature lookback if needed);
- block bootstrap by day/symbol;
- multiple-testing adjustment if many features/thresholds are tested.

## Recommended implementation spec for the next probe

### File

`scripts/non_hft_toxic_overlay_probe.py`

### Inputs

- `data/pacifica_silver_partitioned`
- `docs/experiments/non-hft-regime-state/regime_state.parquet` if present, or a regenerated regime-state table

### Outputs

- `docs/experiments/toxic-regime-overlay/README.md`
- `toxic_bucket_summary.csv`
- `overlay_scorecard.csv`
- `symbol_summary.csv`
- `hour_summary.csv`

### Default config

```text
horizons_minutes = [5, 15, 30, 60]
toxicity_cutoffs = [0.90, 0.80, 0.70]
primary_cutoff = 0.90
primary_metric = p05_adverse_excursion
bucket = 1min
verdicts = PASS, FAIL, INSUFFICIENT_SAMPLE, INCONCLUSIVE_MIXED
```

### Metrics by horizon

For each horizon and cutoff:

- `n_baseline`
- `n_removed`
- `n_accepted`
- `retention_rate`
- `n_days`
- `max_day_concentration`
- `mean_forward_return_baseline`
- `mean_forward_return_accepted`
- `mean_forward_return_removed`
- `p05_long_adverse_excursion_baseline`
- `p05_long_adverse_excursion_accepted`
- `p05_long_adverse_excursion_removed`
- `p05_short_adverse_excursion_baseline`
- `p05_short_adverse_excursion_accepted`
- `p05_short_adverse_excursion_removed`
- `downside_deviation_baseline`
- `downside_deviation_accepted`
- `sortino_proxy_baseline`
- `sortino_proxy_accepted`
- `tail_loss_probability_baseline`
- `tail_loss_probability_accepted`
- `delta_downside_deviation`
- `delta_sortino_proxy`
- `delta_p05_adverse_excursion`

### Initial verdict policy

Use conservative labels:

- `INSUFFICIENT_SAMPLE`: sample/day gates fail.
- `FAIL`: enough data, but high toxicity is not associated with worse downside/adverse excursion.
- `INCONCLUSIVE_MIXED`: some horizons improve, others do not, or monotonicity fails.
- `PASS`: enough data and downside/opportunity gates pass out-of-sample.

Given today's archive size, expect the first run to be `INSUFFICIENT_SAMPLE_DIAGNOSTIC`. That is fine. The value is building the repeatable probe now so future full days can be evaluated without changing the rules.

## What this changes versus the previous plan

The prior next step was already toxic-regime/no-trade overlay. This research pass changes the implementation details:

1. The primary label should be adverse excursion/downside-risk, not directional return.
2. The overlay should be evaluated as a risk filter, not alpha.
3. Top-decile exclusion should be primary; top-20/top-30 are sensitivity checks.
4. The report must distinguish `INSUFFICIENT_SAMPLE` from `FAIL`.
5. The eventual feature set needs OFI, queue imbalance, microprice, liquidation stress, funding/OI stress, and mark/oracle dislocation ablations.
6. VPIN-like flow toxicity can be included later, but not as a trusted standalone trigger.
7. Validation must become chronological/purged once enough days accumulate.

## Next action

Implement the probe with strict TDD:

1. create `tests/scripts/test_non_hft_toxic_overlay_probe.py`;
2. write failing tests for forward paths, adverse excursion, toxicity bucketing, overlay metrics, verdict policy, and markdown rendering;
3. implement `scripts/non_hft_toxic_overlay_probe.py`;
4. run on the current partitioned silver/regime-state data;
5. commit the probe and first diagnostic report.

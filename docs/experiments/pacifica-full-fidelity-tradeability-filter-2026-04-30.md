# Pacifica full-fidelity tradeability filter

Date: 2026-04-30
Role: skeptical trading researcher
Question: given the new full-fidelity Pacifica archive, which brainstorm candidates are likely actually tradeable after fees, slippage, and adverse selection?

## Starting constraints from prior evidence

- There is real tape signal. The strongest proven result is cascade-onset prediction, not profitable direction.
- Naive direction trading is dead under realistic economics: 0/300 strict survivor cells at 60% accuracy in prior taker-side tests; H10 uniformly dead.
- Maker execution is not a free lunch. Prior maker-style tests saw severe fill-conditional adverse selection, with E[realized | filled] around -7.89 bp in the tested framing and 0/300 maker-style survivor cells.
- Cascade-onset alone is not an alpha trade because cascade direction failed: direction AUC about 0.441, confidence-threshold accuracy about 0.482, marginal-long variants net-negative.
- Default Pacifica fees are cheaper than the old 6 bp assumption but still material: taker 4.0 bp/side, maker 1.5 bp/side, no retail maker rebate. Round-trip taker cost is 8 bp before slippage. Round-trip maker fee is 3 bp before adverse selection and missed-fill opportunity cost.
- The new archive adds fields that can transform the problem: mark/oracle/funding/open_interest, BBO order IDs, book raw order counts/nonces, raw trades with li/history IDs, and high-fidelity message timing.

## Ranking principle

Prefer candidates that transform the task away from "predict next direction and pay spread/fees" into one of:

1. risk avoidance: do not trade or do not quote during toxic regimes;
2. event/risk-transfer detection: identify when others are forced to trade;
3. relative-value convergence: trade a mechanically anchored spread, not naked direction;
4. execution-quality selection: decide when an existing order should be marketable vs passive vs absent.

Penalize candidates that require:

- high hit-rate directional prediction;
- passive fills exactly when the market is moving in our favor;
- frequent taker turnover;
- inference from fields without a clear executable edge;
- selection on the already-consumed Apr 1-26 cascade holdout.

## DO FIRST

### 1. Toxic-regime / no-quote filter using cascade-onset plus full-fidelity book/BBO features

Verdict: do first.

Why this transforms the problem:
- It uses the proven signal where it is naturally useful: avoiding adverse selection, not choosing direction.
- A no-trade decision has no fee/slippage cost and can improve Sortino by removing left-tail trades from any baseline strategy.
- New BBO IDs, book nonces, order counts, li/history IDs, and raw timing can measure book instability directly: quote churn, queue pull, BBO ID turnover, book update bursts, nonce gaps, trade-through pressure, and book-count collapse.

Tradeability thesis:
- We do not need to predict post-cascade direction. We need to show that conditional on a high toxic-regime score, maker fills or taker entries have materially worse realized PnL/downside than otherwise.
- If true, the immediate paper-trading product is a kill-switch/risk overlay: avoid entry, cancel passive orders, reduce size, or widen any quote during predicted toxic windows.

Minimum implementation:
- Build event-time panel from trades + bbo + book + prices.
- Features: cascade-onset baseline score; BBO ID lifetime/turnover; queue size/order-count delta at touch; book nonce/li jump rate; bid/ask cancel imbalance proxy; mark-mid/oracle-mid basis; OI/funding z-scores; raw recv lag gaps.
- Labels: not direction. Use realized toxicity of entering/quoting: next 1/5/15 min adverse excursion, fill-conditional adverse move at touch, spread widening, book-depth collapse, and realized slippage for simulated taker entry.

Hard falsification gates:
- Kill if top-decile toxic score does not increase next-5/15m adverse excursion or spread/slippage by at least 2x versus bottom half, day-blocked OOS.
- Kill if a maker quote-cancel overlay cannot reduce simulated fill-conditional adverse selection by at least 30% while retaining at least 50% of benign quoting minutes.
- Kill if paper-trading overlay on a simple baseline cannot improve downside deviation or max drawdown by at least 25% without reducing gross opportunity by more than 50%.
- Kill if effects vanish under day-clustered bootstrap or are carried by fewer than 3 independent high-volume symbols/days.

### 2. Mark/oracle/mid dislocation reversion, gated by liquidity and toxicity

Verdict: do first, but keep the first pass brutally small.

Why this transforms the problem:
- It trades a mechanically anchored spread rather than raw direction. The question becomes: when mark or mid deviates from oracle, does it converge enough to clear costs before the next funding/cascade regime?
- New full-fidelity prices contain mark, oracle, mid, funding, next_funding, OI, and volume_24h; these fields were not available in the older lossy tape framing.

Tradeability thesis:
- On a perp venue, extreme mark/oracle or mid/oracle dislocations may encode temporary microstructure stress, stale index movement, funding pressure, or forced-flow overshoot.
- The only plausible executable form is rare, high-threshold mean reversion with taker entry only when expected convergence exceeds 10-15 bp after cost, or passive entry only when toxicity is low.

Minimum implementation:
- Define basis_bps = 1e4 * (mid - oracle) / oracle and mark_basis_bps = 1e4 * (mark - oracle) / oracle.
- Bucket extreme absolute basis by symbol/liquidity, toxicity score, spread, and OI/funding pressure.
- Label convergence to oracle/mark over 1/5/15/60 minutes and adverse excursion.
- Backtest sparse rules, not ML: enter against extreme basis only when spread < threshold, book depth sufficient, toxicity low, and expected convergence/cost ratio > 2.

Hard falsification gates:
- Kill if extreme basis events are too rare: fewer than 100 independent events/month after liquidity/toxicity filters across the tradable universe.
- Kill if median convergence net of 8 bp taker round trip plus measured slippage is not positive by at least 5 bp in day-blocked OOS.
- Kill if 95th percentile adverse excursion before convergence exceeds expected convergence by more than 2x.
- Kill if the signal is just cascade direction in disguise: performance must remain when high cascade-risk windows are excluded or treated separately.
- Kill if oracle/mark fields update at a cadence or construction that makes the apparent edge non-causal.

### 3. Liquidation/forced-flow absorption and post-event stabilization, not pre-event direction

Verdict: do first as an event study; promote only if sample size grows.

Why this transforms the problem:
- Prior cascade-onset was real but directionless. The tradeable event may be after the forced flow has printed: absorption, stabilization, volatility compression, or refill, not pre-cascade direction.
- Full raw trades with li/history IDs and book counts can identify exact forced-flow bursts, repeated liquidations, and whether the book refills or continues to fail.

Tradeability thesis:
- After a liquidation burst, the sign of the forced flow is known from the trade side. A reversal/continuation rule can be conditional on observed absorption: large liquidation notional, little further price progress, depth refill, BBO ID stabilization, spread normalization.
- This is not dead direction prediction if entry waits for evidence that forced selling/buying has been absorbed.

Minimum implementation:
- Detect liquidation bursts from `tc`/cause fields and `li` sequence clustering.
- Features after the burst: price progress per liquidation notional, spread normalization half-life, depth refill at 1/5/10 levels, BBO ID stability, OI drop/change, mark-oracle basis snapback.
- Labels: post-burst reversal/continuation PnL using a delayed entry after stabilization, not at the onset.

Hard falsification gates:
- Kill if <50 independent liquidation burst events/month after clustering; prior post-cascade strict burst probe had only 2 events and was underpowered.
- Kill if delayed-entry rules cannot show positive expectancy after 8 bp taker round trip plus slippage in day-blocked OOS.
- Kill if edge requires entering before the stabilization features are observable.
- Kill if return distribution is dominated by one or two days/symbols.
- Kill if signal collapses when using event clustering that removes overlapping bursts.

## MAYBE

### 4. Funding/OI carry with microstructure timing overlay

Verdict: maybe.

Why it might transform the problem:
- It is a carry/risk-premium hypothesis rather than high-frequency direction prediction.
- New fields include funding, next_funding, OI, and mark/oracle basis, so the archive can test whether funding extremes are compensation for crowding or a trap.

Why skeptical:
- Funding magnitudes are likely small relative to 4 bp taker fees unless holding periods are long or entries are very sparse.
- Carry trades can have ugly left tails exactly during cascades, so Sortino may fail unless the toxic-regime filter is strong.

Hard gates:
- Kill if expected funding capture over intended holding period is <3x round-trip cost plus expected adverse move.
- Kill if funding extreme buckets have worse downside-adjusted returns than neutral buckets after accounting for mark/oracle basis.
- Promote only if a low-turnover rule clears costs with less than weekly turnover and survives symbol/day blocking.

### 5. Cross-symbol contagion / leader-follower risk states

Verdict: maybe.

Why it might transform the problem:
- It uses cascade/onset as a market-state predictor: if SUI/AVAX/PENGU/XRP carry cascade signal while BTC/ETH/HYPE are chance, maybe the edge is in contagion timing or risk-off detection, not per-symbol direction.
- Could be used as an exposure throttle rather than alpha.

Why skeptical:
- Cross-symbol direction trading still pays costs and likely turns into generic beta prediction.
- Need external venue/index data to separate Pacifica-local effects from market-wide moves.

Hard gates:
- Kill if leader toxic states do not predict follower spread/slippage/adverse-excursion by at least 2x versus baseline.
- Kill if any directional follower trade does not clear 8 bp round-trip taker cost plus slippage with a 2x expected-edge/cost ratio.
- Promote only as a risk filter unless there is fresh OOS evidence of net PnL.

### 6. Queue/fill-quality model for order placement

Verdict: maybe, but only as execution research for a known edge.

Why it might transform the problem:
- BBO IDs, book order counts, and raw nonces can estimate whether a passive order at touch is likely to be filled benignly or toxically.
- The useful output is an order type choice: ALO at touch, ALO one tick back, taker, or skip.

Why skeptical:
- It cannot create alpha. It only improves execution conditional on an existing positive expected signal.
- The prior maker adverse-selection result is bad enough that this must be treated as a defensive tool, not a maker-alpha thesis.

Hard gates:
- Kill as standalone alpha.
- Promote only if, on an existing non-random entry signal, the fill-quality model improves net execution by at least 2 bp/trade or reduces fill-conditional adverse move by at least 30% OOS.
- Kill if simulated queue assumptions cannot be bounded with public data; no private queue position means conservative assumptions must be used.

### 7. Venue/data-quality anomaly detection

Verdict: maybe as infrastructure, not alpha.

Why it might matter:
- Full raw messages preserve nonces, `li`, raw envelopes, and recv timing. This can detect gaps, stale streams, crossed/locked books, delayed oracle updates, and duplicate/missing trades.

Why skeptical:
- Most anomalies will be untradeable after latency and fees. But they are valuable to prevent false research results and bad paper/live orders.

Hard gates:
- Promote only as a guardrail unless anomalies recur with >10 bp predictable convergence and enough time to execute.
- Kill any alpha claim if the anomaly is only visible after a delayed message or non-causal restatement.

## DON'T DO

### 8. More generic direction ML/RL on the new fields

Verdict: don't do.

Reason:
- This is the dead framing. Prior realistic direction accuracy produced 0/300 strict survivor cells, encoder/RL framing failed, and cascade direction failed.
- New fields can help define better labels and states, but using them as more inputs to a direction classifier is unlikely to clear fees and slippage.

Hard stop:
- Do not start unless a non-ML rule already has positive net EV after costs on fresh OOS. RL is only allowed later for sizing/execution/risk control, not edge discovery.

### 9. Maker-pivot alpha from posting predicted direction

Verdict: don't do.

Reason:
- Directional maker posting gets filled when wrong. That is the Albers/Maker's Dilemma problem and already appeared in project evidence as severe adverse selection.
- Pacifica default maker fee is 1.5 bp, not a rebate. The economics require both high fill rate and low adverse selection, which are exactly negatively correlated.

Hard stop:
- Do not evaluate using naive maker fees only. Any maker test must condition on fill and include missed-fill opportunity cost.
- Kill if E[realized | filled] remains more negative than -2 bp before fees or if fill probability is highest in losing regimes.

### 10. Taker scalp / H10-H50 micro-direction

Verdict: don't do.

Reason:
- H10 was uniformly dead and high-frequency taker turnover is fee dominated. Even actual Pacifica 4 bp taker fees are too large for weak micro-direction.

Hard stop:
- Reject any candidate whose expected gross per trade is not at least 2x 8 bp round-trip plus measured slippage before paper trading.

### 11. Liquidation-onset directional front-run

Verdict: don't do in its naive form.

Reason:
- Cascade onset prediction is real, but direction given cascade failed. Front-running onset requires direction and fast taker execution into widening spreads.

Hard stop:
- Only revisit as risk-off/no-quote or post-event absorption. Do not trade naked pre-cascade direction without a fresh, pre-registered direction gate clearing costs.

### 12. Funding arbitrage / VIP rebate / points farming assumptions

Verdict: don't do for this bankroll/research goal.

Reason:
- Retail tier has no maker rebate. VIP thresholds are $100M+ 14-day volume and not planning assumptions. Points/referral/MM pools are not robust PnL.

Hard stop:
- Do not include points or inaccessible fee tiers in Sortino calculations.
- Do not use funding carry unless realized net PnL clears normal fees without promotional rewards.

### 13. Broad paper-trading infrastructure before a net-edge gate

Verdict: don't do.

Reason:
- Paper infra can hide absence of edge behind engineering motion. The goal is non-HFT paper-trading Sortino > 2, but no candidate should enter paper until a simple causal backtest clears cost and downside gates.

Hard stop:
- No multi-strategy paper bot until at least one sparse rule passes day-blocked OOS, conservative cost model, and sample-size gates.

## Recommended sequence

1. Build a full-fidelity derived panel for a small liquid subset first: BTC, ETH, SOL, SUI, AVAX, XRP, PENGU if available in the full archive.
2. Implement toxic-regime labels and BBO/book instability features.
3. Run the no-quote/risk-overlay study before any alpha study.
4. In parallel, run the mark/oracle/mid dislocation event study with strict causality checks.
5. Only after enough fresh liquidation events accrue, run the post-liquidation absorption study.
6. Stop immediately if all three do-first candidates fail their gates. Do not fall back to generic direction ML.

## Bottom line

The most promising use of the new full-fidelity archive is not better directional prediction. It is converting Pacifica microstructure signal into (a) toxicity avoidance, (b) mechanically anchored dislocation/convergence trades, and (c) post-forced-flow stabilization trades. If those do not clear hard cost/downside gates on fresh day-blocked OOS, the project should remain archived rather than revived as another ML/RL trading program.

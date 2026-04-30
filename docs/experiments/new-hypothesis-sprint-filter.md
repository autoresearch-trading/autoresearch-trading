# New-Hypothesis Sprint Filter — Skeptical Trading-Research Review

**Date:** 2026-04-30
**Scope:** candidate hypotheses after closure of v1 direction/representation learning, Goal-A taker/maker economics, and Goal-A v2 cascade-onset encoder/adapters.
**Reviewer stance:** no more model work unless the hypothesis first clears paper economics and falsifiability. Existing Apr 14-26 holdout is consumed; any empirical result on current data is discovery-only, not final validation.

## Closed evidence that binds this review

Known positives:
- Cascade-onset is statistically real: flat LR AUC ~0.815 in-sample, ~0.778 OOS, unified day-blocked CV ~0.837.
- Pacifica tape has useful microstructure structure.

Known negatives:
- Direction trading is too weak: v1 found only about +1pp balanced-accuracy extractability at H500.
- Realistic taker economics fail: 0/300 cells survive at 55%, 57.5%, or 60% directional accuracy; PUMP $1k H500 at 60% is only a near-miss (+0.51 bp median, 51.1% positive, below the 55% frac-positive bar).
- Maker execution fails from adverse selection: universe median E[realized | filled] ~ -7.89 bp; 0/300 cells with breakeven below 55% or 60%.
- Cascade direction failed: conditional direction AUC ~0.441; marginal-long variants net negative.
- Pacifica-unique axes are exhausted as standalone alpha: open-flow imbalance and encoder confidence did not pass.
- Encoder/adapters are not the bottleneck: flat hand features beat random-init and nonlinear cascade adapters by large paired deltas.
- Holdout is consumed: no clean OOS remains until fresh post-Apr-26 data accrues.

## Scoring rubric

Score each candidate 0-5 on each dimension. Maximum score: 30.

1. Novel executable label
   - 0: relabels closed direction/cascade-onset/maker/taker problem.
   - 1: small slice of a closed label with no new execution path.
   - 3: genuinely different target, but execution still ambiguous.
   - 5: new target with explicit instrument, side, horizon, and action.

2. Execution economics before ML
   - 0: no cost/slippage/adverse-selection model.
   - 1: depends on maker fills or >60% direction accuracy already shown implausible.
   - 3: plausible paper edge but missing fill/latency/capacity assumptions.
   - 5: positive EV can be shown from label/base-rate math before training.

3. Independence from closed failures
   - 0: killed directly by prior evidence.
   - 1: mostly a renamed direction/maker/cascade-direction bet.
   - 3: uses a closed signal only as context/risk filter.
   - 5: orthogonal mechanism not requiring direction, maker fills, or encoder lift.

4. Falsifiability in one day
   - 0: needs weeks of infra, RL, live trading, or fresh data before any kill test.
   - 1: needs new external data feeds or a simulator that does not exist.
   - 3: can build a discovery-only backtest on existing data, but final OOS awaits fresh data.
   - 5: can be killed or provisionally advanced from existing artifacts in one day with fixed code and no tuning loop.

5. Statistical cleanliness
   - 0: uses consumed holdout as if clean or invites broad search.
   - 1: many degrees of freedom, no multiple-test control.
   - 3: fixed tiny grid, day-blocked CV, embargo, clear multiple-test accounting.
   - 5: pre-registered single test plus untouched future validation plan.

6. Upside if true
   - 0: too small to trade or only intellectual.
   - 1: small-notional/noisy near-miss.
   - 3: useful risk overlay or capacity-limited alpha.
   - 5: standalone positive-EV strategy with credible capacity.

Minimum pass bars:
- One-day sprint candidate: total score >= 20/30 AND no dimension below 3.
- Model-building candidate: must additionally show paper EV > +3 bp/trade after fees/slippage OR risk-avoidance value > 2x expected cost of false positives, before any ML training.
- Reopen trading program: requires fresh untouched OOS or changed venue economics. Current-data results are discovery-only.
- Automatic reject: any candidate requiring maker fills as alpha, cascade direction, H10 taker direction, generic RL-first, new encoder/adapters, or broad grid search over already-touched data.

## Candidate triage

| Candidate | Score | Sprint? | Verdict |
|---|---:|---|---|
| 1. Volatility/instability prediction as non-directional event label | 22 | Yes, narrowly | Worth one-day paper sprint only if action is options/perp risk overlay or trade/no-trade throttle, not naked direction. |
| 2. Liquidity-vacuum / depth-refill state detection | 21 | Yes, narrowly | Worth one-day sprint if the action is avoid toxic taker entries, delay exits, or condition order size; reject as standalone maker alpha. |
| 3. Post-cascade mean reversion | 20 | Yes, but high kill risk | Worth one-day sprint because it is a new timed label after liquidation events; must clear costs without relying on predicting cascade direction. |
| 4. Failed-cascade / liquidation absorption | 20 | Yes, but only with strict label | Worth one-day sprint if label is defined mechanically and entry occurs after failure confirmation; otherwise too close to cascade-direction fishing. |
| 5. Cross-symbol contagion / leader-follower cascades | 19 | Borderline/no unless fixed to one pair family | Potentially interesting, but likely scope-creepy. Permit only a single pre-registered leader set and target set; otherwise reject. |
| 6. Cross-venue lead/lag execution | 18 | No for this repo-only sprint | Good idea in principle, but needs external synchronized feeds and execution assumptions not present here. Defer until data exists. |
| 7. PUMP-only H500 direction rescue | 12 | No | Reject. It is the known near-miss and requires >60% direction where v1 saw ~51%; too narrow, capacity-poor, and likely overfit. |
| 8. Cascade-onset top-tail long/short direction or marginal-long | 5 | No | Reject. Cascade direction and marginal-long already failed. |
| 9. Maker quoting avoidance / spread widening using cascade risk | 16 | No unless base strategy exists | As standalone alpha reject. As risk filter for an independently profitable maker strategy, allow later. |
| 10. New encoder, adapter, transformer, RL policy, or end-to-end simulator | 3 | No | Reject. Architecture/RL is explicitly not the bottleneck. |

## Hypotheses worth a one-day sprint

### 1. Volatility / instability prediction, not direction

Why it survives paper review:
- It does not require predicting sign.
- Cascade-onset AUC says instability regimes are learnable from tape.
- The economic action can be risk reduction, position sizing, or optionality-style exposure rather than fee-paying round trips on every signal.

Pre-registered label:
- At anchor t, positive if max absolute mid/VWAP move over H100-H500 exceeds a fixed threshold, e.g. top 5% of symbol-day normalized realized absolute return, excluding windows that overlap the label construction lookahead.
- Alternative: positive if realized range over H500 exceeds k times trailing median range.

Allowed action:
- Risk overlay only unless a separate volatility-monetization instrument exists.
- Examples: reduce inventory, avoid initiating trades, cut leverage, widen stop bands, or trade options/vol instruments if available.

One-day sprint output:
- Base rates by symbol/day.
- Flat-feature LR/GBM discovery AUC with day-blocked CV.
- Confusion table at top 1%, 5%, 10% risk scores.
- Economic proxy: avoided tail-loss bp versus opportunity-cost bp.

Kill criteria:
- Reject if day-blocked AUC < 0.70.
- Reject if top-5% bucket captures < 2.0x baseline tail-event rate.
- Reject if false-positive avoidance cost exceeds avoided tail loss under conservative assumptions.
- Reject if signal is only same-timestamp volatility leakage or dominated by realized return in the immediate label boundary.

Minimum pass to continue beyond one day:
- AUC >= 0.75 day-blocked discovery CV.
- Top-5% precision >= 3x base tail rate.
- At least 5 symbols with positive avoided-loss economics.
- Fresh-data validation plan locked before further tuning.

### 2. Liquidity-vacuum / depth-refill state detection

Why it survives paper review:
- It targets execution quality and state avoidance, not directional alpha.
- Existing economics show slippage/depth matter by size, especially alts and stress windows.
- It can be useful as a sizing/filter layer even when alpha is external.

Pre-registered label:
- Liquidity vacuum: next H50/H100 book-walk cost for $1k/$10k exceeds trailing symbol quantile, or visible L10 fillability falls below fixed threshold.
- Refill: after a vacuum, depth/spread normalizes within H100 without adverse mid continuation beyond a fixed bp threshold.

Allowed action:
- Reduce order size, delay execution, switch venue, or abstain.
- Not allowed: passive maker alpha, unless a separate base maker strategy exists.

One-day sprint output:
- Label prevalence and duration.
- Predictability from pre-event flat features only.
- Cost avoided by not trading top-risk buckets.
- Capacity by symbol and size.

Kill criteria:
- Reject if vacuum events are not predictable before the book state is already visible.
- Reject if top-decile predicted-risk bucket does not at least double future high-slippage incidence.
- Reject if avoided slippage is < 2 bp median at realistic trade sizes.
- Reject if result depends on stale 24s orderbook artifacts.

Minimum pass:
- Top-decile lift >= 2.0x for future high-slippage events.
- Median avoided slippage >= 3 bp for target size.
- Works on at least 5 liquid symbols at $10k or at least 5 high-vol symbols at $1k.

### 3. Post-cascade mean reversion

Why it survives paper review:
- It is after the cascade, not before/onset direction.
- It has a plausible microstructure mechanism: forced liquidations may overshoot, followed by liquidity refill and partial reversion.
- Entry can be delayed until cascade is observed, avoiding the failed cascade-direction problem.

Pre-registered label:
- Identify actual liquidation cascade event from `cause` flags.
- At event end plus a fixed delay d, enter contrarian to the cascade move if the cascade move exceeds threshold q.
- Evaluate signed reversion over H50/H100/H500 after delay, net of taker costs and book-walk slippage.

Allowed action:
- Taker contrarian entry after event confirmation only.
- No maker fills.
- No predicting cascade direction before onset.

One-day sprint output:
- Event count by symbol/day.
- Net reversion distribution after fixed delays: d in a tiny pre-registered set, e.g. 0, 10, 50 events.
- Taker-cost-adjusted median and frac-positive.

Kill criteria:
- Reject if n < 400 events total or < 5 symbols with n >= 30.
- Reject if median net reversion <= 0 after 12 bp fees plus slippage.
- Reject if frac-positive < 0.55.
- Reject if best result appears only after trying many delays/thresholds.

Minimum pass:
- Median net >= +3 bp after costs.
- Frac-positive >= 0.57.
- Day-clustered CI lower bound for median net > 0 or sign-test p survives multiple-test correction.
- Pre-register fresh OOS before additional refinement.

### 4. Failed-cascade / liquidation absorption

Why it survives paper review:
- It asks whether forced flow gets absorbed without continuation, a different label from cascade onset or direction.
- It may create a contrarian setup after confirmation, not before the event.

Pre-registered label:
- During a liquidation burst, define absorption if liquidation notional is high but net price displacement over the following H100 is small or reverses by a fixed fraction.
- Entry is after the absorption condition is observable, not during the ambiguous cascade.

Allowed action:
- Small taker contrarian trade only after absorption confirmation.
- Or risk-off/no-trade classification for existing strategy.

One-day sprint output:
- Mechanical event definition.
- Base rate and sample size.
- Net post-confirmation PnL after costs.
- Sensitivity to exactly one threshold pair, not a grid search.

Kill criteria:
- Reject if label definition requires tuning more than two thresholds.
- Reject if sample size < 400 or concentrated in fewer than 5 symbols.
- Reject if net post-confirmation median <= 0 or frac-positive < 0.55.
- Reject if it degenerates into pre-cascade direction prediction.

Minimum pass:
- Median net >= +3 bp after costs.
- Frac-positive >= 0.57.
- Robust to dropping top 5 event-days.

## Borderline: cross-symbol contagion / leader-follower cascades

This is not approved as an open-ended sprint. It becomes acceptable only if narrowed to a single pre-registered design.

Permitted narrow version:
- Leaders: BTC, ETH, SOL only.
- Targets: high-beta alts only, pre-listed before the run.
- Label: target cascade/instability within fixed H100/H500 after leader cascade/instability.
- Action: risk-off or delayed target-side trade after leader event.

Kill criteria:
- Reject if multiple leader sets, target sets, horizons, or thresholds are explored.
- Reject if top-risk lift < 2x base rate.
- Reject if net trade simulation cannot clear costs without direction accuracy >60%.

Reason for skepticism:
- Cross-symbol search spaces explode quickly.
- Consumed holdout makes broad discovery particularly dangerous.

## Rejected on paper

### Cross-venue lead/lag execution

Not rejected forever, but rejected for this one-day repo sprint.
- Needs external synchronized CEX/DEX feeds, latency assumptions, venue-specific fees, and routing/fill modeling.
- Current repo evidence cannot validate it cleanly.
- Reopen only after data ingestion exists and the first sprint is data-quality/clock-sync, not alpha.

### PUMP-only H500 direction rescue

Rejected.
- It is not a new hypothesis; it is the known near-miss from the taker headroom table.
- Requires materially >60% directional skill on one illiquid small-notional cell.
- v1 demonstrated ~51% direction, and cascade direction was worse than chance.
- Capacity is poor and overfit risk is high.

### Cascade top-tail long/short, marginal-long, or direction gating

Rejected.
- Cascade direction AUC ~0.441 and marginal-long variants were net-negative.
- Onset AUC is real but directionless; direction is the binding variable.

### Maker alpha, passive quoting, or maker-fee rescue

Rejected as standalone alpha.
- Empirical E[realized | filled] ~ -7.89 bp and 99.7% negative cells.
- 0/300 maker cells clear realistic breakeven bars.
- Only acceptable as a risk-avoidance overlay for an already profitable maker strategy.

### More encoders/adapters/transformers/RL

Rejected.
- Flat features beat encoder/adapters on the strongest known signal.
- RL lacks a profitable base policy and would optimize simulator artifacts.
- Architecture cannot fix fee economics, directionlessness, or adverse selection.

## One-day sprint protocol

Hard rules:
1. Pick at most two approved hypotheses; prefer one.
2. Write the label, action, costs, horizons, symbols, and thresholds before running code.
3. Use only day-blocked CV / discovery analysis on touched data. Do not call it validation.
4. No architecture work. Flat LR/GBM or direct conditional tables only.
5. No broad grids. At most two thresholds and two horizons per hypothesis.
6. Every result must include base rate, sample size, day clustering, per-symbol concentration, and net-cost proxy.
7. If economics fail on paper, stop before ML.

Recommended sprint order:
1. Post-cascade mean reversion: fastest path to direct PnL table and hard kill.
2. Liquidity-vacuum/depth-refill: useful even as execution-risk overlay.
3. Volatility/instability: only if there is a concrete risk-overlay or vol-monetization action.
4. Failed-cascade/absorption: only if label can be made mechanical in under 30 minutes.

Global stop conditions:
- No candidate with score >=20 and no subscore below 3: stop.
- No paper economics > +3 bp/trade or avoided-loss value > conservative opportunity cost: stop.
- Any need for RL/new encoder/new simulator to make the idea look viable: stop.
- Any claim of final success without fresh post-Apr-26 OOS: stop.

## Bottom line

The only one-day sprints worth considering are non-directional or post-event hypotheses with explicit cost-aware actions: volatility/instability as a risk overlay, liquidity-vacuum/refill execution filtering, post-cascade mean reversion, and tightly-defined failed-cascade absorption. Everything that asks for better direction, maker fills, cascade direction, new encoders, or RL should be rejected on paper.

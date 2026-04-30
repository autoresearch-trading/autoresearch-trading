# New Hypothesis Sprint — Post-Closure Trading Research

**Date:** 2026-04-30  
**Repo state:** current v1/v2/Goal-A ML trading line is closed via `trading-program-closed`.  
**Purpose:** identify the smallest set of genuinely new executable hypotheses worth testing after the closed evidence killed more model/encoder/RL work on the current framing.

---

## Binding context

The sprint starts from the final verdict, not from a blank slate.

Closed facts:

- Cascade-onset prediction is statistically real.
  - Flat LR AUC approximately 0.837 unified day-blocked CV.
  - Prior OOS AUC approximately 0.778.
- Direction prediction is too weak for taker trading.
  - v1 direction signal was only about +1pp balanced accuracy.
  - 0/300 taker cells survived strict economics at 55%, 57.5%, or 60% direction accuracy.
- Cascade direction failed.
  - Conditional direction AUC approximately 0.441.
  - Marginal-long variants were net-negative.
- Maker alpha is dead under current assumptions.
  - Fill-conditional adverse selection approximately -7.89 bp.
- Learned encoders/adapters are not the bottleneck.
  - Flat hand features beat the encoder/adapter stack for cascade-onset.
- April 14-26 holdout is consumed.
  - Current-data experiments are discovery/calibration only.
  - Any binding OOS claim requires fresh post-Apr-26 data or a pre-declared future holdout.

Therefore, the sprint must not ask:

> Can a better model predict direction/cascade harder?

It must ask:

> Is there a new label/action pair where the execution economics can be positive before ML?

---

## Sprint rule

No code/model work is justified unless the hypothesis has all of these:

1. Exact mechanical label.
2. Exact allowed action.
3. Trade venue/instrument.
4. Fee/slippage/adverse-selection assumption.
5. One-day discovery test.
6. Hard kill criteria.
7. Fresh-OOS plan if it survives discovery.

Any hypothesis requiring a broad grid search, maker fills as alpha, cascade direction, or RL-first is rejected.

---

## Candidate scorecard

| Rank | Hypothesis | Sprint verdict | Why |
|---:|---|---|---|
| 1 | Post-cascade exhaustion / snapback | Run first | Direction is observed after liquidation impulse instead of predicted before it. Cleanest Pacifica-only reframing. |
| 2 | Liquidity-vacuum / depth-refill state | Run second | Turns the dataset into execution/risk signal; may avoid toxic states rather than predict alpha. |
| 3 | Failed-cascade / liquidation absorption | Run third | Different label: stress that fails to break. Potential contrarian setup after confirmation. |
| 4 | Cross-symbol leader/laggard contagion | Borderline | Mechanically plausible but multiple-testing risk is high. Only run if narrowed. |
| 5 | Cross-venue lead/lag / basis convergence | Defer, high upside | Probably the best big idea, but it needs synchronized CEX data that this repo does not currently have. First sprint would be data/clock-sync, not alpha. |
| 6 | Volatility/instability risk overlay | Defer unless base strategy exists | Predictable, but monetization without options/base strategy is unclear. Useful as risk control, not standalone alpha. |
| 7 | PUMP-only H500 direction rescue | Reject | Known near-miss, capacity-poor, needs >60% direction where prior work saw ~51%. |
| 8 | More encoders/adapters/RL | Reject | Architecture/RL is explicitly not the bottleneck. |

---

# Recommended one-day sprint

Run a Pacifica-only event-study sprint called:

> Post-cascade regime test: exhaustion, refill, and failed-break behavior after observed liquidation events.

This is the best immediate sprint because it uses existing repo data and avoids the known dead branches:

- It does not predict cascade direction before onset.
- It does not require maker fills.
- It does not require a new encoder.
- It uses the actual liquidation event to reveal direction.
- It can be killed quickly with an economics table.

The output should be a discovery report only, not a trading claim.

Suggested artifact path:

`docs/experiments/post-cascade-regime-sprint.md`

Suggested script path if implementation follows:

`scripts/post_cascade_regime_probe.py`

Suggested tests if implementation follows:

`tests/scripts/test_post_cascade_regime_probe.py`

---

## Hypothesis 1 — Post-cascade exhaustion / snapback

### Thesis

The prior work failed because it tried to predict cascade direction before the event. Instead, wait until the liquidation burst reveals direction, then test whether forced liquidation overshoots and partially mean-reverts after the burst.

### Mechanical event definition

For each symbol/day:

1. Identify liquidation burst starts using `cause` flags.
2. Group liquidation events into a burst if consecutive liquidation events are within a fixed event gap or time gap.
3. Define burst end as the last liquidation event in the group plus a fixed confirmation delay.
4. Define cascade sign from realized signed price move during the burst:
   - up-cascade if cumulative log return from burst start to burst end is greater than `+x` bp.
   - down-cascade if cumulative log return is less than `-x` bp.
   - ignore small/ambiguous bursts.

Use a tiny fixed threshold set only:

- impulse threshold: 10 bp or symbol-day 75th percentile absolute burst move.
- confirmation delay: 0, 10, or 50 events.
- horizons: H50, H100, H500.

Do not expand the grid without writing a new preregistration.

### Label

`post_cascade_snapback = 1` if the forward return after burst end is opposite the burst sign and clears the cost band.

Equivalent signed metric:

`reversion_bps = -sign(burst_return) * forward_return_bps`

Net metric:

`net_reversion_bps = reversion_bps - taker_round_trip_fee_bps - book_walk_slippage_bps`

### Allowed action

Taker contrarian entry after burst confirmation.

No maker fills.
No pre-cascade direction prediction.
No encoder.

### Economics gate

Advance only if all pass:

- total independent burst count >= 400, or if not, mark as underpowered and require fresh data before continuing;
- at least 5 symbols with >= 30 bursts each;
- median `net_reversion_bps` >= +3 bp;
- frac-positive net >= 0.57;
- day-clustered lower CI for median net > 0, or sign-test survives fixed multiple-test correction;
- result robust after dropping top 5 event-days.

### Kill criteria

Kill if any of these occur:

- median net <= 0 after fees/slippage;
- frac-positive < 0.55;
- best result only appears after trying many thresholds/delays;
- edge is concentrated in one day or one symbol;
- confirmation delay enters after most snapback already happened.

---

## Hypothesis 2 — Liquidity-vacuum / depth-refill state

### Thesis

The strategy may be less about direction and more about avoiding or exploiting execution states. After liquidation bursts, the book may enter one of two regimes:

1. vacuum persists, continuation/slippage risk remains high;
2. depth refills, adverse selection decays and contrarian/re-entry trades become safer.

### Mechanical event definition

At burst end and subsequent orderbook snapshots, compute:

- spread versus trailing symbol median;
- L1/L5/L10 depth versus trailing symbol median;
- book-walk slippage for $1k and $10k;
- same-side depth depletion relative to cascade sign;
- refill time to 50% and 80% of trailing median depth.

`vacuum = 1` if:

- spread is above symbol rolling p90, or
- same-side L5 depth is below rolling p10, or
- $1k/$10k book-walk cost is above rolling p90.

`refill_success = 1` if:

- depth/spread normalizes within H100/H500,
- and mid does not continue adversely by more than a fixed threshold before refill.

### Allowed action

Primary action: execution filter.

- reduce size,
- delay entry,
- avoid exits into vacuum,
- abstain from trading top-risk states.

Secondary action only if proven by data:

- taker contrarian trade after refill confirmation.

No standalone maker alpha unless a future base maker strategy exists.

### Economics gate

Advance only if:

- top-decile predicted vacuum risk doubles future high-slippage incidence;
- median avoided slippage >= 3 bp at target size;
- effect holds on at least 5 liquid symbols at $10k or 5 high-vol symbols at $1k;
- result is not an artifact of the 24s orderbook cadence.

### Kill criteria

Kill if:

- vacuum is only observable after the damage is already done;
- avoided slippage is less than 2 bp median;
- top-risk bucket lift < 2x;
- result depends on stale orderbook snapshots;
- conditional maker E[PnL | filled] remains materially negative if maker usage is proposed.

---

## Hypothesis 3 — Failed-cascade / liquidation absorption

### Thesis

The known cascade-onset model predicts stress. Some high-stress windows do not produce follow-through liquidation or directional continuation. Those failures may represent absorption: large effort, poor result, then reversal.

### Mechanical event definition

Define an attempted-break / failed-cascade window using existing flat features and/or liquidation activity:

- high effort / volume / open-flow stress;
- recent signed move exceeds a fixed bp threshold;
- no real liquidation cascade occurs within the confirmation horizon, or liquidation burst occurs but fails to extend price;
- price displacement over the confirmation window is small relative to effort.

A cleaner post-event version:

During a liquidation burst, define absorption if liquidation notional is high but price displacement over the following H100 is small or reverses by a fixed fraction.

### Label

`absorption_reversal = 1` if post-confirmation forward return is opposite the attempted move and clears cost.

### Allowed action

Taker contrarian trade only after failure/absorption is observable.

No pre-cascade direction model.

### Economics gate

Advance only if:

- sample size >= 400 windows/events, or mark underpowered;
- median net after costs >= +3 bp;
- frac-positive >= 0.57;
- robust after dropping top 5 event-days;
- trigger frequency is high enough to matter.

### Kill criteria

Kill if:

- label requires more than two tuned thresholds;
- sample concentrated in fewer than 5 symbols;
- net median <= 0;
- signal reduces to generic mean reversion with no lift from stress/absorption conditions.

---

## Hypothesis 4 — Cross-symbol leader/laggard contagion

### Thesis

A liquidation/stress event in a leader symbol may precede stress or price movement in laggard symbols. Direction is inherited from observed leader impulse, not inferred from laggard tape alone.

### Strictly allowed narrow design

Leaders:

- BTC
- ETH
- SOL

Targets:

Pre-list a small set before running, e.g.:

- SUI
- XRP
- AVAX
- LINK
- BNB
- HYPE

Label:

- leader has observed cascade/instability event;
- target has not yet moved its beta-adjusted amount;
- target experiences same-sign move or liquidation/stress within fixed H100/H500.

Allowed action:

- risk-off signal for targets, or
- taker same-sign target trade only if net economics clear costs.

### Economics gate

Advance only if:

- top-risk lift over same-day shuffled leader times >= 2x;
- median net after costs > 0;
- at least 5 target symbols show positive discovery economics;
- results survive multiple-test correction for the fixed pair set.

### Kill criteria

Kill if:

- signal disappears versus same-day shuffled leader times;
- edge is just broad market beta drift;
- entry delay is longer than median contagion lag;
- net PnL is concentrated in one day/pair.

### Sprint verdict

Do not run this in the first implementation unless hypotheses 1-3 are dead or obviously underpowered. It is attractive but has more multiple-testing risk.

---

## Deferred high-upside hypothesis — Cross-venue lead/lag or basis convergence

### Thesis

The best big pivot may be external: Pacifica may lag larger CEX venues, or Pacifica/CEX basis may mean-revert. This would add genuinely exogenous information instead of re-mining Pacifica tape.

### Why it is deferred

The current repo does not appear to have synchronized CEX feeds in place. A one-day alpha sprint would mostly test data-quality assumptions, not edge.

### First sprint if pursued later

Data/clock-sync sprint:

1. Ingest Binance/OKX/Bybit trades + BBO for BTC/ETH/SOL/XRP/SUI/BNB.
2. Align to Pacifica timestamps.
3. Measure cross-correlation lead/lag around large moves.
4. Estimate realistic latency and executable spread.
5. Only then define catch-up/basis labels.

### Kill criteria

Kill if:

- lead peak is below executable latency;
- Pacifica already tracks external fair value before a trade can enter;
- apparent catch-up disappears after taker fees/slippage;
- edge requires illiquid symbols with poor capacity.

---

## Implementation plan for the first sprint

If proceeding to code, implement only the first Pacifica-only event-study script.

### Script

Create:

`scripts/post_cascade_regime_probe.py`

Responsibilities:

1. Load cached per-symbol/day arrays and raw orderbook where needed.
2. Identify liquidation bursts from `cause`/event fields.
3. Infer burst sign from realized signed return during burst.
4. Compute post-burst signed returns at H50/H100/H500.
5. Estimate taker net using fixed fee assumptions plus available book-walk slippage.
6. Compute vacuum/refill features from orderbook snapshots if available.
7. Produce CSV tables and markdown report.

### Tests

Create:

`tests/scripts/test_post_cascade_regime_probe.py`

Minimum unit tests:

1. Burst grouping respects max-gap rule.
2. Burst sign is inferred correctly from signed returns.
3. Reversion metric has correct sign.
4. Net PnL subtracts round-trip fee and slippage.
5. Ambiguous/small bursts are excluded.
6. Threshold grid is fixed and cannot silently expand.
7. Day-clustered aggregation handles one-day concentration.

### Report

Create:

`docs/experiments/post-cascade-regime-sprint.md`

Required tables:

1. Burst counts by symbol/day.
2. Post-burst reversion/continuation by horizon and delay.
3. Net bps after costs by symbol and universe-pooled.
4. Day-clustered confidence intervals.
5. Top-event-day ablation.
6. Vacuum/refill descriptive stats if orderbook alignment is reliable.
7. Final PASS/KILL verdict.

### No-go limits

Do not:

- add encoder/model training;
- use RL;
- optimize a broad parameter grid;
- treat consumed Apr14-26 as clean final OOS;
- claim tradeability without post-cost net bps and sample-size gates.

---

## Final sprint decision

Proceed with one narrow Pacifica-only discovery sprint:

> Test whether observed liquidation events create post-event regimes — exhaustion/snapback, persistent vacuum, refill, or failed-break absorption — with positive post-cost economics.

If that sprint fails, stop again. The next plausible continuation after that is not more Pacifica-only ML; it is a cross-venue data/clock-sync sprint.

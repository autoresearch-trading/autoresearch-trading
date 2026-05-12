# Pacifica Research Idea Registry

Purpose: force every future Pacifica edge idea to be falsifiable before implementation. This registry is not evidence of alpha and does not authorize paper/live trading. It is a pre-registration gate for ideas that must later pass the existing economics, eligibility, no-trade governor, feature parity, walk-forward validation, baseline, random-control, and concentration checks.

Current maturity: diagnostic only. The current archive is still too young, current eligibility is 0 symbols, and no registered idea may be described as an edge until the fixed validation gates pass.

## Required fields for every idea

Each `IDEA-###` entry must include all of these fields:

- Hypothesis
- Mechanical label
- Trade/risk action
- Cost model
- Validation window
- Frozen parameters
- Kill criteria
- OOS plan
- Result/verdict

## IDEA-001: Toxic regime no-trade overlay

- **Hypothesis:** High toxicity minutes have worse forward downside than ordinary minutes after costs.
- **Mechanical label:** Forward 60 minute adverse excursion and downside deviation from bucket close.
- **Trade/risk action:** Skip or reduce size during the fixed top toxicity buckets; no directional entry signal.
- **Cost model:** Pacifica locked execution economics v1: 4 bps taker per side, slippage/adverse-selection bps, and funding debits included.
- **Validation window:** Purged chronological OOS windows after at least 30 distinct archive days; 60+ preferred.
- **Frozen parameters:** Top 10/20/30 percent toxicity cuts, fixed before additional maturity reruns.
- **Kill criteria:** Fails if post-cost Sortino/drawdown does not improve versus same-frequency controls or if retention/sample/concentration gates fail.
- **OOS plan:** Use run_pacifica_walk_forward_validation with random same-frequency controls and no threshold retuning.
- **Result/verdict:** INSUFFICIENT_SAMPLE_DIAGNOSTIC; current archive is too young for an edge claim.

## IDEA-002: Event-risk no-trade overlay

- **Hypothesis:** Known macro, venue, or market-structure event windows have worse adverse excursion and slippage than ordinary minutes after costs, so the system should avoid or reduce risk during those windows.
- **Mechanical label:** Event-window indicator joined to 1 minute regime rows with forward adverse excursion, realized spread/slippage proxy, and downside deviation after the event window.
- **Trade/risk action:** Skip new entries or reduce size during configured HIGH severity event windows; never create a directional entry solely from the event flag.
- **Cost model:** Pacifica locked execution economics v1 with fees, slippage/adverse-selection bps, and funding included; event windows must outperform skip/reduce alternatives after costs.
- **Validation window:** Only evaluate once a local production event calendar is configured and at least 30 distinct archive days exist; 60+ preferred.
- **Frozen parameters:** Event severities and pre/post windows must be fixed in the local calendar before outcome evaluation.
- **Kill criteria:** Fails if event filtering does not improve post-cost drawdown/Sortino versus same-frequency controls or if too few event rows make the result concentrated.
- **OOS plan:** Use purged chronological walk-forward windows, compare event-window risk controls against random same-frequency skipped windows, and report event/day/symbol concentration.
- **Result/verdict:** PENDING_DIAGNOSTIC; infrastructure exists but no production local event calendar is configured yet.

## IDEA-003: Reference-market dislocation governor

- **Hypothesis:** Pacifica-local price or volatility dislocations versus BTC/ETH/reference markets identify periods with worse post-cost execution quality and downside risk.
- **Mechanical label:** Cross-venue premium/discount, BTC/ETH beta proxy, reference volatility, and forward 60 minute adverse excursion/downside deviation measured on aligned 1 minute buckets.
- **Trade/risk action:** Skip or reduce size during severe local/reference dislocation states; no latency arbitrage or next-tick trading.
- **Cost model:** Pacifica locked execution economics v1 with taker/maker fees, slippage/adverse selection, and funding debits included.
- **Validation window:** Evaluate after reference feeds are wired locally and at least 30 distinct overlapping Pacifica/reference days exist; 60+ preferred.
- **Frozen parameters:** Dislocation thresholds, reference symbols, volatility buckets, and beta proxy formulas fixed before OOS evaluation.
- **Kill criteria:** Fails if post-cost Sortino/drawdown does not improve, if missing reference coverage dominates, or if pass results depend on one symbol/day/event.
- **OOS plan:** Run purged chronological walk-forward validation with random same-frequency controls from the same OOS population and explicit missing-reference accounting.
- **Result/verdict:** PENDING_DIAGNOSTIC; reference-context builder exists but production external reference feeds are not wired yet.

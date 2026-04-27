# Phase 2 Pre-Registration — c-1 Methodology + DSR Accounting Section

**Date:** 2026-04-27
**Reviewer:** Council-1 (Lopez de Prado / financial-ML methodology)

## 1. DSR effective N for Phase 2

The binding count is **N = 5**, derived as follows:

| # | Trial | Source |
|---|---|---|
| 1 | Gate 1 (frozen-encoder linear probe, Feb+Mar held-out) | Ratified pre-reg |
| 2 | Gate 2 (fine-tuning, Feb+Mar held-out) | Ratified pre-reg |
| 3 | Gate 4 (two-half temporal stability, Feb+Mar held-out) | Ratified pre-reg |
| 4 | Multi-probe-battery calibration peek (Feb+Mar covariate inspection that triggered Path D drop) | Council-1 multi-probe-calibration accounting (`docs/council-reviews/2026-04-26-multi-probe-calibration-c1.md` §A) |
| 5 | Phase 2 backtest on April 14+ untouched data | This pre-registration |

**Per-horizon table and RankMe are zero-cost (descriptive re-display of Gate 1 artifacts on already-touched April 1–13 + already-touched Feb+Mar);** they do not increment N. This is the c-3 Kyle-lens disclosure ratified in `docs/council-reviews/2026-04-26-prepublication-strengthening-c3.md` §3 and is why no surcharge applies.

The **calibration peek is binding** (AFML §7.4, §11.6): inspecting Feb+Mar covariate scale to determine label feasibility is selection against the same temporal slice and counts as one trial even though no encoder forward pass occurred. April 14+ has not been peeked at the covariate level (untouched-clause discipline preserved), so Phase 2 itself adds exactly +1 to N rather than +1+selection-surcharge.

**N = 5** is the number entered into Bailey & López de Prado (2014) Eq. 9.

## 2. Sortino threshold derivation under PSR ≥ 0.95 with N = 5

Per Bailey & López de Prado (2014), the deflated Sharpe ratio adjusts for the maximum-of-N order statistic of the trial-Sharpe distribution. For a per-symbol Sortino with bootstrap-empirical skew and kurtosis on April 14+ daily-return distributions (BTC reference: skew ≈ −0.4 to −0.7, excess kurtosis ≈ 4–8 over 12 trading days; alts higher), the PSR ≥ 0.95 condition rearranges to:

```
Sortino_required ≈ √( (Z_{0.95} + √(2·ln(N)) · √(1 - γ·Z + (κ-1)/4 · Z²)) / T_eff )
```

For T_eff = 12 trading days (April 14–25 untouched window assumed), N = 5, Z₀.₉₅ = 1.645, and bootstrap-empirical skew/kurtosis at the median of the 25-symbol distribution:

- √(2·ln(5)) ≈ 1.794 → expected-max-Sharpe inflation factor for N=5
- Required per-symbol Sortino point estimate ≈ **0.85 to 1.10 daily-annualized-equivalent** to clear PSR ≥ 0.95 *per symbol*.
- For the spec's "≥10 symbols" criterion, the threshold is the **median-of-symbols Sortino ≥ 0.95** with at least 10 symbols above 0 individually AND the joint Sortino across the pooled return series at PSR ≥ 0.95.

**Practical floor:** **per-symbol Sortino > 1.0** on **≥ 10/24 symbols**, with at least one of those 10 also clearing PSR ≥ 0.95 individually. The "positive Sortino" reading from the spec is mathematically too weak under N=5 — c-1 binds it to "Sortino > 1.0 on ≥ 10 symbols" as the operational floor for "positive after DSR." This number must be re-derived from the actual bootstrap null at run time using the empirical per-symbol skew/kurtosis from April 14+ daily returns; the 1.0 figure is the prior-derived floor and may move ±0.15 once the null is bootstrapped.

## 3. Pre-committed knobs (no post-hoc sweeps)

The following must be ratified BEFORE the first Phase 2 backtest run. Defaults are c-1 binding choices.

| Knob | Pre-committed default | Rationale |
|---|---|---|
| **Trading horizon** | **H500 only** | Gate 1's binding +1pp lives at H500; H10/H50/H100 are descriptive only per c-3 disclosure. Sweeping horizons post-hoc is the canonical garden-of-forking-paths. |
| **Abstention threshold** | **None (trade every signal)** OR **fixed 0.55 sigmoid threshold** chosen by coin-flip and committed in the ratification commit message. | A sweep over abstention thresholds is N×K trials. One default, ratified ex ante. |
| **Position sizing** | **Fixed-notional, equal across symbols** | Vol-targeted and Kelly-fractional are both legitimate but each is a separate trial. Pick one ex ante; fixed-notional is the most defensible against sizing-as-data-mining accusations. |
| **Per-symbol vs pooled probe** | **Per-symbol probe** (matches Gate 1 binding result) | Pooled is a different estimator; running both is +1 trial. |
| **Retraining cadence** | **No retraining inside Phase 2 window** (frozen probe trained on Oct-Mar, evaluated on April 14+) | Walk-forward retraining is a sweep over retraining frequency. Frozen is the cleanest single-trial design. |
| **Fee model** | **Pacifica taker-fee + slippage 1bp per side, ratified ex ante** | Fee-model sweeps inflate any positive result. (NOTE: c-2 has separately recommended 6bps mixed round-trip + size-dependent slippage; the pre-reg should adopt the stricter c-2 model.) |
| **Symbol set** | **All 24 non-AVAX + AVAX as pre-designated held-out** | Gate 3 still un-run; AVAX is the universality probe. Reporting AVAX separately is legitimate; cherry-picking which 10 symbols count as "≥10" post-hoc is not. |

**Hard rule:** any deviation from these defaults requires an amendment with council-1 + council-5 sign-off and **adds +1 to N for the deviated trial** under the post-resolution amendment surcharge from `multi-probe-calibration-c1.md` §B.

## 4. Stop conditions (mirroring the post-Gate-2 pre-reg)

**STOP A — encoder fails Phase 2.** Median per-symbol Sortino < 0 OR fewer than 10/24 symbols positive OR PSR < 0.95 at N=5.
- *Action:* publish current end-state (`step4-program-end-state.md`) augmented with the Phase 2 negative result. Headline becomes *"+1pp linearly-extractable direction signal at H500, temporally stable, NOT tradeable after fees and DSR adjustment."* Research-quality positive substructure stands; tradeable claim falsified.

**STOP B — encoder marginally clears at N=4 but fails honest N=5.** Median Sortino > 0 with ≥ 10/24 symbols positive but PSR < 0.95 once N=5 (including the calibration peek) is honestly entered.
- *Action:* publish as *inconclusive*. Disclose N=5 derivation explicitly. Do NOT round up to claim a tradeable result. The calibration peek is real and must not be retroactively erased to inflate PSR.

**CONTINUE A — encoder clears Phase 2 cleanly.** Median per-symbol Sortino > 1.0 with ≥ 10/24 symbols individually positive AND pooled PSR ≥ 0.95 at N=5.
- *Action:* council-2 (microstructure realism) + council-5 (skeptic) sign-off required for paper-trading deployment. **Live capital is NOT authorized by this pre-registration.** A separate live-capital pre-reg with fresh DSR accounting would be required.

## 5. What this pre-registration does NOT authorize

- **Re-pretraining on April 14+ data.** The held-out clause is permanent for this program; any encoder retrained on April 14+ data cannot be evaluated against April 14+ in this pre-reg or any successor.
- **Post-hoc horizon sweeps.** H500 only. No "we noticed H100 was better, let's report that."
- **Post-hoc threshold sweeps.** One abstention threshold, one position-sizing rule, one fee model, ratified before run.
- **Per-symbol cherry-picking after seeing per-symbol P&L.** The "≥ 10 symbols" criterion is on the full 24-symbol set declared ex ante. Excluding XPL because "it's too illiquid" after seeing XPL lose money is a forking path.
- **Bootstrap CI lower-bound substitution for point-estimate test.** Same rule as post-Gate-2 pre-reg §"Bootstrap CI handling": the test is the point estimate against the threshold.

## 6. Honest read — should Phase 2 run at all?

**c-1 recommends NOT running Phase 2 as currently scoped.** Reasoning:

The Gate 1 edge is +1pp balanced accuracy at H500 — i.e., classification accuracy of ~0.515 vs ~0.505 for controls. Translating to expected per-trade edge: roughly 1% directional accuracy advantage, which at typical crypto-perp half-spread costs (5–15 bps round-trip taker) requires per-trade expected move > ~10× the edge to be net-positive after fees. The H500 horizon at stride=50 means roughly 2,500 events per trade, which on liquid symbols is 5–25 minutes. The implied move magnitude required to net out fees against a 1% accuracy edge is **on the order of 0.5–1.0% per H500 window**, which is achievable on liquid alts but marginal on BTC/ETH and non-existent on truly illiquid names.

Combined with N=5 DSR adjustment requiring per-symbol Sortino > 1.0 on ≥ 10/24 symbols (a substantially harder bar than "positive"), **the probability that Phase 2 produces a clean CONTINUE-A result is low — c-1 estimates < 15%** based on the edge-size-vs-fee-budget arithmetic alone, before even considering microstructure realism (c-2's domain).

The honest end-state is **publish current Gate-1/2/4 result as research-quality finding; do not run Phase 2.** If the user insists on running Phase 2, the binding accounting above stands and STOP A is the most likely outcome (≥ 70% probability under c-1's prior). The publication value of running Phase 2 is the negative-result disclosure itself, which is worth roughly the cost of the backtest (~1 day compute, no money) but does not change the publishable headline.

## Summary

DSR effective N = 5 (Gates 1, 2, 4, calibration peek, Phase 2 backtest). Per-symbol Sortino threshold ≈ 1.0 on ≥ 10/24 symbols with PSR ≥ 0.95, derived from bootstrap-empirical skew/kurtosis on April 14+ daily returns (re-derive at run time). Seven knobs must be ratified ex ante (horizon=H500, fixed-notional sizing, per-symbol probe, no retraining, taker fee + 1bp slippage [or c-2's stricter 6bps mixed], full 24-symbol set, single abstention threshold). Stop A / Stop B / Continue A mirror the post-Gate-2 pre-reg structure. **c-1's honest read: do not run Phase 2; the +1pp edge size against N=5 DSR adjustment makes a clean positive result infeasible by construction at probability < 15%.** If the user runs anyway, STOP A is the binding response to the most likely outcome and the publication value is the negative result.

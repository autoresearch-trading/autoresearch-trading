---
title: Per-Symbol Surrogate Sweep — n=1 → n=5 on the Transfer Claim
date: 2026-04-24
status: completed
result: success
sources:
  - docs/experiments/step5-surrogate-sweep.md
  - docs/council-reviews/council-5-gate3-avax-falsifiability.md
  - docs/council-reviews/council-3-avax-microstructure.md
last_updated: 2026-04-24
---

# Experiment: Per-Symbol Surrogate Sweep

## Hypothesis

If the Gate 3 AVAX failure is anomalous to AVAX specifically (council-3
Option B framing), then other in-sample Tier-2/Tier-3 symbols under the same
stride=50 single-symbol 1-month protocol should pass. If the failure is
protocol-underpowered (independent of symbol), then they should fail
identically. This converts the transfer-claim falsifier from n=1 to n=5.

## Setup

- **Checkpoint:** `runs/step3-r2/encoder-best.pt`
- **Script:** `scripts/avax_gate3_probe.py` (generalized in commit `ea07bda`)
- **Protocol:** same as Gate 3 triage — time-ordered 80/20 per month,
  stride=50, balanced accuracy, 1000-resample bootstrap 95% CI, N=50
  shuffled-null, class_prior reported.
- **Symbols (spanning council-3 microstructure tiers):**
  - **ASTER** — launchpad / Tier-3 memecoin zone
  - **LDO** — DeFi Tier 2/3
  - **DOGE** — retail/memecoin Tier 2
  - **PENGU** — pure memecoin Tier 3
  - **UNI** — DeFi/DEX Tier 2
- **Cells:** 5 symbols × 2 months (Feb + Mar) × 2 horizons (H100 + H500) = 20.
- **Pre-commitment:** ratified in spec amendment v2 (commit `9c91f85`).
- **Runtime:** ~2 minutes total on M4 Pro MPS.

## Result

| Question | Count |
|---|---|
| Encoder point-estimate > PCA | 11/20 |
| **Encoder CI strictly above PCA CI** | **1/20** (ASTER Feb H500 only) |
| Encoder CI strictly above 51.4% | 3/20 |
| Encoder CI includes 51.4% | 14/20 |
| Shuffled null within μ±2σ of 0.500 | **20/20** (σ all ≤ 0.038) |

**1/20 CI separations is exactly the chance rate at α=0.05 for a null
encoder-vs-PCA comparison.** The single ASTER Feb H500 win (+9.3pp, CI
strictly separated) is consistent with one such spurious tail draw; 1/20
cannot distinguish real signal from sampling variance.

Per-symbol narrative:
- **ASTER** — only "strong" surrogate, Feb H500 passes everything; Mar H500
  returns to overlap. 1/4 cells doesn't support a consistent claim.
- **LDO, DOGE, PENGU** — point-estimate winners on most cells but always
  inside CIs. Directionally consistent with Gate 1 signal at small n.
- **UNI** — encoder loses to PCA on 3/4 cells. Would have "failed" a
  per-symbol Gate 3 individually — confirming the underpower reading.

## What We Learned

1. **AVAX is inside the surrogate distribution.** 0/4 CI separations on
   AVAX vs 1/20 on surrogates; AVAX's failure pattern is protocol-typical,
   not symbol-specific. Closes the last gap on the "AVAX is anomalous"
   alternative hypothesis.
2. **The encoder's Gate-1 signal (~1–2pp) is invisible under this
   protocol on in-sample symbols.** Gate-1 pass was on pooled 24-symbol
   monthly sets (~16K test windows per month). Pooled down to a single
   symbol at ~300–460 test windows, the variance floor (CI width ~0.09–0.12)
   exceeds signal amplitude.
3. **The surrogate sweep is the empirical null distribution of the
   protocol.** Its 1/20 CI-separation rate defines the false-positive rate
   any single-symbol single-month probe must beat to claim transfer.
4. **Two cells clear 51.4% CI lower bound:** LDO Mar H500 (enc 0.571, CI
   [0.521, 0.621]) and PENGU Feb H500 (enc 0.566, CI [0.516, 0.620]) —
   sparse but nonzero evidence that encoder+LR has some signal on Tier-3
   symbols at H500 under this protocol. NOT enough to reactivate Gate 3
   (re-activation requires n_test ≥ 2000, currently unmet).

## Verdict

**SUCCESS — pre-commitment discharged; Gate 3 reframe is correct.** The
protocol is underpowered for the encoder's signal on any single-symbol
1-month pool, regardless of whether the symbol is held-out (AVAX) or
in-sample (ASTER/LDO/DOGE/PENGU/UNI). Running Gate 3 as binding pass/fail
on any of these would be random outcome selection; the amendment's
informational-only reframe is the right methodological call.

## Related

- [Gate 3 retired to informational](../decisions/gate3-retired-to-informational.md)
- [Underpowered single-symbol probe](../concepts/underpowered-single-symbol-probe.md)
- [Bootstrap methodology](../concepts/bootstrap-methodology.md)
- [Gate 3 triage experiment](gate3-avax-triage.md)
- [Cluster cohesion experiment](cluster-cohesion-diagnostic.md)

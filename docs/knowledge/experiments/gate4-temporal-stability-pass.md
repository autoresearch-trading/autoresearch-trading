---
title: Gate 4 Temporal Stability PASS — Frozen Encoder Stable Across Training Halves
date: 2026-04-26
status: completed
result: success
sources:
  - docs/experiments/step4-gate4-temporal-stability.md
last_updated: 2026-04-27
---

# Experiment: Gate 4 Temporal Stability PASS

## Hypothesis

The +1pp Gate 1 directional signal at H500 is reproducible across
disjoint training-period halves — i.e., it is a temporally-stable
representation property, not a regime-conditional artifact of the
specific training window.

## Setup

- Frozen encoder: `runs/step3-r2/encoder-best.pt`
- Two probes trained on disjoint halves of the training period:
  - **Oct-Nov 2025** half
  - **Dec-Jan 2026** half
- Both probes evaluated on the same Feb+Mar 2026 held-out window at H500
- Spec threshold: <3pp drop on 10+ symbols (balanced accuracy)

## Result — PASS

- **<3pp drop on 19/24 non-AVAX symbols** (threshold: 10+/24)
- Mean drop: +0.6pp (Dec-Jan-trained probe slightly stronger than Oct-Nov)
- 5 per-symbol "failures" all in direction Dec-Jan stronger:
  - Mean Dec-Jan bal_acc: 0.5063
  - Mean Oct-Nov bal_acc: 0.5001

Sign-of-life pattern: more recent training data carries more signal for
the evaluation period, NOT encoder non-stationarity.

## Verdict — PASS

The +1pp directional signal at H500 is **temporally stable**. It does not
depend on a specific training-period regime. Two probes trained 60+ days
apart, evaluated on the same out-of-sample window, agree within 3pp on
19 of 24 symbols.

## What We Learned

1. **Frozen-encoder representation is regime-robust** under the spec's
   training distribution. The +1pp linear-probe signal is not a
   regime-conditional artifact.
2. **Probe training drift is asymmetric** — recent data > older data —
   consistent with weak directional momentum being part of what's
   linearly extractable, not pure microstructure tape state.
3. **Gate 4 alone does not establish tradeable edge.** A +1pp signal
   stable across training halves is still a +1pp signal; DSR-adjusted
   it does not clear DEX perp fees (60bps round-trip on Pacifica).

## Related

- [Gate 1 Pass — Feb+Mar H500](../experiments/gate1-pass-feb-mar-h500.md)
- [Gate 2 Fine-Tuning FAIL](../experiments/gate2-finetune-fail.md)
- [Gate 4 Rewrite for Coherence](../decisions/gate4-rewrite-for-coherence.md)

---
title: Manifold Deficiency
topics: [representation-learning, falsifiability, evaluation]
sources:
  - docs/experiments/goal-a-v2/random_init_probe_validator_report.md
  - docs/experiments/goal-a-v2/cascade_adapter_validator_report.md
  - docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md
last_updated: 2026-04-27
---

# Manifold Deficiency

## What It Is

When a fixed-architecture neural encoder maps raw inputs into an embedding
space that **does not contain a downstream task's signal extractable by ANY
plausible head** — neither linear nor non-linear. Distinct from a
**linearity artifact** (where the embedding does contain signal but only a
non-linear head can recover it).

The diagnostic for distinguishing the two:

| Test | Linearity artifact | Manifold deficiency |
|---|---|---|
| Linear probe vs hand features | Loses | Loses |
| Non-linear adapter vs hand features | **Wins** (or matches) | Loses |
| Gap closure (linear → non-linear) | Large (~50%+ of the gap) | Small (<25% of the gap) |

If the non-linear adapter does not close most of the gap, no amount of
end-to-end fine-tuning or pretraining is plausibly going to fix the
encoder for that task without changing the architecture or the input
features.

## Our Implementation

First diagnosed in Goal-A v2 (2026-04-27) on the cascade-onset prediction
task:

- **Phase 0**: random-init `TapeEncoder` (376K-param dilated CNN, 256-dim
  global embedding) + linear LR. Pooled AUC 0.6463 vs flat-LR 0.8373;
  paired delta −0.181.
- **Phase 1**: same frozen embeddings + non-linear adapter
  (`Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)`). Pooled AUC 0.6941;
  paired delta −0.131.
- **Gap closure: 4pp of an 18pp gap (~22%)**. Below the 50% threshold for
  "linearity artifact"; manifold-deficiency confirmed.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-27 | Adopt 5b adapter test as standard cheap arbiter | $0 CPU-hour cost; cleanly distinguishes linearity-artifact from manifold-deficiency before committing GPU | [synthesis review](../../council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md) |
| 2026-04-27 | Use 50% gap-closure as the threshold | Council-6's three-outcome decision tree maps gap closure to "GREENLIGHT / MATCHED / KILL" tiers | [synthesis review](../../council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md) |

## Gotchas

1. **Low gap closure is not necessarily a death sentence for the
   architecture.** It's a death sentence for *that architecture on that task
   with that input feature set*. Changing input features (e.g. pre-tokenizing
   the tape into Wyckoff-state tokens) could re-enable the manifold.
2. **High gap closure is not a guarantee that pretraining will help.** The
   non-linear adapter could be exploiting random-projection structure that
   pretraining might destroy. Re-test with MEM-pretrained embeddings before
   committing to fine-tune.
3. **The 50% threshold is heuristic.** No theoretical basis. Council-6
   chose it; if a future experiment falsifies the threshold, raise the bar.

## Phenomenology connection

The cascade-onset case (council-4 review): hand features capture
summary-statistic regime change (mean(kyle_lambda), max(climax_score),
last(is_open)) trivially. A random projection of (200, 17) to 256-dim does
NOT preserve those nonlinear summaries — the manifold is genuinely
deficient for THIS task. A trained encoder would need to LEARN the
summaries, but with only 169 positives, gradient signal is too weak.

## Related Concepts

- [Phase 0 Random-Init Probe](../experiments/phase0-random-init-probe.md)
- [Phase 1 Adapter Test](../experiments/phase1-adapter-test.md)
- [Random-Init Probe Protocol](../decisions/random-init-probe-protocol.md)

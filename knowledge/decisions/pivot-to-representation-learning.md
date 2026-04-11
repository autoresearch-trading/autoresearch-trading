---
title: Pivot to Representation Learning
date: 2026-04-10
status: accepted
decided_by: user + council (unanimous)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/repr-learning-synthesis.md
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Decision: Pivot to Representation Learning

## What Was Decided

Pivot the entire project from supervised direction prediction (Sortino optimization)
to self-supervised representation learning. The model learns to observe the tape
like a human tape reader develops intuition — direction prediction becomes a
downstream probing task, not the primary objective.

## Why

1. **MLP ceiling reached:** v11 MLP hit Sortino=0.353, walk-forward=0.261, only 9/23
   symbols passing. Every incremental change (20+ experiments) made it worse.

2. **Data reframe:** 40GB of raw trades is marginally sufficient for detecting a 2%
   directional edge, but *massive* for self-supervised representation learning — the
   same way GPT doesn't need labels to learn language structure.

3. **User insight:** The goal is research/exploration, not live trading. Learning to
   observe (representations) is more valuable than learning to predict (signals).

4. **Universality problem:** 9/23 symbols means the model learned symbol-specific
   patterns, not universal microstructure. Representation learning with equal-symbol
   sampling forces universality.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Continue supervised MLP sweeps | At local optimum — every change regressed |
| Bigger supervised model (LSTM/Transformer) | v7 attention overfit on H100 — more data won't help without better objectives |
| Semi-supervised hybrid | Adds complexity without the philosophical shift to observation-first |
| Transfer from equity LOB models | DEX microstructure (is_open, liquidations) differs fundamentally |

## Impact

- **New spec:** `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- **Branch renamed:** `tape-reading` → `representation-learning`
- **Architecture:** 91K → ~400K params, MEM + contrastive pretraining
- **Primary metric:** representation quality (probing tasks, CKA, cluster analysis) instead of Sortino
- **Evaluation:** 5 pre-registered gates with hard thresholds instead of "did Sortino go up?"
- **All agent prompts, skills, contexts reframed** around the new direction

## Related Concepts

- [Effort vs Result](../concepts/effort-vs-result.md) — master signal for representation learning
- [Climax Score](../concepts/climax-score.md) — phase transition detection
- [v11 MLP Baseline](../experiments/v11-baseline.md) — the ceiling that motivated the pivot

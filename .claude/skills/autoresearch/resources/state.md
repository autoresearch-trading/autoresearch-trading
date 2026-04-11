# Research State — Representation Learning Branch

## Environment
- Spec: `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`
- Branch: `representation-learning`
- Stack: Python 3.12+, PyTorch, NumPy, Pandas, DuckDB
- Data: 40GB raw trades, 160 days, 25 symbols (AVAX held out from pretraining)
- Primary metric: representation quality (probing tasks, cluster analysis), NOT Sortino
- Compute cap: 1 H100-day before evaluation gates

## Current State
- Spec written (2026-04-10)
- Council designed self-supervised framework (MEM + contrastive)
- Next steps: Step 0 (data validation) → Step 2 (baselines/Gate 0) → Step 3 (pretraining)

## Architecture
- Self-supervised encoder: ~400K params, dilated CNN, 256-dim embeddings
- Pretraining: MEM (block masking, weight 0.70) + SimCLR contrastive (weight 0.30)
- Fine-tuning: freeze → probe → unfreeze at lr=5e-5

## Evaluation Gates (pre-registered)
| Gate | Threshold | Status |
|------|-----------|--------|
| 0 | PCA + random encoder baselines | Not started |
| 1 | Linear probe > 51.4% on 15+/25 symbols | Not started |
| 2 | Fine-tuned > logistic regression by ≥ 0.5pp | Not started |
| 3 | AVAX (held out) > 51.4% | Not started |
| 4 | Temporal stability < 3pp drop | Not started |

## Key Context
- Prior supervised MLP on main branch hit a local optimum — motivated the pivot to representation learning
- 100-trade batching destroyed tape signals — this branch works with raw order events
- `is_open` has autocorrelation half-life of 20 trades (strongest persistent signal)

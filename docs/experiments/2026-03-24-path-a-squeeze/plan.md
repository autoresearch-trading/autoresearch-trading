# Experiment: Path A — Squeeze v9

## Hypothesis
v9 has edge (PF=1.54) but Sortino≈0 because variance eats returns. Three independent mechanisms should reduce variance without destroying edge:
1. Combining VPIN gate + MAX_HOLD=200 (best passing + best Sortino from sweep — never tested together)
2. Kelly position sizing (Theorem 14 proves G(f*)>0 when PF>1)
3. Confidence gating (only trade on high logit margin)

## Scoring
`score = mean_sortino * 0.6 + (passing / 25) * 0.4`

## Phases

### Phase 1: Combine best sweep winners
| Run | Config delta (vs current code) | Purpose |
|-----|-------------------------------|---------|
| 1   | None (MAX_HOLD=300, vpin_max_z=1.5) | Control — reproduce current best |
| 2   | MAX_HOLD=200 | Combine VPIN gate + shorter holds |

### Phase 2: Kelly position sizing
Depends on Phase 1 winner. Modify evaluate() to scale position by Kelly fraction instead of all-in.
Kelly f* = p*(PF-1)/PF where p=win_rate, PF=profit_factor. Use half-Kelly for safety.
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 3   | + half-Kelly sizing | Does fractional sizing reduce variance? |

### Phase 3: Confidence gating
Depends on Phase 2 result. Only take trades when ensemble logit margin > threshold.
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 4   | + logit_margin_min=1.0 | Filter low-conviction trades |
| 5   | + logit_margin_min=2.0 | More aggressive filtering |

## Decision Logic
- Phase 1: Pick config with higher score. If tied, prefer higher Sortino.
- Phase 2: Keep Kelly if Sortino improves without dropping passing below 20.
- Phase 3: Sweep margin thresholds. Keep if Sortino improves.
- Each phase inherits winning config from prior phase.

## Budget
5 runs × ~10 min = ~50 min total

## Gotchas
- Kelly sizing requires modifying evaluate() in prepare.py — but NOT changing features, so cache stays valid.
- Confidence gating requires modifying make_ensemble_fn() in train.py to return logits, then threshold in eval_policy.
- One run at a time (memory constraints per feedback_parallel_experiments.md).

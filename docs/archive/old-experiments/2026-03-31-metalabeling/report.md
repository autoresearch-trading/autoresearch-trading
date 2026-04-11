# Experiment Report: Metalabeling

## Results
| Run | Name | Threshold | Sortino | Passing | Trades | WR | PF | Score |
|-----|------|-----------|---------|---------|--------|------|------|-------|
| 1 | control | — | 0.353 | 9/23 | 1270 | 0.55 | 1.71 | 0.368 |
| 2 | meta_t0.5 | 0.5 | 0.340 | 7/23 | 1617 | 0.57 | 1.87 | 0.326 |
| 3 | meta_t0.7 | 0.7 | 0.122 | 9/23 | 1489 | 0.53 | 1.35 | 0.229 |

## Eval Result

| Metric | Control | Best (t=0.5) | Criterion | Status |
|--------|---------|--------------|-----------|--------|
| Sortino | 0.353 | 0.340 | >= 0.353 | FAIL |
| WR | 0.55 | 0.57 | > 0.55 | PASS |

**Verdict:** DISCARD
**Reason:** Metalabeling doesn't improve Sortino at either threshold. The meta-model fails to learn a useful gating signal.

## Analysis

The meta-model was supposed to filter losing trades, improving precision. Instead:

- **At t=0.5:** More trades (1617 vs 1270), slightly higher WR/PF, but lower Sortino and fewer passing symbols. The gate is too permissive — it lets through most trades including bad ones.
- **At t=0.7:** Same passing count (9), but much lower Sortino (0.122). The gate vetoes some good trades on high-quality symbols while letting through marginal ones.

Why metalabeling failed here:
1. **The primary model is a single-seed lottery winner.** Its predictions are already noisy — the meta-model is trying to learn patterns in noise.
2. **Val data is only 25 days.** The meta-model has very little data to learn from, especially since it only sees directional predictions (not flat).
3. **The meta-features (softmax probs + temporal stats) overlap heavily with what the primary already uses.** There's no new information for the meta-model to exploit.

The research doc's prerequisite was met (WR=55%, positive expectancy), but the implementation hit practical limitations: insufficient val data and no novel features for the meta-model.

## Conclusion

Metalabeling doesn't help in this regime. The approach needs either (1) more data for the meta-model, (2) novel features not available to the primary (e.g., cross-symbol signals, regime indicators), or (3) a stronger primary model.

## Recommended Config
No change — keep use_metalabeling=False.

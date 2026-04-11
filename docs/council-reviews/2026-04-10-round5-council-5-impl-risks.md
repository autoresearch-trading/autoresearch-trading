# Council-5 Round 5: Implementation Risk Analysis

**Reviewer:** Council-5 (Practitioner Quant — Falsifiability)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

The two highest-priority risks are: (1) cum_ofi_5 naive formula produces anti-correlated signal in 60-80% of trending snapshots, and (2) kyle_lambda signed_notional must aggregate at snapshot level, not event level.

## Feature Implementation Risk Ranking

### Tier 1 — Will Silently Corrupt Training

| Feature | Risk | Detection Difficulty |
|---|---|---|
| cum_ofi_5 (piecewise OFI) | Naive formula anti-correlated during trends | HIGH — values look plausible |
| kyle_lambda (signed_notional) | Event-level aggregation → near-constant zero for 95% of pre-April | HIGH — feature appears computed |
| climax_score (rolling sigma) | Day-parallel breaks rolling state | MEDIUM — opening artifact |
| effort_vs_result (3 error modes) | Wrong normalization → symbol identifier; wrong epsilon; wrong median scope | MEDIUM — distributional check catches it |
| prev_seq_time_span | DataLoader batch vs chronological prior window | MEDIUM — temporal check catches it |

### Tier 2 — Detectable With Sanity Checks

| Feature | Risk | Detection |
|---|---|---|
| trade_vs_mid | NaN from one-sided book mid | Assert no NaN in cache |
| depth_ratio | Extreme outliers without clip → BatchNorm dominated | Add clip(-10, 10) |
| imbalance_L5 | Assumes 10 levels always present | Validate on illiquid symbols |
| log_spread | Crossed book or null mid | Assert spread >= 0 |

## Lookahead Leak Vectors (Beyond Documented)

1. **BatchNorm stats at evaluation:** Any `model.train()` call during embedding extraction contaminates running stats
2. **Global statistics in Wyckoff self-labels:** Rolling percentile thresholds must be causal, not global
3. **Contrastive window jitter at day boundaries:** Must check and fall back to zero jitter
4. **MEM targets shift as BN stats converge:** Consider freezing BN stats after epoch 5
5. **Embargo anchor:** Must measure from final training EVENT, not window start (off by 200 events)

## MEM Training Failure Modes

1. **Shortcut learning via autocorrelation:** 5-event blocks are trivially interpolatable → enlarge to 10-15 events. Monitor per-feature reconstruction MSE.
2. **Representation collapse to symbol identity:** Run symbol probe every 5 epochs; stop if > 30%
3. **Temporal encoding instead of market state:** Add hour-of-day probe as diagnostic; compare to PCA baseline

## Gate 0 Baseline Pitfalls

- C selection must use temporal inner split, never April data
- StandardScaler on training data before PCA (matching encoder's BatchNorm)
- n=50 PCA may be too few or too many — sweep {20, 50, 100, 200}
- Add liquid-symbol sub-gate: 5+/6 liquid must individually exceed 51.4%

## 6-Point Pipeline Validation Checklist (Before Any Training)

1. Manually inspect 17 features for 5 consecutive BTC windows
2. Verify rolling median is actually rolling (plot 5,000 events)
3. Check no window crosses midnight boundary
4. Per-symbol dedup rate should be 45-55%
5. cum_ofi_5 sign positive during BTC uptrend
6. Random encoder produces near-uniform embeddings

## Compute Budget

- ~3.5M windows, batch=256, ~1,500 samples/sec conservative
- 20 epochs ≈ 13 hours, 40 epochs ≈ 26 hours
- 1 H100-day budget fits 20-35 epochs comfortably
- Cache build: 3-5 hours local CPU (must be done before H100 session)

## Risk Register Summary

| Risk | Probability | Impact | Time to Validate |
|---|---|---|---|
| cum_ofi_5 sign error | High | Severe | 2 hours |
| kyle_lambda aggregation level | High | Severe | 1 hour |
| MEM shortcut via interpolation | High | Moderate | Detected at Gate 1 |
| Symbol identity collapse | Medium | Severe | Monitor during training |
| Gate 0 C on April data | Low | Severe | Code review |

# Step 0 Falsifiability Prerequisites

**Date:** 2026-04-14
**Data:** Pre-April only (2025-10-16 to 2026-03-25)
**Runtime:** 684.3s

---

## Summary

| Gate | Measurement | Verdict | Action |
|------|-------------|---------|--------|
| 1 | Stress label firing rate | **PASS** | Mean firing rate 1.626% >= 0.5% on BTC/ETH. Stress label is sufficiently sensitive. |
| 2 | Informed flow firing rate | **PASS** | Firing rate 8.80%–13.75% within 2–15% target range. |
| 3 | Climax date-diversity | **PASS** | 2σ: 25/25 symbols ≥15 dates. 3σ: 25/25 ≥15 dates. |
| 4 | Spring threshold recalibration | **PASS** | sigma_mult=3.0 achieves ≤8% on 25/25 symbols. Use sigma_mult=3.0 in Steps 1-2 spring label. |
| 5 | Feature autocorr at lag 5 | **RECALIBRATE** | Features with r>0.8 at lag 5: ['kyle_lambda', 'prev_seq_time_span']. MEM block size must grow to 10–15 events for these features, or use random position masking instead of block masking. |

---

## Measurement 1: Stress Label Firing Rate

**Definition:** Fires if `log_spread > rolling_p90(log_spread)` AND `|depth_ratio| > rolling_p90(|depth_ratio|)` per snapshot.
**Council target:** ≥0.5% on 20+/25 symbols. Red flag if BTC/ETH fire <0.1%.

| Symbol | Date | Snapshots | Firing Rate |
|--------|------|-----------|-------------|
| BTC | 2025-10-20 | 3585 | 1.199% |
| BTC | 2025-12-15 | 3593 | 1.698% |
| BTC | 2026-02-10 | 3512 | 2.819% |
| ETH | 2025-10-20 | 3585 | 0.725% |
| ETH | 2025-12-15 | 3593 | 1.976% |
| ETH | 2026-02-10 | 3512 | 1.338% |

**Verdict: PASS**
Mean firing rate 1.626% >= 0.5% on BTC/ETH. Stress label is sufficiently sensitive.

---

## Measurement 2: Informed Flow Label Firing Rate

**Definition:** `kyle_lambda > rolling_p75` AND `|cum_ofi_5| > rolling_p50` AND sign consistency over 3 consecutive snapshots.
**Council target:** 2–15% firing. Flag if <1% or >20%.

| Symbol | Date | Snapshots | Firing Rate |
|--------|------|-----------|-------------|
| BTC | 2025-10-20 | 3585 | 9.707% |
| BTC | 2025-12-15 | 3593 | 9.240% |
| BTC | 2026-02-10 | 3512 | 8.798% |
| ETH | 2025-10-20 | 3585 | 9.930% |
| ETH | 2025-12-15 | 3593 | 12.302% |
| ETH | 2026-02-10 | 3512 | 13.753% |

**Verdict: PASS**
Firing rate 8.80%–13.75% within 2–15% target range.

---

## Measurement 3: Climax Date-Diversity per Symbol

**Definition:** At least one event fires climax_score = min(z_qty, z_return) > threshold on that calendar date.
**Council target:** ≥15 distinct dates per symbol. Symbols below 15 flag (`!!!`).

| Symbol | Dates ≥2σ | Dates ≥3σ | Total Dates |
|--------|-----------|-----------|-------------|
| 2Z | 148 | 128 | 161 |
| AAVE | 150 | 124 | 161 |
| ASTER | 157 | 143 | 161 |
| AVAX | 158 | 132 | 161 |
| BNB | 159 | 149 | 161 |
| BTC | 160 | 160 | 161 |
| CRV | 157 | 142 | 161 |
| DOGE | 157 | 140 | 161 |
| ENA | 157 | 128 | 161 |
| ETH | 160 | 159 | 161 |
| FARTCOIN | 160 | 150 | 161 |
| HYPE | 160 | 156 | 161 |
| KBONK | 153 | 138 | 161 |
| KPEPE | 155 | 137 | 161 |
| LDO | 156 | 113 | 161 |
| LINK | 152 | 124 | 161 |
| LTC | 157 | 127 | 161 |
| PENGU | 144 | 119 | 161 |
| PUMP | 159 | 133 | 161 |
| SOL | 160 | 153 | 161 |
| SUI | 160 | 148 | 161 |
| UNI | 153 | 128 | 161 |
| WLFI | 160 | 150 | 161 |
| XPL | 156 | 140 | 161 |
| XRP | 159 | 151 | 161 |

**Verdict: PASS**
2σ: 25/25 symbols ≥15 dates. 3σ: 25/25 ≥15 dates.

---

## Measurement 4: Spring Threshold Recalibration

**Definition:** `min(last_50_returns) < -N*σ` AND `evr_at_min > 1.0` AND `is_open_at_min > 0.5` AND `mean(last_10_returns) > 0`.
**Council target:** ≤8% firing on 24+/25 symbols. Symbols where 2σ fires >8% are flagged.

| Symbol | -2σ | -2.5σ | -3σ | Flags |
|--------|-----|-------|-----|-------|
| 2Z | 5.37% | 4.34% | 3.17% |  |
| AAVE | 5.86% | 3.32% | 1.83% |  |
| ASTER | 5.39% | 3.69% | 2.34% |  |
| AVAX | 3.66% | 2.53% | 1.54% |  |
| BNB | 5.27% | 3.45% | 2.13% |  |
| BTC | 12.05% | 8.79% | 6.14% | 2σ>8% |
| CRV | 3.60% | 3.06% | 2.39% |  |
| DOGE | 3.35% | 2.34% | 1.50% |  |
| ENA | 5.13% | 3.56% | 2.29% |  |
| ETH | 11.78% | 8.22% | 5.49% | 2σ>8% |
| FARTCOIN | 3.14% | 2.46% | 1.89% |  |
| HYPE | 8.43% | 5.71% | 3.65% | 2σ>8% |
| KBONK | 1.45% | 1.03% | 0.62% |  |
| KPEPE | 1.36% | 0.94% | 0.60% |  |
| LDO | 3.23% | 2.67% | 2.08% |  |
| LINK | 6.18% | 3.66% | 2.10% |  |
| LTC | 6.51% | 3.85% | 1.85% |  |
| PENGU | 2.49% | 1.83% | 1.33% |  |
| PUMP | 4.50% | 3.25% | 2.14% |  |
| SOL | 10.16% | 6.53% | 3.78% | 2σ>8% |
| SUI | 3.80% | 2.44% | 1.46% |  |
| UNI | 5.64% | 3.77% | 2.09% |  |
| WLFI | 4.48% | 3.43% | 2.50% |  |
| XPL | 6.09% | 4.24% | 2.52% |  |
| XRP | 5.05% | 3.29% | 1.99% |  |

**Verdict: PASS**
sigma_mult=3.0 achieves ≤8% on 25/25 symbols. Use sigma_mult=3.0 in Steps 1-2 spring label.

**Recommended sigma_mult for Steps 1-2:** `3.0`

---

## Measurement 5: Feature Autocorrelation at Lag 5

**Definition:** Average lag-5 autocorrelation over ≥1000 random 200-event windows per symbol.
**Council threshold:** r>0.8 means MEM 5-event block masking is trivially solvable — block size must grow.

#### BTC (1000 windows)

| Feature | Lag-5 Autocorr |
|---------|----------------|
| log_return | 0.0285 |
| log_total_qty | 0.0301 |
| is_open | 0.0264 |
| time_delta | 0.0065 |
| num_fills | 0.0068 |
| book_walk | 0.0000 |
| effort_vs_result | -0.0038 |
| climax_score | -0.0032 |
| prev_seq_time_span | 0.9068 *** |
| log_spread | 0.7130 |
| imbalance_L1 | 0.3976 |
| imbalance_L5 | 0.4069 |
| depth_ratio | 0.4357 |
| trade_vs_mid | 0.2880 |
| delta_imbalance_L1 | -0.0425 |
| kyle_lambda | 0.8121 *** |
| cum_ofi_5 | 0.1128 |
#### ETH (1000 windows)

| Feature | Lag-5 Autocorr |
|---------|----------------|
| log_return | 0.0065 |
| log_total_qty | 0.0449 |
| is_open | 0.0399 |
| time_delta | 0.0213 |
| num_fills | 0.0161 |
| book_walk | 0.0000 |
| effort_vs_result | 0.0031 |
| climax_score | 0.0075 |
| prev_seq_time_span | 0.9174 *** |
| log_spread | 0.6622 |
| imbalance_L1 | 0.3715 |
| imbalance_L5 | 0.3919 |
| depth_ratio | 0.4053 |
| trade_vs_mid | 0.2431 |
| delta_imbalance_L1 | -0.0455 |
| kyle_lambda | 0.0919 |
| cum_ofi_5 | 0.7864 |
#### SOL (1000 windows)

| Feature | Lag-5 Autocorr |
|---------|----------------|
| log_return | 0.0079 |
| log_total_qty | 0.0497 |
| is_open | 0.0325 |
| time_delta | 0.0468 |
| num_fills | 0.0156 |
| book_walk | 0.0000 |
| effort_vs_result | 0.0121 |
| climax_score | 0.0060 |
| prev_seq_time_span | 0.9401 *** |
| log_spread | 0.4345 |
| imbalance_L1 | 0.2679 |
| imbalance_L5 | 0.2892 |
| depth_ratio | 0.2606 |
| trade_vs_mid | 0.1509 |
| delta_imbalance_L1 | -0.0512 |
| kyle_lambda | 0.0282 |
| cum_ofi_5 | 0.7068 |

**Verdict: RECALIBRATE**
Features with r>0.8 at lag 5: ['kyle_lambda', 'prev_seq_time_span']. MEM block size must grow to 10–15 events for these features, or use random position masking instead of block masking.

---

## Actionable Parameters for Steps 1-2 Plan

1. **MEM block size:** Increase MEM block size to 10-15 for: kyle_lambda, prev_seq_time_span.
2. **Spring threshold:** Use `sigma_mult = 3.0` (set `evr > 1.0`, `is_open > 0.5`, `mean_recent10 > 0` unchanged unless ≥24 symbols still fire >8% at this mult).
3. **Stress label:** Keep joint p90 AND p90 as specified.
4. **Informed flow label:** Keep as specified.
5. **Climax probe:** Keep. Use 2σ threshold.

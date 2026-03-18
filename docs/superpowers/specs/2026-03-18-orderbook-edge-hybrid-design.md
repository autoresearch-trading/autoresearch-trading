# Orderbook Edge Features + Hybrid TCN Architecture — Design Spec

## Goal

Add 7 new features (3 orderbook-priority, 4 trade-based) and introduce a hybrid architecture (flat MLP + tiny 1D TCN) that can exploit temporal patterns in the new features. The orderbook features are the moat — they require data competitors can't freely obtain.

## Motivation

- Orderbook data (10-level snapshots every ~24s) is our competitive edge. Trades are freely available everywhere (Binance, Bybit), but L2 orderbook tick data is scarce and expensive.
- Current 6 OB features (indices 12-17) underutilize this data. Research papers (Cont et al. 2021, Elomari-Kessab et al. 2024) show per-level OFI decomposition and liquidity mode separation substantially improve return prediction.
- The flat MLP destroys temporal ordering. A small TCN branch (9K params) can learn local temporal patterns like "3 consecutive batches of rising buy pressure" without the overfitting risk of full attention (v7: 5M params, overfit to 0.061 Sortino).

## New Features (7)

### Orderbook-priority features (indices 39-41)

All computed at orderbook snapshot resolution (~24s), forward-filled to trade batches.

| Index | Feature | Formula | Description |
|-------|---------|---------|-------------|
| 39 | integrated_ofi | `sum(weights_exp * (delta_bid - delta_ask))` where `weights_exp = [1.0, 0.6, 0.36, 0.22, 0.13]` (exponential decay=0.6 across 5 levels) | Per-level OFI with exponential weighting. Approximates PC1 of multi-level OFI per Cont et al. Coexists with existing OFI (feature 16) which uses harmonic decay. |
| 40 | ob_symmetric_mode | `sum(weights_exp * (delta_bid + delta_ask))` | Total liquidity change (sum, not difference). Positive = liquidity added both sides (market calming). Negative = liquidity pulled (market tensing). Existing OFI only captures the asymmetric component. |
| 41 | eff_liquidity_density | Accumulated `run_volume / run_displacement` during monotonic VWAP runs | Volume per unit price displacement during sustained moves. High values = market absorbing heavy flow without accelerating. Different from price_level_absorption (feature 37) which is single-batch. |

**Data requirements:** All use existing `bids`/`asks` arrays from orderbook snapshots (already loaded). Features 39-40 reuse the `prev_bid_vols`/`prev_ask_vols` delta mechanism from existing OFI computation. Feature 41 uses VWAP and notional from trade batches.

**OB snapshot sparsity:** Snapshots arrive every ~24s. Trade batches are ~1-2s (BTC). Features 39-40 update only when a new OB snapshot arrives (~every 15-25 batches); intermediate batches see the forward-filled value. Feature 41 updates every batch (computed from trades, not OB).

### Trade features (indices 42-45)

| Index | Feature | Formula | Description |
|-------|---------|---------|-------------|
| 42 | high_time_frac | `argmax(prices_batched[i]) / trade_batch` | Where in the 100-trade batch did the highest price occur? Late = momentum, early = reversal. Per Tepelyan (Bloomberg, 2025). |
| 43 | low_time_frac | `argmin(prices_batched[i]) / trade_batch` | Where in the 100-trade batch did the lowest price occur? Complementary to feature 42. |
| 44 | hawkes_ratio | `(ewm_buy - ewm_sell) / (ewm_buy + ewm_sell + 1e-10)` with `halflife=10` batches | Captures buy/sell clustering across batches via EWM approximation of Hawkes process. Different from aggressor_imbalance (feature 36) which is single-batch. Per Anantha & Jain (2025). |
| 45 | push_response_asym | `mean_response_after_down - mean_response_after_up` over rolling 50-batch window | Asymmetric mean-reversion strength. Positive = stronger bounce after drops. Per Vlasiuk & Smirnov (2025). |

**Data requirements:** All computed from existing batch-level arrays (`prices_batched`, `is_buy_batched`, `returns`, `total_batch_notional`). No new data sources.

### Feature implementation details

**Feature 39-40 (integrated OFI + symmetric mode):**

Computed inside the existing OB loop (prepare.py lines 397-504). Uses the same `prev_bid_vols`/`prev_ask_vols` arrays and delta mechanism as feature 16 (OFI). New weights array:

```python
weights_exp = np.array([1.0, 0.6, 0.36, 0.22, 0.13])  # exponential decay

# Inside OB loop, after computing delta_bid and delta_ask:
ob_edge_features[i, 0] = (weights_exp[:n_bid_lvls] * (delta_bid[:n_bid_lvls] - delta_ask[:n_ask_lvls])).sum()
ob_edge_features[i, 1] = (weights_exp[:n_bid_lvls] * (delta_bid[:n_bid_lvls] + delta_ask[:n_ask_lvls])).sum()
```

**Feature 41 (effective liquidity density):**

```python
run_volume = np.zeros(num_batches)
run_displacement = np.zeros(num_batches)
for i in range(1, num_batches):
    if np.sign(returns[i]) == np.sign(returns[i-1]) and returns[i] != 0:
        run_volume[i] = run_volume[i-1] + total_batch_notional[i]
        run_displacement[i] = run_displacement[i-1] + abs(vwap[i] - vwap[i-1])
    else:
        run_volume[i] = total_batch_notional[i]
        run_displacement[i] = abs(vwap[i] - vwap[i-1])

eff_liq_density = run_volume / np.maximum(run_displacement, 1e-10)
# Zero out negligible displacement (same guard as feature 37)
eff_liq_density[run_displacement < 1e-8] = 0.0
```

**Feature 42-43 (OHLC timing):**

```python
high_time_frac = np.argmax(prices_batched, axis=1).astype(np.float32) / trade_batch
low_time_frac = np.argmin(prices_batched, axis=1).astype(np.float32) / trade_batch
```

**Feature 44 (Hawkes intensity ratio):**

```python
buy_count = is_buy_batched.sum(axis=1).astype(np.float64)
sell_count = trade_batch - buy_count
alpha = 1 - np.exp(-np.log(2) / 10)  # halflife=10 batches
hawkes_buy = np.zeros(num_batches)
hawkes_sell = np.zeros(num_batches)
hawkes_buy[0] = buy_count[0]
hawkes_sell[0] = sell_count[0]
for i in range(1, num_batches):
    hawkes_buy[i] = alpha * buy_count[i] + (1 - alpha) * hawkes_buy[i-1]
    hawkes_sell[i] = alpha * sell_count[i] + (1 - alpha) * hawkes_sell[i-1]
denom = hawkes_buy + hawkes_sell + 1e-10
hawkes_ratio = (hawkes_buy - hawkes_sell) / denom
```

**Feature 45 (push-response asymmetry):**

```python
push_response_asym = np.zeros(num_batches)
lookback = 50
for i in range(lookback + 1, num_batches):
    window_returns = returns[i - lookback:i]
    window_responses = returns[i - lookback + 1:i + 1]  # 1-step forward response
    down_mask = window_returns < 0
    up_mask = window_returns > 0
    mean_resp_down = window_responses[down_mask[:-1]].mean() if down_mask[:-1].any() else 0.0
    mean_resp_up = window_responses[up_mask[:-1]].mean() if up_mask[:-1].any() else 0.0
    push_response_asym[i] = mean_resp_down - mean_resp_up
```

## Normalization

| Feature | Index | Distribution | Scaling |
|---------|-------|-------------|---------|
| integrated_ofi | 39 | Tail-heavy (notional deltas) | Robust (IQR) |
| ob_symmetric_mode | 40 | Tail-heavy (same) | Robust (IQR) |
| eff_liquidity_density | 41 | Unbounded ratio, heavy tails | Robust (IQR) |
| high_time_frac | 42 | Bounded [0, 1] | Z-score |
| low_time_frac | 43 | Bounded [0, 1] | Z-score |
| hawkes_ratio | 44 | Bounded [-1, 1], symmetric | Z-score |
| push_response_asym | 45 | Symmetric, small values | Z-score |

`ROBUST_FEATURE_INDICES` adds `{39, 40, 41}`.

## Architecture: Hybrid Flat MLP + 1D TCN

### Overview

```
Input: (batch, window=50, features=46)
         │
    ┌─────┴─────┐
    │            │
  [TCN]      [Flat branch]
    │            │
  Conv1d       Flatten: 50×46 = 2,300
  2 layers     + mean(46) + std(46) = 92
    │          = 2,392
  Pool → 16      │
    │            │
    └─────┬──────┘
          │
    Concatenate: 2,392 + 16 = 2,408
          │
    Linear(2408, 256) → ReLU
    Linear(256, 256) → ReLU
    Linear(256, 3)
```

### TCN branch

```python
self.tcn = nn.Sequential(
    nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
)
```

- **kernel_size=5**: sees 5 consecutive batches (~5-10 seconds for BTC). Captures local temporal patterns a tape reader would see.
- **16 output channels**: 16 learned temporal patterns.
- **Dropout 0.2**: regularization to prevent overfitting.
- **AdaptiveAvgPool1d**: collapses time dimension into a fixed-size summary.

### Forward pass

```python
def forward(self, x):
    # x: (batch, window=50, features=46)

    # TCN branch
    tcn_in = x.transpose(1, 2)                  # (batch, 46, 50)
    tcn_out = self.tcn(tcn_in).squeeze(-1)       # (batch, 16)

    # Flat branch
    flat = x.reshape(x.size(0), -1)              # (batch, 2300)
    feat_mean = x.mean(dim=1)                    # (batch, 46)
    feat_std = x.std(dim=1)                      # (batch, 46)
    flat = torch.cat([flat, feat_mean, feat_std], dim=1)  # (batch, 2392)

    # Combine
    combined = torch.cat([flat, tcn_out], dim=1) # (batch, 2408)
    return self.mlp(combined)
```

### Parameter count

| Component | Parameters |
|-----------|-----------|
| Conv1d(46→32, k=5) | 7,392 |
| Conv1d(32→16, k=3) | 1,552 |
| Linear(2408→256) | 616,704 |
| Linear(256→256) | 65,792 |
| Linear(256→3) | 771 |
| **Total** | **~692K** |

Up from ~600K. Still CPU-trainable. TCN adds <2% of parameters.

### Why this won't overfit like v7 attention

| | v7 Attention | Hybrid TCN |
|---|---|---|
| Window | 2,000 steps | 50 steps |
| Temporal params | ~5M | ~9K |
| Compute | H100 GPU | CPU |
| Receptive field | Global | Local (5 steps) |
| Regularization | Minimal | Dropout 0.2 |

### Initialization

- TCN conv layers: Kaiming normal (default for Conv1d with ReLU)
- MLP layers: orthogonal init (same as v6)

## Ensemble

Unchanged: 5 seeds, logit sum argmax. Each seed may learn different temporal patterns in the TCN branch — this increases ensemble diversity.

## Cache Invalidation

- Bump `_FEATURE_VERSION`: `"v6"` → `"v7"`
- v6 caches remain on disk (keyed by version), v7 computed lazily on first run
- Full recompute: ~20-30 min for 25 symbols

## File Changes

| File | Change |
|------|--------|
| prepare.py | Add 7 features (39-45) to `compute_features()`, bump `_FEATURE_VERSION` to `"v7"`, update `ROBUST_FEATURE_INDICES` (add `{39, 40, 41}`), update docstring feature layout |
| train.py | Add `HybridClassifier` class (flat MLP + TCN branch), replace `DirectionClassifier` usage, update `flat_dim` calculation |
| tests/test_features.py | Update expected feature count 39→46, add tests for new features (bounds, NaN-free, shape, OB forward-fill behavior) |

## Success Criteria

**Minimum**: Sortino ≥ 0.230, ≥ 18/25 passing (match v5 baseline).

**Target**: Improvement in Sortino or more passing symbols, with the hybrid TCN exploiting temporal patterns in the orderbook-edge features.

**Validation**: Follow program.md experiment loop:
1. Run v7 (46 features + hybrid TCN) with current hyperparameters
2. Record in results.tsv, compare to v6 history
3. If promising: Optuna search over TCN hyperparameters (major arch change)
4. If worse: isolate by running v7 features with flat MLP to determine if it's features or architecture

## Risks

1. **Two changes at once.** New features + new architecture. If results change, unclear which caused it. Mitigated by: fallback test (v7 features + flat MLP) to isolate.

2. **TCN overfitting.** 9K temporal params with 145 days of data. Mitigated by: dropout 0.2, tiny model, monitor train/val gap.

3. **OFI redundancy.** Integrated OFI (39) and existing OFI (16) use different weights on the same deltas. If correlation > 0.95 after normalization, one is redundant. Not harmful but wastes a feature slot.

4. **OB snapshot sparsity for TCN.** Features 39-40 are constant for 15-25 consecutive batches. TCN conv filters may see long flat stretches. The model can learn to ignore these channels in the TCN and rely on them via the flat branch instead.

5. **Push-response asymmetry noise.** Rolling conditional means over 50 batches with ~25 samples per bucket (up/down). Inherently noisy. If signal is weak, increase lookback to 100.

6. **Cache rebuild time.** ~20-30 min on first run for 25 symbols.

## References

- Cont, Cucuringu, Zhang (2021): "Cross-Impact of Order Flow Imbalance" — integrated OFI via PCA
- Elomari-Kessab, Maitrier, Bonart, Bouchaud (2024): "Microstructure Modes" — symmetric/antisymmetric decomposition
- Federico-Anastasi (2025): "Effective Liquidity Density in Limit Order Books"
- Tepelyan (Bloomberg, 2025): "Enhancing OHLC Data with Timing Features"
- Anantha & Jain (2025): "Forecasting Order Flow Imbalance using Hawkes Processes"
- Vlasiuk & Smirnov (2025): "Push-response anomalies in high-frequency price series"

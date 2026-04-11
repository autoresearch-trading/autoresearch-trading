# Orderbook Edge Features + Hybrid TCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 new features (39-45) and replace the flat MLP with a hybrid flat MLP + 1D TCN architecture.

**Architecture:** 7 new features computed in `prepare.py` (2 orderbook-derived, 1 liquidity-density, 4 trade-based). New `HybridClassifier` in `train.py` concatenates a flat branch (existing) with a tiny 2-layer TCN branch (9K params) that captures local temporal patterns. Total ~692K params, CPU-trainable.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, Pandas, DuckDB, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-orderbook-edge-hybrid-design.md`

---

### Task 1: Add orderbook-derived features (39-40) to prepare.py

**Files:**
- Modify: `prepare.py:382-504` (OB feature section)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for features 39-40**

Add to `tests/test_features.py`:

```python
class TestOBEdgeFeatures:
    """Tests for v8 orderbook edge features (indices 39-40)."""

    def test_integrated_ofi_changes_on_bid_increase(self):
        """Integrated OFI should be positive when bid depth increases."""
        trades = pd.DataFrame({
            "ts_ms": np.arange(200) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(200)],
            "side": ["open_long"] * 200,
            "qty": [1.0] * 200,
            "price": [100.0] * 200,
            "recv_ms": np.arange(200) * 1000 + 1_000_010,
        })
        bids1 = np.array([{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(5)], dtype=object)
        asks1 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object)
        bids2 = np.array([{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(5)], dtype=object)
        asks2 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object)
        ob = pd.DataFrame([
            {"ts_ms": 1_050_000, "symbol": "TEST", "bids": bids1, "asks": asks1, "recv_ms": 1_050_010, "agg_level": 1},
            {"ts_ms": 1_150_000, "symbol": "TEST", "bids": bids2, "asks": asks2, "recv_ms": 1_150_010, "agg_level": 1},
        ])
        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        assert features[1, 39] > 0  # integrated OFI positive on bid increase

    def test_symmetric_mode_negative_on_liquidity_pull(self):
        """Symmetric mode should be negative when depth decreases on both sides."""
        trades = pd.DataFrame({
            "ts_ms": np.arange(200) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(200)],
            "side": ["open_long"] * 200,
            "qty": [1.0] * 200,
            "price": [100.0] * 200,
            "recv_ms": np.arange(200) * 1000 + 1_000_010,
        })
        bids1 = np.array([{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(5)], dtype=object)
        asks1 = np.array([{"price": 101.0 + i * 0.1, "qty": 5.0} for i in range(5)], dtype=object)
        bids2 = np.array([{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(5)], dtype=object)
        asks2 = np.array([{"price": 101.0 + i * 0.1, "qty": 2.0} for i in range(5)], dtype=object)
        ob = pd.DataFrame([
            {"ts_ms": 1_050_000, "symbol": "TEST", "bids": bids1, "asks": asks1, "recv_ms": 1_050_010, "agg_level": 1},
            {"ts_ms": 1_150_000, "symbol": "TEST", "bids": bids2, "asks": asks2, "recv_ms": 1_150_010, "agg_level": 1},
        ])
        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        assert features[1, 40] < 0  # symmetric mode negative when liquidity pulled

    def test_ob_edge_features_zero_without_orderbook(self, make_trades, empty_df, make_funding):
        features, _, _ = compute_features(make_trades(), empty_df, make_funding())
        assert features[0, 39] == 0.0  # integrated_ofi
        assert features[0, 40] == 0.0  # symmetric_mode
```

Do NOT update `NUM_FEATURES` yet — it stays at 39 until Task 3 when all 7 features are in place. These tests use hardcoded indices and only run against their own test class.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py::TestOBEdgeFeatures -v`
Expected: FAIL — features array only has 39 columns, index 39 out of bounds.

- [ ] **Step 3: Implement features 39-40 in prepare.py**

In `prepare.py`, add `ob_edge_features = np.zeros((num_batches, 3))` before the OB loop (around line 383). This is a 3-column array: columns 0-1 for features 39-40 (this task), column 2 for feature 41 (Task 2). Add `weights_exp = np.array([1.0, 0.6, 0.36, 0.22, 0.13])` alongside existing weights arrays (around line 394).

Inside the existing OB loop, after the OFI computation (after line 489, where `prev_bid_vols` and `prev_ask_vols` are set), add:

```python
                    if prev_ob_valid:
                        # Feature 39: integrated OFI (exponential weights)
                        ob_edge_features[i, 0] = (weights_exp * (delta_bid - delta_ask)).sum()
                        # Feature 40: symmetric mode (total liquidity change)
                        ob_edge_features[i, 1] = (weights_exp * (delta_bid + delta_ask)).sum()
```

Note: `delta_bid` and `delta_ask` are already computed as `curr_bid_vols - prev_bid_vols` and `curr_ask_vols - prev_ask_vols` for existing OFI (feature 16). They are zero-padded 5-element arrays. Reuse them directly.

Add `ob_edge_features` to the final `np.hstack` (between `tape_reading_features` and the closing bracket). Do NOT bump `_FEATURE_VERSION` yet — defer to Task 3.

- [ ] **Step 4: Run the new tests only**

Run: `uv run pytest tests/test_features.py::TestOBEdgeFeatures -v`
Expected: PASS

Note: existing `TestFeatureShape` tests will now fail because the feature count is 42 (39 + 3 from `ob_edge_features`) but `NUM_FEATURES` still says 39. This is expected — it will be fixed in Task 3.

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add integrated OFI and symmetric mode features (39-40)"
```

---

### Task 2: Add effective liquidity density (feature 41) to prepare.py

**Files:**
- Modify: `prepare.py` (after tape reading features section, ~line 611)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for feature 41**

Add to `tests/test_features.py`:

```python
class TestEffLiquidityDensity:
    """Tests for effective liquidity density (index 41)."""

    def test_eff_liq_density_non_negative(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        assert np.all(features[:, 41] >= 0.0)

    def test_eff_liq_density_zero_at_batch_zero(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Batch 0 has no prior return to compare, run resets
        assert features[0, 41] == 0.0

    def test_eff_liq_density_higher_during_runs(self):
        """During a sustained price run, density accumulates."""
        # Monotonically increasing price -> single long run
        trades = pd.DataFrame({
            "ts_ms": np.arange(500) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(500)],
            "side": ["open_long"] * 500,
            "qty": [1.0] * 500,
            "price": [100.0 + 0.01 * i for i in range(500)],
            "recv_ms": np.arange(500) * 1000 + 1_000_010,
        })
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        # Later batches in the run should have higher accumulated density
        assert features[3, 41] > features[1, 41]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py::TestEffLiquidityDensity -v`
Expected: FAIL

- [ ] **Step 3: Implement feature 41 in prepare.py**

After the tape reading features section (after line 611), add:

```python
    # Feature 41: effective liquidity density (volume per unit price displacement in runs)
    run_volume = np.zeros(num_batches)
    run_displacement = np.zeros(num_batches)
    for i in range(1, num_batches):
        if np.sign(returns[i]) == np.sign(returns[i - 1]) and returns[i] != 0:
            run_volume[i] = run_volume[i - 1] + total_batch_notional[i]
            run_displacement[i] = run_displacement[i - 1] + abs(vwap[i] - vwap[i - 1])
        else:
            run_volume[i] = total_batch_notional[i]
            run_displacement[i] = abs(vwap[i] - vwap[i - 1])
    eff_liq_density = run_volume / np.maximum(run_displacement, 1e-10)
    eff_liq_density[run_displacement < 1e-8] = 0.0
```

Store `eff_liq_density` into `ob_edge_features[:, 2]` (the 3-column array was already created in Task 1 with `np.zeros((num_batches, 3))`). Add after the loop:

```python
    ob_edge_features[:, 2] = eff_liq_density
```

This keeps features 39-41 together in `ob_edge_features` for the final `np.hstack`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py::TestEffLiquidityDensity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add effective liquidity density feature (41)"
```

---

### Task 3: Add trade features (42-45) to prepare.py

**Files:**
- Modify: `prepare.py` (after feature 41 computation)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for features 42-45**

Add to `tests/test_features.py`:

```python
class TestV8TradeFeatures:
    """Tests for v8 trade features (indices 42-45)."""

    def test_high_low_time_frac_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert np.all(features[:, 42] >= 0.0) and np.all(features[:, 42] <= 1.0)
        assert np.all(features[:, 43] >= 0.0) and np.all(features[:, 43] <= 1.0)

    def test_high_time_frac_late_for_rising_prices(self):
        """Rising prices within batch -> high occurs late."""
        trades = pd.DataFrame({
            "ts_ms": np.arange(100) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(100)],
            "side": ["open_long"] * 100,
            "qty": [1.0] * 100,
            "price": [100.0 + 0.1 * i for i in range(100)],
            "recv_ms": np.arange(100) * 1000 + 1_000_010,
        })
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        assert abs(features[0, 42] - 0.99) < 1e-6  # argmax=99, 99/100=0.99

    def test_hawkes_ratio_bounded(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        assert np.all(features[:, 44] >= -1.0) and np.all(features[:, 44] <= 1.0)

    def test_hawkes_ratio_one_for_all_buys(self):
        """All-buy batches should have hawkes_ratio approaching 1.0."""
        trades = pd.DataFrame({
            "ts_ms": np.arange(500) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(500)],
            "side": ["open_long"] * 500,
            "qty": [1.0] * 500,
            "price": [100.0] * 500,
            "recv_ms": np.arange(500) * 1000 + 1_000_010,
        })
        features, _, _ = compute_features(trades, pd.DataFrame(), pd.DataFrame(), trade_batch=100)
        # After several all-buy batches, hawkes_ratio should be ~1.0
        assert features[-1, 44] > 0.9

    def test_push_response_asym_zero_during_warmup(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        # Only 2 batches, lookback=50, so all zeros
        assert features[0, 45] == 0.0
        assert features[1, 45] == 0.0

    def test_push_response_asym_finite(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=10000), make_orderbook(n=100), make_funding(n=10), trade_batch=100
        )
        assert np.all(np.isfinite(features[:, 45]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py::TestV8TradeFeatures -v`
Expected: FAIL — index 42-45 out of bounds.

- [ ] **Step 3: Implement features 42-45 in prepare.py**

After the feature 41 computation, add:

```python
    # ── v8 trade features ──────────────────────────────────────────
    # Feature 42-43: OHLC timing (where in batch did high/low occur)
    high_time_frac = np.argmax(prices_batched, axis=1).astype(np.float32) / trade_batch
    low_time_frac = np.argmin(prices_batched, axis=1).astype(np.float32) / trade_batch

    # Feature 44: Hawkes intensity ratio (EWM buy/sell clustering)
    buy_count_f44 = is_buy_batched.sum(axis=1).astype(np.float64)
    sell_count_f44 = trade_batch - buy_count_f44
    alpha_hawkes = 1 - np.exp(-np.log(2) / 10)  # halflife=10 batches
    hawkes_buy = np.zeros(num_batches)
    hawkes_sell = np.zeros(num_batches)
    hawkes_buy[0] = buy_count_f44[0]
    hawkes_sell[0] = sell_count_f44[0]
    for i in range(1, num_batches):
        hawkes_buy[i] = alpha_hawkes * buy_count_f44[i] + (1 - alpha_hawkes) * hawkes_buy[i - 1]
        hawkes_sell[i] = alpha_hawkes * sell_count_f44[i] + (1 - alpha_hawkes) * hawkes_sell[i - 1]
    hawkes_denom = hawkes_buy + hawkes_sell + 1e-10
    hawkes_ratio = (hawkes_buy - hawkes_sell) / hawkes_denom

    # Feature 45: push-response asymmetry
    push_response_asym = np.zeros(num_batches)
    pra_lookback = 50
    for i in range(pra_lookback + 1, num_batches):
        pushes = returns[i - pra_lookback : i - 1]
        responses = returns[i - pra_lookback + 1 : i]
        down_mask = pushes < 0
        up_mask = pushes > 0
        mean_resp_down = responses[down_mask].mean() if down_mask.any() else 0.0
        mean_resp_up = responses[up_mask].mean() if up_mask.any() else 0.0
        push_response_asym[i] = mean_resp_down - mean_resp_up
```

Create the v8 feature block and add to the final stack:

```python
    # === v8 FEATURES (indices 39-45) ===
    v8_features = np.column_stack([
        high_time_frac,       # 42
        low_time_frac,        # 43
        hawkes_ratio,         # 44
        push_response_asym,   # 45
    ])
```

The final `np.hstack` becomes:
```python
    features = np.hstack([
        trade_features,          # 0-11
        ob_features,             # 12-17
        extra_features,          # 18-19
        longer_features,         # 20-24
        cutting_edge_features,   # 25-30
        tape_reading_features,   # 31-38
        ob_edge_features,        # 39-41 (integrated_ofi, symmetric_mode, eff_liq_density)
        v8_features,             # 42-45
    ])
```

- [ ] **Step 4: Finalize v8 feature stack — version bump, NUM_FEATURES, normalization, docstring**

Now that all 7 features are implemented, do all the v8 bookkeeping:

1. Change `_FEATURE_VERSION = "v6"` to `_FEATURE_VERSION = "v8"` in prepare.py
2. Add `{39, 40, 41}` to `ROBUST_FEATURE_INDICES` in prepare.py
3. Update the `compute_features` docstring to describe all 46 features
4. In `tests/test_features.py`, change `NUM_FEATURES = 39` to `NUM_FEATURES = 46`
5. Add a comprehensive NaN/Inf check for all new features:

```python
class TestV8Comprehensive:
    """Comprehensive tests for all v8 features."""

    def test_all_new_features_nan_free(self, make_trades, make_orderbook, make_funding):
        features, _, _ = compute_features(
            make_trades(n=10000), make_orderbook(n=200), make_funding(n=20), trade_batch=100
        )
        assert not np.any(np.isnan(features[:, 39:46]))
        assert not np.any(np.isinf(features[:, 39:46]))

    def test_ob_edge_forward_filled_between_snapshots(self):
        """Features 39-40 should hold last value between OB snapshots."""
        trades = pd.DataFrame({
            "ts_ms": np.arange(500) * 1000 + 1_000_000,
            "symbol": "TEST",
            "trade_id": [f"t{i}" for i in range(500)],
            "side": ["open_long"] * 500,
            "qty": [1.0] * 500,
            "price": [100.0] * 500,
            "recv_ms": np.arange(500) * 1000 + 1_000_010,
        })
        # Two OB snapshots: one at batch 0, one at batch 3. Batches 1-2 should forward-fill.
        bids1 = np.array([{"price": 100.0 - i * 0.1, "qty": 2.0} for i in range(5)], dtype=object)
        asks1 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object)
        bids2 = np.array([{"price": 100.0 - i * 0.1, "qty": 5.0} for i in range(5)], dtype=object)
        asks2 = np.array([{"price": 101.0 + i * 0.1, "qty": 3.0} for i in range(5)], dtype=object)
        ob = pd.DataFrame([
            {"ts_ms": 1_050_000, "symbol": "TEST", "bids": bids1, "asks": asks1, "recv_ms": 1_050_010, "agg_level": 1},
            {"ts_ms": 1_350_000, "symbol": "TEST", "bids": bids2, "asks": asks2, "recv_ms": 1_350_010, "agg_level": 1},
        ])
        features, _, _ = compute_features(trades, ob, pd.DataFrame(), trade_batch=100)
        # Between snapshots (batches 1-2), features 39-40 should be 0 (no new OB delta)
        assert features[1, 39] == 0.0
        assert features[2, 39] == 0.0
```

- [ ] **Step 5: Run ALL feature tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: ALL PASS — existing tests now see `NUM_FEATURES = 46` matching the 46-column feature array.

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add trade features 42-45, finalize v8 feature stack (46 features)"
```

---

### Task 4: Add HybridClassifier to train.py

**Files:**
- Modify: `train.py:43-68` (network section)

- [ ] **Step 1: Add HybridClassifier class after DirectionClassifier**

Keep `DirectionClassifier` for fallback testing. Add the new class:

```python
class HybridClassifier(nn.Module):
    """Flat MLP + tiny 1D TCN for local temporal pattern detection."""

    def __init__(self, obs_shape, n_classes, hidden_dim, num_layers, tcn_channels=16, tcn_dropout=0.2):
        super().__init__()
        n_time, n_feat = obs_shape
        flat_dim = n_time * n_feat + 2 * n_feat  # flat + temporal mean + std

        # TCN branch: 2-layer conv, captures local temporal patterns
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
            nn.Conv1d(32, tcn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # MLP on combined features
        combined_dim = flat_dim + tcn_channels
        layers = [_ortho_init(nn.Linear(combined_dim, hidden_dim)), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x):
        # x: (batch, time, feat)
        # TCN branch
        tcn_in = x.transpose(1, 2)                     # (batch, feat, time)
        tcn_out = self.tcn(tcn_in).squeeze(-1)          # (batch, tcn_channels)

        # Flat branch
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        flat = torch.cat([flat, t_mean, t_std], dim=1)

        # Combine
        combined = torch.cat([flat, tcn_out], dim=1)
        return self.head(self.trunk(combined))
```

- [ ] **Step 2: Replace DirectionClassifier usage with HybridClassifier**

In `train_one_model`, change `model = DirectionClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)` to:
```python
    model = HybridClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
```

- [ ] **Step 3: Fix model.eval() for inference**

The `HybridClassifier` has `nn.Dropout(0.2)` in the TCN branch. Dropout must be disabled during evaluation. In `make_ensemble_fn` (the ensemble policy function), set each model to eval mode:

```python
def make_ensemble_fn(models, device):
    """Create ensemble policy function using argmax of summed logits."""
    for m in models:
        m.eval()

    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits_sum = None
            for m in models:
                logits = m(obs_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            return logits_sum.argmax(dim=-1).item()

    return fn
```

- [ ] **Step 4: Verify it runs**

Run: `uv run python3 -c "
import torch
from train import HybridClassifier
m = HybridClassifier((50, 46), 3, 256, 2)
x = torch.randn(4, 50, 46)
out = m(x)
print(f'Output shape: {out.shape}')
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
"`
Expected: `Output shape: torch.Size([4, 3])` and `Params: ~692,000`

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add HybridClassifier (flat MLP + 1D TCN)"
```

---

### Task 5: Update CLAUDE.md and program.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `program.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the features table to show all 46 features (add features 39-45). Update the architecture section to describe the hybrid TCN. Update `ROBUST_FEATURE_INDICES` to include `{39, 40, 41}`. Change version references from v6 to v8. Update `_FEATURE_VERSION` reference. Update flat_dim to 2,408.

- [ ] **Step 2: Update program.md**

Change "39 features per step, v6" to "46 features per step, v8". Update the feature table to include features 39-45. Update "Current Approach" section to mention hybrid TCN architecture instead of flat MLP.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md program.md
git commit -m "docs: update CLAUDE.md and program.md for v8 (46 features + hybrid TCN)"
```

---

### Task 6: Run full training and record results

**Files:**
- Run: `train.py`
- Modify: `results.tsv`

- [ ] **Step 1: Clear v6 caches (optional — v8 caches will be created alongside)**

The v6 caches are keyed differently (`_FEATURE_VERSION = "v6"` vs `"v8"`), so they won't conflict. First run will recompute all features (~20-30 min).

- [ ] **Step 2: Run training**

```bash
uv run train.py 2>&1 | tee run.log
```

Expected: ~20-30 min for cache rebuild + ~5 min training. Watch for:
- Feature shape printed as `(N, 46)` for each symbol
- No NaN/Inf warnings
- PORTFOLIO SUMMARY with sortino, symbols_passing, etc.

- [ ] **Step 3: Record results in results.tsv**

Add a row with the commit hash, sortino, trades, drawdown, passing count, status, and description like `"v8: 46 features + hybrid TCN"`.

- [ ] **Step 4: Commit results**

```bash
git add results.tsv
git commit -m "experiment: v8 orderbook edge features + hybrid TCN results"
```

- [ ] **Step 5: Evaluate and decide next step**

Compare to v6 results in results.tsv history:
- If better: proceed with Optuna search over TCN hyperparameters
- If worse: run fallback test with `DirectionClassifier` to isolate features vs architecture
- If comparable: the features are adding value but TCN may need tuning

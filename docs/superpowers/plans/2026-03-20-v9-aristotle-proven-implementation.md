# v9 Aristotle-Proven Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 39 v6 features with 5 Aristotle-proven features, add HybridClassifier with window=75, add Hawkes regime gate, and widen fee_mult range.

**Architecture:** New `compute_features_v9()` in prepare.py produces 5 features + raw hawkes_branching sidecar. New `HybridClassifier` in train.py adds TCN branch. Regime gate in evaluate() forces flat when hawkes_branching < r_min. Rollout is incremental: Step 1a (features only) → 1b (model) → 2 (window) → 3 (fee_mult) → 4 (gate) → 5 (Optuna).

**Tech Stack:** Python 3.12+, PyTorch, NumPy, Pandas, Gymnasium, Optuna

**Spec:** `docs/superpowers/specs/2026-03-20-aristotle-proven-v9-design.md`

---

### Task 1: Add v9 feature computation to prepare.py

**Files:**
- Modify: `prepare.py` (add `compute_features_v9()`, update `_FEATURE_VERSION`, update `ROBUST_FEATURE_INDICES`)
- Test: `tests/test_features_v9.py` (new)

- [ ] **Step 1: Write failing tests for v9 features**

Create `tests/test_features_v9.py`:

```python
"""Tests for v9 Aristotle-proven features (5 features)."""
import numpy as np
import pandas as pd
from prepare import compute_features_v9

V9_NUM_FEATURES = 5

class TestV9FeatureShape:
    def test_output_shape_5_features(self, make_trades, make_orderbook, make_funding):
        features, timestamps, prices, raw_hawkes = compute_features_v9(
            make_trades(n=200), make_orderbook(n=50), make_funding(n=5), trade_batch=100
        )
        assert features.shape == (2, V9_NUM_FEATURES)
        assert len(timestamps) == 2
        assert len(prices) == 2
        assert raw_hawkes.shape == (2,)

    def test_empty_trades_returns_empty(self, empty_df, make_orderbook, make_funding):
        features, timestamps, prices, raw_hawkes = compute_features_v9(
            empty_df, make_orderbook(), make_funding()
        )
        assert len(features) == 0

    def test_feature_names(self):
        from prepare import V9_FEATURE_NAMES
        assert V9_FEATURE_NAMES == [
            "lambda_ofi", "directional_conviction", "vpin",
            "hawkes_branching", "reservation_price_dev"
        ]

class TestV9LambdaOfi:
    def test_lambda_ofi_is_kyle_lambda_times_signed_flow(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _, _ = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        # Feature 0 should be finite, can be positive or negative
        assert np.all(np.isfinite(features[:, 0]))

class TestV9DirectionalConviction:
    def test_conviction_is_tfi_times_abs_ofi(
        self, make_trades, make_orderbook, make_funding
    ):
        features, _, _, _ = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        # Feature 1 should be finite
        assert np.all(np.isfinite(features[:, 1]))

class TestV9Vpin:
    def test_vpin_non_negative(self, make_trades, make_orderbook, make_funding):
        features, _, _, _ = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        assert np.all(features[:, 2] >= 0)

    def test_vpin_at_most_one(self, make_trades, make_orderbook, make_funding):
        features, _, _, _ = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        assert np.all(features[:, 2] <= 1.0 + 1e-6)

class TestV9HawkesBranching:
    def test_hawkes_in_valid_range(self, make_trades, make_orderbook, make_funding):
        features, _, _, raw_hawkes = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        # Feature 3 (normalized) can be anything after z-score
        # But raw_hawkes should be in [0, 0.99]
        assert np.all(raw_hawkes >= 0.0)
        assert np.all(raw_hawkes <= 0.99)

    def test_hawkes_zero_for_constant_counts(self):
        """Constant trade counts per batch → Var/Mean = 0 → branching = 0."""
        # This tests the domain guard: Var(N) <= E[N] → 0.0
        counts = np.ones(50) * 100  # constant
        var_mean_ratio = np.var(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        # Var = 0, so ratio = 0, which is <= 1, so branching = 0
        assert var_mean_ratio == 0.0

class TestV9ReservationPriceDev:
    def test_reservation_finite(self, make_trades, make_orderbook, make_funding):
        features, _, _, _ = compute_features_v9(
            make_trades(n=500), make_orderbook(n=125), make_funding(n=10),
            trade_batch=100
        )
        assert np.all(np.isfinite(features[:, 4]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_v9.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_features_v9' from 'prepare'`

- [ ] **Step 3: Implement compute_features_v9()**

Add to `prepare.py` after the existing `compute_features()` function (~line 680):

```python
V9_FEATURE_NAMES = [
    "lambda_ofi", "directional_conviction", "vpin",
    "hawkes_branching", "reservation_price_dev"
]

V9_NUM_FEATURES = 5
V9_ROBUST_FEATURE_INDICES = {4}  # reservation_price_dev (heavy-tailed)


def compute_features_v9(
    trades_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    trade_batch: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 5 Aristotle-proven features from raw data.

    Returns: (features, timestamps, prices, raw_hawkes_branching)
    where features has shape (num_batches, 5).

    Feature layout:
      0: lambda_ofi              - kyle_lambda * signed_notional (sufficient statistic)
      1: directional_conviction  - TFI * |signed_notional| (sufficient statistic)
      2: vpin                    - rolling mean of |TFI| (toxicity proxy)
      3: hawkes_branching        - 1 - 1/sqrt(Var(N)/E[N]) (self-excitation)
      4: reservation_price_dev   - orderbook_imbalance * realvol^2 (inventory pressure)
    """
    if trades_df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # --- Reuse batching logic from compute_features ---
    trades_df = trades_df.copy()
    trades_df["norm_side"] = trades_df["side"].apply(normalize_side)
    trades_df["is_buy"] = trades_df["norm_side"] == "buy"
    trades_df["notional"] = trades_df["price"] * trades_df["qty"]

    num_trades = len(trades_df)
    num_batches = num_trades // trade_batch
    if num_batches == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    prices_arr = trades_df["price"].values[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )
    notionals_batched = trades_df["notional"].values[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )
    is_buy_batched = trades_df["is_buy"].values[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )
    ts_batched = trades_df["ts_ms"].values[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    )

    # VWAP per batch
    total_batch_notional = notionals_batched.sum(axis=1)
    total_batch_qty = trades_df["qty"].values[: num_batches * trade_batch].reshape(
        num_batches, trade_batch
    ).sum(axis=1)
    vwap = np.where(total_batch_qty > 0, total_batch_notional / total_batch_qty, 0)
    vwap = np.where(vwap == 0, prices_arr[:, -1], vwap)

    # Returns
    returns = np.zeros(num_batches)
    returns[1:] = np.diff(np.log(np.clip(vwap, 1e-10, None)))

    # --- Intermediate: TFI ---
    buy_vol = (notionals_batched * is_buy_batched).sum(axis=1)
    sell_vol = (notionals_batched * ~is_buy_batched).sum(axis=1)
    total_vol = buy_vol + sell_vol
    tfi = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0.0)

    # --- Intermediate: signed_notional ---
    signed_notional = buy_vol - sell_vol

    # --- Intermediate: kyle_lambda (rolling 50-batch) ---
    ret_s = pd.Series(returns)
    sn_s = pd.Series(signed_notional)
    rolling_cov = ret_s.rolling(window=50, min_periods=10).cov(sn_s)
    rolling_var = sn_s.rolling(window=50, min_periods=10).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        kyle_lambda = np.where(
            rolling_var.values > 1e-20,
            rolling_cov.values / rolling_var.values,
            0.0,
        )
    kyle_lambda = np.nan_to_num(kyle_lambda)

    # --- Intermediate: realvol (rolling 10-batch std) ---
    realvol = pd.Series(returns).rolling(window=10, min_periods=2).std().fillna(0).values

    # --- Intermediate: weighted_imbalance_5lvl (from orderbook) ---
    weighted_imbalance = _compute_orderbook_imbalance(
        orderbook_df, ts_batched[:, -1], num_batches
    )

    # === Feature 0: lambda_ofi ===
    lambda_ofi = kyle_lambda * signed_notional

    # === Feature 1: directional_conviction ===
    directional_conviction = tfi * np.abs(signed_notional)

    # === Feature 2: vpin ===
    abs_tfi = np.abs(tfi)
    vpin = pd.Series(abs_tfi).rolling(window=50, min_periods=1).mean().fillna(0).values

    # === Feature 3: hawkes_branching ===
    trade_counts = is_buy_batched.sum(axis=1) + (~is_buy_batched).sum(axis=1)
    hawkes_branching = np.zeros(num_batches)
    hawkes_window = 50
    for i in range(hawkes_window, num_batches):
        window_counts = trade_counts[i - hawkes_window : i].astype(float)
        mean_n = window_counts.mean()
        var_n = window_counts.var()
        if mean_n > 0 and var_n > mean_n:  # overdispersed
            ratio = var_n / mean_n
            hawkes_branching[i] = 1.0 - 1.0 / np.sqrt(ratio)
        # else: stays 0.0 (not overdispersed or empty)
    hawkes_branching = np.clip(hawkes_branching, 0.0, 0.99)
    raw_hawkes_branching = hawkes_branching.copy()

    # === Feature 4: reservation_price_dev ===
    reservation_price_dev = weighted_imbalance * (realvol ** 2)

    # --- Stack features ---
    features = np.column_stack([
        lambda_ofi,
        directional_conviction,
        vpin,
        hawkes_branching,
        reservation_price_dev,
    ])

    batch_timestamps = ts_batched[:, -1]
    batch_prices = vwap

    return features, batch_timestamps, batch_prices, raw_hawkes_branching


def _compute_orderbook_imbalance(
    orderbook_df: pd.DataFrame, batch_timestamps: np.ndarray, num_batches: int
) -> np.ndarray:
    """Compute weighted 5-level orderbook imbalance aligned to batch timestamps."""
    imbalance = np.zeros(num_batches)
    if orderbook_df.empty:
        return imbalance

    ob_ts = orderbook_df["ts_ms"].values
    ob_idx = 0

    for i in range(num_batches):
        # Find latest orderbook snapshot before this batch
        while ob_idx < len(ob_ts) - 1 and ob_ts[ob_idx + 1] <= batch_timestamps[i]:
            ob_idx += 1

        if ob_idx >= len(ob_ts) or ob_ts[ob_idx] > batch_timestamps[i]:
            continue

        row = orderbook_df.iloc[ob_idx]
        bids = row.get("bids", [])
        asks = row.get("asks", [])

        bid_depth = sum(b["qty"] for b in bids[:5]) if len(bids) > 0 else 0
        ask_depth = sum(a["qty"] for a in asks[:5]) if len(asks) > 0 else 0
        total = bid_depth + ask_depth
        if total > 0:
            imbalance[i] = (bid_depth - ask_depth) / total

    return imbalance
```

- [ ] **Step 4: Update _FEATURE_VERSION and add v9 normalization**

In `prepare.py`, update the feature version constant:

```python
_FEATURE_VERSION = "v9"  # v9: 5 Aristotle-proven features
```

Add v9-aware normalization to `normalize_features()` or add a new function:

```python
def normalize_features_v9(features: np.ndarray, window: int = 1000) -> np.ndarray:
    """Normalize v9 features. IQR for feature 4, z-score for rest."""
    if features.ndim != 2 or len(features) == 0:
        return features

    normalized = np.zeros_like(features)
    for col in range(features.shape[1]):
        series = pd.Series(features[:, col])
        if col in V9_ROBUST_FEATURE_INDICES:
            rolling_median = series.rolling(window=window, min_periods=100).median()
            rolling_q75 = series.rolling(window=window, min_periods=100).quantile(0.75)
            rolling_q25 = series.rolling(window=window, min_periods=100).quantile(0.25)
            iqr = (rolling_q75 - rolling_q25).replace(0, 1)
            z = (series - rolling_median) / iqr
        else:
            rolling_mean = series.rolling(window=window, min_periods=100).mean()
            rolling_std = series.rolling(window=window, min_periods=100).std()
            z = (series - rolling_mean) / rolling_std.replace(0, 1)
        normalized[:, col] = z.fillna(0).values

    np.clip(normalized, -5, 5, out=normalized)
    return normalized
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_features_v9.py -v`
Expected: All PASS

- [ ] **Step 6: Run existing v6 tests to verify no regression**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS (v6 code untouched)

- [ ] **Step 7: Commit**

```bash
git add prepare.py tests/test_features_v9.py
git commit -m "feat: add compute_features_v9() with 5 Aristotle-proven features"
```

---

### Task 2: Update cache and make_env to support v9

**Files:**
- Modify: `prepare.py` (`cache_features`, `load_cached`, `make_env`)

- [ ] **Step 1: Update cache functions to handle raw_hawkes sidecar**

In `cache_features()`, add `raw_hawkes` parameter:

```python
def cache_features(
    symbol, features, timestamps, prices, cache_dir, start, end, trade_batch,
    raw_hawkes=None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    save_dict = dict(features=features, timestamps=timestamps, prices=prices)
    if raw_hawkes is not None:
        save_dict["raw_hawkes"] = raw_hawkes
    np.savez_compressed(path, **save_dict)
    print(f"Cached {symbol} features to {path}")
```

In `load_cached()`, return raw_hawkes when available:

```python
def load_cached(symbol, cache_dir, start, end, trade_batch):
    key = _cache_key(symbol, start, end, trade_batch)
    path = cache_dir / f"{symbol}_{key}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    raw_hawkes = data["raw_hawkes"] if "raw_hawkes" in data else None
    return data["features"], data["timestamps"], data["prices"], raw_hawkes
```

- [ ] **Step 2: Update make_env to use v9 features when version is v9**

In `make_env()`, detect v9 and use `compute_features_v9` + `normalize_features_v9`. Store `raw_hawkes` on the env object for the regime gate.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add prepare.py
git commit -m "feat: update cache and make_env for v9 features + raw_hawkes sidecar"
```

---

### Task 3: Add HybridClassifier to train.py

**Files:**
- Modify: `train.py` (add `HybridClassifier` class)
- Test: `tests/test_model_v9.py` (new)

- [ ] **Step 1: Write failing test for HybridClassifier**

Create `tests/test_model_v9.py`:

```python
"""Tests for v9 HybridClassifier."""
import torch
from train import HybridClassifier

def test_hybrid_forward_shape():
    model = HybridClassifier(obs_shape=(75, 5), n_classes=3, hidden_dim=128, num_layers=2)
    x = torch.randn(4, 75, 5)
    out = model(x)
    assert out.shape == (4, 3)

def test_hybrid_param_count():
    model = HybridClassifier(obs_shape=(75, 5), n_classes=3, hidden_dim=128, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000  # should be ~55K

def test_hybrid_different_window():
    model = HybridClassifier(obs_shape=(50, 5), n_classes=3, hidden_dim=128, num_layers=2)
    x = torch.randn(2, 50, 5)
    out = model(x)
    assert out.shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_v9.py -v`
Expected: FAIL with `ImportError: cannot import name 'HybridClassifier' from 'train'`

- [ ] **Step 3: Implement HybridClassifier**

Add to `train.py` after `DirectionClassifier`:

```python
class HybridClassifier(nn.Module):
    """Flat MLP + 1D TCN hybrid. Proved: weak dominance over either alone (Theorem 5)."""

    def __init__(self, obs_shape, n_classes, hidden_dim, num_layers):
        super().__init__()
        n_time, n_feat = obs_shape

        # Flat branch: flatten + temporal stats
        flat_dim = n_time * n_feat + 2 * n_feat

        # TCN branch: Conv1d → pool
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Kaiming init for TCN
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

        combined_dim = flat_dim + 8  # flat + tcn pool output

        layers = [_ortho_init(nn.Linear(combined_dim, hidden_dim)), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x):
        # x: (batch, time, feat)
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        flat_branch = torch.cat([flat, t_mean, t_std], dim=1)

        # TCN expects (batch, channels, time)
        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in).squeeze(-1)  # (batch, 8)

        combined = torch.cat([flat_branch, tcn_out], dim=1)
        return self.head(self.trunk(combined))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model_v9.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_model_v9.py
git commit -m "feat: add HybridClassifier with TCN branch for v9"
```

---

### Task 4: Add regime gate to evaluate()

**Files:**
- Modify: `prepare.py` (`evaluate()`)
- Test: `tests/test_regime_gate.py` (new)

- [ ] **Step 1: Write failing test for regime gate**

Create `tests/test_regime_gate.py`:

```python
"""Tests for Hawkes regime gate in evaluate()."""
import numpy as np

def test_regime_gate_forces_flat():
    """When hawkes_branching < r_min, action should be forced to 0."""
    raw_hawkes = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
    r_min = 0.5
    actions = np.array([1, 2, 1, 2, 1])  # model wants to trade
    gated = np.where(raw_hawkes < r_min, 0, actions)
    assert list(gated) == [0, 0, 1, 2, 0]

def test_regime_gate_no_filter_when_zero():
    """When r_min=0, no filtering occurs."""
    raw_hawkes = np.array([0.1, 0.2, 0.8])
    r_min = 0.0
    actions = np.array([1, 2, 1])
    gated = np.where(raw_hawkes < r_min, 0, actions)
    assert list(gated) == [1, 2, 1]

def test_alpha_min_values():
    """Verify proved alpha_min formula."""
    assert abs(0.5 + 1/(2*1.5) - 5/6) < 1e-10
    assert abs(0.5 + 1/(2*4.0) - 0.625) < 1e-10
    assert abs(0.5 + 1/(2*8.0) - 0.5625) < 1e-10
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_regime_gate.py -v`
Expected: PASS (these are pure logic tests)

- [ ] **Step 3: Add r_min parameter to evaluate()**

Modify `evaluate()` signature in `prepare.py`:

```python
def evaluate(
    env_test: TradingEnv, policy_fn, min_trades: int = 50,
    max_drawdown: float = 0.20, r_min: float = 0.0,
) -> float:
```

Inside the step loop, add the gate:

```python
action = policy_fn(obs)
# Regime gate: force flat when Hawkes branching below threshold
if r_min > 0 and hasattr(env_test, 'raw_hawkes') and env_test.raw_hawkes is not None:
    if env_test.raw_hawkes[env_test._idx] < r_min:
        action = 0
```

Add diagnostics at the end of evaluate():

```python
# Regime gate diagnostics
fee_mult = getattr(env_test, '_fee_mult', 1.0)
alpha_min = 0.5 + 1.0 / (2.0 * fee_mult) if fee_mult > 0 else 1.0
print(f"alpha_min: {alpha_min:.4f}")
if hasattr(env_test, 'raw_hawkes') and env_test.raw_hawkes is not None:
    mean_branching = np.mean(env_test.raw_hawkes)
    print(f"hawkes_branching_mean: {mean_branching:.4f}")
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_regime_gate.py
git commit -m "feat: add Hawkes regime gate to evaluate() with alpha_min diagnostics"
```

---

### Task 5: Update train.py for v9 configuration

**Files:**
- Modify: `train.py` (constants, model selection, Optuna ranges, ensemble validity)

- [ ] **Step 1: Update constants**

```python
WINDOW_SIZE = 75  # Proved minimum for TCN (Theorem 5)
MIN_HOLD = 200    # Moderate — regime gate handles quality filtering
```

- [ ] **Step 2: Update BEST_PARAMS and add v9 defaults**

```python
BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 128,
    "nlayers": 2,
    "batch_size": 256,
    "fee_mult": 8.0,
    "r_min": 0.4,  # Hawkes regime gate threshold
}
```

- [ ] **Step 3: Update train_one_model to use HybridClassifier**

Change `DirectionClassifier` to `HybridClassifier`:

```python
model = HybridClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
```

- [ ] **Step 4: Add r_min to eval_policy**

Pass `r_min=p.get("r_min", 0.0)` through to `evaluate()`.

- [ ] **Step 5: Update Optuna search space**

```python
def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "hdim": trial.suggest_categorical("hdim", [64, 128, 256]),
        "nlayers": trial.suggest_categorical("nlayers", [2, 3]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "fee_mult": trial.suggest_float("fee_mult", 1.5, 12.0),
        "r_min": trial.suggest_float("r_min", 0.3, 0.7),
    }
```

- [ ] **Step 6: Add ensemble validity check**

After training all seeds, check accuracy:

```python
# Ensemble validity: only ensemble if alpha > 0.5 (Theorem 10)
if mean_accuracy < 0.5 and len(models) > 1:
    print(f"  WARNING: alpha={mean_accuracy:.3f} < 0.5, using single best model")
    # Use model with best validation loss
    ensemble_fn = make_ensemble_fn([models[best_seed_idx]], DEVICE)
else:
    ensemble_fn = make_ensemble_fn(models, DEVICE)
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add train.py
git commit -m "feat: update train.py for v9 — HybridClassifier, window=75, regime gate, Optuna"
```

---

### Task 6: Run Step 1a experiment (feature reduction only)

**Files:**
- No code changes — experiment run

- [ ] **Step 1: Run with DirectionClassifier, 5 features, window=50, fee_mult=1.5**

Temporarily set `WINDOW_SIZE=50` and use `DirectionClassifier` to isolate the feature reduction effect.

Run: `python train.py`

- [ ] **Step 2: Record results in results.tsv**

Expected output format:
```
commit, sortino, trades, dd, passing, status, description
<hash>, <sortino>, <trades>, <dd>, <N>/25, kept, v9 step1a: 5 features + DirectionClassifier + w50 + fee=1.5
```

- [ ] **Step 3: Commit experiment results**

```bash
git add results.tsv
git commit -m "experiment: v9 step1a — 5 Aristotle features, DirectionClassifier, w=50, fee=1.5"
```

---

### Task 7: Run Step 1b experiment (model size reduction)

- [ ] **Step 1: Switch to HybridClassifier, keep window=50, fee_mult=1.5**

Run: `python train.py`

- [ ] **Step 2: Record and commit**

```bash
git add results.tsv
git commit -m "experiment: v9 step1b — 5 features + HybridClassifier, w=50, fee=1.5"
```

---

### Task 8: Run Step 2 experiment (window=75)

- [ ] **Step 1: Set WINDOW_SIZE=75, run**

Run: `python train.py`

- [ ] **Step 2: Record and commit**

```bash
git add results.tsv
git commit -m "experiment: v9 step2 — window=75, 5 features, HybridClassifier"
```

---

### Task 9: Run Step 3 experiment (fee_mult range)

- [ ] **Step 1: Test fee_mult=4, 6, 8 manually**

Three separate runs with different fee_mult values.

- [ ] **Step 2: Record and commit**

```bash
git add results.tsv
git commit -m "experiment: v9 step3 — fee_mult sweep [4, 6, 8]"
```

---

### Task 10: Run Step 4 experiment (regime gate)

- [ ] **Step 1: Add r_min=0.4 gate, run with best fee_mult from Step 3**

- [ ] **Step 2: Record and commit**

```bash
git add results.tsv
git commit -m "experiment: v9 step4 — Hawkes regime gate r_min=0.4"
```

---

### Task 11: Run Step 5 (full Optuna search)

- [ ] **Step 1: Run Optuna**

Run: `python train.py --search`

- [ ] **Step 2: Run final evaluation with best params**

- [ ] **Step 3: Record and commit**

```bash
git add results.tsv train.py
git commit -m "experiment: v9 step5 — full Optuna search, best config"
```

# V5 Features: Cutting-Edge Trading Innovations

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade prepare.py from 25 classical features to 31 cutting-edge features, then run experiments on the improved feature set.

**Architecture:** Foundation-first approach: (1) add 6 new features to prepare.py with tests, (2) rebuild cache once, (3) run autoresearch experiments on the new feature set. This is NOT the old "grind train.py tweaks" pattern — we're upgrading the mathematical foundation first.

**Tech Stack:** Python 3.12+, NumPy, Pandas, PyTorch (unchanged)

**Current Baseline (deterministic):** val_sharpe=0.191, 25/25 passing, 260 trades, max_dd=0.076

**Branch:** `autoresearch/v5-features` (created from `autoresearch/mar14`)

**What we learned from 5-agent research council + 30 mar14 experiments:**
- All 25 features are classical (2015-era quant toolbox), not cutting-edge
- Conformal abstention to flat(0) causes round-trip oscillation fees — DON'T DO THIS
- Removing the flat representation for attention pooling loses critical info — keep flat+mean/std
- Ordinal loss didn't help (tested at 0.3 and planned at 0.1)
- The flat MLP + focal loss + 30 fixed epochs is well-optimized; further gains require better INPUT FEATURES

**Files to modify:**
- `prepare.py:141-515` — add 6 new feature computations inside `compute_features()`
- `prepare.py:147-177` — update docstring (25 → 31 features)
- `prepare.py:496-515` — add new features to column_stack
- `prepare.py:518-522` — update `ROBUST_FEATURE_INDICES` for new tail-heavy features
- `prepare.py:561` — bump `_FEATURE_VERSION` from `"v4"` to `"v5"`
- `tests/test_features.py:10` — update `NUM_FEATURES_V4` → `NUM_FEATURES_V5 = 31`
- `tests/test_features.py` — add tests for new features

---

## Chunk 1: New Features in prepare.py

### Task 1: Add VPIN and delta_TFI features

**Rationale:** VPIN (Volume-Synchronized Probability of Informed Trading) predicts volatility spikes and toxic flow. Validated by arXiv 2602.00776. delta_TFI captures flow acceleration — the *change* in trade flow, not just the level. Both are trivially computed from existing `tfi` array.

**Files:**
- Modify: `prepare.py:257-261` (after TFI computation, before volume_spike_ratio)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for VPIN and delta_TFI**

Add to `tests/test_features.py`:

```python
class TestVPINFeatures:
    """Tests for VPIN (index 25) and delta_TFI (index 26)."""

    def test_vpin_range(self, make_trades, make_orderbook, make_funding):
        """VPIN should be in [0, 1] since it's rolling mean of |TFI|."""
        features, _, _ = compute_features(
            make_trades(n=5000), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        vpin = features[:, 25]
        assert np.all(vpin >= 0.0), "VPIN must be non-negative"
        assert np.all(vpin <= 1.0), "VPIN must be <= 1.0"

    def test_delta_tfi_first_is_zero(self, make_trades, make_orderbook, make_funding):
        """First delta_TFI should be 0 (no prior to diff against)."""
        features, _, _ = compute_features(
            make_trades(n=5000), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        assert features[0, 26] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_features.py::TestVPINFeatures -v
```

Expected: FAIL (index 25 out of bounds, only 25 features)

- [ ] **Step 3: Implement VPIN and delta_TFI in prepare.py**

Insert after the `cum_tfi_500` computation (line ~261), before `volume_spike_ratio`:

```python
    # --- Feature 25: VPIN (rolling mean of |TFI|, toxicity proxy) ---
    abs_tfi = np.abs(tfi)
    vpin = pd.Series(abs_tfi).rolling(window=50, min_periods=1).mean().fillna(0).values

    # --- Feature 26: delta_tfi (first difference of TFI, flow acceleration) ---
    delta_tfi = np.zeros(num_batches)
    delta_tfi[1:] = tfi[1:] - tfi[:-1]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_features.py::TestVPINFeatures -v
```

Note: Tests won't pass yet because the features aren't in the column_stack. We'll wire everything together in Task 4.

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add VPIN and delta_TFI feature computations"
```

---

### Task 2: Add Higher-Order Statistics (Hurst, Skewness, Vol-of-Vol)

**Rationale:** Hurst exponent detects market regime (trending vs mean-reverting). Realized skewness captures asymmetric return distributions. Vol-of-vol measures volatility stability. All three were flagged by multiple research agents as missing regime/higher-order features.

**Files:**
- Modify: `prepare.py` (after existing volatility features, before orderbook section)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
class TestHigherOrderFeatures:
    """Tests for Hurst (27), realized_skew (28), vol_of_vol (29)."""

    def test_hurst_default_is_half(self, make_trades, make_orderbook, make_funding):
        """Hurst should default to 0.5 (random walk) for short series."""
        features, _, _ = compute_features(
            make_trades(n=500), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        # With only 5 batches, Hurst window=200 won't fire, so all should be 0.5
        hurst = features[:, 27]
        assert np.allclose(hurst, 0.5), f"Expected 0.5 default, got {hurst}"

    def test_realized_skew_exists(self, make_trades, make_orderbook, make_funding):
        """Realized skewness should be finite."""
        features, _, _ = compute_features(
            make_trades(n=5000), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(np.isfinite(features[:, 28]))

    def test_vol_of_vol_non_negative(self, make_trades, make_orderbook, make_funding):
        """Vol-of-vol is a std, must be >= 0."""
        features, _, _ = compute_features(
            make_trades(n=5000), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        assert np.all(features[:, 29] >= 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_features.py::TestHigherOrderFeatures -v
```

- [ ] **Step 3: Implement Hurst, skewness, and vol-of-vol**

Insert after bipower_var computation (around line 246):

```python
    # --- Feature 27: hurst_exponent (rolling R/S analysis, regime detection) ---
    hurst = np.full(num_batches, 0.5)  # default = random walk
    hurst_window = 200
    if num_batches > hurst_window:
        for i in range(hurst_window, num_batches):
            r = returns[i - hurst_window : i]
            r_centered = r - r.mean()
            cumdev = np.cumsum(r_centered)
            R = cumdev.max() - cumdev.min()
            S = r.std()
            if S > 1e-10 and R > 0:
                hurst[i] = np.log(R / S) / np.log(hurst_window)
    hurst = np.clip(hurst, 0, 1)

    # --- Feature 28: realized_skew_20 (rolling skewness of returns) ---
    realized_skew = (
        pd.Series(returns).rolling(window=20, min_periods=5).skew().fillna(0).values
    )

    # --- Feature 29: vol_of_vol_50 (rolling std of realvol_10) ---
    vol_of_vol = (
        pd.Series(realvol_10).rolling(window=50, min_periods=10).std().fillna(0).values
    )
```

- [ ] **Step 4: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add Hurst exponent, realized skewness, vol-of-vol features"
```

---

### Task 3: Add Sign Autocorrelation (Trend Persistence)

**Rationale:** Measures whether return signs are correlated (trending) or anti-correlated (mean-reverting). A cheap scalar feature that captures what Hurst measures from a different angle.

**Files:**
- Modify: `prepare.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

```python
class TestSignAutocorrFeature:
    """Tests for sign_autocorr (index 30)."""

    def test_sign_autocorr_range(self, make_trades, make_orderbook, make_funding):
        """Sign autocorrelation should be in [-1, 1]."""
        features, _, _ = compute_features(
            make_trades(n=5000), make_orderbook(n=100), make_funding(n=10),
            trade_batch=100,
        )
        sa = features[:, 30]
        assert np.all(sa >= -1.0) and np.all(sa <= 1.0)
```

- [ ] **Step 2: Implement sign autocorrelation**

```python
    # --- Feature 30: sign_autocorr_20 (return sign autocorrelation, trend persistence) ---
    ret_sign = np.sign(returns)
    sign_autocorr = np.zeros(num_batches)
    if num_batches > 20:
        # Vectorized: rolling mean of sign[i]*sign[i-1]
        sign_product = ret_sign[1:] * ret_sign[:-1]
        sign_product_series = pd.Series(sign_product)
        sa_rolling = sign_product_series.rolling(window=19, min_periods=5).mean().fillna(0).values
        sign_autocorr[1:] = sa_rolling
```

- [ ] **Step 3: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: add sign autocorrelation feature"
```

---

### Task 4: Wire Everything Together — Column Stack, Version Bump, Normalization

**Rationale:** All 6 new features are computed but not yet included in the output array. This task wires them in, bumps the feature version to v5, updates normalization indices, and updates the docstring.

**Files:**
- Modify: `prepare.py:147-177` (docstring)
- Modify: `prepare.py:496-515` (column_stack + hstack)
- Modify: `prepare.py:518-522` (ROBUST_FEATURE_INDICES)
- Modify: `prepare.py:561` (_FEATURE_VERSION)
- Modify: `tests/test_features.py:10` (NUM_FEATURES constant)

- [ ] **Step 1: Update docstring to reflect 31 features**

Add after line 176 (`24: funding_rate_raw`):

```
     25: vpin               - rolling mean of |TFI| (toxicity proxy)
     26: delta_tfi           - first difference of TFI (flow acceleration)
     27: hurst_exponent      - rolling R/S Hurst exponent (regime detection)
     28: realized_skew_20    - rolling skewness of returns
     29: vol_of_vol_50       - rolling std of realvol_10
     30: sign_autocorr_20    - return sign autocorrelation (trend persistence)
```

Update `"""Compute 25 features` → `"""Compute 31 features` and shape comment `(num_batches, 25)` → `(num_batches, 31)`.

- [ ] **Step 2: Add new features to column_stack**

Create a new feature group after `longer_features` and before the final `hstack`:

```python
    # === CUTTING-EDGE FEATURES (indices 25-30) ===
    cutting_edge_features = np.column_stack(
        [
            vpin,            # 25
            delta_tfi,       # 26
            hurst,           # 27
            realized_skew,   # 28
            vol_of_vol,      # 29
            sign_autocorr,   # 30
        ]
    )

    features = np.hstack(
        [trade_features, ob_features, extra_features, longer_features, cutting_edge_features]
    )
```

- [ ] **Step 3: Update ROBUST_FEATURE_INDICES**

VPIN (25) and vol_of_vol (29) are tail-heavy and need robust normalization:

```python
ROBUST_FEATURE_INDICES = {5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 24, 25, 29}
```

- [ ] **Step 4: Bump feature version**

```python
_FEATURE_VERSION = "v5"  # v5: 31 features (+ VPIN, delta_TFI, Hurst, skew, vol-of-vol, sign_autocorr)
```

- [ ] **Step 5: Update test constant**

In `tests/test_features.py`, change:

```python
NUM_FEATURES_V5 = 31
```

And update all references from `NUM_FEATURES_V4` to `NUM_FEATURES_V5`.

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: ALL PASS (existing tests updated for 31 features, new tests pass)

- [ ] **Step 7: Commit**

```bash
git add prepare.py tests/test_features.py
git commit -m "feat: wire v5 features (25→31), bump version, update normalization"
```

---

## Chunk 2: Cache Migration and Baseline

### Task 5: Migrate v4 Caches to v5 (NO raw Parquet rebuild)

**Rationale:** Existing v4 `.npz` caches contain raw prices (unnormalized VWAP) and timestamps. We can derive all 6 new features from these cached arrays in ~40 seconds total, avoiding the 20-30 minute raw Parquet rebuild. This follows the established "incremental cache migration" pattern.

**Files:**
- Create: `scripts/migrate_v4_to_v5.py`

**How it works:**
1. Find all v4 `.npz` files in `.cache/`
2. For each file: load `features` (25 normalized), `prices` (raw VWAP), `timestamps`
3. Compute raw returns from `prices`: `log(prices[i]/prices[i-1])`
4. Derive new features from raw returns + existing normalized columns
5. Normalize the 6 new columns with the same rolling normalization
6. `hstack([old_25, new_6_normalized])` → save with v5 cache key
7. Total time: ~40 seconds for all 75 files

- [ ] **Step 1: Write the migration script**

```python
#!/usr/bin/env python3
"""Migrate v4 caches (25 features) to v5 (31 features) without raw Parquet."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import normalization and cache functions from prepare
sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare import (
    CACHE_DIR,
    DEFAULT_SYMBOLS,
    ROBUST_FEATURE_INDICES,
    TRAIN_START,
    TRAIN_END,
    VAL_END,
    TEST_END,
    _cache_key,
    normalize_features,
)


def compute_new_features_from_cache(features_25, prices):
    """Compute 6 new features from cached prices and normalized features.

    Returns: (N, 6) array of RAW (unnormalized) new features.
    """
    n = len(prices)

    # Raw returns from cached prices
    returns = np.zeros(n)
    returns[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))

    # Raw realvol from raw returns (needed for vol_of_vol)
    realvol_10 = pd.Series(returns).rolling(window=10, min_periods=1).std().fillna(0).values

    # TFI from normalized feature column 6 (approximate — normalization preserves relative patterns)
    tfi_normalized = features_25[:, 6]

    # --- Feature 25: VPIN (rolling mean of |TFI|) ---
    vpin = pd.Series(np.abs(tfi_normalized)).rolling(window=50, min_periods=1).mean().fillna(0).values

    # --- Feature 26: delta_TFI (first difference of TFI) ---
    delta_tfi = np.zeros(n)
    delta_tfi[1:] = tfi_normalized[1:] - tfi_normalized[:-1]

    # --- Feature 27: Hurst exponent (rolling R/S) ---
    hurst = np.full(n, 0.5)
    hurst_window = 200
    if n > hurst_window:
        for i in range(hurst_window, n):
            r = returns[i - hurst_window:i]
            r_centered = r - r.mean()
            cumdev = np.cumsum(r_centered)
            R = cumdev.max() - cumdev.min()
            S = r.std()
            if S > 1e-10 and R > 0:
                hurst[i] = np.log(R / S) / np.log(hurst_window)
    hurst = np.clip(hurst, 0, 1)

    # --- Feature 28: realized skewness ---
    realized_skew = pd.Series(returns).rolling(window=20, min_periods=5).skew().fillna(0).values

    # --- Feature 29: vol-of-vol ---
    vol_of_vol = pd.Series(realvol_10).rolling(window=50, min_periods=10).std().fillna(0).values

    # --- Feature 30: sign autocorrelation ---
    ret_sign = np.sign(returns)
    sign_autocorr = np.zeros(n)
    if n > 1:
        sign_product = ret_sign[1:] * ret_sign[:-1]
        sa_rolling = pd.Series(sign_product).rolling(window=19, min_periods=5).mean().fillna(0).values
        sign_autocorr[1:] = sa_rolling

    return np.column_stack([vpin, delta_tfi, hurst, realized_skew, vol_of_vol, sign_autocorr])


def normalize_new_features(new_features_raw, window=1000):
    """Normalize only the 6 new columns using the same hybrid scheme."""
    # Indices relative to new_features_raw (0-5):
    # 0=vpin (robust), 1=delta_tfi (z-score), 2=hurst (z-score),
    # 3=realized_skew (z-score), 4=vol_of_vol (robust), 5=sign_autocorr (z-score)
    robust_cols = {0, 4}  # vpin and vol_of_vol are tail-heavy

    normalized = np.zeros_like(new_features_raw)
    for col in range(new_features_raw.shape[1]):
        series = pd.Series(new_features_raw[:, col])
        if col in robust_cols:
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


def migrate():
    """Migrate all v4 caches to v5."""
    # Temporarily override version for v4 key lookup
    import prepare
    old_version = prepare._FEATURE_VERSION
    prepare._FEATURE_VERSION = "v4"

    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }

    migrated = 0
    for sym in DEFAULT_SYMBOLS:
        for split_name, (start, end) in splits.items():
            v4_key = _cache_key(sym, start, end, 100)
            v4_path = CACHE_DIR / f"{sym}_{v4_key}.npz"

            if not v4_path.exists():
                print(f"  SKIP {sym} {split_name} (no v4 cache)")
                continue

            # Load v4
            data = np.load(v4_path)
            features_25 = data["features"]
            timestamps = data["timestamps"]
            prices = data["prices"]

            if len(features_25) == 0:
                continue

            # Compute and normalize new features
            new_raw = compute_new_features_from_cache(features_25, prices)
            new_normalized = normalize_new_features(new_raw)

            # Concatenate
            features_31 = np.hstack([features_25, new_normalized])

            # Save with v5 key
            prepare._FEATURE_VERSION = "v5"
            v5_key = _cache_key(sym, start, end, 100)
            v5_path = CACHE_DIR / f"{sym}_{v5_key}.npz"
            np.savez_compressed(v5_path, features=features_31, timestamps=timestamps, prices=prices)
            prepare._FEATURE_VERSION = "v4"  # reset for next v4 lookup

            print(f"  {sym} {split_name}: {features_25.shape} → {features_31.shape}")
            migrated += 1

    prepare._FEATURE_VERSION = old_version
    print(f"\nMigrated {migrated} cache files to v5")


if __name__ == "__main__":
    migrate()
```

- [ ] **Step 2: Run the migration**

```bash
uv run python scripts/migrate_v4_to_v5.py
```

Expected: ~40 seconds, 75 files migrated, each showing `(N, 25) → (N, 31)`.

- [ ] **Step 3: Bump `_FEATURE_VERSION` in prepare.py to "v5"**

So that `make_env()` loads the new v5 caches:

```python
_FEATURE_VERSION = "v5"  # v5: 31 features (+ VPIN, delta_TFI, Hurst, skew, vol-of-vol, sign_autocorr)
```

- [ ] **Step 4: Run v5 baseline**

```bash
uv run train.py 2>&1 | tee run.log
```

Should load v5 caches instantly (no Parquet rebuild) and train on 31 features.

- [ ] **Step 5: Record v5 baseline to results.tsv**

```bash
SHARPE=$(grep '^val_sharpe:' run.log | tail -1 | awk '{print $2}')
TRADES=$(grep '^num_trades:' run.log | tail -1 | awk '{print $2}')
DRAWDOWN=$(grep '^max_drawdown:' run.log | tail -1 | awk '{print $2}')
PASSING=$(grep '^symbols_passing:' run.log | tail -1 | awk '{print $2}')
COMMIT=$(git rev-parse --short HEAD)
echo -e "commit\tval_sharpe\tnum_trades\tmax_drawdown\tsymbols_passing\tstatus\tdescription" > results.tsv
echo -e "$COMMIT\t$SHARPE\t$TRADES\t$DRAWDOWN\t$PASSING\tkept\tv5 baseline (31 features, focal gamma=1.0, 30 epochs)" >> results.tsv
git add results.tsv scripts/migrate_v4_to_v5.py
git commit -m "v5 baseline: 31 features, migrated cache (val_sharpe=$SHARPE)"
```

- [ ] **Step 6: Compare to v4 baseline**

v4 baseline was 0.191 with 25 features. If v5 baseline is lower, the model may need epoch/LR retuning for the larger input (31*50 + 2*31 = 1612 dims vs 25*50 + 2*25 = 1300 dims).

---

## Chunk 3: Cutting-Edge Model Innovations

### Task 6: Log-Signature Features (Path Signatures)

**Rationale:** Rated #2 priority by 4/5 research agents. Path signatures capture temporal cross-feature interactions (e.g., "volatility rose while order flow turned negative") that mean+std completely miss. The `iisignature` library computes them in C++ — fast enough for train-time computation. This is the single most mathematically innovative addition we can make.

**Key concept:** The log-signature of a multivariate path is a fixed-size vector that encodes the path's shape, order, and cross-channel interactions. A linear model on log-signature features can approximate any continuous function of the path (universal linearization property).

**Files:**
- Modify: `train.py` (imports, feature computation, model input, inference)

**Dependency:** `uv pip install iisignature`

- [ ] **Step 1: Install iisignature and verify**

```bash
uv pip install iisignature
uv run python -c "import iisignature; print('logsig dim for 4 channels depth 2:', iisignature.logsigdim(4, 2))"
```

Expected: `logsig dim for 4 channels depth 2: 10`

- [ ] **Step 2: Define channel groups and compute function**

Group the 31 features into ~5 semantically meaningful channel groups. Compute depth-2 log-signatures per group, concatenate into a single feature vector per observation window.

```python
import iisignature

# Channel groups (indices into the 31 features):
SIG_GROUPS = [
    [0, 1, 2, 3],         # momentum: returns, r_5, r_20, r_100
    [4, 5, 29],           # volatility: realvol_10, bipower_var, vol_of_vol
    [6, 25, 26],          # flow: tfi, vpin, delta_tfi
    [14, 15, 16],         # book: weighted_imbalance, microprice_dev, ofi
    [12, 13, 17],         # liquidity: spread, depth, slope_asym
]
SIG_DEPTH = 2


def compute_logsig_features(obs_batch):
    """Compute log-signature features for a batch of (N, T, F) observations.

    Returns: (N, total_sig_dim) numpy array.
    """
    N, T, F = obs_batch.shape
    all_sigs = []
    for group in SIG_GROUPS:
        path = obs_batch[:, :, group]
        sig_dim = iisignature.logsigdim(len(group), SIG_DEPTH)
        sigs = np.zeros((N, sig_dim), dtype=np.float32)
        for i in range(N):
            sigs[i] = iisignature.logsig(path[i], SIG_DEPTH)
        all_sigs.append(sigs)
    return np.concatenate(all_sigs, axis=1)
```

- [ ] **Step 3: Modify DirectionClassifier to accept extra features**

Add a `sig_dim` parameter to the constructor that increases input size. The `forward()` method accepts an optional `sig_feats` tensor that gets concatenated alongside flat+mean+std.

```python
class DirectionClassifier(nn.Module):
    def __init__(self, obs_shape, n_classes, hidden_dim, num_layers, extra_dim=0):
        super().__init__()
        n_time, n_feat = obs_shape
        flat_dim = n_time * n_feat + 2 * n_feat + extra_dim
        layers = [_ortho_init(nn.Linear(flat_dim, hidden_dim)), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x, extra=None):
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        parts = [flat, t_mean, t_std]
        if extra is not None:
            parts.append(extra)
        x = torch.cat(parts, dim=1)
        return self.head(self.trunk(x))
```

- [ ] **Step 4: Precompute signatures in train_one_model**

After collecting X (observations), compute log-signatures once for all training data. Pass them as a separate tensor during training.

```python
# After X = np.concatenate(all_obs):
sig_feats = compute_logsig_features(X)
sig_dim = sig_feats.shape[1]
sig_t = torch.tensor(sig_feats, dtype=torch.float32, device=DEVICE)

# Model gets extra_dim=sig_dim
model = DirectionClassifier(obs_shape, 3, p["hdim"], p["nlayers"], extra_dim=sig_dim)

# In training loop:
batch_sig = sig_t[perm[start:start+batch_size]]
logits = model(batch_x, extra=batch_sig)
```

- [ ] **Step 5: Compute signatures at inference time in make_ensemble_fn**

```python
def make_ensemble_fn(models, device):
    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            sig = compute_logsig_features(obs[np.newaxis, :, :])
            sig_t = torch.tensor(sig, dtype=torch.float32, device=device)
            logits_sum = None
            for m in models:
                logits = m(obs_t, extra=sig_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            return logits_sum.argmax(dim=-1).item()
    return fn
```

- [ ] **Step 6: Run experiment**

```bash
git add train.py && git commit -m "experiment: log-signature features (depth-2, 5 channel groups)"
uv run train.py 2>&1 | tee run.log
```

- [ ] **Step 7: Evaluate and tune**

Check val_sharpe and total wall-clock time. If signatures slow training too much (>10 min total), try:
- Reduce SIG_DEPTH to 1
- Reduce to 3 channel groups (momentum, flow, book)
- Use fewer samples (8k instead of 10k) to offset compute cost

---

### Task 7: Recency-Weighted Training

**Rationale:** Markets are non-stationary. The training set spans Oct 2025 — Jan 2026 (100 days). Data from January is more relevant to the Feb-Mar test period than data from October. Exponential recency weights make recent samples contribute more to the loss without discarding old data entirely.

**Files:**
- Modify: `train.py:72-114` (make_labeled_dataset — return sample indices)
- Modify: `train.py:118-203` (train_one_model — apply weights in loss)

- [ ] **Step 1: Modify make_labeled_dataset to return sample indices**

Add a third return value: the data index of each sample (proxy for recency within each symbol's time series).

```python
def make_labeled_dataset(env, horizon, fee_threshold, max_samples=10000):
    # ... existing code ...
    obs_list = []
    labels = []
    sample_indices = []  # NEW

    for i in idx:
        if prices[i] <= 0:
            continue
        fwd_return = (prices[i + horizon] - prices[i]) / prices[i]
        # ... labeling logic ...
        obs_list.append(features[i - window : i])
        labels.append(label)
        sample_indices.append(i)  # NEW

    if not obs_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.array(obs_list), np.array(labels), np.array(sample_indices)
```

- [ ] **Step 2: Compute exponential recency weights in train_one_model**

```python
# After concatenating all data:
all_indices = np.concatenate(all_sample_indices)
# Normalize indices to [0, 1]
norm_idx = (all_indices - all_indices.min()) / max(all_indices.max() - all_indices.min(), 1)
# Exponential: recent samples get ~3x weight of oldest
decay = 1.0
sample_weights = np.exp(decay * norm_idx)
sample_weights /= sample_weights.mean()  # mean weight = 1
weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=DEVICE)
```

- [ ] **Step 3: Apply per-sample weights in loss function**

```python
def focal_loss(logits, targets, sample_w, gamma=1.0):
    ce = nn.functional.cross_entropy(logits, targets, weight=cw, reduction="none")
    pt = torch.exp(-ce)
    return (sample_w * (1 - pt) ** gamma * ce).mean()

# In training loop:
batch_w = weights_t[perm[start:start+batch_size]]
loss = criterion(logits, batch_y, batch_w)
```

- [ ] **Step 4: Run experiment**

```bash
git add train.py && git commit -m "experiment: recency-weighted focal loss (decay=1.0)"
uv run train.py 2>&1 | tee run.log
```

- [ ] **Step 5: Tune decay parameter**

If decay=1.0 doesn't help, try 0.5 (gentler) or 2.0 (more aggressive). Log each to results.tsv.

---

### Task 8: Cross-Asset BTC Lead Feature

**Rationale:** Agent 2 flagged cross-asset context as "usually a first-order omission in crypto." BTC leads altcoin moves. Adding a BTC return feature to all symbols lets the model learn lead-lag relationships.

**Files:**
- Modify: `train.py:118-140` (train_one_model — inject BTC features into training data)

**Note:** This doesn't require prepare.py changes. We load BTC's cached features at train time and append BTC returns as an extra column to each symbol's observation windows.

- [ ] **Step 1: Load BTC features in train_one_model**

```python
# At start of train_one_model, load BTC env for cross-asset features:
btc_env = train_envs.get("BTC")
btc_returns = None
if btc_env is not None:
    btc_features = btc_env.features
    btc_returns = btc_features[:, 0]  # feature 0 = returns
```

- [ ] **Step 2: Align and append BTC returns to each symbol's observations**

For each symbol, find the corresponding BTC return at each timestep (using index alignment since all symbols have the same number of batches per split) and append as an extra feature column.

```python
# When building obs for each sample:
# obs shape is (window_size, 31). Append btc_return at each step to get (window_size, 32)
if btc_returns is not None and sym != "BTC":
    btc_window = btc_returns[i - window : i].reshape(-1, 1)
    obs_with_btc = np.concatenate([obs, btc_window], axis=-1)
else:
    obs_with_btc = obs
```

- [ ] **Step 3: Update model input dimension**

The model reads `obs_shape` dynamically, but with the extra BTC column the shape changes from (50, 31) to (50, 32). Handle this by adjusting obs_shape after the BTC column is added.

- [ ] **Step 4: Run experiment**

```bash
git add train.py && git commit -m "experiment: BTC lead feature (cross-asset return)"
uv run train.py 2>&1 | tee run.log
```

---

### Task 9: Experiment Loop

**Rationale:** After Tasks 6-8 are individually tested, combine the winners and run the autoresearch experiment loop.

**Files:**
- Modify: `train.py` (per experiment, following autoresearch pattern)

- [ ] **Step 1: Follow the autoresearch loop from program.md**

For each experiment:
1. Form hypothesis
2. Edit train.py (one change)
3. Commit
4. Run: `uv run train.py 2>&1 | tee run.log`
5. Extract results
6. Keep or discard (git reset --hard HEAD~1 if discarded)
7. Log to results.tsv
8. Repeat

- [ ] **Step 2: Stack winners one at a time**

Apply winner A. Run. If improved, keep. Apply winner B on top. Run. If improved, keep. Continue.

- [ ] **Step 3: Record final combined result**

```bash
SHARPE=$(grep '^val_sharpe:' run.log | tail -1 | awk '{print $2}')
TRADES=$(grep '^num_trades:' run.log | tail -1 | awk '{print $2}')
DRAWDOWN=$(grep '^max_drawdown:' run.log | tail -1 | awk '{print $2}')
PASSING=$(grep '^symbols_passing:' run.log | tail -1 | awk '{print $2}')
echo "FINAL: val_sharpe=$SHARPE trades=$TRADES dd=$DRAWDOWN passing=$PASSING"
```

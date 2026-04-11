# Realism Improvements: Funding Costs, Conditional Slippage, Correlation, Position Sizing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four layers of realism to the backtesting model: (1) funding rate costs in P&L, (2) conditional slippage that accounts for spread widening during signals, (3) correlated drawdown analysis across the portfolio, and (4) position sizing under capital constraints. Each layer is backed by an Aristotle proof (T42-T45) and informed by external research.

**Architecture:** Modify `prepare.py` (TradingEnv.step, compute_features_v9) and `train.py` (evaluate, portfolio construction) to incorporate funding costs and conditional slippage. Add analysis scripts for correlation and position sizing. Research tasks run in parallel and feed into the proofs.

**Tech Stack:** Python 3.12+, NumPy, Pandas, existing Pacifica data (trades/orderbook/funding)

---

## File Structure

| File | Changes | Responsibility |
|------|---------|----------------|
| `prepare.py:726-962` | Extend `compute_features_v9()` | Return `funding_rates` alongside features |
| `prepare.py:1118-1160` | Update cache/load | Store `funding_rates` in .npz |
| `prepare.py:1360-1412` | `TradingEnv.step()` | Charge funding cost per step when holding |
| `prepare.py:1620-1690` | `make_env()` | Pass `funding_rates` to env |
| `scripts/analyze_funding.py` | Create | Analyze funding rate distributions from our data |
| `scripts/analyze_correlation.py` | Create | Compute correlation matrix and stress correlation |
| `scripts/analyze_conditional_spread.py` | Create | Measure E[spread|signal] vs E[spread] |
| `tests/test_funding_cost.py` | Create | Tests for funding cost in TradingEnv |
| `docs/superpowers/specs/2026-03-27-aristotle-T42-T45.md` | Create | Formal proofs |

---

## Research Tasks (parallel, run first)

These 4 research tasks can run simultaneously. Each produces findings that inform the corresponding proof and implementation.

### Research A: Funding Rate Distributions on DEX Perps

**Tool:** `/exa-research` with deep research pro

**Query:** Funding rate distributions on Solana DEX perps (Drift, Jupiter, Pacifica). Typical range in bps/hour, positive vs negative frequency, distribution shape, extreme values during stress, hourly vs 8-hour comparison, predictability.

**Also:** Run `scripts/analyze_funding.py` on our own data to get empirical distributions for all 25 symbols.

**Output needed for T42:** Median funding rate per symbol, % of time funding is against longs, worst-case funding during stress.

---

### Research B: Conditional Spread Modeling in Crypto

**Tool:** `/web-search-advanced-research-paper` (2025-2026 papers)

**Query:** Academic papers on spread behavior during high-activity periods in crypto markets. Almgren-Chriss calibration for crypto. Relationship between trade arrival rate and spread widening on DEX orderbooks. Conditional spread estimation methods.

**Output needed for T43:** Academic formula or empirical multiplier for E[spread|high activity] / E[spread].

---

### Research C: Correlation Structure of Crypto Perps During Stress

**Tool:** `/exa-research` with deep research pro

**Query:** Pairwise correlation of crypto perps during normal vs stress. "Correlation goes to 1" in crypto. Effective number of independent bets with 20-25 crypto symbols. Diversification illusion.

**Also:** Run `scripts/analyze_correlation.py` on our own data.

**Output needed for T44:** Average ρ normal, average ρ stress, effective independent count.

---

### Research D: Execution Latency on Solana DEX Perps

**Tool:** `/exa-research` (balanced model)

**Query:** Actual fill latency on Solana DEX perps. Block time vs execution time. MEV on Solana. Priority fees. Professional MM latency benchmarks.

**Output needed:** Whether 400ms assumption is correct. Whether we need to model latency as a cost.

---

## Task 1: Analyze Our Own Funding Data (T42 prep)

**Files:**
- Create: `scripts/analyze_funding.py`

- [ ] **Step 1: Write funding analysis script**

```python
#!/usr/bin/env python3
"""Analyze funding rate distributions from our Pacifica data."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, load_funding, TRAIN_START, TEST_END

def main():
    all_rates = []
    for sym in DEFAULT_SYMBOLS:
        df = load_funding(sym, TRAIN_START, TEST_END)
        if df.empty:
            continue
        rates = df["rate"].values
        nonzero = rates[rates != 0]
        if len(nonzero) == 0:
            continue
        median_rate = np.median(nonzero)
        pct_positive = (nonzero > 0).mean() * 100
        pct_negative = (nonzero < 0).mean() * 100
        p5 = np.percentile(nonzero, 5)
        p95 = np.percentile(nonzero, 95)
        # Convert to bps (rate is usually a fraction like 0.0001 = 1 bps)
        median_bps = median_rate * 10000
        p5_bps = p5 * 10000
        p95_bps = p95 * 10000
        print(f"{sym:<12} median={median_bps:>7.2f}bps  pos={pct_positive:>5.1f}%  neg={pct_negative:>5.1f}%  "
              f"p5={p5_bps:>7.2f}bps  p95={p95_bps:>7.2f}bps  n={len(nonzero)}")
        all_rates.extend(nonzero.tolist())

    all_rates = np.array(all_rates)
    print(f"\nAGGREGATE: median={np.median(all_rates)*10000:.2f}bps  "
          f"mean={np.mean(all_rates)*10000:.2f}bps  "
          f"pos={( all_rates > 0).mean()*100:.1f}%  "
          f"p5={np.percentile(all_rates, 5)*10000:.2f}bps  "
          f"p95={np.percentile(all_rates, 95)*10000:.2f}bps")

    # Estimate hourly cost for a long position
    # Pacifica funding is hourly. If median funding = X bps/hour, a 1-hour hold costs X bps.
    median_hourly_bps = np.median(all_rates) * 10000
    print(f"\nEstimated funding cost for LONG position:")
    print(f"  Per hour: {median_hourly_bps:.2f} bps")
    print(f"  Per trade (15 min hold): {median_hourly_bps * 0.25:.2f} bps")
    print(f"  Per trade (1 hour hold): {median_hourly_bps:.2f} bps")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `uv run python scripts/analyze_funding.py`
Expected: Per-symbol funding rate distributions with bps values.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_funding.py
git commit -m "analysis: funding rate distributions from Pacifica data (T42 prep)"
```

---

## Task 2: T42 — Prove Funding Rate P&L Impact + Implement

**Files:**
- Create: `tests/test_funding_cost.py`
- Modify: `prepare.py:726-962` (return funding_rates from compute_features_v9)
- Modify: `prepare.py:1118-1160` (cache funding_rates)
- Modify: `prepare.py:1360-1412` (charge funding in TradingEnv.step)
- Modify: `prepare.py:1620-1690` (pass funding_rates through make_env)
- Create: `docs/superpowers/specs/2026-03-27-aristotle-T42-T45.md` (T42 section)

- [ ] **Step 1: Write T42 proof**

**Claim:** For a position with direction d ∈ {+1, -1}, funding rate r_f per period, held for T periods, the funding cost is:
```
funding_pnl = -d × r_f × T
```
Longs (d=+1) pay when r_f > 0. Shorts (d=-1) pay when r_f < 0.

**Condition for funding drag to exceed alpha:** If per-trade alpha (expected excess return) is α and average hold is T̄ steps with H funding settlements per step, then funding drag exceeds alpha when:
```
|E[r_f]| × T̄ × H > α
```

- [ ] **Step 2: Write failing test for funding cost**

```python
# tests/test_funding_cost.py
"""Tests for funding rate cost in TradingEnv."""
import numpy as np
from prepare import TradingEnv

def test_funding_charged_when_holding_long():
    """Long position should pay positive funding rate."""
    n = 100
    features = np.random.randn(n, 13).astype(np.float32)
    prices = np.full(n, 100.0)  # flat price — no P&L from price movement
    env = TradingEnv(features, prices, window_size=10, fee_bps=0, min_hold=1)
    env.spread_bps = None  # no slippage
    # Positive funding rate: 10 bps per period
    env.funding_rates = np.full(n, 0.001)  # 0.1% = 10 bps

    env.reset(seed=42, options={"sequential": True})
    # Enter long
    obs, reward, done, trunc, info = env.step(1)
    # Hold for 5 more steps — should accumulate funding cost
    total_pnl = reward
    for _ in range(5):
        obs, reward, done, trunc, info = env.step(1)
        total_pnl += reward
    # Long pays positive funding: should be negative P&L
    assert total_pnl < 0, f"Long should pay positive funding, got {total_pnl}"

def test_funding_not_charged_when_flat():
    """Flat position should not pay funding."""
    n = 100
    features = np.random.randn(n, 13).astype(np.float32)
    prices = np.full(n, 100.0)
    env = TradingEnv(features, prices, window_size=10, fee_bps=0, min_hold=1)
    env.spread_bps = None
    env.funding_rates = np.full(n, 0.001)

    env.reset(seed=42, options={"sequential": True})
    # Stay flat for 5 steps
    total_pnl = 0
    for _ in range(5):
        obs, reward, done, trunc, info = env.step(0)
        total_pnl += reward
    assert total_pnl == 0.0, f"Flat should not pay funding, got {total_pnl}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_funding_cost.py -v`
Expected: FAIL (funding_rates attribute not used yet)

- [ ] **Step 4: Extract funding_rates from compute_features_v9**

In `prepare.py`, modify `compute_features_v9()` to also return the raw funding rate per step:

After the existing funding handling (around line 870 area), add forward-filled funding rate extraction:
```python
    # Extract raw funding rate per step for execution cost modeling
    funding_rates = np.zeros(num_batches)
    if not funding_df.empty:
        fund_ts = funding_df["ts_ms"].values
        fund_rate = funding_df["rate"].values
        indices = np.searchsorted(fund_ts, batch_timestamps_final, side="right") - 1
        valid = indices >= 0
        funding_rates[valid] = fund_rate[indices[valid]]
```

Update return to include 6th value:
```python
    return features, batch_timestamps_final, batch_prices, raw_hawkes_branching, spread_bps_arr, funding_rates
```

Update all callers (cache, load, make_env) to handle the 6th value.

- [ ] **Step 5: Add funding cost to TradingEnv.step()**

In `TradingEnv.__init__`, add:
```python
self.funding_rates = None  # set externally for funding cost modeling
```

In `TradingEnv.step()`, after the position P&L calculation and before the position change block:
```python
        # Funding cost: charged every step when holding a position
        if self._position != 0 and self.funding_rates is not None:
            if self._idx < len(self.funding_rates):
                fr = self.funding_rates[self._idx]
                # Long pays positive funding, short pays negative funding
                if self._position == 1:  # long
                    step_pnl -= fr
                elif self._position == 2:  # short
                    step_pnl += fr  # shorts receive positive funding
```

- [ ] **Step 6: Update cache to store funding_rates**

Add `funding_rates` to `cache_features()` save_dict and `load_cached()` return. Same pattern as `spread_bps`.

- [ ] **Step 7: Update make_env to pass funding_rates**

```python
env.funding_rates = funding_rates_arr  # for funding cost modeling
```

- [ ] **Step 8: Bump cache version**

```python
_FEATURE_VERSION = "v11c"  # v11c: 13 features + spread_bps + funding_rates
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/test_funding_cost.py tests/ -v`
Expected: All PASS

- [ ] **Step 10: Run experiment with funding costs**

```bash
uv run python -u train.py 2>&1 | tee docs/experiments/2026-03-27-funding/run_with_funding.log
```

- [ ] **Step 11: Commit**

```bash
git add prepare.py tests/test_funding_cost.py train.py
git commit -m "feat: T42 — charge funding rate as P&L cost in TradingEnv"
```

---

## Task 3: Analyze Conditional Spread (T43 prep)

**Files:**
- Create: `scripts/analyze_conditional_spread.py`

- [ ] **Step 1: Write conditional spread analysis**

```python
#!/usr/bin/env python3
"""Measure E[spread | high activity] vs E[spread] from our data (T43 prep)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, make_env

EXCLUDED = {"CRV", "XPL"}

def main():
    ratios = []
    for sym in [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED]:
        env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=1200)
        if env.spread_bps is None:
            continue
        spread = env.spread_bps
        nonzero = spread > 0

        # Use arrival rate feature (index 11) as activity proxy
        # High activity = top 25th percentile of arrival rate
        arrival = env.features[:, 11]  # trade_arrival_rate (unnormalized in features)
        p75 = np.percentile(arrival[arrival != 0], 75) if (arrival != 0).any() else 0

        high_activity = arrival >= p75
        low_activity = ~high_activity

        spread_high = spread[high_activity & nonzero]
        spread_low = spread[low_activity & nonzero]
        spread_all = spread[nonzero]

        if len(spread_high) > 0 and len(spread_all) > 0:
            ratio = np.median(spread_high) / np.median(spread_all)
            ratios.append(ratio)
            print(f"{sym:<12} E[spread|high]={np.median(spread_high):.1f}bps  "
                  f"E[spread]={np.median(spread_all):.1f}bps  ratio={ratio:.2f}x")

    if ratios:
        print(f"\nMean conditional ratio: {np.mean(ratios):.2f}x")
        print(f"This means our slippage model should multiply spread by ~{np.mean(ratios):.1f}x during signal periods")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and record**

Run: `uv run python scripts/analyze_conditional_spread.py`

- [ ] **Step 3: Write T43 proof using the measured ratio**

If ratio R = E[spread|signal] / E[spread], then the correct slippage is:
```
slippage = R × half_spread + impact_buffer
```
Currently we use `1.0 × half_spread + 3bps`. If R > 1, we're underestimating slippage.

- [ ] **Step 4: Update TradingEnv.step() if R is material (>1.3)**

- [ ] **Step 5: Commit**

---

## Task 4: Analyze Correlation Structure (T44 prep)

**Files:**
- Create: `scripts/analyze_correlation.py`

- [ ] **Step 1: Write correlation analysis**

```python
#!/usr/bin/env python3
"""Compute correlation matrix of symbol returns, normal vs stress (T44 prep)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, make_env

EXCLUDED = {"CRV", "XPL"}

def main():
    returns = {}
    for sym in [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED]:
        env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=1200)
        px = env.prices
        ret = np.diff(np.log(np.clip(px, 1e-10, None)))
        returns[sym] = ret[:min(len(ret), 50000)]  # align lengths

    # Align to shortest
    min_len = min(len(v) for v in returns.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in returns.items()})

    # Full period correlation
    corr = df.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"Average pairwise correlation (full period): {avg_corr:.3f}")

    # Stress correlation (worst 5% of BTC returns)
    if "BTC" in df.columns:
        btc_ret = df["BTC"]
        stress_mask = btc_ret <= btc_ret.quantile(0.05)
        stress_corr = df[stress_mask].corr()
        avg_stress = stress_corr.values[np.triu_indices_from(stress_corr.values, k=1)].mean()
        print(f"Average pairwise correlation (BTC worst 5%): {avg_stress:.3f}")

        # T44 formula: portfolio DD ≈ single DD × sqrt(1 + (n-1)*rho) / sqrt(n)
        n = len(returns)
        for rho, label in [(avg_corr, "normal"), (avg_stress, "stress")]:
            multiplier = np.sqrt(1 + (n-1) * rho) / np.sqrt(n)
            print(f"\nT44 ({label}, rho={rho:.3f}, n={n}):")
            print(f"  DD multiplier vs independent: {multiplier:.2f}x")
            print(f"  If per-symbol DD = 20%: portfolio DD = {20 * multiplier:.1f}%")
            effective_n = n / (1 + (n-1) * rho)
            print(f"  Effective independent bets: {effective_n:.1f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and record**

Run: `uv run python scripts/analyze_correlation.py`

- [ ] **Step 3: Write T44 proof using measured correlation**

- [ ] **Step 4: Commit**

---

## Task 5: Write Formal T42-T45 Proofs Document

**Files:**
- Create: `docs/superpowers/specs/2026-03-27-aristotle-T42-T45.md`

- [ ] **Step 1: Write proofs document**

Combine findings from Tasks 1-4 with the mathematical proofs. Each theorem should have: Claim, Setup, Proof, Edge cases, Implementation status, Empirical verification.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-03-27-aristotle-T42-T45.md
git commit -m "feat: Aristotle T42-T45 — realism proofs (funding, slippage, correlation, sizing)"
```

---

## Task 6: Run Experiment with All Realism Improvements

- [ ] **Step 1: Run with funding costs + existing slippage**

```bash
uv run python -u train.py 2>&1 | tee docs/experiments/2026-03-27-realism/run_full_realism.log
```

- [ ] **Step 2: Parse and compare to current best (Sortino=0.353)**

- [ ] **Step 3: Update results.tsv and state.md**

- [ ] **Step 4: Commit**

---

## Dependency Graph

```
Research A (funding)  ──→ Task 1 (analyze data) ──→ Task 2 (T42 implement) ──→ Task 6 (experiment)
Research B (spreads)  ──→ Task 3 (analyze + T43)  ──────────────────────────→ Task 6
Research C (corr)     ──→ Task 4 (analyze + T44)  ──→ Task 5 (proofs doc) ──→ Task 6
Research D (latency)  ──→ Task 5 (T45 proof)      ──────────────────────────→ (future)
```

Research A-D are independent (parallel). Tasks 1-4 can partially parallelize. Task 5 needs 1-4 results. Task 6 needs Task 2 (the code change).

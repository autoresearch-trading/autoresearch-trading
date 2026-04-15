# Analysis: Cache NaN Contamination — Root-Cause Investigation

**Date:** 2026-04-15
**Script:** `scripts/analysis/analyze_nan_contamination.py`
**Scope:** 4003 shards in `data/cache/`, 117 contaminated (2.9%)

---

## Method

1. Loaded 5 representative contaminated shards and ran per-column NaN attribution.
2. Cross-referenced each shard's raw OB parquet for truncated level arrays.
3. Ran a full corpus scan of all 4003 shards, attributing NaN counts to each of the 17 feature columns.
4. Traced each NaN-generating code path back to `tape/io_parquet.py` and `tape/features_ob.py`.

---

## Results

### Per-feature NaN count (corpus-wide)

| Feature | Col | NaN count | Inf count | Total |
|---|---|---|---|---|
| `depth_ratio` | 12 | 722 | 0 | **722** |
| `imbalance_L5` | 11 | 141 | 0 | **141** |
| `cum_ofi_5` | 16 | 11 | 0 | **11** |
| `delta_imbalance_L1` | 14 | 5 | 0 | **5** |
| All 13 other features | — | 0 | 0 | **0** |

Total bad cells across 117 shards: **879**. No NaN in any trade feature (cols 0-8), `log_spread`, `imbalance_L1`, `trade_vs_mid`, or `kyle_lambda`.

### Representative shard breakdown

| Shard | N_events | NaN features | NaN count | Clustering |
|---|---|---|---|---|
| `2Z__2026-02-02` | 2,929 | depth_ratio, imbalance_L5 | 2 | single row 343 |
| `KBONK__2026-02-06` | 4,264 | depth_ratio | 8 | rows 57-73 (clustered) |
| `CRV__2026-02-03` | 4,015 | depth_ratio, imbalance_L5 | 33 | scattered (17 short snaps) |
| `UNI__2025-10-16` | 8,259 | depth_ratio | 4 | rows 2482-2485 (single snap) |
| `WLFI__2026-03-12` | 3,900 | depth_ratio | 1 | single row 2031 |

### Adjacent values at NaN rows

All NaN occurrences show clean finite values immediately before and after, with no trend anomalies. Example, `CRV 2026-02-03` `depth_ratio` at row 1288:
```
rows 1285-1291: 0.00229, 0.00340, -0.00070, [NaN], 0.03034, 0.08737, 0.36553
```
The NaN is isolated — not a sign of runaway computation or overflow.

### Raw OB data at affected dates

All 5 target shards have non-empty trades and OB parquets. The OB snapshots themselves are non-empty but some have fewer than 10 bid or ask levels:

| Shard | OB snaps | Short snaps (< 10 levels) | Min levels seen |
|---|---|---|---|
| `2Z__2026-02-02` | 2,685 | 4 (snaps 280-283) | 4 bid / 4 ask |
| `KBONK__2026-02-06` | 3,492 | 2 (snaps 28, 32) | 5 ask / 7 bid |
| `CRV__2026-02-03` | 3,569 | 17 | 3 bid / 8 ask |
| `UNI__2025-10-16` | 1,857 | 1 (snap 565) | 9 bid |
| `WLFI__2026-03-12` | 3,474 | 1 (snap 1831) | 9 bid |

Additionally, `XPL__2026-02-02` has 4 snapshots with **0 bid levels** (snaps 280-283: bids=[], asks=10). These represent a completely one-sided book during a flash-crash or data-feed outage.

No timestamp gaps > 5 min were found in these five shards. No zero prices or negative quantities.

---

## Root-Cause Hypotheses

### Root Cause 1 (PRIMARY): `expand_ob_levels` initializes to `np.nan` and leaves short arrays unfilled

**Location:** `tape/io_parquet.py`, `expand_ob_levels()`, lines 39-42.

The function creates `np.full((n_rows, 10), np.nan)` arrays and fills only `min(10, len(bids))` entries. If a raw OB snapshot has 4 bid levels, bid levels 5-10 remain `np.nan` in the flattened representation.

**How it produces NaN in `depth_ratio`:**
In `features_ob.py` `compute_snapshot_features()`, `depth_ratio` is computed as:
```python
for lvl in range(1, 11):
    bid_not_total += ob[f"bid{lvl}_price"] * ob[f"bid{lvl}_qty"]
```
Any `NaN * NaN = NaN` addition poisons `bid_not_total`, which then propagates through `np.log(max(bid_not_total, 1e-6) / ...)` — the `max()` with `1e-6` does NOT guard against NaN input, only against zero.

**How it produces NaN in `imbalance_L5`:**
Same pattern: the L5 loop runs `for lvl in range(1, 6)`. A 4-level book has `NaN` at index 4 (level 5), poisoning the sum. A 9-level book is safe for `imbalance_L5` (levels 1-5 all present) but still produces `depth_ratio` NaN (level 10 missing).

This perfectly explains the count asymmetry: 722 `depth_ratio` NaN (any book < 10 levels on either side) vs. 141 `imbalance_L5` NaN (only books with < 5 levels, which are rarer and more severe market events).

**The valid-mask bypass:** `cache.py` line 258 filters events via `np.isfinite(ob_aligned["mid"])`. Since `mid = (bid1_p + ask1_p)/2` uses only level-1 data, which is always present in any non-empty snapshot, truncated snapshots (1-9 levels) produce finite `mid` and pass the filter. The NaN-carrying columns are not checked.

### Root Cause 2 (SECONDARY): `_piecewise_cont_ofi` NaN propagation through rolling sum

**Location:** `tape/features_ob.py`, `_piecewise_cont_ofi()` + `compute_snapshot_features()` lines 137-146.

The OFI function operates on `bid1_p, bid1_q, ask1_p, ask1_q` extracted from the snapshot DataFrame. For 0-level snapshots (e.g., XPL snaps 280-283 with empty bids), `bid1_p` is `NaN`, making every OFI term at that snapshot `NaN`.

The critical amplifier: `pd.Series(ofi).rolling(OFI_WINDOW, min_periods=1).sum()` with `OFI_WINDOW=5`. A NaN in the OFI series propagates through the rolling sum for up to 5 subsequent snapshots. Events aligned to those 5 follow-on snapshots inherit `cum_ofi_5 = NaN` — even though their aligned snapshot itself has a full 10-level book. This is why `cum_ofi_5` NaN appears at rows 389-397 in `KPEPE__2025-11-22` while `depth_ratio` NaN spans rows 371-416.

Note: `pd.Series.rolling().sum()` with `min_periods=1` still propagates NaN from the window — it only sets a lower bound on the minimum observations required before computing, but `NaN + finite = NaN` within the window regardless.

### Root Cause 3 (TERTIARY): `delta_imbalance_L1` inherits NaN from 0-level predecessor snapshot

**Location:** `tape/features_ob.py`, `compute_snapshot_features()` lines 120-126.

`imbalance_L1` is computed at the snapshot level as `(bid1_not - ask1_not) / (bid1_not + ask1_not)`. For a 0-level bid snapshot, `bid1_price = NaN`, so `bid1_not = NaN`, making `imb_l1[s] = NaN`.

The delta is `delta_imb_l1 = np.concatenate([[first_delta], np.diff(imb_l1)])`, so `delta_imb_l1[s+1] = imb_l1[s+1] - imb_l1[s] = finite - NaN = NaN`. After the sequence of 0-level snaps recovers, the first full-10-level snapshot after them produces a `NaN` delta even though it is fully valid. This NaN is then forward-filled to any events aligned to that transition snapshot.

Confirmed on `XPL__2026-02-02` row 335: event ts 1770018967652 aligns to snap 284 (10-level clean book), but `delta_imbalance_L1 = NaN` because snap 283 immediately before it had 0 bid levels (`imb_l1[283] = NaN`).

### Root Cause 4 (RANGE VIOLATION, not NaN): Genuine extreme spreads in thin books

**Location:** Data quality issue, not a code bug.

`KBONK__2026-02-02` events 357-359 have `log_spread = 0.0743`, violating the validator's `log_spread < 0` sanity check. This is a real market event: snap 284 of that day has **1 bid level** at price 0.0032 and **2 ask levels** starting at 0.01067. The spread-to-mid ratio is 1.077, giving `log(1.077) = 0.074`. This is not a code error — it is an accurately computed feature value for an extremely illiquid moment (KBONK during very low-volume period). Only 1 shard (3 events) is affected.

---

## Proposed Fixes (for Builder-8)

### Fix 1: `expand_ob_levels` — treat missing levels as zero, not NaN

**File:** `tape/io_parquet.py`, `expand_ob_levels()`.

- Change initialization from `np.full((n_rows, _N_LEVELS), np.nan, dtype=float)` to `np.zeros((n_rows, _N_LEVELS), dtype=float)` for both `_qty` arrays.
- For `_price` arrays, initialize to 0 as well (or keep NaN then replace with 0 before returning).
- Rationale: a missing level is economically equivalent to zero quantity at that level. A bid-level-10 with qty=0 contributes 0 notional and does not change depth_ratio or imbalance_L5. This treats thin books correctly — fewer levels means less depth, not undefined depth.
- This single change fixes Root Causes 1, 2, and 3 simultaneously because all downstream formulas already use epsilon guards that handle zero inputs.

### Fix 2: `compute_snapshot_features` — guard `depth_ratio` and `imbalance_L5` against partial NaN explicitly

**File:** `tape/features_ob.py`, `compute_snapshot_features()`.

- As belt-and-suspenders after Fix 1: `np.nansum` instead of direct sum in the `depth_ratio` and `imbalance_L5` loops, or equivalently use `np.nan_to_num(level_array, nan=0.0)` before accumulating.
- This makes the code robust to any future NaN source in the OB levels (e.g., corrupt parquet row).

### Fix 3: `compute_snapshot_features` — forward-fill `imb_l1` before computing delta

**File:** `tape/features_ob.py`, `compute_snapshot_features()` around line 126.

- After computing `imb_l1`, apply `pd.Series(imb_l1).fillna(method='ffill').fillna(0)` before `np.diff`.
- Or equivalently: after Fix 1, `imb_l1` will never be NaN (because bid1_not = 0 gives imb_l1 = -1.0, not NaN), so Fix 1 alone resolves this.
- If Fix 1 is not applied, an explicit guard is needed here.

### Fix 4: `_piecewise_cont_ofi` — handle 0-level snapshots in rolling sum

**File:** `tape/features_ob.py`, `compute_snapshot_features()` lines 138-140.

- After computing `ofi`, replace `NaN` values with `0` before the rolling sum: `ofi = np.nan_to_num(ofi, nan=0.0)`.
- Rationale: an OFI of 0 is the correct imputation for a snapshot with no visible bid/ask (no order flow information available).
- Fix 1 also resolves this, since bid1_price=0 gives `bid_up=0`, `bid_same=0`, etc.

### Fix 5: Range-clamp `log_spread` in `compute_snapshot_features` (optional)

**File:** `tape/features_ob.py`, line 82.

- Add `log_spread = np.clip(log_spread, -10.0, 0.0)` after computing.
- Rationale: `log_spread > 0` means `ask > 3× bid`, which is an extreme market-structure anomaly that should not train the encoder. Clipping to 0 preserves the "extreme widening" signal without producing out-of-distribution values.
- Alternatively, expose these events to the valid-mask drop in `cache.py` by checking `log_spread > 0`.
- This is cosmetic — these 3 events will not break training, but they violate the pre-registered sanity bound.

---

## Severity and Recommendation

**Severity: LOW-MEDIUM. Patch the code and rebuild.**

- 879 bad cells out of ~125M total cells across 4003 shards = **0.0007%** contamination rate.
- 117 shards affected = **2.9%** of corpus. But the NaN cells are scattered (1-91 per shard), not concentrated in entire shards.
- The NaN values represent real thin-book events (market stress periods) — dropping entire shards would discard valid Wyckoff stress signals.
- Fix 1 alone (change `np.nan` initialization to `np.zeros`) resolves all four NaN-producing code paths. It requires a full cache rebuild (cheap: no model training needed, ~1-2h on CPU).
- Do NOT drop shards — patch and rebuild. Dropping 117 shards discards ~450K events of potentially informative market stress data.

**Rebuild scope:** All 4003 shards must be rebuilt. Fix 1 changes `expand_ob_levels` which is called for every shard. There is no way to patch individual shards without access to the raw parquet.

---

## Appendix: KBONK `log_spread = 0.074` Anomaly

Snap 284 of `KBONK__2026-02-02` (ts=1770018946975) has:
- 1 bid level: price=0.0032, qty=15000
- 2 ask levels: price=0.01067, qty=185150 and price=0.0149, qty=738

This is a real data snapshot. The bid side shows a single stale limit order at 0.0032 (approximately 3.3× below the ask1). This is not an instrument-level book inversion — KBONK was trading at ~0.0050-0.0067 on that date (confirmed from trades: price range [0.0052, 0.0067]). The 0.0032 bid is a deep stale order, effectively creating an empty-bid book at market-relevant prices. The `log_spread > 0` is a correct computation of this anomalous state. **The raw OB data is the source; the code is correct.**

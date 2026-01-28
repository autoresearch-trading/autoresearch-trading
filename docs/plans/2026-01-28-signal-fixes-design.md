# Signal Engine Bug Fixes Design

**Date**: 2026-01-28
**Scope**: 6 bug fixes for signal calculators and settings
**Approach**: Conservative fixes - minimal changes to solve specific bugs

---

## Fix #1: OFI Formula (Critical)

**File**: `signal-engine/src/signals/ofi.py`
**Lines**: 119-143

**Problem**: `_bid_component` and `_ask_component` have unreachable code and don't match the MASTER_IDEA.md OFI formula. When price is unchanged, both indicator functions should be 1, returning `curr_qty - prev_qty`.

**Fix**: Handle three distinct cases:
```python
@staticmethod
def _bid_component(curr_price, curr_qty, prev_price, prev_qty) -> float:
    if curr_price > prev_price:      # Price improved: new buying pressure
        return curr_qty
    elif curr_price < prev_price:    # Price worsened: buying pressure left
        return -prev_qty
    else:                            # Price unchanged: net change in depth
        return curr_qty - prev_qty
```

Same pattern for `_ask_component` with inverted logic.

---

## Fix #2: TFI Floating-Point Drift (Critical)

**File**: `signal-engine/src/signals/tfi.py`
**Lines**: 68-76

**Problem**: `buy_volume` and `sell_volume` accumulate floating-point errors after thousands of subtractions in `_evict_old_trades()`.

**Fix**: Periodic recalculation every 100 evictions:
```python
def __init__(self, ...):
    ...
    self._eviction_count: int = 0
    self._recalc_interval: int = 100

def _evict_old_trades(self, current_ts):
    # ... existing eviction logic ...
    self._eviction_count += 1

    if self._eviction_count >= self._recalc_interval:
        self._recalculate_volumes()
        self._eviction_count = 0

def _recalculate_volumes(self):
    self.buy_volume = sum(t.qty for t in self.trade_window if t.side == "buy")
    self.sell_volume = sum(t.qty for t in self.trade_window if t.side == "sell")
```

---

## Fix #3: OFI Stats/History Mismatch (Important)

**File**: `signal-engine/src/signals/ofi.py`
**Lines**: 113-117

**Problem**: `_history` is bounded to 100 items but `_stats` (Welford) accumulates forever. Z-score uses stale mean/variance.

**Fix**: Compute stats directly from bounded history instead of Welford:
```python
def _track(self, value: float) -> None:
    self._history.append(value)
    if len(self._history) > self.history_size:
        self._history.pop(0)

    if len(self._history) >= 2:
        self._stats.mean = float(np.mean(self._history))
        self._stats.variance = float(np.var(self._history, ddof=1)) * (len(self._history) - 1)
        self._stats.count = len(self._history)
```

---

## Fix #4: CVD Divergence Edge Case (Important)

**File**: `signal-engine/src/signals/cvd.py`
**Lines**: 78-90

**Problem**: When CVD is near zero, relative divergence calculation explodes (1e-9 denominator causes huge values).

**Fix**: Use minimum denominator threshold:
```python
def __init__(self, ...):
    ...
    self.min_divergence_denom: float = 1.0

# In _detect_divergence:
denom = max(abs(first_cvd_high), self.min_divergence_denom)
```

---

## Fix #5: TFI Missing reset() Method (Important)

**File**: `signal-engine/src/signals/tfi.py`

**Problem**: CVD and OFI have `reset()` for regime transitions, but TFI doesn't.

**Fix**: Add reset method:
```python
def reset(self) -> None:
    """Reset internal state, e.g. when switching regimes."""
    self.trade_window.clear()
    self.buy_volume = 0.0
    self.sell_volume = 0.0
    self._eviction_count = 0
```

---

## Fix #6: Settings Type Annotation (Minor)

**File**: `signal-engine/src/config/settings.py`
**Line**: 32

**Problem**: `symbols: list[str] = None` is incorrect typing.

**Fix**: Use proper optional typing:
```python
symbols: list[str] | None = None
```

---

## Implementation Order

1. **#1 OFI formula** (unblocks #3)
2. **#4 TFI drift** (unblocks #5)
3. **#2 CVD edge case** (independent)
4. **#6 Settings type** (independent)
5. **#3 OFI stats** (blocked by #1)
6. **#5 TFI reset** (blocked by #4)

## Testing

- Run existing tests after each fix
- Add test cases for edge conditions (CVD near zero, TFI after many trades)
- Verify OFI formula matches MASTER_IDEA.md specification

# Pacifica Activity-Gate Redesign Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after explicit approval. Do not implement this plan during the current diagnostic-only state unless Diego asks.

**Goal:** Redesign the paper-trading eligibility activity gate so it distinguishes sparse-but-valid trade activity from missing trade lineage, without loosening gates to force eligible symbols on the current early sample.

**Architecture:** Keep the current eligibility report as the canonical gate until a new pre-registered V2 activity gate is implemented and tested. Add V2 metrics beside the current all-row median metric, run V1/V2 side-by-side, and require the 30-day sample gate plus lineage consistency before any V2 output can be used for paper-trading eligibility. Treat V2 as diagnostic until it survives fresh OOS data and downstream economics checks.

**Tech Stack:** Python, pandas, pytest, parquet/CSV reports, existing Pacifica regime state and eligibility scripts.

---

## Current evidence and constraints

Current canonical state as of 2026-05-19:

- `docs/experiments/paper-trading-eligibility/README.md`: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`, eligible symbols `0/66`.
- Current sample maturity: up to 19 distinct dates, below the fixed 30-day provisional gate.
- Current activity gate uses all-row median 1-minute `trade_notional` plus median BBO updates.
- `docs/experiments/trade-activity-lineage/README.md`: `LINEAGE_AUDIT_PASS_DIAGNOSTIC`; audited raw/silver/regime trade-count mismatches are zero.
- Sparse-trade zero-median explanations: 8/10 audited symbols.

Non-negotiable constraints:

- Do not paper/live trade from this work.
- Do not tune thresholds to make the current 19-day sample pass.
- Do not lower `min_days=30` as part of this redesign.
- Do not replace liquidity, spread/cost, stability, concentration, lineage, or post-cost validation gates.
- V2 activity-gate thresholds must be declared before using future data to evaluate eligibility.

## Proposed V2 activity metrics

Add these diagnostic columns to `symbol_eligibility.csv` before changing any `eligible` semantics:

1. `trade_active_row_share`
   - Definition: fraction of symbol regime rows where `trade_count > 0` or `trade_notional > 0`.
   - Purpose: separates sparse trading from missing lineage.

2. `active_minute_median_trade_notional`
   - Definition: median `trade_notional` over rows where `trade_notional > 0`.
   - Purpose: measures typical notional when trades occur.
   - Fail closed to 0/null-safe failure when there are no active minutes.

3. `active_minute_p25_trade_notional`
   - Definition: 25th percentile `trade_notional` over rows where `trade_notional > 0`.
   - Purpose: avoids a few large trade minutes carrying the median.

4. `trade_active_days`
   - Definition: number of distinct dates with at least one positive `trade_count` or positive `trade_notional` row.
   - Purpose: prevents one-day activity from passing.

5. `median_daily_trade_count`
   - Definition: median daily sum of `trade_count`.
   - Purpose: catches symbols with occasional isolated trades.

6. `median_daily_trade_notional`
   - Definition: median daily sum of `trade_notional`.
   - Purpose: coarse daily execution/liquidity proxy.

7. `activity_v2_gate_pass_diagnostic`
   - Initial diagnostic-only boolean, separate from current `activity_gate_pass`.
   - Must not feed `eligible` until a later approved implementation step.

## Candidate V2 gate shape, to freeze before use

A symbol may pass V2 activity only if all are true:

- sample gate passes: `n_days >= 30` and `n_observations >= 10000`;
- lineage gate passes for the symbol or the symbol is not yet eligible for final paper trading;
- `trade_active_days >= 20` once `min_days=30` is met;
- `trade_active_row_share >= 0.20`;
- `active_minute_median_trade_notional >= 25`;
- `active_minute_p25_trade_notional >= 5`;
- `median_daily_trade_count >= 100`;
- `median_daily_trade_notional >= 1000`;
- existing `median_bbo_updates_per_min >= 10` still passes.

These numbers are starting candidates for review, not tuned results. Before implementation, either freeze them explicitly or replace them with a separately justified pre-registration. Do not adjust them after inspecting pass counts on the current sample.

## Task 1: Add metric helper tests

**Objective:** Prove V2 metric semantics on small synthetic regime-state fixtures.

**Files:**

- Modify: `tests/scripts/test_build_pacifica_eligibility_gates.py`
- Modify: `scripts/build_pacifica_eligibility_gates.py`

**Step 1: Write failing tests**

Add tests covering:

- all rows inactive -> active-minute metrics are 0/null-safe and V2 gate fails;
- sparse rows with correct raw/regime trade counts -> `trade_active_row_share` reflects active rows, not missing data;
- active minutes with one outlier -> p25 metric prevents outlier-only pass;
- active trades concentrated in one date -> `trade_active_days` fails.

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/scripts/test_build_pacifica_eligibility_gates.py -q
```

Expected: new tests fail because V2 metrics are not implemented.

## Task 2: Implement V2 diagnostic metrics only

**Objective:** Add V2 activity metric columns without changing `eligible` or current `activity_gate_pass`.

**Files:**

- Modify: `scripts/build_pacifica_eligibility_gates.py`
- Modify: `tests/scripts/test_build_pacifica_eligibility_gates.py`

**Implementation notes:**

- Compute active rows per symbol with `(trade_count > 0) | (trade_notional > 0)`.
- Require `trade_count` as a numeric input if V2 metrics are enabled; fail closed if missing.
- Add columns to every symbol row:
  - `trade_active_row_share`
  - `active_minute_median_trade_notional`
  - `active_minute_p25_trade_notional`
  - `trade_active_days`
  - `median_daily_trade_count`
  - `median_daily_trade_notional`
  - `activity_v2_gate_pass_diagnostic`
- Keep current `activity_gate_pass` unchanged.

**Verification:**

```bash
uv run pytest tests/scripts/test_build_pacifica_eligibility_gates.py -q
```

Expected: tests pass.

## Task 3: Update report outputs without changing eligibility semantics

**Objective:** Make V2 diagnostics visible in reports while preserving current no-trade status.

**Files:**

- Modify: `scripts/build_pacifica_eligibility_gates.py`
- Modify: `docs/experiments/paper-trading-eligibility/README.md` only after running the report generator.

**Steps:**

1. Add V2 thresholds to `thresholds.csv` under clearly named diagnostic keys.
2. Add V2 gate count row: `activity_v2_gate_pass_diagnostic`.
3. Add README language: V2 metrics are diagnostic and do not feed `eligible` yet.
4. Keep verdict logic unchanged.

**Verification:**

```bash
uv run python scripts/build_pacifica_eligibility_gates.py
uv run pytest tests/scripts/test_build_pacifica_eligibility_gates.py -q
git diff --check
```

Expected: report still says `INSUFFICIENT_SAMPLE_DIAGNOSTIC` and `eligible_symbols=0` while V2 diagnostic metrics are visible.

## Task 4: Add lineage dependency for final eligibility use

**Objective:** Prevent V2 activity from being used for paper eligibility unless trade lineage is current and clean.

**Files:**

- Modify: `scripts/build_pacifica_eligibility_gates.py` or add a wrapper script if cleaner.
- Modify: `scripts/audit_pacifica_trade_activity_lineage.py` only if it needs machine-readable summary output.
- Test: add/update tests under `tests/scripts/`.

**Rule:**

Before `activity_v2_gate_pass_diagnostic` may become a final activity gate, the latest lineage audit must show:

- `raw/silver mismatches = 0` for candidate symbols;
- `silver/regime trade-count mismatches = 0`;
- no `unexplained_zero_medians` for candidate symbols;
- lineage artifact timestamp no older than the eligibility report timestamp unless both are produced in the same run.

**Verification:**

Add tests for missing/stale/failing lineage summary causing fail-closed diagnostic verdicts.

## Task 5: Run side-by-side V1/V2 diagnostics on the next 30+ day refresh

**Objective:** Evaluate V2 only after the fixed 30-day sample gate can pass.

**Files:**

- Output: `docs/experiments/paper-trading-eligibility/`
- Output: `docs/experiments/trade-activity-lineage/`

**Steps:**

1. Refresh raw -> silver -> regime through the verified side-by-side promotion flow.
2. Run eligibility with V2 diagnostics.
3. Run lineage audit for any symbols that would pass all non-sample/non-activity gates.
4. Report pass counts without changing paper-trading permission.

**Verdict discipline:**

- If V2 produces eligible candidates before 30 days: still `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- If V2 produces candidates at 30+ days: still diagnostic until post-cost baselines and walk-forward validation are present.
- If V2 admits many illiquid/sparse symbols: kill or tighten before any paper logger.

## Task 6: Only after V2 survives fresh data, wire it into final eligibility

**Objective:** Replace current activity gate only after a reviewed V2 diagnostic run on fresh data.

**Files:**

- Modify: `scripts/build_pacifica_eligibility_gates.py`
- Modify: tests and README.

**Required approval:**

Diego must explicitly approve changing final `activity_gate_pass` semantics. Until then, V2 remains a diagnostic column only.

**Verification:**

```bash
uv run pytest tests/scripts/test_build_pacifica_eligibility_gates.py tests/scripts/test_audit_pacifica_trade_activity_lineage.py -q
uv run python scripts/build_pacifica_eligibility_gates.py
uv run python scripts/audit_pacifica_trade_activity_lineage.py
git diff --check
```

Expected: final report still clearly distinguishes paper-trading eligibility from alpha/trading permission.

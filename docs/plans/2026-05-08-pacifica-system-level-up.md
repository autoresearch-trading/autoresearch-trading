# Pacifica System Level-Up Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the Pacifica full-fidelity research stack from a diagnostic data pipeline into an institutional-grade, non-HFT paper-trading research system without claiming edge before sample/eligibility gates pass.

**Architecture:** Build the accounting and validation spine before strategy alpha. Start with reusable execution economics, a strategy-neutral paper ledger, no-trade regime governance, online/offline parity, and walk-forward validation. Then add symbol lifecycle, cross-venue context, event risk, external alerting, and a falsifiable research registry.

**Tech Stack:** Python 3.12, pandas/pyarrow, pytest, Markdown/CSV artifacts under `docs/experiments/`, scripts under `scripts/`, tests under `tests/scripts/`.

---

## Phase 0: Non-negotiable constraints

- Keep Hermes as the primary workflow; do not revive Claude Code assets.
- Keep strategies non-HFT. Use high-frequency data for slower state/risk decisions.
- Keep raw archival append-only and do not delete active `raw/` or `ops/` R2 prefixes.
- Do not paper-trade while eligibility remains `0` symbols or reports remain `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Every strategy result must include fees, slippage/adverse selection, funding, drawdown, Sortino, trade/day/symbol concentration, dumb baselines, and random same-frequency controls.

## Phase 1: Execution economics simulator

**Objective:** Create a reusable cost/fill simulator that all future backtests and paper-ledger runs must use.

**Files:**
- Create: `scripts/simulate_pacifica_execution.py`
- Create: `tests/scripts/test_simulate_pacifica_execution.py`
- Output: `docs/experiments/execution-simulator/README.md`

**Task 1.1: Write failing tests for cost components**

Test behavior:
- taker/taker round trip charges 8 bps before slippage;
- maker/maker round trip charges 3 bps before adverse selection;
- funding payment reduces PnL for long positions when funding is positive;
- slippage/adverse-selection bps are applied against notional.

Run:
`uv run pytest tests/scripts/test_simulate_pacifica_execution.py -q`

Expected first result: fail because module does not exist.

**Task 1.2: Implement minimal execution simulator**

Core dataclasses:
- `ExecutionAssumptions`
- `TradeIntent`
- `SimulatedFill`

Core function:
- `simulate_round_trip(intent, assumptions) -> SimulatedFill`

**Task 1.3: Add report writer**

Function:
- `write_execution_simulator_report(out_dir)`

Artifacts:
- `README.md`
- `assumptions.csv`
- `example_round_trips.csv`

## Phase 2: Strategy-neutral paper ledger

**Objective:** Build the accounting spine independent of any strategy.

**Files:**
- Create: `scripts/build_pacifica_paper_ledger.py`
- Create: `tests/scripts/test_build_pacifica_paper_ledger.py`
- Output: `docs/experiments/paper-ledger/README.md`

Ledger records:
- orders
- fills
- positions
- funding
- fees
- realized/unrealized PnL
- equity curve
- drawdown
- exposure by symbol
- no-trade / halted reasons

Acceptance tests:
- opening and closing a long updates realized PnL after fees;
- equity curve max drawdown is computed from chronological snapshots;
- funding debits are included in net PnL;
- ledger refuses fills for ineligible symbols unless explicitly diagnostic/dry-run.

## Phase 3: No-trade regime governor

Status: implemented 2026-05-08.

**Objective:** Convert regime-state diagnostics into explicit live/paper-trading decisions.

**Files:**
- Create: `scripts/build_pacifica_regime_governor.py`
- Create: `tests/scripts/test_build_pacifica_regime_governor.py`
- Output: `docs/experiments/regime-governor/README.md`

States:
- `TRADABLE_DIAGNOSTIC`
- `REDUCE_SIZE_DIAGNOSTIC`
- `SKIP_TOXIC_REGIME`
- `SKIP_WIDE_SPREAD`
- `SKIP_THIN_DEPTH`
- `SKIP_STALE_DATA`
- `SKIP_MARK_DISLOCATION`
- `SKIP_FORCED_FLOW_AFTERSHOCK`

Rules must be fixed and documented before serious validation windows accrue.

## Phase 4: Online/offline parity harness

Status: implemented 2026-05-08.

**Objective:** Prove live microbatch features match historical rebuilt features.

**Files:**
- Create: `scripts/check_pacifica_feature_parity.py`
- Create: `tests/scripts/test_check_pacifica_feature_parity.py`
- Output: `docs/experiments/feature-parity/README.md`

Required columns:
- `available_ts`
- `computed_at`
- `watermark_ts`
- `feature_version`
- `provisional_final_flag`

Acceptance tests:
- equal fixtures pass within tolerance;
- changed spread/depth/toxicity features fail with clear mismatch rows;
- missing feature versions fail.

## Phase 5: Walk-forward validation harness

Status: implemented 2026-05-08.

**Objective:** Standardize chronological validation for every future idea.

**Files:**
- Create: `scripts/run_pacifica_walk_forward_validation.py`
- Create: `tests/scripts/test_run_pacifica_walk_forward_validation.py`
- Output: `docs/experiments/walk-forward-validation/README.md`

Verdicts:
- `INSUFFICIENT_SAMPLE_DIAGNOSTIC`
- `EARLY_SANITY_ONLY`
- `PROVISIONAL_PASS`
- `PROVISIONAL_FAIL`
- `VALIDATION_GRADE_PASS`
- `VALIDATION_GRADE_FAIL`

Must include purged chronological windows, concentration gates, and random same-frequency controls.

## Phase 6: Symbol lifecycle promotion/demotion

Status: implemented 2026-05-08.

**Objective:** Turn eligibility into a lifecycle instead of a static report.

**Files:**
- Create: `scripts/build_pacifica_symbol_lifecycle.py`
- Create: `tests/scripts/test_build_pacifica_symbol_lifecycle.py`
- Output: `docs/experiments/symbol-lifecycle/README.md`

States:
- `COLLECTED`
- `RESEARCHABLE`
- `ELIGIBLE`
- `PROBATION`
- `DISABLED`
- `RETIRED`

Reasons:
- insufficient days
- insufficient activity
- spread too wide
- depth too thin
- unstable feed
- too concentrated
- bad post-cost baseline

## Phase 7: Cross-venue/reference market context

Status: implemented 2026-05-08.

**Objective:** Distinguish Pacifica-local states from broad market states.

**Files:**
- Create: `scripts/build_pacifica_reference_context.py`
- Create: `tests/scripts/test_build_pacifica_reference_context.py`
- Output: `docs/experiments/reference-market-context/README.md`

Features:
- BTC/ETH market beta proxy
- reference volatility
- cross-venue premium/discount
- broad crypto risk-on/risk-off state
- funding divergence

Start with pluggable CSV/parquet inputs; do not hardwire paid APIs.

## Phase 8: External ops alerting

Status: implemented 2026-05-08.

**Objective:** Separate health checks from actual notification delivery.

**Files:**
- Create: `scripts/plan_pacifica_ops_alerts.py`
- Create: `tests/scripts/test_plan_pacifica_ops_alerts.py`
- Output: `docs/ops/pacifica-alerting/README.md`

Alert conditions:
- raw freshness stale
- free disk below 50 GiB
- lifecycle upload/verify failures
- R2 sidecar mismatch
- watchdog status stale
- API surface changed
- archive inventory stale
- research refresh failed

No external delivery credentials should be committed.

## Phase 9: Event/calendar risk layer

Status: implemented 2026-05-08.

**Objective:** Mark known event-risk windows in the regime table.

**Files:**
- Create: `scripts/build_pacifica_event_risk_calendar.py`
- Create: `tests/scripts/test_build_pacifica_event_risk_calendar.py`
- Output: `docs/experiments/event-risk-calendar/README.md`

Start with local CSV input:
- event timestamp
- event type
- pre-window minutes
- post-window minutes
- severity
- source note

## Phase 10: Research idea registry

Status: implemented 2026-05-08.

**Objective:** Force every future edge idea to be falsifiable before implementation.

**Files:**
- Create: `docs/research/pacifica-idea-registry.md`
- Create: `scripts/validate_pacifica_idea_registry.py`
- Create: `tests/scripts/test_validate_pacifica_idea_registry.py`

Each idea must include:
- hypothesis
- mechanical label
- trade/risk action
- cost model
- validation window
- frozen parameters
- kill criteria
- OOS plan
- result/verdict

## Execution order

1. Implement Phase 1 and Phase 2 first because all future strategy work needs cost accounting.
2. Implement Phase 3 before any live paper decisions.
3. Implement Phase 4 before using live features for decisions.
4. Implement Phase 5 before claiming validation.
5. Implement Phases 6-10 as hardening layers after the spine exists. Phases 6-10 are now complete; next work is to let the archive mature, rerun fixed validation, then only add sparse strategy adapters after eligibility/economics/governor/registry gates pass.

## Verification command set

After each phase:

```bash
uv run pytest tests/scripts/test_<new_script>.py -q
python -m py_compile scripts/<new_script>.py
git diff --check
```

Before any handoff:

```bash
uv run pytest tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
python -m py_compile scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
git status --short
```

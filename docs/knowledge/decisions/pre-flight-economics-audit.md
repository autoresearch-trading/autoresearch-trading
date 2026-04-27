---
title: Pre-Flight Economics Audit (Mandatory Before Any Encoder Retrain)
date: 2026-04-27
status: accepted
decided_by: v2 program closure synthesis — council-5 STOP argument formalized as standing rule
sources:
  - docs/experiments/goal-a-v2-program-end-state.md
  - docs/experiments/goal-a-feasibility/maker_adverse_selection.md
  - docs/experiments/goal-a-feasibility/headroom_table.csv
  - docs/experiments/goal-a-feasibility/survivors.md
last_updated: 2026-04-27
---

# Decision: Mandatory Pre-Flight Economics Audit

## What Was Decided

Before any encoder-retrain or representation-learning effort on this
codebase, the proposal MUST first answer:

> **Is there a fee-clearing edge in the LABELS alone, evaluated at the
> highest plausible prediction accuracy AND the relevant execution
> regime (taker / maker / hybrid)?**

If the answer is no — under both maker AND taker assumptions — the
encoder retrain question is moot. Pivot the labels OR pivot the execution
regime BEFORE writing model code.

The audit is CPU-only and produces three artifacts at minimum:

1. **Per-symbol oracle headroom table** (cf. `headroom_table.csv`) at the
   plausible accuracy ceiling, enumerating (symbol × size × horizon) cells.
2. **Adverse-selection sim** (cf. `maker_adverse_selection.md`) for any
   maker-fee execution: E[realized|filled], breakeven distribution, fill
   rate, post-fee envelope.
3. **Survivors documentation** (cf. `survivors.md`) — explicit count and
   identity of cells that clear breakeven under each fee regime.

## Why

v1 and v2 BOTH found real +1pp signal that fee-economics killed:

- **v1 direction prediction:** linearly-extractable +1pp at H500;
  fee-blocked at every framing tested. Closed `v1-program-closed`
  (commit `522fc1e`).
- **v2 cascade-onset prediction:** flat-LR AUC=0.8373; top-1% precision
  25.4% OOS; under E[realized|filled] = −7.89bp adverse selection,
  post-fee envelope is consumed. Closed `v2-program-closed`
  (commit `77e29f4`).

Both programs spent weeks of work on architecture / representation /
training-protocol design BEFORE establishing that the venue economics
were the binding constraint. The pre-flight audit is the methodology
correction: do the cheap audit FIRST.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Trust the spec's economic assumptions implicitly | Both v1 and v2 violated this; the spec's "tradeable signal" claim was unverified |
| Run audit AFTER baseline + before encoder | Still wastes baseline compute on tasks the venue blocks; the audit is cheap enough to run first |
| Make the audit informational, not gating | v2 council-5 review demonstrated lead-0 underestimating prior negative results; binding gate prevents repeat |

## Concrete Audit Recipe

For any new label proposal, before any encoder code is written:

1. **Compute per-symbol forward-return distributions** at H10, H50, H100,
   H500 conditional on the new label's positive class.
2. **Compute breakeven accuracy** at each (symbol × size × horizon) cell
   under both taker fees (4bp + 1bp slip per side = 10bp round-trip) and
   maker fees (1.5bp + adverse-selection drag).
3. **Cross-check against an oracle accuracy ceiling** of 0.55-0.60. If
   oracle accuracy at THIS ceiling produces 0/N survivors, the encoder
   cannot make economics work. Stop.
4. **If maker-side**: simulate adverse-selection conditional on fill at
   reasonable queue position; report E[realized|filled].
5. **Document the audit** as a standalone artifact in
   `docs/experiments/<goal>-feasibility/` BEFORE any architecture work.

## Impact

- Adopted as standing rule for any future Goal-* program on this codebase.
- Combined with [random-init-probe-protocol](random-init-probe-protocol.md):
  the two pre-flight gates a new proposal must pass.
- Methodology pattern documented in
  [v2-program-closure](v2-program-closure.md) §"Impact" and the v2 end-state
  document.

## Related

- [V2 Program Closure](v2-program-closure.md)
- [Random-Init Probe Protocol](random-init-probe-protocol.md)
- [V1 Path A Program Closure](path-a-program-closure.md)
- v2 feasibility chain: `docs/experiments/goal-a-feasibility/`

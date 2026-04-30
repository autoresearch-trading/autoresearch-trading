# Final Trading Research Verdict

**Date:** 2026-04-30  
**Scope:** `autoresearch-trading` v1 representation-learning program, Goal-A feasibility program, and Goal-A v2 cascade-onset program.  
**Related closure tags:** `v1-program-closed`, `v2-program-closed`  
**Recommendation:** close the current ML/RL trading-research line. Do not run more model, encoder, adapter, or RL experiments under the current framing.

---

## Executive verdict

The project has found real predictive structure in Pacifica perp tape, especially around liquidation-cascade onset. However, the tested signals do not map to an executable trading strategy after direction, fees, slippage, fill selection, adverse selection, and headroom constraints.

The correct conclusion is not "there is no signal." The correct conclusion is:

> Pacifica tape contains statistically real microstructure signal, but the currently tested signal classes are not tradeable under the available execution regimes. More ML/RL on this framing is not warranted.

The current research program should therefore be closed as an ML trading experiment and preserved as a research asset / negative result.

---

## What is real

### 1. Cascade-onset prediction is real

Goal-A v2 found a real cascade-onset signal using flat hand features.

Evidence:

- `docs/experiments/goal-a-v2-program-end-state.md`
- `docs/experiments/goal-a-feasibility/cascade_precursor_real.md`
- `docs/experiments/goal-a-feasibility/cascade_precursor_oos.md`

Key reported results:

- In-sample flat-LR AUC: approximately 0.815 on Apr 1-13.
- OOS flat-LR AUC: approximately 0.778 on Apr 14-26.
- Unified day-blocked CV flat-LR AUC: approximately 0.8373.
- Top-tail precision showed meaningful lift versus the base cascade rate.

Interpretation:

The tape can predict cascade-onset regimes. This is the strongest positive result in the project.

### 2. The data has useful microstructure information

The experiment archive shows that Pacifica-specific fields are useful as labels/context and that standard order-flow / flat microstructure features capture meaningful structure.

Interpretation:

The dataset is valuable. The failure is not that the data is random. The failure is that the discovered signals do not clear the trading/economics layer.

---

## What is dead under the current framing

### 1. v1 representation learning as a direction-trading strategy

Evidence:

- `docs/experiments/step4-program-end-state.md`
- `docs/experiments/step4-gate2-finetune.md`

Key reported results:

- The MEM + SimCLR encoder was non-collapsed and did learn weak structure.
- Frozen encoder produced only a small H500 direction signal, roughly +1pp balanced accuracy over controls.
- Fine-tuning failed versus flat baselines:
  - flat-LR H500 balanced accuracy: approximately 0.5115
  - frozen encoder LR: approximately 0.5061
  - fine-tuned CNN: approximately 0.4947
  - CNN minus flat-LR: approximately -1.7pp

Verdict:

The representation learned something, but not enough to trade. Direction prediction from the v1 encoder is closed.

### 2. Universal tape-reading representation

Evidence:

- `docs/experiments/step4-program-end-state.md`
- `docs/experiments/step5-gate3-triage.md`

Key reported results:

- Representation was strongly symbol-clustered rather than universal.
- Symbol-ID probe was high.
- Held-out/single-symbol probes did not produce a robust flat/PCA-beating edge.

Verdict:

The universal representation-learning thesis is closed for this dataset/framing.

### 3. Taker-side direction trading at realistic accuracy

Evidence:

- `docs/experiments/goal-a-feasibility/README.md`
- `docs/experiments/goal-a-feasibility/survivors.md`
- `docs/experiments/goal-a-feasibility/headroom_table.csv`

Key reported results:

- Perfect-direction H500 gross headroom exists at $1k/$10k across the universe.
- H10 is uniformly dead.
- Once realistic directional accuracy is imposed:
  - 55% accuracy: 0/300 survivor cells
  - 57.5% accuracy: 0/300 survivor cells
  - 60% accuracy: 0/300 strict survivor cells
- Closest near-miss:
  - PUMP $1k H500 at 60% accuracy
  - median headroom approximately +0.51 bp
  - frac-positive approximately 51.1%, below the 55% gate

Verdict:

The issue is not only fees or slippage. The issue is signal strength. The observed direction signal is much too weak to convert gross volatility/headroom into net profit.

### 4. Maker-side pivot

Evidence:

- `docs/experiments/goal-a-feasibility/maker_adverse_selection.md`
- `docs/experiments/goal-a-v2-program-end-state.md`

Key reported results:

- Maker fill-conditional adverse selection is severe.
- Reported E[realized | filled]: approximately -7.89 bp.
- 0/300 maker-style cells survived the tested economics.

Verdict:

The maker path is closed under the tested conditions. This is a venue/fill-conditional economics problem, not a better-model problem.

### 5. Cascade direction

Evidence:

- `docs/experiments/goal-a-feasibility/cascade_direction.md`

Key reported results:

- Cascade-onset prediction is real.
- Conditional cascade-direction model failed:
  - directional AUC approximately 0.441
  - confidence-threshold direction accuracy approximately 0.482
- Marginal-long variants were net-negative.

Verdict:

Cascade onset alone is not a trade. Direction remains the binding problem.

### 6. Pacifica-specific open-flow imbalance as standalone edge

Evidence:

- `docs/experiments/goal-a-feasibility/open_imbalance.md`
- `docs/experiments/goal-a-v2-program-end-state.md`

Key reported results:

- Extreme open-flow imbalance did not robustly beat standard OFI.
- Universe-level frac-positive was effectively around chance relative to the tradeability gate.

Verdict:

Pacifica-specific `is_open` / open-flow axes are useful for labeling/context, but not a standalone executable edge in the tested framing.

### 7. Encoder confidence gating

Evidence:

- `docs/experiments/goal-a-feasibility/encoder_confidence.md`

Key reported results:

- Encoder confidence was not a reliable tradeability selector.
- Top-confidence windows did not consistently identify profitable windows.

Verdict:

The encoder does not reliably "know when it knows."

### 8. More encoder / adapter work for cascade-onset

Evidence:

- `docs/experiments/goal-a-v2/random_init_probe_validator_report.md`
- `docs/experiments/goal-a-v2/cascade_adapter_validator_report.md`
- `docs/experiments/goal-a-v2-program-end-state.md`

Key reported results:

- Flat-LR cascade baseline: AUC approximately 0.8373.
- Random-init encoder linear probe: AUC approximately 0.6463, roughly -18.1pp versus flat-LR.
- Nonlinear cascade adapter: AUC approximately 0.6941, roughly -13.1pp versus flat-LR.

Verdict:

Architecture is not the bottleneck. Flat hand features beat the learned encoder stack for the strongest known signal.

---

## Why RL is not the next step

RL should not be used to rescue this program right now.

Reason:

RL optimizes a policy inside a simulator. The project has not established a simple profitable policy or a robust executable edge for RL to optimize. The binding failures are upstream:

1. direction signal is too weak,
2. cascade onset is directionless,
3. maker fills are adversely selected,
4. taker economics fail at realistic accuracy,
5. the consumed holdout prevents clean re-testing of many ad hoc variants.

Running RL now would likely optimize simulator artifacts, selection effects, or already-touched data rather than discover a durable trading edge.

Re-entry condition for RL:

Only consider RL after a simple non-RL policy has positive expected value under realistic costs and survives fresh untouched OOS validation. RL can then be used for sizing, inventory, execution, or risk control. It should not be used for first discovery of the edge.

---

## Final decision

Close the current trading-research line.

Do not spend more time on:

- new CNNs,
- representation-learning retries,
- cascade adapters,
- random-init probes,
- extra AUC hunting,
- maker-vs-taker pivots on the same labels,
- RL environments for the current signal set,
- broad paper-trading infrastructure before a new executable hypothesis exists.

The correct next artifact is this verdict, plus a clean repository state.

---

## What remains valuable

The project should be preserved because it contains a strong research result:

1. DEX perp tape has real predictive structure.
2. Cascade-onset prediction is learnable from flat microstructure features.
3. Learned encoders did not beat flat hand features.
4. Direction is the hard missing variable.
5. Fill-conditional maker economics are severely adverse-selected.
6. Realistic taker accuracy does not clear costs.
7. Predictive AUC and tradeable PnL are different problems.

This is a useful negative/partial-positive result, not a failed repo.

---

## Re-entry conditions

Only reopen the trading program if at least one of the following is true.

### 1. Fresh untouched data

Accumulate a genuinely untouched post-Apr-26 validation set large enough to support a clean OOS test.

Minimum spirit of the requirement:

- fresh data not used in hypothesis generation,
- enough events for statistical power,
- pre-registered hypothesis,
- pre-registered kill criteria,
- no iterative peeking.

### 2. Changed venue economics

Reopen if Pacifica or the execution venue changes materially:

- lower fees,
- rebates/tier improvements,
- protected execution mode,
- better visible/hidden liquidity,
- lower adverse selection,
- cross-venue routing that changes fill economics.

### 3. New executable label class

A new hypothesis must satisfy both conditions:

1. learnable from tape structure, and
2. executable with positive expected value after costs.

Examples worth considering only after a napkin-economics pass:

- cross-venue lead/lag execution,
- volatility/instability prediction rather than direction,
- post-cascade mean reversion,
- failed-cascade detection,
- liquidation absorption,
- liquidity-vacuum/refill states,
- inventory/risk-avoidance overlay for an already-profitable base strategy,
- cross-symbol contagion / leader-follower structure.

### 4. Existing profitable base strategy

If a separate base strategy exists, cascade-onset prediction may be useful as a risk filter rather than an alpha signal.

Example:

- reduce inventory before predicted cascade regimes,
- avoid maker quoting when adverse selection risk is high,
- widen spreads or disable fills in known toxic regimes.

This is a risk-management use case, not a standalone alpha use case.

---

## Recommended next human decision

Choose one of three paths:

### Path A — Archive

Accept closure. Keep the repo as a research archive. No more trading work unless a re-entry condition appears.

### Path B — Writeup

Turn the result into a research memo/blog/paper:

> DEX perp tape contains real cascade-onset signal, but realistic execution economics and direction uncertainty kill the naive trading strategy.

This is likely the highest-value output if the goal is intellectual/research value.

### Path C — New-hypothesis sprint

Spend one short, bounded session generating new executable hypotheses. For each candidate, require:

- exact label,
- exact instrument/venue to trade,
- fee/slippage model,
- minimum required accuracy/precision,
- fresh-OOS validation plan,
- kill criteria.

If no hypothesis passes that pre-flight economics gate, stop.

---

## One-line final verdict

Stop model experimentation. Preserve the work. Reopen only for fresh data, changed economics, or a genuinely new executable label/execution mechanism.

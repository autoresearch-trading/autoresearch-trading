---
name: health-check
description: Consistency audit for active full-fidelity Pacifica paper-trading docs, code, tests, and agent instructions.
---

# Health Check — active full-fidelity branch

Use this to find drift, contradictions, and stale references across the current project.

## Sources of truth

Read:

1. `CLAUDE.md`
2. `docs/NEXT_SESSION_HANDOFF.md`
3. `docs/AGENT_OPERATING_MAP.md`
4. `README.md`
5. `docs/ops/pacifica-full-fidelity-archival.md`
6. `docs/experiments/non-hft-regime-state/README.md`
7. `docs/experiments/toxic-regime-overlay/README.md`

## Checks

- Active direction is full-fidelity Pacifica non-HFT paper trading, not old 25-symbol representation learning.
- Symbol universe is dynamic from `/info`; no hard-coded 63/65/66 as configuration.
- Current data-sample caveats are explicit; 1-2 day probes are diagnostic only.
- Toxic thresholds are not tuned on diagnostic sample.
- Trading eligibility gates require liquidity, costs, sample size, stability, concentration, and post-cost economics.
- Agent files in `.claude/` do not silently point agents back to the old program.
- No credentials, local allowlists, tokens, SSH keys, or raw secrets are tracked.
- Tests exist and are run for changed scripts.

## Output

Write a concise PASS/FAIL report with exact files/lines for any drift.

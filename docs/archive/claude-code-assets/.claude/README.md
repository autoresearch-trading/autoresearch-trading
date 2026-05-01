# `.claude` assets — active branch status

This directory originally supported the old 25-symbol representation-learning program. The active repo direction is now the full-fidelity Pacifica non-HFT paper-trading program.

Use these files first:

1. `../CLAUDE.md`
2. `../docs/NEXT_SESSION_HANDOFF.md`
3. `../docs/AGENT_OPERATING_MAP.md`

## Current status

Active/re-targeted agents:

- `agents/lead-0.md`
- `agents/builder-8.md`
- `agents/reviewer-10.md`
- `agents/analyst-9.md`
- `agents/validator-11.md`
- `agents/researcher-14.md`

Legacy/optional agents:

- `agents/council-1.md` through `agents/council-6.md`
- `agents/runpod-7.md`
- `agents/prover-12.md`

Legacy/closed skills:

- `skills/autoresearch/` is historical unless rewritten for fresh full-fidelity data and economics-first paper trading.

Security:

- `settings.local.json` is ignored by git and must remain local-only. Do not copy credentials, local permission allowlists, SSH paths, or API tokens into tracked files.

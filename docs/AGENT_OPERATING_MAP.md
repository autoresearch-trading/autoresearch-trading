# Agent Operating Map — current Pacifica full-fidelity paper-trading branch

Last updated: 2026-05-01

This file maps the practical Hermes/tool/skill workflow for the current active program. It is not a codebase map. For code paths and pipeline state, read `docs/NEXT_SESSION_HANDOFF.md` first, then `AGENTS.md`.

## Active program

The active program is the economics-first, non-HFT Pacifica paper-trading system built on the new full-fidelity live public market-data archive.

Do not treat the old 25-symbol representation-learning setup as authoritative for current work. It remains historical context only. The active workflow is:

1. Keep full-fidelity collection running across all live public Pacifica symbols from `/info`.
2. Build raw archive → silver parquet → non-HFT regime-state tables.
3. Run diagnostic toxicity/no-trade overlays without tuning on tiny samples.
4. Add explicit tradeability and economics gates before paper trading any symbol.
5. Paper trade only eligible symbols, not the entire collected universe.

## Hermes / current assistant arsenal

The current Hermes workflow has broadly useful capabilities for this repo:

- File tools: read, search, write, and targeted patching.
- Terminal tools: run tests, git commands, scripts, collectors, validators, launchd checks, and process checks.
- Python execution: compact multi-step checks and repo audits.
- Skills: load procedural workflows such as `requesting-code-review`, `systematic-debugging`, `test-driven-development`, `writing-plans`, `deep-research`, and market/research skills when relevant.
- Delegation: spawn isolated subagents for parallel review/research/analysis, then verify their claims before reporting success.
- Session search: recover prior cross-session context when the user references earlier work.
- Cron/background jobs: schedule recurring checks only when explicitly useful and self-contained.
- Browser/web tools: use for current external facts or interactive web workflows; do not use from memory for live facts.

For this repository, the most useful Hermes patterns are:

- `requesting-code-review` before committing meaningful code changes.
- `systematic-debugging` for pipeline/data correctness bugs.
- `test-driven-development` for fixes like liquidation classification, eligibility gates, and backtester logic.
- `writing-plans` for multi-step implementation plans.
- `deep-research` / web research only when a decision needs external evidence.

## Repo-level agent instruction file

`AGENTS.md` is the canonical repo-level instruction file.

There is no active `CLAUDE.md`. Do not recreate it unless Diego explicitly decides to support Claude Code again.

Fresh-session reading order:

1. `docs/NEXT_SESSION_HANDOFF.md`
2. `AGENTS.md`
3. `docs/AGENT_OPERATING_MAP.md`

## Archived Claude Code assets

Claude Code is no longer used in this repo.

The old tracked `.claude` assets have been moved to:

- `docs/archive/claude-code-assets/.claude/README.md`
- `docs/archive/claude-code-assets/.claude/agents/`
- `docs/archive/claude-code-assets/.claude/skills/`

These files are historical reference only. Do not execute or revive them as active workflow without an explicit decision from Diego.

Local-only `.claude/settings.local.json` was not archived or committed because it may contain machine-specific permissions or sensitive values. If it still exists in the working tree, it is local-only and should remain out of git or be deleted manually.

## Security note

Do not copy local permission allowlists, shell history, SSH paths, API keys, OAuth tokens, or machine-specific credentials into tracked docs.

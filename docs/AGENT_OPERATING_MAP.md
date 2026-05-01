# Agent Operating Map — current Pacifica full-fidelity paper-trading branch

Last updated: 2026-04-30

This file maps the repo's agent assets and the practical tool/skill arsenal for the current active program. It is not a codebase map. For code paths and pipeline state, read `docs/NEXT_SESSION_HANDOFF.md` first, then `README.md` and `CLAUDE.md`.

## Active program

The active program is the economics-first, non-HFT Pacifica paper-trading system built on the new full-fidelity live public market-data archive.

Do not treat the old 25-symbol representation-learning agent setup as authoritative for current work. It remains useful historical context, but the active workflow is:

1. Keep full-fidelity collection running across all live public Pacifica symbols from `/info`.
2. Build raw archive → silver parquet → non-HFT regime-state tables.
3. Run diagnostic toxicity/no-trade overlays without tuning on tiny samples.
4. Add explicit tradeability and economics gates before paper trading any symbol.
5. Paper trade only eligible symbols, not the entire collected universe.

## Hermes / current assistant arsenal

The current Hermes session has broadly useful capabilities for this repo:

- File tools: read, search, write, and targeted patching.
- Terminal tools: run tests, git commands, scripts, collectors, validators, and process checks.
- Python execution: compact multi-step checks and repo audits.
- Skills: load procedural workflows such as `hermes-agent`, `requesting-code-review`, `systematic-debugging`, `test-driven-development`, `writing-plans`, `deep-research`, and market/research skills when relevant.
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

## Repo-local `.claude/agents` inventory

Active/re-targeted repo-local `.claude` agents:

- `lead-0`
- `builder-8`
- `reviewer-10`
- `analyst-9`
- `validator-11`
- `researcher-14`

Legacy/optional repo-local `.claude` agents:

- `council-1..6`
- `runpod-7`
- `prover-12`

| Agent | Intended role | Current status for active program |
|---|---|---|
| `lead-0` | Orchestrator | Retargeted for the active full-fidelity paper-trading program. |
| `builder-8` | Implementation/data-pipeline builder | Retargeted for collector/silver/regime/probe/eligibility/backtester work. |
| `reviewer-10` | Code reviewer | Retargeted for dynamic symbols, causality, costs, liquidation classification, tests, and security. |
| `analyst-9` | Statistical analysis/reporting | Retargeted to regime-state, toxicity, tradeability, costs, concentration, and post-cost diagnostics. |
| `validator-11` | Gate/pass-fail validation | Retargeted to economics/tradeability/sample-size/robustness gates, not old Gates 0-4. |
| `researcher-14` | Web/literature/code researcher | Retargeted to non-HFT microstructure, execution costs, toxicity, and regime filters. |
| `runpod-7` | GPU/RunPod operator | Marked legacy/optional unless GPU training returns. |
| `prover-12` | Lean/Aristotle theorem prover | Marked legacy/optional; not needed for current pipeline operations. |
| `council-1..6` | Expert review council | Marked legacy/optional; domain ideas remain useful but prompts should not override active docs. |

## Repo-local `.claude/skills` inventory

The repo contains project-specific Claude skills:

| Skill | Current status |
|---|---|
| `autoresearch` | Legacy/closed for Goal-A/v2. Do not use as active loop unless rewritten for fresh full-fidelity data and economics-first paper trading. |
| `research-first` | Reusable principle, but examples are representation-learning-oriented. |
| `experiment-eval` | Reusable principle. Should be rewritten around post-cost PnL, Sortino, drawdown, trade count, concentration, sample size, and causal thresholding. |
| `health-check` | Useful idea, stale source list. Should include `docs/NEXT_SESSION_HANDOFF.md`, active scripts, full-fidelity docs, and current experiment READMEs. |
| `compile-knowledge` | Reusable if the knowledge wiki remains active. |
| `runpodctl` / `flash` | Only relevant if GPU/RunPod/serverless work is resumed. |

## Should Hermes need its own repo config?

No separate Hermes config is required right now. The active repo-level guidance is already in `CLAUDE.md`, which Hermes receives as project context, and this file can act as the explicit operating map.

Avoid adding a separate `AGENTS.md` unless we want a tool-agnostic duplicate of `CLAUDE.md`. Duplicates can drift. If we do add one later, it should be short and simply point to:

1. `docs/NEXT_SESSION_HANDOFF.md`
2. `CLAUDE.md`
3. `docs/AGENT_OPERATING_MAP.md`

## Recommended updates before using `.claude` agents again

If the user wants Claude Code agents to actively work on this branch, the core agents have been retargeted. Remaining optional cleanup:

1. Expand specialized non-HFT versions of `council-*` if we need a new expert council.
2. Reactivate `runpod-7` only if GPU training returns.
3. Reactivate `prover-12` only for formal arithmetic/proof tasks.
4. Keep `.claude/settings.local.json` out of git. It is local-only and may contain secrets or machine-specific permissions.

## Security note

`.claude/settings.local.json` is ignored by git and should remain local-only. Do not copy local permission allowlists, shell history, SSH paths, API keys, OAuth tokens, or machine-specific credentials into tracked docs.

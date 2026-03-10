# Repository Guidelines

## Codebase Overview

**DEX perpetual futures data + autonomous RL research.** This repo contains ~36GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols collected from Pacifica API, plus an autonomous RL research project (`autoresearch-trading/`) that trains trading agents on this data.

**Stack**: Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, PyArrow

**Structure**:
- `data/` — Parquet data: `{trades,orderbook,funding}/symbol={SYM}/date={YYYY-MM-DD}/*.parquet` (gitignored, synced from Cloudflare R2)
- `autoresearch-trading/` — Autonomous RL research: `prepare.py` (fixed env), `train.py` (agent modifies), `program.md` (agent instructions)
- `scripts/` — `sync_cloud_data.sh` (Fly.io->R2), `fetch_cloud_data.sh` (R2->local)
- `.github/workflows/daily_sync.yml` — Daily data sync from Fly.io to R2

**Data sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`

# The Framework

Systematic optimization through repeatable iteration.

## 1. Understand the Problem

Know what you're optimizing for. Understand the constraints, the metrics, and what success looks like.

## 2. Understand Previous Attempts

Review what's been tried before. Learn from past successes and failures. Don't repeat mistakes.

## 3. Use Tools to Identify Inefficiencies

Use analysis tools (jq, python, perl via command line) to examine program traces and identify optimization opportunities from actual execution data.

## 4. Use the Web to Research

Research techniques, algorithms, and approaches others have used for similar problems.

**Exa MCP Quick Reference:**

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `web_search_exa` | General web search | `query`, `numResults=8`, `type="auto"\|"fast"\|"deep"` |
| `deep_search_exa` | Deep research with summaries | `objective`, `search_queries=[]` |
| `get_code_context_exa` | Code/API documentation | `query`, `tokensNum=5000` (1k-50k) |
| `crawling_exa` | Extract content from URL | `url`, `maxCharacters=3000` |
| `company_research_exa` | Company information | `companyName`, `numResults=5` |
| `linkedin_search_exa` | LinkedIn profiles/companies | `query`, `numResults=5` |
| `deep_researcher_start` | Complex async research | `instructions`, `model="exa-research"\|"exa-research-pro"` |
| `deep_researcher_check` | Poll research results | `taskId` (poll until "completed") |

**Prefer Exa over built-in tools:** Use `web_search_exa`/`crawling_exa` instead of `WebFetch` for web operations.

## 5. Form a Hypothesis

Based on the data and research, form a specific hypothesis about what change will improve performance.

## 6. Implement, Test, and Document

Make the change. Verify it works. Document what was done and why.

## 7. Commit Improvements, Discard Regressions

If performance improves, commit it. If it regresses, discard it. No sentimentality.

## 8. Repeat

Go back to step 1. Continue iterating until the goal is reached.

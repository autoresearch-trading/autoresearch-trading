# Repository Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strip the repo down to data + autoresearch-trading, removing all old signal-engine, collector, deployment, and CI infrastructure.

**Architecture:** Delete everything not related to the current autoresearch-trading work or the data pipeline (R2 sync). Update CLAUDE.md, .gitignore, and README to reflect the new lean repo. Keep `data/` (gitignored Parquet), `autoresearch-trading/`, data sync scripts, and the daily_sync workflow.

**Tech Stack:** Git, bash

---

### Task 1: Remove old signal-engine

The entire `signal-engine/` directory is the old Bytewax/QuestDB trading strategy. No longer needed.

**Files:**
- Delete: `signal-engine/` (entire directory)

**Step 1: Delete signal-engine**

```bash
rm -rf signal-engine/
```

**Step 2: Stage the deletion**

```bash
git add signal-engine/
```

**Step 3: Commit**

```bash
git commit -m "chore: remove signal-engine (old Bytewax/QuestDB strategy)"
```

---

### Task 2: Remove old collector, config, and tests

The `src/collector/`, `config/`, and `tests/` directories are the old data collection and configuration layer. The collector ran on Fly.io — we only need the R2 sync scripts now.

**Files:**
- Delete: `src/` (entire directory — only contains `collector/`)
- Delete: `config/` (entire directory — old Pydantic settings for API, trading, signals)
- Delete: `tests/` (entire directory — old collector tests)

**Step 1: Delete directories**

```bash
rm -rf src/ config/ tests/
```

**Step 2: Stage and commit**

```bash
git add src/ config/ tests/
git commit -m "chore: remove old collector, config, and tests"
```

---

### Task 3: Remove deployment infrastructure

The Dockerfiles, deploy configs, docker/, dashboards/, and monitoring/ are all for the old Fly.io collector deployment. No longer needed.

**Files:**
- Delete: `Dockerfile`
- Delete: `Dockerfile.arm64`
- Delete: `Dockerfile.cloud`
- Delete: `deploy/` (fly.toml, docker-compose.yml, railway.toml, render.yaml)
- Delete: `docker/`
- Delete: `dashboards/`
- Delete: `monitoring/`
- Delete: `setup-arm64.sh`

**Step 1: Delete all deployment files**

```bash
rm -f Dockerfile Dockerfile.arm64 Dockerfile.cloud setup-arm64.sh
rm -rf deploy/ docker/ dashboards/ monitoring/
```

**Step 2: Stage and commit**

```bash
git add Dockerfile Dockerfile.arm64 Dockerfile.cloud setup-arm64.sh deploy/ docker/ dashboards/ monitoring/
git commit -m "chore: remove deployment infrastructure (Dockerfiles, fly.toml, monitoring)"
```

---

### Task 4: Remove old build/lint config files

Old Python tooling config that was for the collector/signal-engine. The autoresearch-trading project has its own pyproject.toml.

**Files:**
- Delete: `requirements.txt` (root-level, old pip requirements)
- Delete: `Makefile`
- Delete: `Makefile.local`
- Delete: `mypy.ini`
- Delete: `.isort.cfg`
- Delete: `.cursorrules` (old Cursor AI rules)
- Delete: `.cursorignore`
- Delete: `.env.example`
- Delete: `.envrc.arm64`
- Delete: `symbols.json` (autoresearch-trading has DEFAULT_SYMBOLS in prepare.py)
- Delete: `symbols.txt`

**Step 1: Delete config files**

```bash
rm -f requirements.txt Makefile Makefile.local mypy.ini .isort.cfg .cursorrules .cursorignore .env.example .envrc.arm64 symbols.json symbols.txt
```

**Step 2: Stage and commit**

```bash
git add requirements.txt Makefile Makefile.local mypy.ini .isort.cfg .cursorrules .cursorignore .env.example .envrc.arm64 symbols.json symbols.txt
git commit -m "chore: remove old build/lint config files"
```

---

### Task 5: Remove old CI workflows

The old CI pipeline (`ci.yml`, `ci-simple.yml`) tests the collector and signal-engine with QuestDB, TA-Lib, etc. No longer relevant. Keep `daily_sync.yml` (syncs data from Fly.io to R2).

**Files:**
- Delete: `.github/workflows/ci.yml`
- Delete: `.github/workflows/ci-simple.yml`

**Step 1: Delete old CI**

```bash
rm -f .github/workflows/ci.yml .github/workflows/ci-simple.yml
```

**Step 2: Stage and commit**

```bash
git add .github/workflows/ci.yml .github/workflows/ci-simple.yml
git commit -m "chore: remove old CI workflows (keep daily_sync)"
```

---

### Task 6: Clean up old collection scripts

Remove collection scripts that were for the Fly.io collector. Keep `sync_cloud_data.sh` (used by daily_sync workflow) and `fetch_cloud_data.sh` (local data fetch from R2).

**Files:**
- Delete: `scripts/collect_all_symbols_cloud.py`
- Delete: `scripts/collect_all_symbols.py`
- Delete: `scripts/collect_data.py`
- Delete: `scripts/validate_cloud_dataset.py`
- Delete: `scripts/test_sync_local.sh`

**Step 1: Delete old collection scripts**

```bash
rm -f scripts/collect_all_symbols_cloud.py scripts/collect_all_symbols.py scripts/collect_data.py scripts/validate_cloud_dataset.py scripts/test_sync_local.sh
```

**Step 2: Stage and commit**

```bash
git add scripts/collect_all_symbols_cloud.py scripts/collect_all_symbols.py scripts/collect_data.py scripts/validate_cloud_dataset.py scripts/test_sync_local.sh
git commit -m "chore: remove old collection scripts (keep sync scripts)"
```

---

### Task 7: Clean up docs

Remove old architecture/operations docs that reference the deleted signal-engine and collector. Keep `docs/plans/` (our autoresearch plans) and `docs/CODEBASE_MAP.md` (will be updated).

**Files:**
- Delete: `docs/architecture/` (old PROJECT_STRUCTURE.md, SIGNAL_ENGINE_QUICKSTART.md, etc.)
- Delete: `docs/operations/` (old CLOUD_DATA_PIPELINE.md, DEPLOY_NOW.md, etc.)
- Delete: `docs/reports/` (old test reports)
- Delete: `docs/vision/` (old MASTER_IDEA.md, agent.md)
- Delete: `docs/archive/` (old CI fix summaries)
- Delete: `docs/README.md` (old docs index)
- Delete: `docs/CODEBASE_MAP.md` (references deleted code)
- Keep: `docs/plans/` (our autoresearch plans and research findings)

**Step 1: Delete old docs**

```bash
rm -rf docs/architecture/ docs/operations/ docs/reports/ docs/vision/ docs/archive/
rm -f docs/README.md docs/CODEBASE_MAP.md
```

**Step 2: Stage and commit**

```bash
git add docs/architecture/ docs/operations/ docs/reports/ docs/vision/ docs/archive/ docs/README.md docs/CODEBASE_MAP.md
git commit -m "chore: remove old docs (keep plans/)"
```

---

### Task 8: Clean up caches and junk files

Remove cached/generated files that shouldn't be in git.

**Files:**
- Delete: `.DS_Store`
- Delete: `.mypy_cache/` (if tracked)
- Delete: `.pytest_cache/` (if tracked)

**Step 1: Delete caches**

```bash
rm -rf .mypy_cache/ .pytest_cache/ .DS_Store
```

**Step 2: Update .gitignore to exclude .DS_Store globally**

Read `.gitignore` and verify `.DS_Store` is already listed. If not, add it.

**Step 3: Stage and commit**

```bash
git add .mypy_cache/ .pytest_cache/ .DS_Store .gitignore
git commit -m "chore: clean up cached/junk files"
```

---

### Task 9: Update CLAUDE.md

The current CLAUDE.md describes the old neurosymbolic trading bot architecture. Update it to reflect the new repo structure: data store + autoresearch-trading.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Replace CLAUDE.md content**

Replace the Codebase Overview section. Keep "The Framework" section (still applicable). Update to:

```markdown
# Repository Guidelines

## Codebase Overview

**DEX perpetual futures data + autonomous RL research.** This repo contains ~36GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols collected from Pacifica API, plus an autonomous RL research project (`autoresearch-trading/`) that trains trading agents on this data.

**Stack**: Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, PyArrow

**Structure**:
- `data/` — Parquet data: `{trades,orderbook,funding}/symbol={SYM}/date={YYYY-MM-DD}/*.parquet` (gitignored, synced from Cloudflare R2)
- `autoresearch-trading/` — Autonomous RL research: `prepare.py` (fixed env), `train.py` (agent modifies), `program.md` (agent instructions)
- `scripts/` — `sync_cloud_data.sh` (Fly.io→R2), `fetch_cloud_data.sh` (R2→local)
- `.github/workflows/daily_sync.yml` — Daily data sync from Fly.io to R2

**Data sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`
```

Keep everything from "# The Framework" onward unchanged.

**Step 2: Stage and commit**

```bash
git add CLAUDE.md
git commit -m "chore: update CLAUDE.md to reflect cleaned-up repo"
```

---

### Task 10: Update .gitignore and README

Simplify `.gitignore` to remove references to deleted tooling. Update or replace README.

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

**Step 1: Replace .gitignore**

```gitignore
# Data (synced from R2, too large for git)
data/
*.parquet

# Python
.venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# Autoresearch caches
autoresearch-trading/.cache/

# Environment
.env

# OS / Editor
.DS_Store
.vscode/
.idea/
*.swp
*~

# Runtime
*.log
logs/

# Git worktrees
.worktrees/
```

**Step 2: Replace README.md**

```markdown
# data-collector

DEX perpetual futures data store + autonomous RL trading research.

## Data

~36GB of Hive-partitioned Parquet data for 25 crypto symbols (2025-10-16 to 2026-03-09):

- **Trades**: `data/trades/symbol={SYM}/date={DATE}/*.parquet`
- **Orderbook**: `data/orderbook/symbol={SYM}/date={DATE}/*.parquet`
- **Funding**: `data/funding/symbol={SYM}/date={DATE}/*.parquet`

Sync from R2: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`

## Autoresearch Trading

Autonomous RL research for DEX perpetual futures. See `autoresearch-trading/program.md`.

Launch: `cd autoresearch-trading && claude --dangerously-skip-permissions -p "$(cat program.md)"`
```

**Step 3: Stage and commit**

```bash
git add .gitignore README.md
git commit -m "chore: update .gitignore and README for cleaned-up repo"
```

---

### Task 11: Verify clean state

**Step 1: Check nothing is broken**

```bash
git status
ls -la
```

Expected: Clean working tree. Only these top-level items remain:
- `.claude/`
- `.env`
- `.git/`
- `.github/`
- `.gitignore`
- `.worktrees/`
- `autoresearch-trading/`
- `CLAUDE.md`
- `data/` (symlinked or real, gitignored)
- `docs/plans/`
- `README.md`
- `scripts/` (only sync_cloud_data.sh and fetch_cloud_data.sh)

**Step 2: Verify autoresearch-trading still works**

```bash
cd autoresearch-trading
uv run python -c "from prepare import DEFAULT_SYMBOLS; print(f'{len(DEFAULT_SYMBOLS)} symbols loaded')"
```

Expected: `25 symbols loaded`

**Step 3: Check data symlink**

```bash
ls autoresearch-trading/data/
```

Expected: `funding  orderbook  trades`

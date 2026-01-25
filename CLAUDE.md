# Repository Guidelines

## Codebase Overview

A **neurosymbolic trading bot for DEX perpetuals** combining tape-reading signals (CVD, TFI, OFI) with ATR-based regime detection. The system spans async data collection from Pacifica API, Bytewax-powered signal processing, QuestDB persistence, event-driven backtesting, and paper trading with circuit breakers.

**Stack**: Python 3.12+, Bytewax (stream processing), QuestDB (time-series DB), Pydantic (config), Parquet (storage), TA-Lib (indicators)

**Structure**: Data collector (`src/collector/`) polls API → Parquet files → Signal engine (`signal-engine/src/`) computes signals → QuestDB → Backtest/Paper trading

For detailed architecture, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

## Project Structure & Module Organization
The pipeline is split across two Python packages. `src/collector/` contains the Pacifica REST client, async live runner, and storage adapters. Operational entry points live under `scripts/` (use `collect_data.py` locally, `collect_all_symbols_cloud.py` in production). Signal analytics are isolated in `signal-engine/src/`, with supporting scripts in `signal-engine/scripts/` and dedicated tests under `signal-engine/tests/`. Shared configuration lives in `config/`; infra manifests live in `deploy/` and `docker/`. Dashboards are in `dashboards/`; `tests/` targets collector flows.

## Build, Test, and Development Commands
Activate `.venv` (`python -m venv .venv && source .venv/bin/activate`). Core workflows are wrapped in the Makefile:

```bash
make install          # install collector + signal-engine deps
make test             # run all collector + signal-engine tests
make lint             # flake8 + black --check + isort --check + mypy
make collect-data     # launch live collector against local Parquet sink
make signal-pipeline  # dry-run signal pipeline with sample inputs
```

For Apple Silicon, use `./setup-arm64.sh` or `make -f Makefile.local install` to provision TA-Lib.

## Coding Style & Naming Conventions
Python 3.12+ (tested on 3.13) with 4-space indentation is standard. Keep modules and variables snake_case, classes in CapWords, and CLI subcommands hyphenated. Type hints are required for new public APIs; match the docstrings already present in `src/collector/`. Format with `black` (line length 88) and organize imports via `isort`. Run `flake8` and `mypy` locally before pushing.

## Testing Guidelines
Unit and integration suites use `pytest`. Place collector tests in `tests/` and signal-engine coverage in `signal-engine/tests/{unit,integration}`. Mirror existing naming (`test_<feature>.py`) and add fixtures under `tests/fixtures/` when shared state is needed. Run `make test`; for quick iteration call `pytest tests/test_transform.py -k <case>` or the narrower make targets (`test-unit`, `test-integration`). Add regression tests before modifying data transforms or Bytewax stages.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat:`, `fix:`, `chore:`). Scope commits narrowly and include updated docs or dashboards when behaviour changes. Pull requests must describe the motivation, list functional or test coverage, and link to tracking issues. Attach screenshots when dashboards change. Request review from a maintainer, ensure CI is green, and confirm configuration files remain free of secrets.

## Environment & Configuration Tips
Copy `.env.example` to `.env` and set `PACIFICA_NETWORK`, API base URLs, and credentials before running collectors. Use `deploy/docker-compose.yml` to boot QuestDB locally via `make docker-up`, and stop services with `make docker-down`. To inspect live Parquet output, run `streamlit run dashboards/app.py` while the collector is active.

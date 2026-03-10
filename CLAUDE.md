# Repository Guidelines

## Codebase Overview

**DEX perpetual futures data + autonomous RL research.** This repo contains ~36GB of Hive-partitioned Parquet data (trades, orderbook, funding) for 25 crypto symbols collected from Pacifica API, plus an autonomous RL research loop that trains trading agents on this data.

**Stack**: Python 3.12+, PyTorch, Gymnasium, NumPy, Pandas, PyArrow

**Structure**:
- `prepare.py` — Fixed: data loading, feature engineering, TradingEnv, evaluate()
- `train.py` — Mutable: agent rewrites this each experiment
- `program.md` — Agent instructions (Karpathy autoresearch pattern)
- `data/` — Parquet data: `{trades,orderbook,funding}/symbol={SYM}/date={YYYY-MM-DD}/*.parquet` (gitignored, synced from Cloudflare R2)
- `scripts/` — `sync_cloud_data.sh` (Fly.io->R2), `fetch_cloud_data.sh` (R2->local)
- `.github/workflows/daily_sync.yml` — Daily data sync from Fly.io to R2

**Data sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`

For detailed architecture, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).


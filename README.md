# autoresearch-trading

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

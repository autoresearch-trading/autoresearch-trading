# Data Collector

Bootstrap project for harvesting public market data from the [Pacifica REST API](https://docs.pacifica.fi/api-documentation/api).

## Setup

- Create the virtual environment (already prepared in `.venv/`):  
  `python3 -m venv .venv`
- Activate it:  
  `source .venv/bin/activate`
- Upgrade packaging tools:  
  `python -m pip install --upgrade pip setuptools`
- Install dependencies:  
  `pip install -r requirements.txt`

### Apple Silicon (M1/M2/M3)

If you're working on Apple Silicon, rebuild the environment with the native toolchain:

```bash
./setup-arm64.sh
```

The script installs TA-Lib via Homebrew, recreates `.venv/`, and verifies key imports using Apple Silicon wheels. Alternatively, run the Makefile overrides manually:

```bash
make -f Makefile.local install
source .envrc.arm64  # loads TA-Lib paths for the current shell
```

Avoid reusing Intel-built virtual environments—remove `.venv/` before installing dependencies on Apple Silicon to prevent `bad CPU type` errors.

## Configuration

1. Copy `.env.example` to `.env`.
2. Set `PACIFICA_NETWORK` to `mainnet` or `testnet`.  
   - Leave `API_BASE_URL` empty to follow the documented defaults (`https://api.pacifica.fi/api/v1` or `https://test-api.pacifica.fi/api/v1`).  
   - Provide `API_BASE_URL` if you need to target a custom environment.
3. Populate `PACIFICA_API_KEY` only if you plan to extend the collector with authenticated endpoints.

The CLI also accepts `--network`, `--base-url`, `--timeout`, and `--api-key` flags to override `.env` values at runtime.

## Usage

After activating the virtual environment, invoke the CLI subcommand that matches the data you need:

```bash
# Exchange metadata
python collect_data.py market-info

# Current pricing snapshot
python collect_data.py prices --out prices.json

# Candles (timestamps accept ms since epoch or ISO-8601)
python collect_data.py kline --symbol BTC --interval 1m \
  --start 2024-09-01T00:00:00Z --end 2024-09-01T01:00:00Z

# Order book snapshot with aggregation
python collect_data.py orderbook --symbol BTC --agg-level 5

# Recent trades
python collect_data.py recent-trades --symbol BTC

# Historical funding (paginated)
python collect_data.py funding --symbol BTC --limit 200 --offset 0

# Ad-hoc calls when experimenting with new endpoints
python collect_data.py raw /info/prices --param symbol=BTC

# Live polling to Parquet datasets (writes under ./data by default)
python collect_data.py live --symbols BTC,ETH --max-rps 4 \
  --poll-prices 2s --poll-trades 1s --poll-orderbook 3s --poll-funding 60s
```

Each command prints JSON to stdout or writes to `--out` when provided. Responses retain the `success` wrapper returned by Pacifica so you can inspect metadata alongside the data payload.

Pacifica currently enforces 300 REST credits per 60 seconds per IP address—monitor your request cadence when automating higher-frequency jobs.

Implementation details live in `src/collector/`, with `PacificaREST` exposing typed helpers for the documented endpoints. Extend this module (or add new subcommands) as Pacifica publishes additional data surfaces.

## Live Mode

The `live` subcommand launches an async polling runner that:

- multiplexes Pacifica REST endpoints with a global rate limiter and exponential-backoff retries,
- normalizes each payload via `pydantic` models,
- writes columnar Parquet files partitioned by `symbol=` and `date=` under the configured `--out-root`,
- emits JSON logs via `structlog` for easy ingestion.

Example:

```bash
python collect_data.py live \
  --symbols BTC,ETH \
  --out-root ./data \
  --max-rps 4 \
  --poll-prices 2s \
  --poll-trades 1s \
  --poll-orderbook 3s \
  --poll-funding 60s \
  --book-depth 25 \
  --agg-level 5
```

Intervals accept `ms`, `s`, or `m` units. Set any polling interval to `0` to disable that stream. The runner stops gracefully on `Ctrl+C`.

## Dashboard

Streamlit renders a lightweight dashboard that tails the Parquet datasets:

```bash
streamlit run dashboards/app.py
```

Use the sidebar to point at a different `data` root, adjust auto-refresh cadence, or narrow tracked symbols. The dashboard plots spot prices, recent trades, top-of-book depth, and funding rates with Plotly.

## Testing

Install the dev dependencies and execute the provided suite:

```bash
pip install -r requirements.txt
pytest
```

Tests cover the transformation helpers, Parquet writer, and a mocked live runner loop to guard against regressions in payload handling and scheduling.

# Cloud Data Retrieval & Validation

Use this playbook after the daily Fly ➜ Cloudflare R2 sync has been running for a few days. It explains how to pull the archived parquet datasets down to your workstation, validate their structure, and confirm the data is ready for backtesting.

## Prerequisites

- AWS CLI installed locally (`aws --version`).
- Environment variables exported:
  - `S3_BUCKET_NAME` – the Cloudflare R2 bucket synced by `scripts/sync_cloud_data.sh`.
  - `S3_ENDPOINT_URL` – the R2 endpoint (omit for native S3).
- Python dependencies are already available via `requirements.txt` (pyarrow, numpy, etc.).

## 1. Sync the archive from Cloudflare R2

```bash
# Make the helper executable once
chmod +x scripts/fetch_cloud_data.sh

# Pull the full archive (defaults to ./cloud-data)
./scripts/fetch_cloud_data.sh ./cloud-data
```

This mirrors the bucket layout locally, preserving the `symbol=*/date=*/*.parquet` structure under `cloud-data/{trades,orderbook,prices,funding}`.

## 2. Validate schema, ordering, and coverage

```bash
python scripts/validate_cloud_dataset.py --data-root ./cloud-data
```

Key checks performed:
- Confirms each dataset includes the expected columns (`ts_ms`, `symbol`, etc.).
- Aggregates row counts and reports the min/max timestamps per dataset.
- Ensures timestamps inside each parquet file are sorted.
- Flags duplicate `trade_id` values.
- Summarises coverage by symbol and trading day.

Use `--max-files 200` when you want a quicker sample scan.

## 3. Point the signal engine at the synced data

Once validation passes, you can backtest or replay signals using the synced directory:

```bash
# Example: run the signal pipeline in dry-run mode against the downloaded data
cd signal-engine
python scripts/run_signal_pipeline.py \
  --symbols BTC ETH SOL \
  --date 2025-10-20 \
  --dry-run \
  --skip-regime \
  --checkpoint-dir ../cloud-data/checkpoints \
  --data-root ../cloud-data
```

Adjust `--data-root` or update your `.env` to point the engine at `../cloud-data` instead of the default `../data` tree.

## Troubleshooting

- **Bucket credentials errors**: confirm `S3_BUCKET_NAME`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are exported in your shell.
- **Missing datasets**: if validation reports "No files found", verify the Cloudflare sync pipeline (`scripts/sync_cloud_data.sh`) completed successfully.
- **Timestamp ordering issues**: rerun the collector or regenerate the affected parquet files before backtesting. Backtests assume monotonic `ts_ms`.

Keep this workflow handy whenever you want to refresh your local data lake from Cloudflare and QA the datasets before analysis or backtesting.

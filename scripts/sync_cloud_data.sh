#!/bin/bash
# Triggers data sync + purge on the Fly.io instance.
# The heavy lifting (S3 upload) runs server-side — GHA is just the trigger.
#
# R2 credentials are stored as Fly secrets, not GHA secrets.
# boto3 must be installed on the Fly container.

set -euo pipefail

APP_NAME="pacifica-collector"
REMOTE_DATA_PATH="/app/data"
DAYS_TO_KEEP=2

echo "▶️ Triggering sync on Fly.io app: $APP_NAME"

flyctl ssh console -q -a "$APP_NAME" -C sh <<REMOTE_SCRIPT
set -e

echo "📦 Syncing today's parquet files to R2..."
python3 - <<'PYTHON'
import os, boto3, pathlib, time

data_dir = pathlib.Path("${REMOTE_DATA_PATH}")
bucket = os.environ["S3_BUCKET_NAME"]
endpoint = os.environ["S3_ENDPOINT_URL"]

s3 = boto3.client("s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

cutoff = time.time() - 86400  # last 24h
uploaded = 0
for f in data_dir.rglob("*.parquet"):
    if f.stat().st_mtime < cutoff:
        continue
    key = str(f.relative_to(data_dir))
    s3.upload_file(str(f), bucket, key)
    uploaded += 1

print(f"✅ Uploaded {uploaded} files to s3://{bucket}")
PYTHON

echo "🗑️ Purging files older than ${DAYS_TO_KEEP} days..."
find ${REMOTE_DATA_PATH} -type f -name '*.parquet' -mtime +$((${DAYS_TO_KEEP} - 1)) -print -delete
echo "✅ Purge complete."
exit 0
REMOTE_SCRIPT

echo "🎉 Sync & Purge finished."

#!/bin/bash
# Syncs data from Fly.io to cloud object storage, then purges old data on Fly.

set -euo pipefail

# --- Configuration ---
APP_NAME="pacifica-collector"
LOCAL_TEMP_DIR="/tmp/fly-sync-data"
REMOTE_DATA_PATH="/app/data"
DAYS_TO_KEEP_ON_FLY=2 # Keep today's and yesterday's data

# --- S3/R2 Configuration (from environment) ---
S3_BUCKET="${S3_BUCKET_NAME}"
# S3_ENDPOINT_URL is only needed for S3-compatible services like Cloudflare R2
S3_ARGS=""
if [ -n "${S3_ENDPOINT_URL-}" ]; then
  S3_ARGS="--endpoint-url ${S3_ENDPOINT_URL}"
fi

# --- Main Logic ---
echo "▶️ Starting data sync for Fly.io app: $APP_NAME"

# 1. Create local temporary directory
rm -rf "$LOCAL_TEMP_DIR"
mkdir -p "$LOCAL_TEMP_DIR"
echo "✅ Created temporary directory: $LOCAL_TEMP_DIR"

# 2. Archive data on the Fly.io instance
ARCHIVE_FILENAME="data-backup-$(date +%Y-%m-%d).tar.gz"
echo "📦 Archiving data on Fly.io instance..."
flyctl ssh console -q -a "$APP_NAME" -C "tar --ignore-failed-read -czf /tmp/${ARCHIVE_FILENAME} -C ${REMOTE_DATA_PATH} ."
echo "✅ Remote archive created: /tmp/${ARCHIVE_FILENAME}"

# 3. Download the archive
echo "⬇️ Downloading archive from Fly.io..."
flyctl ssh sftp get "/tmp/${ARCHIVE_FILENAME}" "${LOCAL_TEMP_DIR}/${ARCHIVE_FILENAME}" -a "$APP_NAME"
echo "✅ Archive downloaded."

# 4. Extract and upload to cloud storage, preserving structure
echo "☁️ Uploading data to bucket: s3://${S3_BUCKET}"
mkdir -p "${LOCAL_TEMP_DIR}/extracted"
tar -xzf "${LOCAL_TEMP_DIR}/${ARCHIVE_FILENAME}" -C "${LOCAL_TEMP_DIR}/extracted"

# Count files to upload
FILE_COUNT=$(find "${LOCAL_TEMP_DIR}/extracted" -type f -name "*.parquet" | wc -l)
echo "📊 Found ${FILE_COUNT} parquet files to upload"

# The `aws s3 sync` command is smart and efficient
aws s3 sync "${LOCAL_TEMP_DIR}/extracted" "s3://${S3_BUCKET}" ${S3_ARGS} --only-show-errors
echo "✅ Upload complete."

# 5. Purge old data on the Fly.io instance
echo "🗑️ Purging data older than ${DAYS_TO_KEEP_ON_FLY} days on Fly.io volume..."
# IMPORTANT: The `find` command deletes files. `-mtime +1` means older than 2 days ago (48h).
PURGE_CMD="find ${REMOTE_DATA_PATH} -type f -name '*.parquet' -mtime +$((DAYS_TO_KEEP_ON_FLY - 1)) -print -delete"

# First, run a dry-run to see what would be deleted
echo "🔍 Dry run of purge command:"
flyctl ssh console -q -a "$APP_NAME" -C "find ${REMOTE_DATA_PATH} -type f -name '*.parquet' -mtime +$((DAYS_TO_KEEP_ON_FLY - 1)) -print"

# Then, execute the actual purge
flyctl ssh console -q -a "$APP_NAME" -C "${PURGE_CMD}"
echo "✅ Purge command executed."

# 6. Cleanup local files
rm -rf "$LOCAL_TEMP_DIR"
echo "🧹 Local cleanup complete."

echo "🎉 Sync & Purge process finished successfully!"
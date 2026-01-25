#!/bin/bash
# Syncs data from Fly.io to cloud object storage, then purges old data on Fly.
#
# ARCHITECTURE: Streams tar directly from Fly.io to local machine to avoid
# filling up limited /tmp space on the remote instance.

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
mkdir -p "$LOCAL_TEMP_DIR/extracted"
echo "✅ Created temporary directory: $LOCAL_TEMP_DIR"

# 2. Stream tar directly from Fly.io (no remote temp file needed)
# This avoids "No space left on device" on Fly.io's limited /tmp
echo "📦 Streaming data from Fly.io instance..."
flyctl ssh console -q -a "$APP_NAME" -C "tar --ignore-failed-read -cf - -C ${REMOTE_DATA_PATH} ." | tar -xf - -C "${LOCAL_TEMP_DIR}/extracted"
echo "✅ Data streamed and extracted locally."

# 3. Count files to upload
FILE_COUNT=$(find "${LOCAL_TEMP_DIR}/extracted" -type f -name "*.parquet" | wc -l)
echo "📊 Found ${FILE_COUNT} parquet files to upload"

# 4. Upload to cloud storage, preserving structure
echo "☁️ Uploading data to bucket: s3://${S3_BUCKET}"
# The `aws s3 sync` command is smart and efficient (only uploads changed files)
aws s3 sync "${LOCAL_TEMP_DIR}/extracted" "s3://${S3_BUCKET}" ${S3_ARGS} --only-show-errors
echo "✅ Upload complete."

# 5. Purge old data on the Fly.io instance
echo "🗑️ Purging data older than ${DAYS_TO_KEEP_ON_FLY} days on Fly.io volume..."
# IMPORTANT: The `find` command deletes files. `-mtime +1` means older than 2 days ago (48h).
PURGE_CMD="find ${REMOTE_DATA_PATH} -type f -name '*.parquet' -mtime +$((DAYS_TO_KEEP_ON_FLY - 1)) -print -delete"

# First, run a dry-run to see what would be deleted
echo "🔍 Dry run of purge command:"
flyctl ssh console -q -a "$APP_NAME" -C "find ${REMOTE_DATA_PATH} -type f -name '*.parquet' -mtime +$((DAYS_TO_KEEP_ON_FLY - 1)) -print" || true

# Then, execute the actual purge
flyctl ssh console -q -a "$APP_NAME" -C "${PURGE_CMD}" || true
echo "✅ Purge command executed."

# 6. Cleanup remote /tmp (in case previous runs left partial archives)
echo "🧹 Cleaning up remote /tmp..."
flyctl ssh console -q -a "$APP_NAME" -C "rm -f /tmp/data-backup-*.tar.gz" || true

# 7. Cleanup local files
rm -rf "$LOCAL_TEMP_DIR"
echo "🧹 Local cleanup complete."

echo "🎉 Sync & Purge process finished successfully!"
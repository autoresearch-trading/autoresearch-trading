#!/bin/bash
# Triggers data sync + purge on the Fly.io instance.
# GHA is just the trigger — all heavy lifting runs server-side.
#
# Syncs one date at a time to keep each SSH session under ~20min
# (flyctl SSH drops after ~35min, and each date has ~30K files).
#
# R2 credentials are stored as Fly secrets, not GHA secrets.
# boto3 must be installed on the Fly container.

set -euo pipefail

APP_NAME="pacifica-collector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_DAYS=2

echo "▶️ Ensuring boto3 is installed on Fly.io..."
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "pip install -q boto3"

echo "▶️ Uploading sync script to Fly.io..."
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "rm -f /tmp/sync.py"
flyctl ssh sftp put -q -a "$APP_NAME" "$SCRIPT_DIR/sync_remote.py" /tmp/sync.py

# Purge FIRST — cleans accumulated old data (already synced in
# previous successful runs), keeping the volume lean for sync.
echo "▶️ Running purge..."
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "python3 /tmp/sync.py --purge-only"

# Generate recent dates (today, yesterday)
DATES=$(python3 -c "
import datetime
for i in range($SYNC_DAYS):
    print((datetime.date.today() - datetime.timedelta(days=i)).isoformat())
")

# Sync each date in a separate SSH session to avoid timeout.
# Retry on SSH disconnects (flyctl ssh occasionally exits with -1
# mid-session). sync_remote.py is idempotent — files already in R2
# are skipped by key+size — so retries are safe.
for date in $DATES; do
  echo "▶️ Syncing date=$date ..."
  for attempt in 1 2 3; do
    if flyctl ssh console -q --pty=false -a "$APP_NAME" -C "python3 /tmp/sync.py $date"; then
      break
    fi
    if [ "$attempt" = "3" ]; then
      echo "❌ Failed after 3 attempts for date=$date"
      exit 1
    fi
    backoff=$((attempt * 30))
    echo "⚠️  attempt $attempt failed (likely SSH drop), retrying in ${backoff}s..."
    sleep "$backoff"
  done
done

echo "🎉 Sync & Purge finished."

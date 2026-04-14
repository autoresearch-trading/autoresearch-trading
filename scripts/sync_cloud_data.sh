#!/bin/bash
# Triggers data sync + purge on the Fly.io instance.
# GHA is just the trigger — all heavy lifting runs server-side.
#
# Each date is launched detached via nohup so flyctl SSH drops
# (which happen around ~30min) cannot kill the upload. We then
# poll a small status file via short SSH calls until done.
#
# R2 credentials are stored as Fly secrets, not GHA secrets.
# boto3 must be installed on the Fly container.

set -euo pipefail

APP_NAME="pacifica-collector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_DAYS=2
POLL_INTERVAL=60
POLL_TIMEOUT=5400  # 90 min per date

ssh_exec() {
  flyctl ssh console -q --pty=false -a "$APP_NAME" -C "$1"
}

echo "▶️ Ensuring boto3 is installed on Fly.io..."
ssh_exec "pip install -q boto3"

echo "▶️ Uploading sync script to Fly.io..."
ssh_exec "rm -f /tmp/sync.py"
flyctl ssh sftp put -q -a "$APP_NAME" "$SCRIPT_DIR/sync_remote.py" /tmp/sync.py

# Purge FIRST — cleans accumulated old data (already synced in
# previous successful runs), keeping the volume lean for sync.
echo "▶️ Running purge..."
ssh_exec "python3 /tmp/sync.py --purge-only"

DATES=$(python3 -c "
import datetime
for i in range($SYNC_DAYS):
    print((datetime.date.today() - datetime.timedelta(days=i)).isoformat())
")

for date in $DATES; do
  echo "▶️ Launching detached sync for date=$date ..."
  log="/tmp/sync_${date}.log"
  status="/tmp/sync_${date}.status"
  ssh_exec "rm -f $status $log; nohup sh -c 'python3 /tmp/sync.py $date > $log 2>&1; echo \$? > $status' >/dev/null 2>&1 &"

  echo "⏳ Polling $status (every ${POLL_INTERVAL}s, max ${POLL_TIMEOUT}s)..."
  elapsed=0
  exit_code=""
  while [ "$elapsed" -lt "$POLL_TIMEOUT" ]; do
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
    if exit_code=$(ssh_exec "cat $status 2>/dev/null" 2>/dev/null); then
      if [ -n "$exit_code" ]; then
        break
      fi
    fi
    # Tail recent progress to keep GHA logs informative.
    ssh_exec "tail -n 3 $log 2>/dev/null" || true
  done

  echo "📜 Final log for date=$date:"
  ssh_exec "tail -n 20 $log" || true

  if [ -z "$exit_code" ]; then
    echo "❌ Timed out after ${POLL_TIMEOUT}s for date=$date"
    exit 1
  fi
  if [ "$exit_code" != "0" ]; then
    echo "❌ Sync exited with code $exit_code for date=$date"
    exit 1
  fi
  echo "✅ date=$date complete."
done

echo "🎉 Sync & Purge finished."

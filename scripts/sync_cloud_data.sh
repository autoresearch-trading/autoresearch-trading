#!/bin/bash
# Triggers data sync + purge on the Fly.io instance.
# GHA is just the trigger — all heavy lifting runs server-side.
#
# R2 credentials are stored as Fly secrets, not GHA secrets.
# boto3 must be installed on the Fly container.

set -euo pipefail

APP_NAME="pacifica-collector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "▶️ Ensuring boto3 is installed on Fly.io..."
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "pip install -q boto3"

echo "▶️ Uploading sync script to Fly.io..."
# sftp won't overwrite, so remove first
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "rm -f /tmp/sync.py"
flyctl ssh sftp put -q -a "$APP_NAME" "$SCRIPT_DIR/sync_remote.py" /tmp/sync.py

echo "▶️ Running sync on Fly.io..."
flyctl ssh console -q --pty=false -a "$APP_NAME" -C "python3 /tmp/sync.py"

echo "🎉 Sync & Purge finished."

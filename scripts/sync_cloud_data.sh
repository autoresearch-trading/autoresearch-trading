#!/bin/bash
# Sync data from Fly.io to local machine

set -e

LOCAL_DATA_DIR="./data"
TEMP_DIR="/tmp/fly-data"
APP_NAME="pacifica-collector"

echo "📥 Syncing data from Fly.io..."

# Create temp directory
mkdir -p "$TEMP_DIR"

# SSH into Fly and create tar archive (using a more reliable approach)
echo "📦 Creating archive on Fly.io..."
flyctl ssh console --app "$APP_NAME" << 'ENDSSH'
cd /app/data
if [ -d "candles" ]; then
  tar czf /tmp/data-backup.tar.gz trades orderbook prices funding candles
else
  tar czf /tmp/data-backup.tar.gz trades orderbook prices funding
fi
exit
ENDSSH

# Download the archive
echo "⬇️  Downloading archive..."
flyctl ssh sftp get /tmp/data-backup.tar.gz "$TEMP_DIR/data-backup.tar.gz" --app "$APP_NAME"

# Extract locally
echo "📦 Extracting data..."
mkdir -p "$LOCAL_DATA_DIR"
tar xzf "$TEMP_DIR/data-backup.tar.gz" -C "$LOCAL_DATA_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

# Show stats
echo ""
echo "✅ Sync complete!"
echo "Data available in: $LOCAL_DATA_DIR"
echo ""
echo "📊 Data summary:"
du -sh "$LOCAL_DATA_DIR"/*/ 2>/dev/null || echo "No data directories found"


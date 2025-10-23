# 🔄 Fly.io Data Sync & Cleanup Guide

Complete guide for syncing data from Fly.io to local storage and cleaning up the cloud instance.

## 📋 Table of Contents

- [Overview](#overview)
- [When to Sync & Cleanup](#when-to-sync--cleanup)
- [Prerequisites](#prerequisites)
- [Step-by-Step Process](#step-by-step-process)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Automation](#automation)

---

## Overview

The Fly.io free tier has limited storage (3GB) and **inode limits (~195K files)**. With 25+ symbols collecting tick data, you'll hit the inode limit every 1-2 days. This guide shows how to:

1. **Sync data** from Fly.io to local storage
2. **Clean up** old data on Fly.io to free space
3. **Verify** both local and remote data integrity

---

## When to Sync & Cleanup

### Sync Frequency

Run sync when:
- ✅ Every 1-2 days (before hitting inode limit)
- ✅ Before backtesting with fresh data
- ✅ When collector health shows low space
- ✅ After significant market events you want to analyze

### Warning Signs

🚨 **Need immediate cleanup if you see:**
- `[Errno 28] No space left on device` in logs
- Inode usage > 90% (`df -i` shows high IUse%)
- Collector health reports no recent files
- Data collection stopped

Check status:
```bash
./monitoring/check_collector.sh
```

---

## Prerequisites

1. **Fly CLI installed and authenticated**
   ```bash
   flyctl auth whoami  # Verify you're logged in
   ```

2. **Local data directory exists**
   ```bash
   cd /Users/diego/Dev/data-collector
   ls -la data/  # Should see: trades, orderbook, prices, funding, candles
   ```

3. **App name configured**
   - Default: `pacifica-collector`
   - Change below if using different name

---

## Step-by-Step Process

### Part 1: Sync Data to Local

#### Step 1: Archive Data on Fly.io

```bash
cd /Users/diego/Dev/data-collector

# Create compressed archive of all data
flyctl ssh console -a pacifica-collector -C 'tar -czf /tmp/data-sync.tar.gz -C /app/data .'
```

**What this does:**
- Creates compressed archive in `/tmp/` on Fly.io
- Includes all data: trades, orderbook, prices, funding
- Takes ~30-60 seconds depending on data size

#### Step 2: Download Archive

```bash
# Download to local machine (~10 MB = 1-2 days of data)
flyctl ssh sftp get "/tmp/data-sync.tar.gz" "./data-sync.tar.gz" -a pacifica-collector
```

**Expected output:**
```
9612930 bytes written to ./data-sync.tar.gz
```

**Typical sizes:**
- 1 day: ~5-10 MB
- 2 days: ~10-20 MB
- 1 week: ~50-70 MB

#### Step 3: Extract to Local Data Folder

```bash
# Extract archive (preserves directory structure)
tar -xzf ./data-sync.tar.gz -C ./data/

# Verify extraction
echo "✅ Extraction complete"
```

**What gets extracted:**
```
data/
├── trades/
│   ├── symbol=BTC/
│   │   ├── date=2025-10-16/
│   │   └── date=2025-10-17/
│   └── symbol=ETH/...
├── orderbook/
├── funding/
└── prices/
```

#### Step 4: Verify Local Data

```bash
cd data

# Check sizes
du -sh trades orderbook prices funding candles

# Count files
for dir in trades orderbook prices funding; do 
  echo "$dir: $(find $dir -type f -name '*.parquet' 2>/dev/null | wc -l | xargs) files"
done
```

**Expected results:**
```
trades:     17M  →  2,233 files
orderbook:  17M  →  2,199 files
prices:      0M  →      0 files
funding:   309M  → 79,127 files
candles:    47M  →  1,196 files (preserved)
```

#### Step 5: Cleanup Local Archive

```bash
cd /Users/diego/Dev/data-collector
rm -f data-sync.tar.gz
echo "🧹 Cleaned up local archive"
```

---

### Part 2: Clean Up Fly.io

#### Step 1: Delete Old Data on Fly.io

Delete everything except today's data:

```bash
# Delete old date partitions (keeps only today)
flyctl ssh console -a pacifica-collector -C 'rm -rf /app/data/trades/*/date=2025-10-16 /app/data/orderbook/*/date=2025-10-16 /app/data/funding/*/date=2025-10-16 2>/dev/null; rm -f /tmp/data-sync.tar.gz; echo "Deleted old data and archive"'
```

**⚠️ Important:**
- Replace `2025-10-16` with the date(s) you want to delete
- Always keep today's date (`date=2025-10-17` or current)
- This frees ~112K files in one command

**Safer alternative (see what will be deleted first):**
```bash
# Dry run - see what would be deleted
flyctl ssh console -a pacifica-collector -C 'find /app/data -type d -name "date=2025-10-16"'

# Then delete if it looks correct
flyctl ssh console -a pacifica-collector -C 'find /app/data -type d -name "date=2025-10-16" -exec rm -rf {} +'
```

#### Step 2: Verify Cleanup

```bash
# Check disk and inode usage
flyctl ssh console -a pacifica-collector -C 'df -h /app/data'
flyctl ssh console -a pacifica-collector -C 'df -i /app/data'
```

**Good cleanup results:**
```
Disk Usage:   13-20% (was 40-42%)
Inode Usage:  40-45% (was 100%) ✅
```

#### Step 3: Check Remaining Data

```bash
# See what dates remain
flyctl ssh console -a pacifica-collector -C 'ls /app/data/trades/symbol=BTC/'

# Should show only: date=2025-10-17 (today)
```

#### Step 4: Verify Collector Health

```bash
# Health check endpoint
flyctl ssh console -a pacifica-collector -C 'curl -s localhost:8080/health'
```

**Expected healthy response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-17T12:00:00.000000",
  "data_dir_exists": true,
  "recent_files_count": 25,
  "healthy": true
}
```

**Key indicators:**
- ✅ `status: "healthy"`
- ✅ `recent_files_count > 0`
- ✅ `healthy: true`

#### Step 5: Check Logs (Optional)

```bash
# View recent logs (should show no errors)
flyctl logs -a pacifica-collector --no-tail 2>&1 | tail -30

# Look for:
# ✅ "HTTP Request: GET .../trades" (collecting data)
# ✅ No "[Errno 28] No space left on device"
# ✅ Regular polling activity
```

---

## Verification

### Local Verification Checklist

- [ ] Archive downloaded successfully
- [ ] Data extracted to `./data/` directories
- [ ] File counts match expected (~84K+ total files for 2 days)
- [ ] Candles directory unchanged (~1,196 files)
- [ ] Local archive deleted

### Fly.io Verification Checklist

- [ ] Inode usage < 50% (was 100%)
- [ ] Only today's date partition remains
- [ ] Collector health returns `"healthy": true`
- [ ] Recent files count > 0
- [ ] No space errors in logs
- [ ] Archive cleaned from `/tmp/`

---

## Troubleshooting

### Issue: "No space left on device" during archive creation

**Cause:** Disk is full

**Solution:**
```bash
# Clean up first, then archive
flyctl ssh console -a pacifica-collector -C 'rm -rf /app/data/*/date=2025-10-16'

# Then try archive again
flyctl ssh console -a pacifica-collector -C 'tar -czf /tmp/data-sync.tar.gz -C /app/data .'
```

### Issue: Archive download is very slow

**Cause:** Large archive size (>50MB)

**Solution:**
- Be patient (can take 2-5 minutes for large archives)
- Or sync more frequently to keep archives smaller
- Or exclude funding data if not needed:
  ```bash
  tar -czf /tmp/data-sync.tar.gz -C /app/data trades orderbook prices
  ```

### Issue: Collector stopped writing after cleanup

**Cause:** Deleted current date accidentally

**Check:**
```bash
flyctl ssh console -a pacifica-collector -C 'ls /app/data/trades/symbol=BTC/'
# Should show today's date
```

**Solution:**
```bash
# Restart the collector
flyctl apps restart pacifica-collector
```

### Issue: Local data looks incomplete

**Verify:**
```bash
# Check if extraction worked
cd data
find trades -name "*.parquet" | head -20

# Re-extract if needed
tar -xzf ../data-sync.tar.gz -C .
```

### Issue: Inode usage still high after cleanup

**Check what's using space:**
```bash
flyctl ssh console -a pacifica-collector -C 'find /app/data -type d -name "date=*"'

# Delete additional old dates if found
flyctl ssh console -a pacifica-collector -C 'rm -rf /app/data/*/date=2025-10-15'
```

---

## Automation

### Option 1: Manual Script (Recommended)

Create `sync_and_clean.sh`:

```bash
#!/bin/bash
set -euo pipefail

APP_NAME="pacifica-collector"
DATE_TO_DELETE=$(date -v-1d +%Y-%m-%d)  # Yesterday
ARCHIVE_NAME="data-sync-$(date +%Y%m%d).tar.gz"

echo "📦 Starting sync and cleanup..."
echo "Will delete data from: $DATE_TO_DELETE"

# 1. Archive
echo "Creating archive on Fly.io..."
flyctl ssh console -a "$APP_NAME" -C "tar -czf /tmp/$ARCHIVE_NAME -C /app/data ."

# 2. Download
echo "Downloading..."
flyctl ssh sftp get "/tmp/$ARCHIVE_NAME" "./$ARCHIVE_NAME" -a "$APP_NAME"

# 3. Extract
echo "Extracting..."
tar -xzf "./$ARCHIVE_NAME" -C ./data/

# 4. Verify
echo "Verifying..."
cd data && du -sh trades orderbook funding

# 5. Cleanup local
cd ..
rm -f "$ARCHIVE_NAME"

# 6. Cleanup remote
echo "Cleaning up Fly.io..."
flyctl ssh console -a "$APP_NAME" -C "rm -rf /app/data/*/date=$DATE_TO_DELETE /tmp/$ARCHIVE_NAME"

# 7. Verify
echo "Checking health..."
flyctl ssh console -a "$APP_NAME" -C 'curl -s localhost:8080/health' | python3 -m json.tool

echo "✅ Sync and cleanup complete!"
```

Make executable:
```bash
chmod +x sync_and_clean.sh
./sync_and_clean.sh
```

### Option 2: Cron Job (Automated)

Add to crontab for daily 2 AM sync:

```bash
crontab -e

# Add this line:
0 2 * * * cd /Users/diego/Dev/data-collector && ./sync_and_clean.sh >> /tmp/sync-log.txt 2>&1
```

### Option 3: GitHub Actions (see daily_sync.yml)

Already configured in `.github/workflows/daily_sync.yml` - runs daily via GitHub Actions.

---

## Best Practices

1. **Sync Frequency:**
   - Daily or every 2 days
   - Before inode usage hits 90%
   - Before important backtests

2. **Data Retention on Fly.io:**
   - Keep only 1-2 days on Fly.io
   - Store historical data locally
   - Archive to S3/R2 for long-term storage

3. **Local Data Management:**
   - Keep candles directory separate
   - Organize by date for easy deletion
   - Consider compressing old local data

4. **Monitoring:**
   - Check collector health daily
   - Monitor inode usage weekly
   - Set up alerts for space issues

5. **Before Backtesting:**
   - Always sync fresh data first
   - Verify data completeness
   - Check for gaps in timestamps

---

## Quick Reference Commands

```bash
# Check Fly.io status
flyctl status -a pacifica-collector
flyctl ssh console -a pacifica-collector -C 'df -i /app/data'

# Health check
./monitoring/check_collector.sh

# Quick sync (one-liner)
flyctl ssh console -a pacifica-collector -C 'tar -czf /tmp/sync.tar.gz -C /app/data .' && \
flyctl ssh sftp get "/tmp/sync.tar.gz" "./sync.tar.gz" -a pacifica-collector && \
tar -xzf ./sync.tar.gz -C ./data/ && rm sync.tar.gz

# Quick cleanup (one-liner - update date)
flyctl ssh console -a pacifica-collector -C 'rm -rf /app/data/*/date=2025-10-16'

# Check what dates exist
flyctl ssh console -a pacifica-collector -C 'ls /app/data/trades/symbol=BTC/'
```

---

## Summary

| Step | Command | Duration | Result |
|------|---------|----------|--------|
| Archive | `tar -czf` | 30-60s | ~10MB archive |
| Download | `sftp get` | 10-30s | Local archive |
| Extract | `tar -xzf` | 5-10s | Data in ./data/ |
| Cleanup Remote | `rm -rf` | 5s | 50%+ space freed |
| Verify | Health check | 2s | Confirm healthy |

**Total time:** ~2-3 minutes for complete sync & cleanup

---

## Related Documentation

- `DEPLOYMENT_GUIDE.md` - Initial Fly.io setup
- `TESTING_SYNC.md` - Advanced sync scenarios
- `collection-status.md` - Current collection status
- `monitoring/check_collector.sh` - Health check script

---

**Last Updated:** October 17, 2025  
**Maintainer:** Diego  
**Fly.io App:** pacifica-collector


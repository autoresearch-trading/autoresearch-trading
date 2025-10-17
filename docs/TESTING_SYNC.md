# Testing the Data Sync & Purge System

This guide helps you test the daily sync workflow and script.

## What the System Does

1. **Archives** data from Fly.io `/app/data` directory
2. **Downloads** the archive to a temporary location
3. **Uploads** to cloud storage (S3 or Cloudflare R2)
4. **Purges** old Parquet files (>2 days old) from Fly.io to save space
5. **Cleans up** temporary files

## Quick Test via GitHub Actions (Easiest)

### Prerequisites
Ensure these secrets are set in GitHub (Settings → Secrets and variables → Actions):
- `FLY_API_TOKEN`
- `STORAGE_ACCESS_KEY_ID`
- `STORAGE_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `STORAGE_ENDPOINT_URL` (for R2; leave empty for S3)

### Steps
1. Go to **Actions** tab in GitHub
2. Select **"Daily Data Sync & Purge"** workflow
3. Click **"Run workflow"** → Select branch → **"Run workflow"**
4. Watch the logs for any errors

### Expected Output
```
✅ Created temporary directory
📦 Archiving data on Fly.io instance...
✅ Remote archive created
⬇️ Downloading archive from Fly.io...
✅ Archive downloaded
☁️ Uploading data to bucket
✅ Upload complete
🗑️ Purging data older than 2 days
🔍 Dry run of purge command: (shows files to delete)
✅ Purge command executed
🧹 Local cleanup complete
🎉 Sync & Purge process finished successfully!
```

## Local Testing (Advanced)

### Prerequisites
```bash
# Install required tools
brew install flyctl awscli  # macOS
# or
sudo apt-get install awscli  # Linux

# Set environment variables
export FLY_API_TOKEN='your-fly-token'
export S3_BUCKET_NAME='your-bucket-name'
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export S3_ENDPOINT_URL='https://xxx.r2.cloudflarestorage.com'  # R2 only
```

### Run Test Script
```bash
./scripts/test_sync_local.sh
```

This will:
- ✅ Verify all required tools are installed
- ✅ Check environment variables are set
- ✅ Test Fly.io connection
- ✅ Test cloud storage connection
- 🚀 Run the actual sync script

## Dry Run Mode (Test Without Purging)

To test without actually deleting files, you can modify the script temporarily:

```bash
# Comment out the actual purge line (line 58) in sync_cloud_data.sh
# flyctl ssh console -q -a "$APP_NAME" -C "${PURGE_CMD}"
```

Or just observe the dry-run output (line 55) to see what would be deleted.

## Monitoring After Deployment

### Check Fly.io Disk Usage
```bash
flyctl status -a pacifica-collector
flyctl volumes list -a pacifica-collector
```

### Check Cloud Storage
```bash
# For R2
aws s3 ls s3://your-bucket/ --endpoint-url https://xxx.r2.cloudflarestorage.com --recursive

# For S3
aws s3 ls s3://your-bucket/ --recursive
```

### Check What Files Would Be Purged (Without Deleting)
```bash
flyctl ssh console -a pacifica-collector -C "find /app/data -type f -name '*.parquet' -mtime +1 -print"
```

## Troubleshooting

### Issue: "No space left on device" on Fly.io
**Solution**: Run the sync workflow immediately to purge old data

### Issue: "Permission denied" on cloud storage
**Solution**: Verify your `STORAGE_ACCESS_KEY_ID` and `STORAGE_SECRET_ACCESS_KEY`

### Issue: "Connection refused" to Fly.io
**Solution**: Verify `FLY_API_TOKEN` is valid: `flyctl auth token`

### Issue: Archive download is very slow
**Expected**: First run with lots of data may take 10-30 minutes

### Issue: Files not being purged
**Check**: The purge uses `-mtime +1` which means "modified >48 hours ago"
- Day 0 (today): Keep ✅
- Day 1 (yesterday): Keep ✅  
- Day 2+: Purge 🗑️

## Configuration

Edit `scripts/sync_cloud_data.sh` to customize:

```bash
DAYS_TO_KEEP_ON_FLY=2  # Keep today's and yesterday's data
```

## Schedule

The workflow runs automatically:
- **Every day at 2:00 AM UTC** (via cron)
- Can be triggered **manually** anytime via GitHub Actions UI

## Safety Features

1. **Dry-run first**: Shows what will be deleted before deleting
2. **Data backed up first**: Upload completes before purge starts
3. **Smart sync**: Only uploads changed/new files (via `aws s3 sync`)
4. **Preserves structure**: Maintains `symbol=X/date=Y` partitioning

## Next Steps

After successful testing:
- ✅ Monitor first automated run (2 AM UTC)
- ✅ Verify data in cloud storage
- ✅ Check Fly.io disk space reduction
- ✅ Set up alerts if needed (GitHub Actions can notify on failure)


#!/bin/bash
# Download parquet datasets from Cloudflare R2 (or any S3-compatible store) into a local directory.
#
# Usage:
#   export S3_BUCKET_NAME="your-bucket"
#   export S3_ENDPOINT_URL="https://<account>.<region>.r2.cloudflarestorage.com"  # optional for native S3
#   ./scripts/fetch_cloud_data.sh ./cloud-data
#
# The script mirrors s3://$S3_BUCKET_NAME into the target directory. Existing files are updated
# in-place and untouched files are left as-is.

set -euo pipefail

if [ -z "${S3_BUCKET_NAME:-}" ]; then
  echo "❌ S3_BUCKET_NAME is not set. Export it before running." >&2
  exit 1
fi

LOCAL_DIR=${1:-./cloud-data}
mkdir -p "${LOCAL_DIR}"

S3_ARGS=("--only-show-errors")
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
  S3_ARGS+=("--endpoint-url" "${S3_ENDPOINT_URL}")
fi

echo "⬇️  Syncing parquet datasets from s3://${S3_BUCKET_NAME} to ${LOCAL_DIR}"
aws s3 sync "s3://${S3_BUCKET_NAME}" "${LOCAL_DIR}" "${S3_ARGS[@]}"
echo "✅ Sync complete"

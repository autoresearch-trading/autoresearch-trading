#!/bin/bash
# Local testing script for sync_cloud_data.sh
# This script helps you test the sync script locally with proper environment setup

set -euo pipefail

# Check for required commands
if ! command -v flyctl &> /dev/null; then
    echo "❌ Error: flyctl is not installed. Install it from: https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "❌ Error: aws CLI is not installed. Install it with: brew install awscli"
    exit 1
fi

# Check for required environment variables
REQUIRED_VARS=("FLY_API_TOKEN" "S3_BUCKET_NAME" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Set them in your shell or create a .env.sync file:"
    echo "  export FLY_API_TOKEN='your-token'"
    echo "  export S3_BUCKET_NAME='your-bucket'"
    echo "  export AWS_ACCESS_KEY_ID='your-key'"
    echo "  export AWS_SECRET_ACCESS_KEY='your-secret'"
    echo "  export S3_ENDPOINT_URL='https://xxx.r2.cloudflarestorage.com'  # For R2 only"
    exit 1
fi

# Verify Fly.io connection
echo "🔍 Verifying Fly.io connection..."
if ! flyctl status -a pacifica-collector > /dev/null 2>&1; then
    echo "❌ Error: Cannot connect to Fly.io app 'pacifica-collector'"
    echo "   Check your FLY_API_TOKEN and app name"
    exit 1
fi
echo "✅ Fly.io connection verified"

# Verify AWS/R2 connection
echo "🔍 Verifying cloud storage connection..."
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
    aws s3 ls "s3://${S3_BUCKET_NAME}" --endpoint-url "${S3_ENDPOINT_URL}" > /dev/null 2>&1 || {
        echo "❌ Error: Cannot access R2 bucket '${S3_BUCKET_NAME}'"
        exit 1
    }
else
    aws s3 ls "s3://${S3_BUCKET_NAME}" > /dev/null 2>&1 || {
        echo "❌ Error: Cannot access S3 bucket '${S3_BUCKET_NAME}'"
        exit 1
    }
fi
echo "✅ Cloud storage connection verified"

# Run the sync script
echo ""
echo "🚀 All checks passed! Running sync script..."
echo "================================================"
echo ""

./scripts/sync_cloud_data.sh


#!/bin/bash
# Health check script - run locally to monitor cloud collector

APP_NAME="pacifica-collector"

echo "🔍 Checking Fly.io collector status..."

# Check if app is running
STATUS=$(flyctl status --app "$APP_NAME" --json 2>/dev/null | jq -r '.Allocations[0].Status' 2>/dev/null || echo "unknown")

if [ "$STATUS" != "running" ]; then
    echo "❌ Collector is $STATUS"
    # Send alert (email, Slack, etc.)
    exit 1
fi

echo "✅ Collector is running"

# Check health endpoint
echo "🏥 Checking health endpoint..."
HEALTH=$(flyctl ssh console --app "$APP_NAME" -C "curl -s localhost:8080/health 2>/dev/null" | jq -r '.status' 2>/dev/null || echo "unknown")

if [ "$HEALTH" != "healthy" ]; then
    echo "⚠️  Health check failed: $HEALTH"
    exit 1
fi

echo "✅ Health check passed"

# Check data freshness
echo "📊 Checking data freshness..."
RECENT_FILES=$(flyctl ssh console --app "$APP_NAME" -C "find /app/data/trades -name '*.parquet' -mmin -5 2>/dev/null | wc -l" 2>/dev/null || echo "0")

if [ "$RECENT_FILES" -lt 1 ]; then
    echo "⚠️  No recent data files (less than 5 minutes old)"
    exit 1
fi

echo "✅ Data is being written ($RECENT_FILES recent files)"

# Check disk space
echo "💾 Checking disk space..."
DISK_USAGE=$(flyctl ssh console --app "$APP_NAME" -C "df -h /app/data | tail -1 | awk '{print \$5}'" 2>/dev/null || echo "unknown")
echo "📊 Disk usage: $DISK_USAGE"

echo ""
echo "🎉 All checks passed!"


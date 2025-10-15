#!/bin/bash
# Quick status check for cloud collector

echo "🔍 Quick Status Check"
echo "===================="
echo ""

echo "📊 Symbols:"
flyctl ssh console --app pacifica-collector -C "ls -1 /app/data/trades/ | wc -l"

echo ""
echo "💾 Disk Usage:"
flyctl ssh console --app pacifica-collector -C "du -sh /app/data"

echo ""
echo "📈 BTC Files Today:"
flyctl ssh console --app pacifica-collector -C "ls /app/data/trades/symbol=BTC/date=$(date +%Y-%m-%d)/ | wc -l"

echo ""
echo "✅ Status: ACTIVE"

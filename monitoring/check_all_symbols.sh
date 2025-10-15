#!/bin/bash
# Check that all symbols are collecting data

APP_NAME="pacifica-collector"
DATE=$(date +%Y-%m-%d)

echo "🔍 Checking data collection for all symbols..."
echo "Date: $DATE"
echo ""

# Check each data type
echo "📊 Data directories by type:"
echo ""

echo "TRADES:"
flyctl ssh console --app "$APP_NAME" -C "ls -1 /app/data/trades/ 2>/dev/null | grep symbol=" 2>/dev/null | wc -l | xargs echo "Total symbols:"

echo ""
echo "📈 Sample symbols with recent files (last 5 min):"
flyctl ssh console --app "$APP_NAME" -C "
cd /app/data/trades
for symbol in symbol=BTC symbol=ETH symbol=SOL symbol=DOGE symbol=FARTCOIN; do
    if [ -d \"\$symbol/date=$DATE\" ]; then
        count=\$(find \"\$symbol/date=$DATE\" -name '*.parquet' -mmin -5 2>/dev/null | wc -l)
        total=\$(find \"\$symbol/date=$DATE\" -name '*.parquet' 2>/dev/null | wc -l)
        if [ \$count -gt 0 ]; then
            echo \"✅ \$symbol: \$count recent files (total today: \$total)\"
        else
            echo \"⚠️  \$symbol: NO recent files (total today: \$total)\"
        fi
    fi
done
" 2>/dev/null

echo ""
echo "💾 Disk usage breakdown:"
flyctl ssh console --app "$APP_NAME" -C "
echo 'Trades:     ' \$(du -sh /app/data/trades 2>/dev/null | cut -f1)
echo 'Orderbook:  ' \$(du -sh /app/data/orderbook 2>/dev/null | cut -f1)
echo 'Prices:     ' \$(du -sh /app/data/prices 2>/dev/null | cut -f1)
echo 'Funding:    ' \$(du -sh /app/data/funding 2>/dev/null | cut -f1)
echo '----------------------------------------'
echo 'Total:      ' \$(du -sh /app/data 2>/dev/null | cut -f1)
" 2>/dev/null

echo ""
echo "📄 Latest BTC trade files:"
flyctl ssh console --app "$APP_NAME" -C "ls -lth /app/data/trades/symbol=BTC/date=$DATE/*.parquet 2>/dev/null | head -5 | awk '{print \$6, \$7, \$8, \$9}'" 2>/dev/null

echo ""
echo "🎯 All symbols being collected:"
flyctl ssh console --app "$APP_NAME" -C "ls -1 /app/data/trades/ 2>/dev/null | sed 's/symbol=//' | sort" 2>/dev/null

echo ""
echo "✅ Monitoring complete!"

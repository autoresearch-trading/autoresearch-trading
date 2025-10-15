# ☁️ Complete Guide: Deploy Multi-Symbol Collector to Cloud

## Quick Start: Deploy to Fly.io in 10 Minutes

### Prerequisites
- Fly.io account (free): https://fly.io/app/sign-up
- Fly CLI installed
- Your data collector working locally

---

## 🚀 Step 1: Install Fly CLI

```bash
# macOS
brew install flyctl

# Linux
curl -L https://fly.io/install.sh | sh

# Windows
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Verify installation
flyctl version
```

---

## 🔐 Step 2: Login to Fly.io

```bash
# Sign up / login (opens browser)
flyctl auth login

# Verify you're logged in
flyctl auth whoami
```

---

## 🚀 Step 3: Deploy Your Collector

```bash
# 1. Create the app
flyctl apps create pacifica-collector

# 2. Create persistent volume for data (3GB free)
flyctl volumes create data_volume \
  --region sjc \
  --size 3 \
  --app pacifica-collector

# 3. Deploy!
flyctl deploy --app pacifica-collector

# This will:
# ✅ Build your Docker image
# ✅ Push to Fly.io registry
# ✅ Start the collector
# ✅ Mount the persistent volume
# ✅ Begin collecting data 24/7
```

---

## 📊 Step 4: Monitor Your Deployment

```bash
# View real-time logs
flyctl logs --app pacifica-collector

# Check status
flyctl status --app pacifica-collector

# View dashboard
flyctl dashboard --app pacifica-collector

# SSH into the machine
flyctl ssh console --app pacifica-collector

# Inside SSH: Check collected data
cd /app/data/trades
ls -lh symbol=*/date=*/
```

---

## 📥 Step 5: Sync Data to Your Local Machine

```bash
# Make sync script executable
chmod +x scripts/sync_cloud_data.sh

# Run sync (do this weekly or when you want to backtest)
./scripts/sync_cloud_data.sh

# Data will be downloaded to ./data/
```

### Automated Daily Sync (Optional)

```bash
# Add to crontab for daily 2 AM sync
crontab -e

# Add this line:
0 2 * * * cd /path/to/data-collector && ./scripts/sync_cloud_data.sh >> /tmp/fly-sync.log 2>&1
```

---

## 🔧 Step 6: Set Up Monitoring

```bash
# Make monitoring script executable
chmod +x monitoring/check_collector.sh

# Run health check
./monitoring/check_collector.sh

# Schedule checks every 15 minutes (optional)
crontab -e

# Add this line:
*/15 * * * * /path/to/data-collector/monitoring/check_collector.sh >> /tmp/collector-health.log 2>&1
```

---

## ⚙️ Configuration

### Update Environment Variables

```bash
# Change rate limits
flyctl secrets set MAX_RPS=4 --app pacifica-collector

# Change poll intervals
flyctl secrets set POLL_TRADES=2s --app pacifica-collector

# Apply changes (restart)
flyctl deploy --app pacifica-collector
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PACIFICA_NETWORK` | mainnet | Network to connect to |
| `MAX_RPS` | 3 | Max requests per second |
| `POLL_TRADES` | 1s | Trade polling interval |
| `POLL_ORDERBOOK` | 3s | Orderbook polling interval |
| `POLL_PRICES` | 2s | Price polling interval |
| `POLL_FUNDING` | 60s | Funding rate polling interval |
| `BOOK_DEPTH` | 25 | Orderbook depth levels |

---

## 💰 Cost & Resource Management

### Free Tier Limits
- ✅ **Machines**: 3 shared-cpu (using 1)
- ✅ **Storage**: 3GB persistent (monitored)
- ✅ **Bandwidth**: 160GB/month (~10GB actual)
- ✅ **RAM**: 512MB per machine

### Data Growth Estimation
```
Per Symbol Per Day:
- Trades: ~8-15 MB
- Orderbook: ~20-30 MB  
- Prices: ~2-5 MB
- Total: ~30-50 MB/day/symbol

8 symbols × 30 days = ~7-12 GB
```

### Managing Disk Space

```bash
# Check current usage
flyctl ssh console --app pacifica-collector
du -sh /app/data/*

# Option 1: Delete old data (keep last 7 days)
flyctl ssh console --app pacifica-collector
cd /app/data
find . -type d -name "date=2025-01-*" -exec rm -rf {} +

# Option 2: Upgrade volume (paid)
flyctl volumes extend data_volume --size 10 --app pacifica-collector
# Cost: $0.15/GB/month = $1.05/month for 7 extra GB

# Option 3: Reduce symbols (edit collect_all_symbols_cloud.py)
# Focus on BTC, ETH, SOL only
```

---

## 🔍 Troubleshooting

### Deployment Fails

```bash
# Check build logs
flyctl logs --app pacifica-collector

# Common fixes:
# 1. Ensure all dependencies in requirements.txt
# 2. Verify Dockerfile.cloud syntax
# 3. Check fly.toml configuration
```

### No Data Being Written

```bash
# SSH in and test API connection
flyctl ssh console --app pacifica-collector

# Test API manually
python3 -c "
from collector.config import APISettings
from collector.api_client import APIClient
settings = APISettings.from_env()
client = APIClient(settings)
print(client.get('/info'))
"

# Check logs for errors
flyctl logs --app pacifica-collector | grep -i error
```

### Out of Disk Space

```bash
# Check disk usage
flyctl ssh console --app pacifica-collector -C "df -h /app/data"

# Clear old data
flyctl ssh console --app pacifica-collector
rm -rf /app/data/*/date=2025-01-*

# Or extend volume
flyctl volumes extend data_volume --size 5 --app pacifica-collector
```

### App Keeps Restarting

```bash
# Check status
flyctl status --app pacifica-collector

# View logs
flyctl logs --app pacifica-collector

# Common causes:
# - Memory limit exceeded (upgrade to 1GB)
# - API rate limiting (reduce MAX_RPS)
# - Network issues (check PACIFICA_BASE_URL)

# Increase memory
flyctl scale memory 1024 --app pacifica-collector
```

---

## 🔄 Alternative Platforms

### Railway.app

```bash
# Install CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Configuration is in railway.toml
```

### Render.com

```bash
# Deploy via dashboard: https://render.com
# Configuration is in render.yaml
# Upload files and deploy
```

---

## 📋 Deployment Checklist

- [ ] Install Fly CLI: `brew install flyctl`
- [ ] Login: `flyctl auth login`
- [ ] Create app: `flyctl apps create pacifica-collector`
- [ ] Create volume: `flyctl volumes create data_volume --region sjc --size 3`
- [ ] Deploy: `flyctl deploy`
- [ ] Verify logs: `flyctl logs`
- [ ] Check status: `flyctl status`
- [ ] Wait 5 minutes, verify data: `flyctl ssh console`
- [ ] Set up sync: `chmod +x scripts/sync_cloud_data.sh`
- [ ] Set up monitoring: `chmod +x monitoring/check_collector.sh`
- [ ] Schedule health checks (optional)
- [ ] Let it run for 7-14 days
- [ ] Sync data weekly: `./scripts/sync_cloud_data.sh`

---

## 🎯 What's Next?

### Week 1: Monitor
```bash
flyctl logs --app pacifica-collector | tail -100
flyctl ssh console --app pacifica-collector -C "du -sh /app/data/trades"
```

### Week 2: First Sync & Test
```bash
./scripts/sync_cloud_data.sh
cd signal-engine
python3 scripts/run_signal_pipeline.py --symbols BTC --dry-run
```

### Week 3-4: Backtest
```bash
./scripts/sync_cloud_data.sh
cd signal-engine
python3 scripts/run_backtest.py --symbols BTC --days 14
```

---

## 🎉 Success!

You now have:
- ✅ 24/7 data collection for all symbols
- ✅ Free cloud hosting (within limits)
- ✅ Automated sync to local machine
- ✅ Health monitoring
- ✅ Production-ready tick data for backtesting

**Let it run for 7-14 days, then sync and backtest with real data!** 🚀


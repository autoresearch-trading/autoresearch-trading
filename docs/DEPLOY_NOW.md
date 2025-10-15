# 🚀 Deploy NOW - Quick Commands

## All files ready! Deploy in 10 minutes:

### 1️⃣ Install Fly CLI (2 min)
```bash
brew install flyctl
flyctl auth login
```

### 2️⃣ Create & Deploy (5 min)
```bash
# Create app
flyctl apps create pacifica-collector

# Create 3GB volume (free tier)
flyctl volumes create data_volume --region sjc --size 3 --app pacifica-collector

# Deploy!
flyctl deploy --app pacifica-collector
```

### 3️⃣ Verify (1 min)
```bash
# Watch logs
flyctl logs --app pacifica-collector

# Check status
flyctl status --app pacifica-collector

# After 5 minutes, check data
flyctl ssh console --app pacifica-collector
ls -lh /app/data/trades/symbol=BTC/
```

### 4️⃣ Setup Monitoring (2 min)
```bash
# Test health check
./monitoring/check_collector.sh

# Optional: Add to crontab for alerts
crontab -e
# Add: */15 * * * * /path/to/data-collector/monitoring/check_collector.sh
```

---

## 📥 Weekly Data Sync

```bash
# Sync cloud data to local
./scripts/sync_cloud_data.sh

# Data available in ./data/
```

---

## 🎯 What You Get

✅ **24/7 data collection** for ALL symbols  
✅ **Free hosting** (3GB storage, 512MB RAM)  
✅ **Automatic health checks**  
✅ **Easy local sync**  
✅ **Production-ready tick data**  

**In 7-14 days, you'll have real data for accurate backtesting!** 🎉

---

## 📖 Full Documentation

- **DEPLOYMENT_GUIDE.md** - Complete setup guide
- **fly.toml** - Fly.io configuration
- **Dockerfile.cloud** - Cloud-optimized Docker image
- **collect_all_symbols_cloud.py** - Cloud collector with health checks
- **scripts/sync_cloud_data.sh** - Sync data to local machine
- **monitoring/check_collector.sh** - Health monitoring script

---

## 🚨 Quick Troubleshooting

### Deployment fails?
```bash
flyctl logs --app pacifica-collector
# Check for missing dependencies or config errors
```

### No data after 5 minutes?
```bash
flyctl ssh console --app pacifica-collector
python3 -c "from collector.api_client import APIClient; from collector.config import APISettings; print(APIClient(APISettings.from_env()).get('/info'))"
```

### Out of space?
```bash
# Delete old data (keep last 7 days)
flyctl ssh console --app pacifica-collector
find /app/data -type d -name "date=2025-01-*" -exec rm -rf {} +
```

---

## 💡 Pro Tips

1. **Start with FREE tier** - 3GB is enough for 7-10 days of data
2. **Sync weekly** - Keep local backups, clear cloud storage
3. **Monitor daily** - Run health checks automatically
4. **Scale when needed** - Upgrade to 10GB for $1.05/month if needed

---

**Ready? Run the commands above and you're live in 10 minutes!** 🚀

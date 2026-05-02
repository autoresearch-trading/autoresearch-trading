# Always-on Pacifica full-fidelity collector on Fly

This is the always-on version of the local laptop collector.

Goal:

- run collection when the laptop is off;
- write raw full-fidelity chunks to a small Fly volume as a spool;
- upload and verify to R2 continuously;
- prune only local files that are already remote-verified;
- keep the laptop as a research client/cache, not the 24/7 collector.

## Files

- `ops/fly/pacifica-full-fidelity/Dockerfile`
- `ops/fly/pacifica-full-fidelity/entrypoint.sh`
- `ops/fly/pacifica-full-fidelity/fly.toml`

The container runs three process groups:

1. Foreground collector:
   - `scripts/collect_pacifica_full_fidelity.py`
   - output: `/data/pacifica_full_fidelity`
   - compact raw payload mode
   - free-space guard

2. Background lifecycle loop every 30 minutes:
   - scan local sealed files
   - upload data objects to R2
   - upload `.sha256` sidecars
   - verify remote size and hash sidecar
   - prune only verified local files when `PACIFICA_R2_PRUNE_EXECUTE=1`

3. Background ops-watchdog loop every hour, with per-operation due markers:
   - API/docs surface watcher: daily by default (`PACIFICA_API_SURFACE_WATCH_INTERVAL_S=86400`);
   - R2 inventory + raw-retention/cold-compaction policy planner: daily by default (`PACIFICA_R2_RETENTION_PLAN_INTERVAL_S=86400`);
   - non-destructive reports under `/data/ops`;
   - optional report upload to `r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest` when `PACIFICA_OPS_UPLOAD_REPORTS=1`.

## Storage model

Fly volume is a spool, not durable source of truth.

R2 path:

```text
r2:pacifica-trading-data/raw/pacifica/full_fidelity/...
```

Local Fly path:

```text
/data/pacifica_full_fidelity/...
```

Default retention on Fly is 1 day because the volume should stay small once R2 verification is working.

## R2 credentials

Do not commit credentials.

Set rclone's R2 remote via Fly secrets. Required secrets are usually:

```bash
fly secrets set \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID='...' \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY='...' \
  RCLONE_CONFIG_R2_ENDPOINT='https://<account-id>.r2.cloudflarestorage.com' \
  -a pacifica-full-fidelity
```

The non-secret rclone settings live in `fly.toml`:

```text
RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_PROVIDER=Cloudflare
RCLONE_CONFIG_R2_ACL=private
RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
```

## Create and deploy

Example:

```bash
fly apps create pacifica-full-fidelity
fly volumes create pacifica_full_fidelity_data --size 100 --region iad -a pacifica-full-fidelity
fly deploy -c ops/fly/pacifica-full-fidelity/fly.toml -a pacifica-full-fidelity
```

Use a volume size comfortably above:

```text
daily raw rate * retention days + free-space guard + upload backlog buffer
```

Fly's Miami region is not available in this account/region list. For Panama-adjacent operation use `iad` / Ashburn, Virginia as the default Americas region. With a 1-day retention and current rough 25 GiB/day rate, an 80 GiB volume is the minimum practical starting point. A 100-150 GiB volume is safer if budget allows; the current deployment template uses `iad` and a 100GB volume recommendation.

## Check status

```bash
fly status -a pacifica-full-fidelity
fly logs -a pacifica-full-fidelity
fly ssh console -a pacifica-full-fidelity -C 'df -h /data'
fly ssh console -a pacifica-full-fidelity -C 'sqlite3 /data/pacifica_full_fidelity_storage.sqlite "select status,count(*),sum(size_bytes) from archive_files group by status"'
fly ssh console -a pacifica-full-fidelity -C 'python scripts/run_pacifica_fly_ops_watchdogs.py --once'
fly ssh console -a pacifica-full-fidelity -C 'sed -n "1,120p" /data/ops/watchdogs/README.md'
```

## Important safety notes

- Do not use `rclone sync` from the Fly spool to the durable raw R2 prefix.
- The lifecycle code uses copy/upload semantics and only prunes after remote verification.
- If R2 upload fails, the collector disk guard stops collection before the volume reaches zero free space.
- The laptop can be off; this app keeps collecting as long as Fly is running and Pacifica/R2 are reachable.
- The local laptop collector should remain stopped unless intentionally used for smoke/debug collection.

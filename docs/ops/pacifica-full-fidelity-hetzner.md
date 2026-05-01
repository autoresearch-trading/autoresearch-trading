# Always-on Pacifica full-fidelity collector on Hetzner

Goal: run the Pacifica full-fidelity raw collector when the laptop is off, using Hetzner as a cheap always-on ingest host and Cloudflare R2 as durable archive.

## Architecture

Hetzner is a spool, not the durable archive.

```text
Pacifica websocket/REST
  -> Hetzner collector service
  -> /mnt/pacifica-spool/pacifica_full_fidelity/*.jsonl.gz
  -> lifecycle timer uploads + verifies to R2
  -> prune only local files that are remote-verified
  -> laptop later reads/rebuilds from R2 or compact silver
```

Durable R2 prefix:

```text
r2:pacifica-trading-data/raw/pacifica/full_fidelity/...
```

Recommended V1 Hetzner shape:

- small CAX/CX VPS;
- location: prefer `ash` / Ashburn, Virginia for Panama/US-market connectivity; Hetzner is not Europe-only and also offers North America locations;
- attached volume mounted at `/mnt/pacifica-spool`;
- total usable spool target: about 100GB;
- local retention: 2 days default, 3 days max;
- min free disk floor: 10GiB on Hetzner spool;
- R2 upload/verify lifecycle every 15 minutes;
- health check every 5 minutes.

At the current rough 25GiB/day raw rate:

- 2 days local raw = about 50GiB;
- 3 days local raw = about 75GiB;
- 100GB spool gives room for retention + upload backlog + free-space guard.

## Files

- `scripts/run_pacifica_full_fidelity_collector.sh`
- `scripts/run_pacifica_full_fidelity_r2_lifecycle.sh`
- `scripts/check_pacifica_full_fidelity_health.py`
- `ops/systemd/pacifica-full-fidelity-collector.service`
- `ops/systemd/pacifica-full-fidelity-r2-lifecycle.service`
- `ops/systemd/pacifica-full-fidelity-r2-lifecycle.timer`
- `ops/systemd/pacifica-full-fidelity-health.service`
- `ops/systemd/pacifica-full-fidelity-health.timer`
- `ops/hetzner/pacifica-full-fidelity.env.example`
- `ops/hetzner/bootstrap_pacifica_full_fidelity.sh`

## Provisioning outline

On the Hetzner host, use Ubuntu/Debian.

Recommended region/location for Diego in Panama:

```text
ash = Ashburn, Virginia, US
```

Hetzner also has European locations such as Falkenstein, Nuremberg, and Helsinki, but those are not the default recommendation here. For this collector, choose Ashburn unless a live latency test against Pacifica shows another location is materially better. The user's physical laptop location matters less than the path from the always-on collector to Pacifica/R2, but Ashburn is a better default than Europe for Panama-adjacent operations.

Create and mount the volume at:

```text
/mnt/pacifica-spool
```

Persist it in `/etc/fstab` by UUID. Example shape:

```bash
blkid
mkdir -p /mnt/pacifica-spool
# Add the UUID line to /etc/fstab, then:
mount -a
```

Do not put raw collector output on the root disk if avoidable.

## Install repo and services

Clone or copy the repo to:

```text
/opt/pacifica-full-fidelity
```

Then run:

```bash
sudo /opt/pacifica-full-fidelity/ops/hetzner/bootstrap_pacifica_full_fidelity.sh
```

The bootstrap script:

- installs `uv`, `rclone`, `sqlite3`, git/build basics;
- creates service user `pacifica`;
- creates `/mnt/pacifica-spool` and `/var/log/pacifica`;
- creates `/etc/pacifica-full-fidelity.env` from the template if missing;
- runs `uv sync --frozen --no-dev`;
- installs systemd units/timers;
- enables lifecycle and health timers, but does not start the collector until credentials are configured.

## Configure credentials

Edit:

```text
/etc/pacifica-full-fidelity.env
```

Set the R2 fields there. Do not print, commit, or paste real secret values into reports.

Required non-secret fields already in the template:

```text
RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_PROVIDER=Cloudflare
RCLONE_CONFIG_R2_ACL=private
RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
```

Required secret fields:

```text
RCLONE_CONFIG_R2_ACCESS_KEY_ID=[REDACTED]
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=[REDACTED]
RCLONE_CONFIG_R2_ENDPOINT=https://[REDACTED].r2.cloudflarestorage.com
```

## Start order

After credentials and mount are ready:

```bash
sudo systemctl daemon-reload
sudo systemctl start pacifica-full-fidelity-r2-lifecycle.service
sudo systemctl status pacifica-full-fidelity-r2-lifecycle.service --no-pager
```

Then start the collector:

```bash
sudo systemctl enable --now pacifica-full-fidelity-collector.service
sudo systemctl enable --now pacifica-full-fidelity-r2-lifecycle.timer
sudo systemctl enable --now pacifica-full-fidelity-health.timer
```

## Status checks

Collector logs:

```bash
journalctl -u pacifica-full-fidelity-collector.service -f
```

Lifecycle logs:

```bash
journalctl -u pacifica-full-fidelity-r2-lifecycle.service -n 100 --no-pager
```

Health logs:

```bash
journalctl -u pacifica-full-fidelity-health.service -n 100 --no-pager
```

Disk:

```bash
df -h /mnt/pacifica-spool
```

Lifecycle DB status:

```bash
sqlite3 /mnt/pacifica-spool/pacifica_full_fidelity_storage.sqlite \
  'select status,count(*),sum(size_bytes) from archive_files group by status;'
```

Latest local raw files:

```bash
find /mnt/pacifica-spool/pacifica_full_fidelity -name '*.jsonl.gz' -type f -printf '%T@ %p\n' \
  | sort -nr | head -20
```

R2 spot check:

```bash
rclone lsf r2:pacifica-trading-data/raw/pacifica/full_fidelity/ --max-depth 2 | head
```

## Unit behavior

### Collector service

`pacifica-full-fidelity-collector.service` runs:

```bash
scripts/run_pacifica_full_fidelity_collector.sh
```

That launches:

```bash
uv run python scripts/collect_pacifica_full_fidelity.py \
  --out-dir /mnt/pacifica-spool/pacifica_full_fidelity \
  --raw-payload-mode compact \
  --min-free-disk-gb 10
```

with additional settings from `/etc/pacifica-full-fidelity.env`.

### Lifecycle timer

`pacifica-full-fidelity-r2-lifecycle.timer` runs every 15 minutes:

```bash
scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
```

It performs:

1. scan local raw files into SQLite;
2. upload pending files to R2 with copy semantics;
3. upload `.sha256` sidecars;
4. verify remote size and sidecar hash;
5. prune only local files that are marked remote-verified and older than retention.

On Hetzner the template sets:

```text
PACIFICA_R2_PRUNE_EXECUTE=1
PACIFICA_FULL_FIDELITY_RETENTION_DAYS=2
```

because the Hetzner volume is intentionally a bounded spool.

### Health timer

`pacifica-full-fidelity-health.timer` runs every 5 minutes and prints JSON with:

- free disk;
- latest raw file age;
- lifecycle DB counts by status;
- unverified bytes;
- failure list.

## Safety rules

- Never use destructive `rclone sync` from local spool to the durable R2 prefix.
- Never prune files that are not remote-verified in the lifecycle DB.
- Do not store credentials in the repo.
- Do not rely on Hetzner volume as archive.
- If lifecycle upload fails, the collector disk floor should stop writes before the spool reaches zero free space.
- Keep laptop collector stopped unless doing explicit smoke/debug runs.

## Cost guardrails

Host cost is small. R2 retention becomes the growing cost.

At ~25GiB/day:

- 30 days full raw in R2: about 750GiB, roughly $12/month run-rate;
- 90 days: about 2.25TiB, roughly $36/month run-rate;
- 180 days: about 4.5TiB, roughly $72/month run-rate;
- 365 days: about 9.1TiB, roughly $147/month run-rate.

Retention decision points:

- local Hetzner prune-after-verified: day 1;
- first channel-volume review: after 7 days;
- first R2 raw retention decision: after 30 days;
- mandatory compression/retention policy: before 90 days.

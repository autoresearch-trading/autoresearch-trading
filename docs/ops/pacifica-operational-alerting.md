# Pacifica Operational Alerting Path

Updated: 2026-05-05

This describes the alert path for the `pacifica-full-fidelity` collector/lifecycle appliance. It is intentionally separate from research reports.

## Current state

- Fly app: `pacifica-full-fidelity`.
- Lifecycle DB: `/data/pacifica_full_fidelity_storage.sqlite`.
- Canonical remote archive: `r2:pacifica-trading-data/raw/pacifica/full_fidelity`.
- Fly-side watchdogs run from the app entrypoint and write JSON/Markdown status under `/data/ops` and R2 `ops/`.
- Hermes/manual checks currently provide human-visible status.

## Alert conditions

Trigger an alert if any of these are true:

1. Fly app is not `started`.
2. Collector newest raw file age exceeds 15 minutes.
3. Free disk falls below 50 GiB.
4. Lifecycle DB has rows with `error is not null` after one full repair cycle.
5. Upload failures are non-zero in the latest lifecycle log.
6. Verify failures keep recurring for current-hour/recent files rather than historical stable files.
7. R2 `raw/` top-level disappears or latest remote object age exceeds expected upload cadence.
8. Local prune-after-verified stops advancing for more than 24 hours while disk use rises.
9. Ops watchdog cannot write its status artifact.

## Non-alert noisy states

Do not page for these by themselves:

- one-off websocket reconnects that recover;
- stable-age skipped files;
- historical mismatch rows that clear on the next repair cycle;
- R2 inventory timeout if collector/lifecycle health is otherwise OK.

## Recommended implementation

Use a scheduled Hermes cron or external scheduler to run a read-only check every 2-6 hours and only deliver a message on failures or material state changes.

The check should collect:

```bash
fly status -a pacifica-full-fidelity
fly logs -a pacifica-full-fidelity --no-tail | tail -200
fly machine exec e2862502a76778 -a pacifica-full-fidelity --timeout 120 'sh -lc "sqlite3 /data/pacifica_full_fidelity_storage.sqlite \"select status,count(*),coalesce(sum(size_bytes),0) from archive_files group by status; select count(*) from archive_files where error is not null;\""'
rclone lsf r2:pacifica-trading-data --max-depth 1
```

Then classify as:

- `OK`: collector healthy, rows_with_errors zero, disk above guard, raw present.
- `WARN`: historical backlog/timeout/noisy but self-clearing.
- `PAGE`: collector stopped, disk below guard, raw missing, upload failures, or persistent errors.

## Delivery

Until a dedicated external channel is configured, deliver alerts into the active Hermes chat/session. If Diego wants push notifications, wire the same classification to Telegram/Discord/email using Hermes cron delivery or platform-specific tools.

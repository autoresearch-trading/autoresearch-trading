#!/bin/bash
# Always-on Pacifica full-fidelity collector + R2 lifecycle loop for Fly/VPS.
# Requires R2 credentials via rclone environment variables or mounted config.

set -euo pipefail

: "${PACIFICA_FULL_FIDELITY_ROOT:=/data/pacifica_full_fidelity}"
: "${PACIFICA_FULL_FIDELITY_STATE_DB:=/data/pacifica_full_fidelity_storage.sqlite}"
: "${PACIFICA_FULL_FIDELITY_R2_PREFIX:=raw/pacifica/full_fidelity}"
: "${PACIFICA_FULL_FIDELITY_REMOTE_BASE:=r2:pacifica-trading-data}"
: "${PACIFICA_FULL_FIDELITY_RETENTION_DAYS:=1}"
: "${PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S:=1800}"
: "${PACIFICA_FULL_FIDELITY_BATCH_LIMIT:=200}"
: "${PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB:=10}"
: "${PACIFICA_R2_PRUNE_EXECUTE:=1}"
: "${PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE:=compact}"

export PACIFICA_FULL_FIDELITY_ROOT
export PACIFICA_FULL_FIDELITY_STATE_DB
export PACIFICA_FULL_FIDELITY_R2_PREFIX
export PACIFICA_FULL_FIDELITY_REMOTE_BASE
export PACIFICA_FULL_FIDELITY_RETENTION_DAYS
export PACIFICA_FULL_FIDELITY_BATCH_LIMIT
export PACIFICA_R2_PRUNE_EXECUTE
export PACIFICA_USE_SYSTEM_PYTHON=1

mkdir -p "$PACIFICA_FULL_FIDELITY_ROOT" /data/logs

lifecycle_loop() {
  while true; do
    echo "[$(date -u +%FT%TZ)] lifecycle scan/upload/verify/prune start"
    if scripts/run_pacifica_full_fidelity_r2_lifecycle.sh; then
      echo "[$(date -u +%FT%TZ)] lifecycle complete"
    else
      echo "[$(date -u +%FT%TZ)] lifecycle failed; collector disk guard remains active" >&2
    fi
    python scripts/check_pacifica_full_fidelity_health.py \
      --root "$PACIFICA_FULL_FIDELITY_ROOT" \
      --state-db "$PACIFICA_FULL_FIDELITY_STATE_DB" \
      --min-free-gb "$PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB" \
      --max-newest-age-min 60 || true
    sleep "$PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S"
  done
}

lifecycle_loop &
LIFECYCLE_PID=$!

cleanup() {
  kill "$LIFECYCLE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec scripts/run_pacifica_full_fidelity_collector.sh

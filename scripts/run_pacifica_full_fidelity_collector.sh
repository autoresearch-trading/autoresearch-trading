#!/bin/bash
# Run the Pacifica full-fidelity collector with environment-driven defaults.
# Intended for always-on hosts such as Hetzner VPS/systemd.

set -euo pipefail

PYTHON_CMD=(uv run python)
if [ "${PACIFICA_USE_SYSTEM_PYTHON:-0}" = "1" ]; then
  PYTHON_CMD=(python)
fi

ROOT="${PACIFICA_FULL_FIDELITY_ROOT:-data/pacifica_full_fidelity}"
RAW_PAYLOAD_MODE="${PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE:-compact}"
MIN_FREE_DISK_GB="${PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB:-10}"
REST_INTERVAL_S="${PACIFICA_FULL_FIDELITY_REST_SNAPSHOT_INTERVAL_S:-60}"
SUBSCRIPTION_BATCH_SIZE="${PACIFICA_FULL_FIDELITY_SUBSCRIPTION_BATCH_SIZE:-50}"
SUBSCRIPTION_DELAY_S="${PACIFICA_FULL_FIDELITY_SUBSCRIPTION_DELAY_S:-0.25}"

args=(
  --out-dir "$ROOT"
  --raw-payload-mode "$RAW_PAYLOAD_MODE"
  --min-free-disk-gb "$MIN_FREE_DISK_GB"
  --rest-snapshot-interval-s "$REST_INTERVAL_S"
  --subscription-batch-size "$SUBSCRIPTION_BATCH_SIZE"
  --subscription-delay-s "$SUBSCRIPTION_DELAY_S"
)

if [ -n "${PACIFICA_FULL_FIDELITY_SYMBOLS:-}" ]; then
  args+=(--symbols "$PACIFICA_FULL_FIDELITY_SYMBOLS")
fi

if [ -n "${PACIFICA_FULL_FIDELITY_INTERVALS:-}" ]; then
  args+=(--intervals "$PACIFICA_FULL_FIDELITY_INTERVALS")
fi

if [ -n "${PACIFICA_FULL_FIDELITY_AGG_LEVELS:-}" ]; then
  args+=(--agg-levels "$PACIFICA_FULL_FIDELITY_AGG_LEVELS")
fi

if [ "${PACIFICA_FULL_FIDELITY_NO_PRICES:-0}" = "1" ]; then
  args+=(--no-prices)
fi

exec "${PYTHON_CMD[@]}" scripts/collect_pacifica_full_fidelity.py "${args[@]}"

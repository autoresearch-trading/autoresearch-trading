#!/bin/bash
# Bootstrap a Hetzner Ubuntu/Debian host for the Pacifica full-fidelity collector.
# Run on the Hetzner host after cloning the repo. This script does not create
# cloud resources and does not contain credentials.

set -euo pipefail

REPO_DIR="${PACIFICA_REPO_DIR:-/opt/pacifica-full-fidelity}"
ENV_FILE="${PACIFICA_ENV_FILE:-/etc/pacifica-full-fidelity.env}"
SPOOL_MOUNT="${PACIFICA_SPOOL_MOUNT:-/mnt/pacifica-spool}"
SERVICE_USER="${PACIFICA_SERVICE_USER:-pacifica}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root on the Hetzner host" >&2
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl git sqlite3 python3 python3-venv build-essential

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  install -m 0755 /root/.local/bin/uv /usr/local/bin/uv
fi

if ! command -v rclone >/dev/null 2>&1; then
  curl https://rclone.org/install.sh | bash
fi

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

mkdir -p "$SPOOL_MOUNT" /var/log/pacifica
mkdir -p "$SPOOL_MOUNT/.uv-cache"
chown -R "$SERVICE_USER:$SERVICE_USER" "$SPOOL_MOUNT" /var/log/pacifica

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Expected repo checkout at $REPO_DIR" >&2
  echo "Clone or copy the repo there first, then rerun." >&2
  exit 1
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR"

if [ ! -f "$ENV_FILE" ]; then
  install -m 0600 -o root -g root "$REPO_DIR/ops/hetzner/pacifica-full-fidelity.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from template. Edit R2 secrets before starting services." >&2
else
  chmod 0600 "$ENV_FILE"
fi

chmod 0755 "$REPO_DIR/scripts/run_pacifica_full_fidelity_collector.sh"
chmod 0755 "$REPO_DIR/scripts/run_pacifica_full_fidelity_r2_lifecycle.sh"
chmod 0755 "$REPO_DIR/scripts/check_pacifica_full_fidelity_health.py"

# Install deps as service user so uv cache/venv ownership is clean.
su -s /bin/bash "$SERVICE_USER" -c "cd '$REPO_DIR' && uv sync --frozen --no-dev"

install -m 0644 "$REPO_DIR/ops/systemd/pacifica-full-fidelity-collector.service" /etc/systemd/system/
install -m 0644 "$REPO_DIR/ops/systemd/pacifica-full-fidelity-r2-lifecycle.service" /etc/systemd/system/
install -m 0644 "$REPO_DIR/ops/systemd/pacifica-full-fidelity-r2-lifecycle.timer" /etc/systemd/system/
install -m 0644 "$REPO_DIR/ops/systemd/pacifica-full-fidelity-health.service" /etc/systemd/system/
install -m 0644 "$REPO_DIR/ops/systemd/pacifica-full-fidelity-health.timer" /etc/systemd/system/

systemctl daemon-reload
systemctl enable pacifica-full-fidelity-r2-lifecycle.timer pacifica-full-fidelity-health.timer

echo "Bootstrap complete. Next steps:" >&2
echo "1. Ensure the Hetzner volume is mounted at $SPOOL_MOUNT and persisted in /etc/fstab." >&2
echo "2. Edit $ENV_FILE and replace [REDACTED] R2 values." >&2
echo "3. Test lifecycle: systemctl start pacifica-full-fidelity-r2-lifecycle.service" >&2
echo "4. Start collector: systemctl enable --now pacifica-full-fidelity-collector.service" >&2

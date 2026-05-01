FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PACIFICA_USE_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && pip install --no-cache-dir requests websockets

WORKDIR /app

COPY scripts ./scripts
COPY ops/fly/pacifica-full-fidelity/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    scripts/run_pacifica_full_fidelity_collector.sh \
    scripts/run_pacifica_full_fidelity_r2_lifecycle.sh

ENTRYPOINT ["/entrypoint.sh"]

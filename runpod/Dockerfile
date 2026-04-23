# runpod/Dockerfile
# Base: CUDA 12.1 + cuDNN 8 + PyTorch 2.2.0 (matches requirements-runpod.txt)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace/repo

# Install rclone (needed by launch.sh to sync cache and push artifacts)
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends curl unzip && \
    curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && \
    unzip -q /tmp/rclone.zip -d /tmp && \
    mv /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && \
    chmod +x /usr/local/bin/rclone && \
    rm -rf /tmp/rclone* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps (no torch — already in base image)
COPY runpod/requirements-runpod.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

# Bake the repo source (big paths excluded via .dockerignore)
COPY tape/ tape/
COPY scripts/ scripts/
COPY runpod/ runpod/
COPY pyproject.toml pyproject.toml

# Install the local package (tape/) in editable mode so imports resolve cleanly
RUN pip install --no-deps -e .

# Bake the git commit SHA at build time so export_checkpoint.py can stamp provenance
# without needing a live .git directory.
# Usage: docker build --build-arg GIT_SHA=$(git rev-parse HEAD) ...
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

CMD ["bash", "runpod/launch.sh"]

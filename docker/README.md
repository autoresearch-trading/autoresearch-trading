# Docker Build Configurations

We maintain three Dockerfile variants:

## Dockerfile (x86_64)
Standard build for x86_64 architecture.
```bash
docker build -f Dockerfile -t pacifica-collector:latest .
```

## Dockerfile.arm64 (Apple Silicon)
Optimized for M1/M2/M3 Macs.
```bash
docker build -f Dockerfile.arm64 -t pacifica-collector:arm64 .
```

## Dockerfile.cloud (Production)
Minimal production image for cloud deployment.
```bash
docker build -f Dockerfile.cloud -t pacifica-collector:cloud .
```

## Why Multiple Dockerfiles?

- **Development needs** (Dockerfile): Includes debugging tools
- **Apple Silicon** (Dockerfile.arm64): Special TA-Lib build process
- **Production** (Dockerfile.cloud): Minimal layers, optimized size

## Building Multi-Architecture Images

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t pacifica-collector:multi \
  -f Dockerfile \
  --push .
```

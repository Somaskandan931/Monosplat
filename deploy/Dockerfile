# ── MonoSplat v2  ·  Dockerfile ──────────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder): install Python deps, compile nothing
#   Stage 2 (runtime): minimal image, copy only dist artifacts
#
# GPU support: run with --gpus all (requires nvidia-container-toolkit).
# CPU-only: works out of the box, training is capped at 1000 iterations.
#
# Usage:
#   docker build -t monosplat:latest .
#   docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data monosplat:latest

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg \
        colmap \
        libgl1 libglib2.0-0 \
        git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN pip install --upgrade pip

WORKDIR /app

# Install deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install gsplat (CUDA wheels published to PyPI)
RUN pip install gsplat --extra-index-url https://pypi.nvidia.com || \
    echo "[Docker] gsplat CUDA install failed — will fall back to CPU renderer"

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        ffmpeg \
        colmap \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/colmap_output \
             models/gaussian models/checkpoints \
             outputs/renders outputs/logs outputs/videos

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.pipeline.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# SafeAging Analytics Service
# Supports: linux/amd64 (dev), linux/arm64 (AI Box 5430 / 6490)
#
# Build (dev, amd64, CPU-only torch):
#   docker compose build
#
# Build for AI Box (arm64, default PyPI torch):
#   docker compose build --build-arg TORCH_INDEX_URL=

FROM python:3.11-slim

# System libraries: OpenCV headless + PyTorch + insightface (needs g++ for Cython ext)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (CPU-only) ────────────────────────────────────────────────────────
# amd64 dev: uses download.pytorch.org/whl/cpu  (~750 MB lighter than CUDA build)
# arm64 AI Box: set TORCH_INDEX_URL="" to fall back to default PyPI arm64 wheels
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ -n "${TORCH_INDEX_URL}" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url "${TORCH_INDEX_URL}"; \
    else \
        pip install --no-cache-dir torch torchvision; \
    fi

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY python/ ./python/

# ── S P2.1 – Pre-cache MobileNetV3-Small weights so container starts offline ──
RUN python - <<'EOF'
import torchvision.models as m
m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)
print("MobileNetV3-Small weights cached")
EOF

# ── Runtime directories ───────────────────────────────────────────────────────
# /app/models  → bind-mounted from ./models (calibration.json, YOLO .pt)
# /app/runtime → named volume for SQLite outbox + other edge state
RUN mkdir -p /app/models /app/runtime

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
# Strip Windows CRLF line endings (safe no-op if already LF)
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 18000

ENTRYPOINT ["/entrypoint.sh"]
